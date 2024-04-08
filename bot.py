"""Defines the bot engines. The meaty stuff."""
import json
from queue import Queue
from typing import Any

import langchain
from anyio.from_thread import start_blocking_portal
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from langfuse.openai import OpenAI

from milvusdb import session_source_summaries
from models import BotRequest, ChatRequest, get_uuid_id
from prompts import COMBINE_TOOL_OUTPUTS_TEMPLATE
from search_tools import search_openai_tool, search_toolset_creator
from vdb_tools import session_query_tool, vdb_openai_tool, vdb_toolset_creator

langchain.debug = True

langfuse_handler = CallbackHandler()

max_num_tools = 8

# OPB bot main function
def opb_bot(r: ChatRequest, bot: BotRequest) -> str:
    """Call bot using langchain engine.

    Parameters
    ----------
    r : ChatRequest
        ChatRequest object, containing the conversation and session data
    bot : BotRequest
        BotRequest object, containing the bot data

    Returns
    -------
    str
        The response from the bot

    Raises
    ------
    ValueError
        toolset cannot be empty, needs at least one tool defined

    """
    class MyCallbackHandler(BaseCallbackHandler):
        def __init__(self, query):
            self.q = query

        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            self.q.put(token)

    if r.history[-1][0].strip() == "":  # TODO: make this a more full check to ensure that the query is relevant
        return "Hi, how can I assist you today?"

    q = Queue()
    job_done = object()

    bot_llm = ChatOpenAI(temperature=0.0, model="gpt-4-turbo-preview", request_timeout=60 * 5, streaming=True,
                            callbacks=[MyCallbackHandler(q)])
    # TODO: fix opb bot memory index
    chat_history = []
    for tup in r.history[1:len(r.history) - 1]:
        if tup[0]:
            chat_history.append(HumanMessage(content=tup[0]))
        if tup[1]:
            chat_history.append(AIMessage(content=tup[1]))

    # memory_llm = ChatOpenAI(temperature=0.0, model='gpt-4-turbo-preview')
    # memory = ConversationSummaryBufferMemory(llm=memory_llm, max_token_limit=2000, memory_key="chat_history", return_messages=True)
    # for i in range(1, len(r.history)-1):
    #     memory.save_context({'input': r.history[i][0]}, {'output': r.history[i][1]})

    # ------- agent definition -------#
    toolset = []
    toolset += search_toolset_creator(bot)
    toolset += vdb_toolset_creator(bot)
    source_summaries = session_source_summaries(r.session_id)
    if source_summaries:
        toolset.append(session_query_tool(r.session_id, source_summaries))
        # system_message += f'The session_query_tool sources have these summaries: {source_summaries}.' #this temporary change for testings
    if len(toolset) == 0:
        error_description = "toolset cannot be empty, needs at least one tool defined"
        raise ValueError(error_description)

    prompt: ChatPromptTemplate = hub.pull("hwchase17/openai-tools-agent")
    prompt.messages[0].prompt.template = COMBINE_TOOL_OUTPUTS_TEMPLATE
    agent = create_openai_tools_agent(bot_llm, toolset, prompt)

    async def task(p):
        # definition of llm used for bot
        p = bot.message_prompt + p
        # Create an agent executor by passing in the agent and tools
        agent_executor = AgentExecutor(agent=agent, tools=toolset, verbose=False, return_intermediate_steps=False)
        ret = await agent_executor.ainvoke({"input": p, "chat_history": chat_history},
                                            config={"callbacks": [langfuse_handler]})
        q.put(job_done)
        return ret

    with start_blocking_portal() as portal:
        portal.start_task_soon(task, r.history[-1][0])
        content = ""
        while True:
            next_token = q.get(True)
            if next_token is job_done:
                return content
            content += next_token


def openai_bot(r: ChatRequest, bot: BotRequest) -> str:
    """Call bot using openai engine.

    Parameters
    ----------
    r : ChatRequest
        ChatRequest object, containing the conversation and session data
    bot : BotRequest
        BotRequest object, containing the bot data

    Returns
    -------
    str
        The response from the bot

    """
    if r.history[-1][0].strip() == "":
        return "Hi, how can I assist you today?"
    client = OpenAI()
    model = "gpt-3.5-turbo-0125"
    messages = []
    for tup in r.history:
        if tup[0]:
            messages.append({"role": "user", "content": tup[0]})
        if tup[1]:
            messages.append({"role": "assistant", "content": tup[1]})

    messages.append({"role": "system", "content": COMBINE_TOOL_OUTPUTS_TEMPLATE})
    trace_id = get_uuid_id()
    toolset = []
    toolset += search_toolset_creator(bot)
    toolset += vdb_toolset_creator(bot)

    # Step 1: send the conversation and available functions to the model
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=toolset,
        tool_choice="auto",  # auto is default, but we'll be explicit
        temperature=0,
        trace_id=trace_id,
        session_id=r.session_id,
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        messages.append(response_message.dict(exclude={"function_call"}))
        tools_used = 0
        while tool_calls and tools_used < max_num_tools:
            # TODO: run tool calls in parallel
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                vdb_tool = next((t for t in bot.vdb_tools if function_name == t["name"]), None)
                search_tool = next((t for t in bot.search_tools if function_name == t["name"]), None)
                # Step 3: call the function
                # Note: the JSON response may not always be valid; be sure to handle errors
                if vdb_tool:
                    tool_response = vdb_openai_tool(vdb_tool, function_args)
                elif search_tool:
                    tool_response = search_openai_tool(search_tool, function_args)
                else:
                    tool_response = "error: unable to run tool"
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_response,
                    },
                )  # extend conversation with function response
                tools_used += 1
            # Step 4: send the info for each function call and function response to the model
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=toolset,
                temperature=0,
                trace_id=trace_id,
                session_id=r.session_id,
            )  # get a new response from the model where it can see the function response
            response_message = response.choices[0].message
            messages.append(response_message)
            tool_calls = response_message.tool_calls
    else:
        print("no tool used")
    return response_message.content
