import json
from queue import Queue
from typing import TYPE_CHECKING, Any

import langchain
from anthropic import Anthropic
from anyio.from_thread import start_blocking_portal
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import OpenAI

import chat_models
from milvusdb import session_source_summaries
from models import BotRequest, ChatRequest, get_uuid_id
from prompts import MAX_NUM_TOOLS, MULTIPLE_TOOLS_TEMPLATE
from search_tools import (
    search_anthropic_tool,
    search_openai_tool,
    search_toolset_creator,
)
from vdb_tools import (
    session_query_tool,
    vdb_anthropic_tool,
    vdb_openai_tool,
    vdb_toolset_creator,
)

if TYPE_CHECKING:
    from langchain_core.prompts import ChatPromptTemplate

langchain.debug = True

langfuse_handler = CallbackHandler()



# OPB bot main function
def opb_bot(r: ChatRequest, bot: BotRequest):
    class MyCallbackHandler(BaseCallbackHandler):
        def __init__(self, query):
            self.q = query

        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            self.q.put(token)

    if r.history[-1][0].strip() == "":  # TODO: make this a more full check to ensure that the query is relevant
        return "Hi, how can I assist you today?"

    q = Queue()
    job_done = object()

    bot_llm = ChatOpenAI(temperature=0.0, model=bot.model, request_timeout=60 * 5, streaming=True,
                            callbacks=[MyCallbackHandler(q)])
    # TODO: fix opb bot memory index
    chat_history = chat_models.messages(r.history[1:len(r.history) - 1], bot.engine)

    # memory_llm = ChatOpenAI(temperature=0.0, model='gpt-4-turbo-preview')
    # memory = ConversationSummaryBufferMemory(llm=memory_llm, max_token_limit=2000, memory_key="chat_history", return_messages=True)
    # for i in range(1, len(r.history)-1):
    #     memory.save_context({'input': r.history[i][0]}, {'output': r.history[i][1]})

    # ------- agent definition -------#
    toolset = search_toolset_creator(bot)
    toolset += vdb_toolset_creator(bot)
    source_summaries = session_source_summaries(r.session_id)
    if source_summaries:
        toolset.append(session_query_tool(r.session_id, source_summaries))
        # system_message += f'The session_query_tool sources have these summaries: {source_summaries}.' #this temporary change for testings
    if len(toolset) == 0:
        raise ValueError("toolset cannot be empty")

    prompt: ChatPromptTemplate = hub.pull("hwchase17/openai-tools-agent")
    prompt.messages[0].prompt.template = MULTIPLE_TOOLS_TEMPLATE
    agent = create_openai_tools_agent(bot_llm, toolset, prompt)

    async def task(p):
        # definition of llm used for bot
        p = bot.message_prompt + p
        # Create an agent executor by passing in the agent and tools
        agent_executor = AgentExecutor(agent=agent, tools=toolset, verbose=False, return_intermediate_steps=False)
        # TODO: make sure opb bot works
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

def openai_bot(r: ChatRequest, bot: BotRequest):
    if r.history[-1][0].strip() == "":
        return "Hi, how can I assist you today?"
    client = OpenAI()
    chatmodel = chat_models.ChatModelParams(bot.engine, bot.model)
    messages = chat_models.messages(r.history, bot.engine)
    messages.append({"role": "system", "content": MULTIPLE_TOOLS_TEMPLATE})
    trace_id = get_uuid_id()
    toolset = search_toolset_creator(bot)
    toolset += vdb_toolset_creator(bot)
    kwargs = {
        "client": client,
        "trace_id": trace_id,
        "tools": toolset,
        "tool_choice": "auto",  # auto is default, but we'll be explicit
        "session_id": r.session_id,
        "temperature": 0,
    }
    # response is a ChatCompletion object
    response = chat_models.chat(messages, chatmodel, **kwargs)
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        messages.append(response_message.model_dump(exclude={"function_call"}))
        tools_used = 0
        while tool_calls and tools_used < MAX_NUM_TOOLS:
            # TODO: run tool calls in parallel
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                vdb_tool = next(
                    (t for t in bot.vdb_tools if function_name == t.name),
                    None,
                )
                search_tool = next(
                    (t for t in bot.search_tools if function_name == t.name),
                    None,
                )
                # Step 3: call the function
                # Note: the JSON response may not always be valid;
                # be sure to handle errors
                if vdb_tool:
                    tool_response = vdb_openai_tool(vdb_tool, function_args)
                elif search_tool:
                    tool_response = search_openai_tool(search_tool, function_args)
                else:
                    tool_response = "error: unable to run tool"
                # Step 4: send the info for each function call and function response to
                # the model
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_response,
                    },
                )  # extend conversation with function response
                tools_used += 1
            # get a new response from the model where it can see the function response
            response = chat_models.chat(messages, chatmodel, **kwargs)
            response_message = response.choices[0].message
            messages.append(response_message)
            tool_calls = response_message.tool_calls
    return response_message.content

@observe()
def anthropic_bot(r: ChatRequest, bot: BotRequest):
    if r.history[-1][0].strip() == "":
        return "Hi, how can I assist you today?"
    messages = chat_models.messages(r.history, bot.engine)
    chatmodel = chat_models.ChatModelParams(bot.engine, bot.model)
    toolset = search_toolset_creator(bot)
    toolset += vdb_toolset_creator(bot)
    client = Anthropic()
    # Step 1: send the conversation and available functions to the model
    kwargs = {
        "tools": toolset,
        "client": client,
        "system": MULTIPLE_TOOLS_TEMPLATE,
        "temperature": 0,
    }
    langfuse_context.update_current_trace(session_id=r.session_id)
    # Anthropic system prompt does not go in messages list,
    # but thats where langfuse expects it, so insert it and add it to the observation
    langfuse_prompt_msg = [{"role": "system", "content": kwargs["system"]}]
    langfuse_context.update_current_observation(input=messages + langfuse_prompt_msg)
    response = chat_models.chat(messages, chatmodel, **kwargs)
    messages.append({"role": response.role, "content": response.content})
    tool_msgs = [msg for msg in response.content if msg.type == "tool_use"]
    # Step 2: check if the model wanted to call a function
    tools_used = 0
    while tool_msgs and tools_used < MAX_NUM_TOOLS:
        for tool_msg in tool_msgs:
            function_name = tool_msg.name
            vdb_tool = next(
                (t for t in bot.vdb_tools if function_name == t.name),
                None,
            )
            search_tool = next(
                (t for t in bot.search_tools if function_name == t.name),
                None,
            )
            tool_response_msg = {
                "type": "tool_result",
                "tool_use_id": tool_msg.id,
            }
            # Step 3: call the function
            if vdb_tool:
                tool_response = vdb_anthropic_tool(vdb_tool, tool_msg.input)
            elif search_tool:
                tool_response = search_anthropic_tool(search_tool, tool_msg.input)
            else:
                tool_response = "error: unable to identify tool"
                tool_response_msg["is_error"] = True
            tool_response_msg["content"] = tool_response
            # extend conversation with function response
            messages.append(
                {
                    "role": "user",
                    "content": [tool_response_msg],
                },
            )
            tools_used += 1
        # Step 4: send info for each function call and function response to the model
        # get a new response from the model where it can see the function response
        response = chat_models.chat(messages, chatmodel, **kwargs)
        messages.append({"role": response.role, "content": response.content})
        tool_msgs = [msg for msg in response.content if msg.type == "tool_use"]
    if isinstance(messages[-1]["content"], list):
        return "\n\n".join([msg.text for msg in messages[-1]["content"]])
    return messages[-1].content
