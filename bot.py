"""Defines the bot engines. The meaty stuff."""
import json
from queue import Queue
from typing import TYPE_CHECKING, Any

import langchain
from anthropic import Anthropic
from anthropic.types.beta.tools import ToolsBetaContentBlock
from anyio.from_thread import start_blocking_portal
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall

import chat_models
from milvusdb import session_source_summaries
from models import BotRequest, ChatRequest
from prompts import (
    COMBINE_TOOL_OUTPUTS_TEMPLATE,
    MAX_NUM_TOOLS,
    MULTIPLE_TOOLS_PROMPT,
    OPB_BOT_PROMPT,
)
from search_tools import (
    run_search_tool,
    search_toolset_creator,
)
from vdb_tools import (
    run_vdb_tool,
    session_query_tool,
    vdb_toolset_creator,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion


langchain.debug = True


# OPB bot main function
@observe(capture_input=False)
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
    if chat_models.moderate(r.history[-1][0].strip()):
        return (
            "I'm sorry, I can't help you with that. "
            "Please modify your message and try again."
        )

    q = Queue()
    job_done = object()

    bot_llm = ChatOpenAI(temperature=0.0, model=bot.chat_model.model, request_timeout=60 * 5, streaming=True,
                            callbacks=[MyCallbackHandler(q)])
    # TODO: fix opb bot memory index
    chat_history = chat_models.messages(r.history[1:len(r.history) - 1], bot.chat_model.engine)

    # memory_llm = ChatOpenAI(temperature=0.0, model='gpt-4-turbo-preview')
    # memory = ConversationSummaryBufferMemory(llm=memory_llm, max_token_limit=2000, memory_key="chat_history", return_messages=True)
    # for i in range(1, len(r.history)-1):
    #     memory.save_context({'input': r.history[i][0]}, {'output': r.history[i][1]})

    # ------- agent definition -------#
    toolset = search_toolset_creator(bot) + vdb_toolset_creator(bot)
    source_summaries = session_source_summaries(r.session_id)
    if source_summaries:
        toolset.append(session_query_tool(r.session_id, source_summaries))
        # system_message += f'The session_query_tool sources have these summaries: {source_summaries}.' #this temporary change for testings
    if len(toolset) == 0:
        error_description = "toolset cannot be empty, needs at least one tool defined"
        raise ValueError(error_description)

    agent = create_openai_tools_agent(bot_llm, toolset, OPB_BOT_PROMPT)

    # tracing
    langfuse_context.update_current_observation(input=r.history[-1][0])
    langfuse_context.update_current_trace(
        session_id=r.session_id,
        metadata={"bot_id": r.bot_id, "engine": bot.chat_model.engine},
    )

    async def task(p):
        # definition of llm used for bot
        p = bot.message_prompt + p
        # Create an agent executor by passing in the agent and tools
        agent_executor = AgentExecutor(agent=agent, tools=toolset, verbose=False, return_intermediate_steps=False)
        # TODO: make sure opb bot works
        ret = await agent_executor.ainvoke({"input": p, "chat_history": chat_history})
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

@observe(capture_input=False)
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
    if chat_models.moderate(r.history[-1][0].strip()):
        return (
            "I'm sorry, I can't help you with that. "
            "Please modify your message and try again."
        )

    client = OpenAI()
    messages = chat_models.messages(r.history, bot.chat_model.engine)
    messages.append({"role": "system", "content": COMBINE_TOOL_OUTPUTS_TEMPLATE})
    toolset = search_toolset_creator(bot) + vdb_toolset_creator(bot)
    kwargs = {
        "client": client,
        "trace_id": langfuse_context.get_current_trace_id(),
        "tools": toolset,
        "tool_choice": "auto",  # auto is default, but we'll be explicit
        "session_id": r.session_id,
        "temperature": 0,
    }

    # langfuse
    langfuse_context.update_current_observation(input=r.history[-1][0])
    langfuse_context.update_current_trace(
        session_id=r.session_id,
        metadata={"bot_id": r.bot_id, "engine": bot.chat_model.engine},
    )

    # response is a ChatCompletion object
    response: ChatCompletion = chat_models.chat(messages, bot.chat_model, **kwargs)
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        messages.append(response_message.model_dump(exclude={"function_call"}))
        response_message = openai_tools(messages, tool_calls, bot, **kwargs)
    return response_message.content

def openai_tools(
    messages: list[dict],
    tool_calls: list[ChatCompletionMessageToolCall],
    bot: BotRequest,
    **kwargs: dict,
):
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
                tool_response = run_vdb_tool(
                    vdb_tool,
                    function_args,
                    bot.chat_model.engine,
                )
            elif search_tool:
                tool_response = run_search_tool(
                    search_tool,
                    function_args,
                    bot.chat_model.engine,
                )
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
        response = chat_models.chat(messages, bot.chat_model, **kwargs)
        response_message = response.choices[0].message
        messages.append(response_message)
        tool_calls = response_message.tool_calls
    return response_message

@observe(capture_input=False)
def anthropic_bot(r: ChatRequest, bot: BotRequest):
    if r.history[-1][0].strip() == "":
        return "Hi, how can I assist you today?"
    if chat_models.moderate(r.history[-1][0].strip(), bot.chat_model):
        return (
            "I'm sorry, I can't help you with that. "
            "Please modify your message and try again."
        )
    messages = chat_models.messages(r.history, bot.chat_model.engine)
    toolset = search_toolset_creator(bot) + vdb_toolset_creator(bot)
    client = Anthropic()
    # Step 1: send the conversation and available functions to the model
    kwargs = {
        "tools": toolset,
        "client": client,
        "system": MULTIPLE_TOOLS_PROMPT,
        "temperature": 0,
    }

    # Step 1.5: tracing
    # Anthropic system prompt does not go in messages list, so add it to the input
    langfuse_prompt_msg = [{"role": "system", "content": kwargs["system"]}]
    langfuse_context.update_current_observation(input={
        "input": messages[-1]["content"],
        "prompt": langfuse_prompt_msg,
    })
    langfuse_context.update_current_trace(
        session_id=r.session_id,
        metadata={"bot_id": r.bot_id, "engine": bot.chat_model.engine},
    )

    response = chat_models.chat(messages, bot.chat_model, **kwargs)
    messages.append({"role": response.role, "content": response.content})
    # Step 2: check if the model wanted to call a function
    tool_calls: list[ToolsBetaContentBlock] = [
        msg for msg in response.content if msg.type == "tool_use"
    ]
    return anthropic_tools(messages, tool_calls, bot, **kwargs)

def anthropic_tools(
    messages: list[dict],
    tool_calls: list[ToolsBetaContentBlock],
    bot: BotRequest,
    **kwargs: dict,
):
    tools_used = 0
    while tool_calls and tools_used < MAX_NUM_TOOLS:
        for tool_call in tool_calls:
            function_name = tool_call.name
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
                "tool_use_id": tool_call.id,
            }
            # Step 3: call the function
            if vdb_tool:
                tool_response = run_vdb_tool(
                    vdb_tool,
                    tool_call.input,
                    bot.chat_model.engine,
                )
            elif search_tool:
                tool_response = run_search_tool(
                    search_tool,
                    tool_call.input,
                    bot.chat_model.engine,
                )
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
        response = chat_models.chat(messages, bot.chat_model, **kwargs)
        messages.append({"role": response.role, "content": response.content})
        tool_calls = [msg for msg in response.content if msg.type == "tool_use"]
    content: list[ToolsBetaContentBlock] = messages[-1]["content"]
    return content[-1].text
