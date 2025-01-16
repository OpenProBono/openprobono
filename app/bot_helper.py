"""Helper functions for bot.py."""
from __future__ import annotations

import ast
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import Context, copy_context
from typing import TYPE_CHECKING

from langfuse.decorators import langfuse_context, observe
from openai.types.chat import ChatCompletionMessageToolCall

from app.chat_models import chat_str
from app.logger import setup_logger
from app.models import BotRequest, ChatModelParams, ChatRequest, EngineEnum
from app.moderation import moderate
from app.prompts import TITLE_CHAT_PROMPT
from app.search_tools import (
    find_search_tool,
    format_search_tool_results,
    run_search_tool,
    search_toolset_creator,
)
from app.vdb_tools import (
    find_vdb_tool,
    format_vdb_tool_results,
    run_vdb_tool,
    vdb_toolset_creator,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from anthropic.types.tool_use_block import ToolUseBlock
    from openai import Stream as OpenAIStream

logger = setup_logger()

def stream_openai_response(response: OpenAIStream):
    full_delta_dict_collection = []
    no_yield = False
    usage = None
    content = ""
    for chunk in response:
        if chunk.usage is not None:
            # last chunk contains usage, choices is an empty array
            usage = chunk.usage
            continue
        if(chunk.choices[0].delta.tool_calls or no_yield):
            no_yield = True
            full_delta_dict_collection.append(chunk.choices[0].delta.to_dict())
        else:
            chunk_content = chunk.choices[0].delta.content
            if(chunk_content):
                content += chunk_content
                if content.endswith("\n"):
                    yield {
                        "type": "response",
                        "content": content,
                    }
                    content = ""

    if content:
        yield {
            "type": "response",
            "content": content,
        }

    tool_calls, current_dict = [], {}
    if(no_yield):
        current_dict = full_delta_dict_collection[0]
        for i in range(1, len(full_delta_dict_collection)):
            merge_dicts_stream_openai_completion(current_dict, full_delta_dict_collection[i])

        tool_calls = [ChatCompletionMessageToolCall.model_validate(tool_call)
                    for tool_call in current_dict["tool_calls"]]
    return tool_calls, current_dict, usage


def merge_dicts_stream_openai_completion(dict1, dict2) -> None:
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                merge_dicts_stream_openai_completion(dict1[key], dict2[key])
            elif isinstance(dict1[key], list) and isinstance(dict2[key], list):
                for item in dict2[key]:
                    item_index= item["index"]
                    del item["index"]
                    if(item_index < len(dict1[key])):
                       if("index" in dict1[key][item_index]):
                           del dict1[key][item_index]["index"]
                       merge_dicts_stream_openai_completion(dict1[key][item_index], item)
                    else:
                        dict1[key].append(item)
            else:
                dict1[key] += dict2[key]
        else:
            dict1[key] = dict2[key]


def handle_empty_or_moderated(
    content: str,
    chat_model: ChatModelParams | None = None,
) -> str | None:
    """Handle empty or moderated messages.

    Parameters
    ----------
    content : str
        User message
    chat_model : ChatModelParams, optional
        The LLM to use for moderation, by default None

    Returns
    -------
    str | None
        Default response if content is empty or moderated

    """
    if content.strip() == "":
        return "Hi, how can I assist you today?"
    if moderate(content.strip(), chat_model):
        return (
            "I'm sorry, I can't help you with that. "
            "Please modify your message and try again."
        )
    return None


def setup_common(r: ChatRequest, bot: BotRequest) -> dict:
    """Set up for both streaming and non-streaming bot functions.

    Parameters
    ----------
    r : ChatRequest
        The chat session
    bot : BotRequest
        The bot

    Returns
    -------
    dict
        Keyword args for LLM API calls

    """
    toolset = search_toolset_creator(bot, r.bot_id)
    toolset += vdb_toolset_creator(bot, r.bot_id, r.session_id)
    kwargs = {"tools": toolset}
    # System prompt
    match bot.chat_model.engine:
        case EngineEnum.openai:
            system_prompt_msg = {"role": "system", "content": bot.system_prompt}
            if system_prompt_msg not in r.history:
                r.history.insert(0, system_prompt_msg)
        case EngineEnum.anthropic:
            kwargs["system"] = bot.system_prompt
    # Setup tracing
    last_user_msg = next(
        (m for m in r.history[::-1] if m["role"] == "user"),
        None,
    )
    langfuse_context.update_current_observation(input=last_user_msg["content"])
    langfuse_context.update_current_trace(metadata={"bot_id": r.bot_id} | kwargs)
    return kwargs


def execute_tool_call(
    tool_call: ChatCompletionMessageToolCall | ToolUseBlock,
    bot: BotRequest,
) -> tuple[str, str, list[dict]]:
    """Call a tool for a bot.

    Parameters
    ----------
    tool_call : ChatCompletionMessageToolCall | ToolUseBlock
        the tool call object, type depends on engine (OpenAI or Anthropic)
    bot : BotRequest
        the bot calling the tool

    Returns
    -------
    tuple[str, str, list[dict]]
        tool call id, tool response, formatted tool results

    """
    if isinstance(tool_call, ChatCompletionMessageToolCall):
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
    else: # ToolUseBlock (anthropic)
        function_name = tool_call.name
        function_args = tool_call.input
    logger.info("Tool %s Called With Args %s", function_name, function_args)
    vdb_tool = find_vdb_tool(bot, function_name)
    search_tool = find_search_tool(bot, function_name)
    # Step 3: call the function
    if vdb_tool:
        tool_response = run_vdb_tool(vdb_tool, function_args)
        formatted_results = format_vdb_tool_results(tool_response, vdb_tool)
    elif search_tool:
        tool_response = run_search_tool(search_tool, function_args)
        formatted_results = format_search_tool_results(
            tool_response,
            search_tool,
        )
    else:
        tool_response = "error: unable to run tool"
        logger.error("Tool %s encountered an error", function_name)
        formatted_results = []
    return tool_call.id, str(tool_response), formatted_results


def execute_tool_calls(
    tool_calls: list[ChatCompletionMessageToolCall],
    bot: BotRequest,
    messages: list,
    *, stream: bool,
) -> Generator[dict, None, list]:
    """Execute tool calls in parallel using ThreadPoolExecutor.

    Parameters
    ----------
    tool_calls : list[ChatCompletionMessageToolCall]
        The tools to run
    bot : BotRequest
        The bot running the tools
    messages : list
        The conversation history
    stream : bool
        if streaming is enabled, yields tool_call and tool_result messages

    Returns
    -------
    list
        The sources that were used in this set of tool calls

    Yields
    ------
    dict
        tool_call + tool_result messages for each tool (only if stream=True)

    """
    tool_id_name = {}
    with ThreadPoolExecutor() as executor:
        futures = []
        for tool_call in tool_calls:
            ctx = copy_context()
            function_name = tool_call.function.name
            tool_id_name[tool_call.id] = function_name
            if stream:
                yield {
                    "type": "tool_call",
                    "id": tool_call.id,
                    "name": function_name,
                    "args": tool_call.function.arguments,
                }

            def task(
                tc: ChatCompletionMessageToolCall = tool_call,
                context: Context = ctx,
            ) -> tuple[str, str, list[dict]]:
                return context.run(execute_tool_call, tc, bot)
            futures.append(executor.submit(task))

        sources = []
        for future in as_completed(futures):
            tool_call_id, tool_response, formatted_results = future.result()

            if stream:
                yield {
                    "type": "tool_result",
                    "id": tool_call_id,
                    "name": tool_id_name[tool_call_id],
                    "results": formatted_results,
                }
            sources += [str(res["id"]) for res in formatted_results]
            # extend conversation with function response
            messages.append({
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": tool_id_name[tool_call_id],
                "content": tool_response,
            })

        return sources


def get_sources(messages: list[dict]) -> list:
    """Extract the cumulative list of source IDs used in the conversation history.

    Parameters
    ----------
    messages : list[dict]
        The conversation history containing source list messages

    Returns
    -------
    list
        The list of source IDs extracted from `messages`

    """
    all_sources = []
    src_msgs = [
        msg for msg in messages
        if msg["role"] == "system" and msg["content"].startswith("**Sources**:\n")
    ]
    for src_msg in src_msgs:
        numbered_srcs = src_msg["content"].split("\n")[1:]
        srcs = [num_src.split(" ")[1] for num_src in numbered_srcs]
        all_sources += srcs
    return all_sources


def update_sources(
    messages: list[dict],
    current_sources: list,
    all_sources: list,
) -> None:
    """Add new sources from `current_sources` not in `all_sources` to `messages`.

    Parameters
    ----------
    messages : list[dict]
        The conversation history getting the updated source list message
    current_sources : list
        The sources used in the current round of tool calls
    all_sources : list
        All of the sources used up to the current round of tool calls

    """
    srcset = set(all_sources)
    current_sources = [
        src
        for src in current_sources
        if not (src in srcset or srcset.add(src))
    ]

    if current_sources:
        source_list = "\n".join([
            f"[{i}] {src}"
            for i, src in enumerate(current_sources, start=len(all_sources) + 1)
        ])
        all_sources.extend(current_sources)
        messages.append({"role": "system", "content": "**Sources**:\n" + source_list})


@observe(capture_input=False)
def title_chat(bot: BotRequest, message: str) -> str:
    """Title a chat for front end display.

    Parameters
    ----------
    bot : BotRequest
        BotRequest object, used to create the title
    message : str
        The users initial message

    Returns
    -------
    str
        A title for the chat

    """
    kwargs = {}
    conv_msg = {"role": "user", "content": message}
    match bot.chat_model.engine:
        case EngineEnum.openai:
            sys_msg = {"role": "system", "content": TITLE_CHAT_PROMPT}
            messages = [sys_msg, conv_msg]
        case EngineEnum.anthropic:
            kwargs["system"] = TITLE_CHAT_PROMPT
            messages = [conv_msg]
    return chat_str(messages, bot.chat_model, **kwargs)


def format_session_history(cr: ChatRequest, bot: BotRequest) -> list:
    """Format messages for front end display.

    Parameters
    ----------
        cr : ChatRequest
            Containing the conversation and session data
        bot : BotRequest
            To look up the tools used in the session

    Returns
    -------
        list
            The conversation history with formatted messages

    """
    history = []
    # session_data and get_source extension tools aren't in bot definition by default,
    # add them here
    _ = vdb_toolset_creator(bot, cr.bot_id, cr.session_id)
    match bot.chat_model.engine:
        case EngineEnum.openai:
            for msg in cr.history:
                if msg["role"] == "system": # ignore system prompts for front end display
                    continue
                if msg["role"] == "assistant" and "tool_calls" in msg: # tool call
                    history += [
                        {
                            "type": "tool_call",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "args": tool_call["function"]["arguments"],
                        }
                        for tool_call in msg["tool_calls"]
                    ]
                elif msg["role"] == "tool": # tool result
                    function_name = msg["name"]
                    tool_result = ast.literal_eval(msg["content"])
                    vdb_tool = find_vdb_tool(bot, function_name)
                    search_tool = find_search_tool(bot, function_name)
                    # Step 3: call the function
                    # Note: the JSON response may not always be valid;
                    # be sure to handle errors
                    if vdb_tool:
                        formatted_results = format_vdb_tool_results(tool_result, vdb_tool)
                    elif search_tool:
                        formatted_results = format_search_tool_results(
                            tool_result,
                            search_tool,
                        )
                    else:
                        formatted_results = []
                    history.append({
                        "type": "tool_result",
                        "id": msg["tool_call_id"],
                        "name": function_name,
                        "results": formatted_results,
                    })
                elif msg["role"] == "user": # user message
                    if msg["content"].startswith("file:"):
                        # user file upload
                        file_id = msg["content"][5:]
                        history += [
                            {"type": "file", "id": file_id},
                            {
                                "type": "file_upload_result",
                                "status": "Success",
                                "id": file_id,
                            },
                        ]
                    else:
                        history.append({"type": "user", "content": msg["content"]})
                elif msg["role"] == "assistant": # assistant response
                    history.append({"type": "response", "content": msg["content"]})
        case EngineEnum.anthropic:
            for msg in cr.history:
                if msg["role"] == "assistant": # assistant response or tool call
                    if isinstance(msg["content"], str): # assistant response
                        history.append({"type": "response", "content": msg["content"]})
                        continue
                    # content is a list of responses and/or tool calls
                    for content in msg["content"]:
                        if content["type"] == "text": # response text
                            history.append({"type": "response", "content": content["text"]})
                        elif content["type"] == "tool_use": # tool call
                            history.append({
                                "type": "tool_call",
                                "id": content["id"],
                                "name": content["name"],
                                "args": str(content["input"]),
                            })
                elif msg["role"] == "user": # user message (tool result, source list, or actual user message)
                    if isinstance(msg["content"], str): # source list or actual user message
                        if msg["content"].startswith("file:"):
                            # user file upload
                            file_id = msg["content"][5:]
                            history += [
                                {"type": "file", "id": file_id},
                                {
                                    "type": "file_upload_result",
                                    "status": "Success",
                                    "id": file_id,
                                },
                            ]
                        elif not msg["content"].startswith("**Sources**:\n"):
                            # ignore sources message
                            # append actual user message
                            history.append({"type": "user", "content": msg["content"]})
                        continue
                    for content in msg["content"]: # tool result
                        tool_call_id = content["tool_use_id"]
                        # find tool name from tool call message
                        tool_call_msg = next(
                            (
                                m for m in history
                                if m["type"] == "tool_call" and m["id"] == tool_call_id
                            ),
                            None,
                        )
                        if tool_call_msg is None:
                            logger.error("Format session history found a tool result without a tool call message in session %s", cr.session_id)
                            return []
                        function_name = tool_call_msg["name"]
                        tool_result = ast.literal_eval(content["content"])
                        vdb_tool = find_vdb_tool(bot, function_name)
                        search_tool = find_search_tool(bot, function_name)
                        # reformat tool results
                        if vdb_tool:
                            formatted_results = format_vdb_tool_results(tool_result, vdb_tool)
                        elif search_tool:
                            formatted_results = format_search_tool_results(
                                tool_result,
                                search_tool,
                            )
                        else:
                            formatted_results = []
                        history.append({
                            "type": "tool_result",
                            "id": tool_call_id,
                            "name": function_name,
                            "results": formatted_results,
                        })
    # add a done event signaling end of a response stream
    history.append({"type": "done"})
    return history
