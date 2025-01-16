"""Defines the bot engines. The meaty stuff."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import openai
from anthropic.types.tool_use_block import ToolUseBlock
from langfuse.decorators import langfuse_context, observe

from app.bot_helper import (
    execute_tool_call,
    execute_tool_calls,
    get_sources,
    handle_empty_or_moderated,
    setup_common,
    stream_openai_response,
    update_sources,
)
from app.chat_models import chat, chat_stream
from app.logger import setup_logger

if TYPE_CHECKING:
    from collections.abc import Generator

    from anthropic import Stream as AnthropicStream
    from anthropic.types import Message as AnthropicMessage
    from openai import Stream as OpenAIStream
    from openai.types.chat import ChatCompletionMessage

    from app.models import BotRequest, ChatRequest

openai.log = "debug"
MAX_NUM_TOOLS = 8

logger = setup_logger()

@observe(capture_input=False)
def openai_bot_stream(r: ChatRequest, bot: BotRequest) -> Generator[dict, None, None]:
    """Call streaming bot using openai engine.

    Parameters
    ----------
    r : ChatRequest
        ChatRequest object, containing the conversation and session data
    bot : BotRequest
        BotRequest object, containing the bot data


    Yields
    ------
    dict
        The response chunks from the bot

    """
    initial_response = handle_empty_or_moderated(r.history[-1]["content"])
    if initial_response:
        yield {
            "type": "response",
            "content": initial_response,
        }
        return
    kwargs = setup_common(r, bot)
    response = chat_stream(r.history, bot.chat_model, **kwargs)
    yield from openai_tools_stream(r.history, response, bot, **kwargs)


@observe(capture_input=False, capture_output=False, as_type="generation")
def openai_tools_stream(
    messages: list[dict],
    response: OpenAIStream,
    bot: BotRequest,
    **kwargs: dict,
) -> Generator[dict, None, None]:
    """Handle tool calls in the conversation for OpenAI engine.

    Parameters
    ----------
    messages : list[dict]
        The conversation messages
    response : OpenAIStream
        The initial response stream
    bot : BotRequest
        BotRequest object
    kwargs : dict
        Keyword arguments for the LLM.

    Yields
    ------
    dict
        The response chunks from the OpenAI LLM

    """
    tools_used = 0
    usage_dict = {"input": 0, "output": 0, "total": 0}
    all_sources = get_sources(messages)

    while tools_used < MAX_NUM_TOOLS:
        tool_calls, current_dict, usage = yield from stream_openai_response(response)
        if usage:
            usage_dict["input"] += usage.prompt_tokens
            usage_dict["output"] += usage.completion_tokens
            usage_dict["total"] += usage.total_tokens

        if not tool_calls:
            break

        #sometimes doesnt capture the role with streaming + multiple tool calls
        if "role" not in current_dict:
            current_dict["role"] = "assistant"
        if "content" not in current_dict:
            current_dict["content"] = None

        messages.append(current_dict)

        current_sources = yield from execute_tool_calls(
            tool_calls,
            bot,
            messages,
            stream=True,
        )
        tools_used += len(tool_calls)

        update_sources(messages, current_sources, all_sources)
        response = chat_stream(messages, bot.chat_model, **kwargs)
    # add usage to tracing after all tools are called
    langfuse_context.update_current_observation(
        model=bot.chat_model.model,
        usage=usage_dict,
    )


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
    initial_response = handle_empty_or_moderated(r.history[-1]["content"])
    if initial_response:
        return initial_response
    kwargs = setup_common(r, bot)
    response = chat(r.history, bot.chat_model, **kwargs)
    response_message = response.choices[0].message
    return openai_tools(r.history, response_message, bot, **kwargs)


def openai_tools(
    messages: list[dict],
    response_message: ChatCompletionMessage,
    bot: BotRequest,
    **kwargs: dict,
) -> str:
    """Handle tool calls in the conversation for OpenAI engine.

    Parameters
    ----------
    messages : list[dict]
        The conversation messages
    response_message : ChatCompletionMessage
        The initial response message
    bot : BotRequest
        BotRequest object
    kwargs : dict
        Keyword arguments for the LLM.

    Returns
    -------
    str
        The response message from the bot, should be final response.

    """
    if not response_message.tool_calls:
        return response_message.content

    tools_used = 0
    all_sources = get_sources(messages)
    while response_message.tool_calls and tools_used < MAX_NUM_TOOLS:
        messages.append(response_message.model_dump())
        new_sources = execute_tool_calls(
            response_message.tool_calls,
            bot,
            messages,
            stream=False,
        )
        tools_used += len(response_message.tool_calls)

        update_sources(messages, new_sources, all_sources)

        response = chat(messages, bot.chat_model, **kwargs)
        response_message = response.choices[0].message

    return response_message.content


@observe(capture_input=False)
def anthropic_bot(r: ChatRequest, bot: BotRequest) -> str:
    """Call bot using Anthropic engine.

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
    current_msg = r.history[-1]["content"].strip()
    initial_response = handle_empty_or_moderated(current_msg, bot.chat_model)
    if initial_response:
        return initial_response
    kwargs = setup_common(r, bot)
    response = chat(r.history, bot.chat_model, **kwargs)
    return anthropic_tools(r.history, response, bot, **kwargs)


def anthropic_tools(
    messages: list[dict],
    response: AnthropicMessage,
    bot: BotRequest,
    **kwargs: dict,
) -> str:
    """Handle tool calls in the conversation for Anthropic engine.

    Parameters
    ----------
    messages : list[dict]
        The conversation messages
    response : AnthropicMessage
        The initial response
    bot : BotRequest
        BotRequest object
    kwargs : dict
        Keyword arguments for the LLM.

    Returns
    -------
    str
        The response message from the bot, should be final response.

    """
    # Step 2: check if the model wanted to call a function
    tool_calls = [
        msg for msg in response.content if msg.type == "tool_use"
    ]
    if not tool_calls:
        return "\n".join([
            block.text for block in response.content if block.type == "text"
        ])

    tools_used = 0
    # get the bot's source list up to this point in the conversation
    all_sources = []
    src_msgs = [
        msg for msg in messages
        if msg["role"] == "user" and \
        isinstance(msg["content"], str) and msg["content"].startswith("**Sources**:\n")
    ]
    for src_msg in src_msgs:
        numbered_srcs = src_msg["content"].split("\n")[1:] # ignore first line
        srcs = [num_src.split(" ")[1] for num_src in numbered_srcs]
        all_sources += srcs
    while tool_calls and tools_used < MAX_NUM_TOOLS:
        new_sources = []
        # add message before tool calls so the last message isn't duplicated in db
        messages.append({"role": response.role, "content": response.content})
        for tool_call in tool_calls:
            tool_call_id, tool_response, formatted_results = execute_tool_call(
                tool_call,
                bot,
            )
            # add sources from this tool call to the overall source list
            new_sources += [str(res["id"]) for res in formatted_results]
            tool_response_msg = {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
            }
            if tool_response == "error: unable to run tool":
                tool_response_msg["is_error"] = True
            tool_response_msg["content"] = str(tool_response)
            # extend conversation with function response
            messages.append({
                "role": "user",
                "content": [tool_response_msg],
            })
            tools_used += 1
        # remove duplicate sources while maintaining original order
        srcset = set(all_sources)
        new_sources = [
            src
            for src in new_sources
            if not (src in srcset or srcset.add(src))
        ]
        # only add the new sources to the bots source list
        source_list = "\n".join([
            f"[{i}] {src}"
            for i, src in enumerate(new_sources, start=len(all_sources) + 1)
        ])
        all_sources += new_sources
        # append the source list as a system message
        if source_list:
            messages.append({"role": "user", "content": "**Sources**:\n" + source_list})
        # Step 4: send info for each function call and function response to the model
        # get a new response from the model where it can see the function response
        response = chat(messages, bot.chat_model, **kwargs)
        tool_calls = [msg for msg in response.content if msg.type == "tool_use"]
    return "\n".join([
        block.text for block in response.content if block.type == "text"
    ])


@observe(capture_input=False)
def anthropic_bot_stream(
    r: ChatRequest,
    bot: BotRequest,
) -> Generator[dict, None, None]:
    """Call streaming bot using Anthropic engine.

    Parameters
    ----------
    r : ChatRequest
        ChatRequest object, containing the conversation and session data
    bot : BotRequest
        BotRequest object, containing the bot data


    Yields
    ------
    dict
        The response chunks from the bot

    """
    initial_response = handle_empty_or_moderated(r.history[-1]["content"])
    if initial_response:
        yield {
            "type": "response",
            "content": initial_response,
        }
        return
    kwargs = setup_common(r, bot)
    response = chat_stream(r.history, bot.chat_model, **kwargs)
    yield from anthropic_tools_stream(r.history, response, bot, **kwargs)


@observe(capture_input=False, capture_output=False, as_type="generation")
def anthropic_tools_stream(
    messages: list[dict],
    response: AnthropicStream,
    bot: BotRequest,
    **kwargs: dict,
) -> Generator[dict, None, None]:
    """Handle tool calls in the conversation for Anthropic engine.

    Parameters
    ----------
    messages : list[dict]
        The conversation messages
    response : AnthropicStream
        The initial response stream
    bot : BotRequest
        BotRequest object
    kwargs : dict
        Keyword arguments for the LLM.

    Yields
    ------
    dict
        The response chunks from the Anthropic LLM

    """
    tools_used = 0
    usage = {"input": 0, "output": 0}
    # get the bot's source list up to this point in the conversation
    all_sources = []
    src_msgs = [
        msg for msg in messages
        if msg["role"] == "user" and \
        isinstance(msg["content"], str) and msg["content"].startswith("**Sources**:\n")
    ]
    for src_msg in src_msgs:
        numbered_srcs = src_msg["content"].split("\n")[1:] # ignore first line
        srcs = [num_src.split(" ")[1] for num_src in numbered_srcs]
        all_sources += srcs
    while tools_used < MAX_NUM_TOOLS:
        new_sources = []
        # Step 2: see if the model wants to call any functions
        current_tool_call = None
        current_text = None
        tool_msg = {"role": "assistant", "content": []}
        content = ""

        try:
            for chunk in response:
                if chunk.type == "content_block_start":
                    if chunk.content_block.type == "tool_use":
                        current_tool_call = ToolUseBlock(
                            type="tool_use",
                            name=chunk.content_block.name,
                            input="",
                            id=chunk.content_block.id,
                        )
                    elif chunk.content_block.type == "text":
                        current_text = {"type": "text", "text": ""}
                elif chunk.type == "content_block_delta":
                    if chunk.delta.type == "text_delta":
                        current_text["text"] += chunk.delta.text
                        content += chunk.delta.text
                        if "\n" in content:
                            index_of_newline = content.rfind("\n")
                            yield {
                                "type": "response",
                                "content": content[:index_of_newline + 1],
                            }
                            content = content[index_of_newline + 1:]
                    elif chunk.delta.type == "input_json_delta" and current_tool_call:
                        current_tool_call.input += chunk.delta.partial_json
                elif chunk.type == "content_block_stop":
                    if current_text:
                        tool_msg["content"].append(current_text)
                        current_text = None
                        if content:
                            yield {
                                "type": "response",
                                "content": content,
                            }
                            content = ""
                    if current_tool_call:
                        # Step 3: call the function for the model
                        yield {
                            "type": "tool_call",
                            "id": current_tool_call.id,
                            "name": current_tool_call.name,
                            "args": current_tool_call.input,
                        }

                        # convert tool call string to object
                        current_tool_call.input = json.loads(current_tool_call.input)

                        tool_call_id, tool_response, formatted_results = execute_tool_call(
                            current_tool_call,
                            bot,
                        )

                        yield {
                            "type": "tool_result",
                            "id": current_tool_call.id,
                            "name": current_tool_call.name,
                            "results": formatted_results,
                        }
                        # add sources from this tool call to the overall source list
                        new_sources += [str(res["id"]) for res in formatted_results]

                        # tool use message
                        tool_msg["content"].append(current_tool_call)
                        messages.append(tool_msg)
                        # tool response message
                        tool_response_msg = {
                            "type": "tool_result",
                            "tool_use_id": current_tool_call.id,
                            "content": tool_response,
                        }
                        messages.append({"role": "user", "content": [tool_response_msg]})

                        tools_used += 1
                        tool_msg = {"role": "assistant", "content": []}
                        break
                elif chunk.type == "message_delta" and \
                chunk.delta.stop_reason == "end_turn":
                    usage["output"] += chunk.usage.output_tokens
                    break
                elif chunk.type == "message_start":
                    usage["input"] += chunk.message.usage.input_tokens
                    usage["output"] += chunk.message.usage.output_tokens
        except Exception as e:
            logger.exception("anthropic_tools_stream exception")
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message=str(e),
            )
            yield {"type": "error"}
            yield {"type": "done"}
            return

        if content:
            yield {
                "type": "response",
                "content": content,
            }

        # update source list if a tool was used
        if current_tool_call is not None and current_tool_call.id:
            # remove duplicate sources while maintaining original order
            srcset = set(all_sources)
            new_sources = [
                src
                for src in new_sources
                if not (src in srcset or srcset.add(src))
            ]
            # only add the new sources to the bots source list
            source_list = "\n".join([
                f"[{i}] {src}"
                for i, src in enumerate(new_sources, start=len(all_sources) + 1)
            ])
            all_sources += new_sources
            # append the source list as a system message
            if source_list:
                messages.append({"role": "user", "content": "**Sources**:\n" + source_list})
            # Step 4: Send function results to the model and get a new response
            response = chat_stream(messages, bot.chat_model, **kwargs)
        else:
            usage["total"] = usage["input"] + usage["output"]
            langfuse_context.update_current_observation(
                model=bot.chat_model.model,
                usage=usage,
            )
            return
