"""Defines the bot engines. The meaty stuff."""
import json
from typing import Generator

import openai
from anthropic import Stream as AnthropicStream
from anthropic.types import Message as AnthropicMessage
from anthropic.types.content_block import ContentBlock
from langfuse.decorators import langfuse_context, observe
from openai import Stream as OpenAIStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)

from app.chat_models import chat, chat_stream
from app.models import BotRequest, ChatRequest
from app.moderation import moderate
from app.prompts import MAX_NUM_TOOLS
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

openai.log = "debug"

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
                    yield json.dumps({
                        "type": "response",
                        "content": content,
                    }) + "\n"
                    content = ""

    if content:
        yield json.dumps({
            "type": "response",
            "content": content,
        }) + "\n"

    tool_calls, current_dict = [], {}
    if(no_yield):
        current_dict = full_delta_dict_collection[0]
        for i in range(1, len(full_delta_dict_collection)):
            merge_dicts_stream_openai_completion(current_dict, full_delta_dict_collection[i])

        tool_calls = [ChatCompletionMessageToolCall.model_validate(tool_call)
                    for tool_call in current_dict["tool_calls"]]
    return tool_calls, current_dict, usage


def merge_dicts_stream_openai_completion(dict1, dict2):
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


@observe(capture_input=False)
def openai_bot_stream(r: ChatRequest, bot: BotRequest):
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
    if r.history[-1]["content"].strip() == "":
        yield "Hi, how can I assist you today?"
        return
    if moderate(r.history[-1]["content"].strip()):
        yield (
            "I'm sorry, I can't help you with that. "
            "Please modify your message and try again."
        )
        return

    toolset = search_toolset_creator(bot, r.bot_id) + vdb_toolset_creator(bot, r.bot_id, r.session_id)

    # Step 0: tracing
    last_user_msg = next(
        (m for m in r.history[::-1] if m["role"] == "user"),
        None,
    )
    langfuse_context.update_current_observation(input=last_user_msg["content"])

    kwargs = {"tools": toolset}
    langfuse_context.update_current_trace(
        session_id=r.session_id,
        metadata={"bot_id": r.bot_id} | kwargs,
    )

    # Step 1: send initial message to the model
    # response is a Stream object
    response: OpenAIStream = chat_stream(r.history, bot.chat_model, **kwargs)
    yield from openai_tools_stream(r.history, response, bot, **kwargs)


@observe(capture_input=False, capture_output=False, as_type="generation")
def openai_tools_stream(
    messages: list[dict],
    response: OpenAIStream,
    bot: BotRequest,
    **kwargs: dict,
) -> Generator:
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

    Returns
    -------
    Generator
        A stream of response chunks from the OpenAI LLM

    """
    tools_used = 0
    usage_dict = {"input": 0, "output": 0, "total": 0}
    all_sources = []
    while tools_used < MAX_NUM_TOOLS:
        # Step 2: see if the bot wants to call any functions
        tool_calls, current_dict, usage = yield from stream_openai_response(response)
        if usage:
            usage_dict["input"] += usage.prompt_tokens
            usage_dict["output"] += usage.completion_tokens
            usage_dict["total"] += usage.total_tokens
        if tool_calls:
            #sometimes doesnt capture the role with streaming + multiple tool calls
            if("role" not in current_dict):
                current_dict["role"] = "assistant"
            if("content" not in current_dict):
                current_dict["content"] = None

            messages.append(current_dict)
        else:
            break

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            vdb_tool = find_vdb_tool(bot, function_name)
            search_tool = find_search_tool(bot, function_name)
            # Step 3: call the function
            # Note: the JSON response may not always be valid;
            # be sure to handle errors
            yield json.dumps({
                "type":"tool_call",
                "name":function_name,
                "args":tool_call.function.arguments,
            }) + "\n"
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
                formatted_results = []

            yield json.dumps({
                "type": "tool_result",
                "name": function_name,
                "results": formatted_results,
            }) + "\n"
            all_sources += [res["id"] for res in formatted_results]
            # Step 4: send the info for each function call and function response to
            # the model
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(tool_response),
            })  # extend conversation with function response
            tools_used += 1
        # remove duplicate sources while maintaining original order
        srcset = set()
        all_sources = [src for src in all_sources if not (src in srcset or srcset.add(src))]
        source_list = "\n".join([f"{i}. {src}" for i, src in enumerate(all_sources, start=1)])
        # append the source list as a system message
        messages.append({"role": "system", "content": "**Sources**:\n" + source_list})
        # get a new response from the model where it can see the function response
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
    if r.history[-1]["content"].strip() == "":
        return "Hi, how can I assist you today?"
    if moderate(r.history[-1]["content"].strip()):
        return (
            "I'm sorry, I can't help you with that. "
            "Please modify your message and try again."
        )

    toolset = search_toolset_creator(bot, r.bot_id) + vdb_toolset_creator(bot, r.bot_id, r.session_id)
    kwargs = {"tools": toolset}
    messages = r.history

    # tracing
    last_user_msg = next(
        (m for m in messages[::-1] if m["role"] == "user"),
        None,
    )
    langfuse_context.update_current_observation(input=last_user_msg["content"])
    langfuse_context.update_current_trace(metadata={"bot_id": r.bot_id} | kwargs)

    # response is a ChatCompletion object
    response: ChatCompletion = chat(messages, bot.chat_model, **kwargs)
    response_message = response.choices[0].message
    return openai_tools(messages, response_message, bot, **kwargs)


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
    ChatCompletionMessage
        The response message from the bot, should be final response.

    """
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if not tool_calls:
        return response_message.content

    tools_used = 0
    all_sources = []
    while tool_calls and tools_used < MAX_NUM_TOOLS:
        messages.append(response_message.model_dump())
        # TODO: run tool calls in parallel
        for tool_call in tool_calls:
            print("RUN TOOL")
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            vdb_tool = find_vdb_tool(bot, function_name)
            search_tool = find_search_tool(bot, function_name)
            # Step 3: call the function
            # Note: the JSON response may not always be valid;
            # be sure to handle errors
            if vdb_tool:
                tool_response, sources = run_vdb_tool(vdb_tool, function_args)
            elif search_tool:
                tool_response, sources = run_search_tool(search_tool, function_args)
            else:
                tool_response = "error: unable to run tool"
            # add sources from this tool call to the overall source list
            if sources:
                # yield "Sources found: \n"
                # for i, src in enumerate(sources, start=len(all_sources) + 1):
                #     yield f"{i}. {src} \n"
                # yield "\n\n"
                all_sources += sources
            # extend conversation with function response
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(tool_response),
            })
            tools_used += 1
        # append the source list as a system message
        source_list = "\n".join([f"{i}. {src}" for i, src in enumerate(all_sources, start=1)])
        messages.append({"role": "system", "content": "**Sources**:\n" + source_list})
        # Step 4: send the function responses to the model
        response: ChatCompletion = chat(messages, bot.chat_model, **kwargs)
        # get a new response from the model where it can see the function response
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
    return response_message.content


@observe(capture_input=False)
def anthropic_bot(r: ChatRequest, bot: BotRequest) -> str:
    if r.history[-1]["content"].strip() == "":
        return "Hi, how can I assist you today?"
    if moderate(r.history[-1]["content"].strip(), bot.chat_model):
        return (
            "I'm sorry, I can't help you with that. "
            "Please modify your message and try again."
        )

    toolset = search_toolset_creator(bot, r.bot_id) + vdb_toolset_creator(bot, r.bot_id, r.session_id)
    kwargs = {"tools": toolset, "system": bot.system_prompt}

    # The Messages API accepts a top-level `system` parameter,
    # not "system" as an input message role
    messages = [msg for msg in r.history if msg["role"] != "system"]
    # Step 0: tracing
    last_user_msg = next(
        (m for m in messages[::-1] if m["role"] == "user"),
        None,
    )
    langfuse_context.update_current_observation(input=last_user_msg["content"])
    langfuse_context.update_current_trace(metadata={"bot_id": r.bot_id} | kwargs)

    # Step 1: send the conversation and available functions to the model
    response = chat(messages, bot.chat_model, **kwargs)
    return anthropic_tools(messages, response, bot, **kwargs)


def anthropic_tools(
    messages: list[dict],
    response: AnthropicMessage,
    bot: BotRequest,
    **kwargs: dict,
) -> str:
    # Step 2: check if the model wanted to call a function
    tool_calls: list[ContentBlock] = [
        msg for msg in response.content if msg.type == "tool_use"
    ]
    if not tool_calls:
        return "\n".join([
            block.text for block in response.content if block.type == "text"
        ])

    tools_used = 0
    all_sources = []
    while tool_calls and tools_used < MAX_NUM_TOOLS:
        # add message before tool calls so the last message isn't duplicated in db
        messages.append({"role": response.role, "content": response.content})
        for tool_call in tool_calls:
            function_name = tool_call.name
            vdb_tool = find_vdb_tool(bot, function_name)
            search_tool = find_search_tool(bot, function_name)
            tool_response_msg = {
                "type": "tool_result",
                "tool_use_id": tool_call.id,
            }
            # Step 3: call the function
            if vdb_tool:
                tool_response, sources = run_vdb_tool(vdb_tool, tool_call.input)
            elif search_tool:
                tool_response, sources = run_search_tool(search_tool, tool_call.input)
            else:
                tool_response = "error: unable to identify tool"
                tool_response_msg["is_error"] = True
            # add sources from this tool call to the overall source list
            if sources:
                # yield "Sources found: \n"
                # for i, src in enumerate(sources, start=len(all_sources) + 1):
                #     yield f"{i}. {src} \n"
                # yield "\n\n"
                all_sources += sources
            tool_response_msg["content"] = str(tool_response)
            # extend conversation with function response
            messages.append({
                "role": "user",
                "content": [tool_response_msg],
            })
            tools_used += 1
        # append the source list as a system message
        source_list = "\n".join([f"{i}. {src}" for i, src in enumerate(all_sources, start=1)])
        messages.append({"role": "system", "content": "**Sources**:\n" + source_list})
        # Step 4: send info for each function call and function response to the model
        # get a new response from the model where it can see the function response
        response = chat(messages, bot.chat_model, **kwargs)
        tool_calls = [msg for msg in response.content if msg.type == "tool_use"]
    return "\n".join([
        block.text for block in response.content if block.type == "text"
    ])


@observe(capture_input=False)
def anthropic_bot_stream(r: ChatRequest, bot: BotRequest) -> Generator:
    if r.history[-1]["content"].strip() == "":
        yield "Hi, how can I assist you today?"
        return
    if moderate(r.history[-1]["content"].strip(), bot.chat_model):
        yield (
            "I'm sorry, I can't help you with that. "
            "Please modify your message and try again."
        )
        return

    toolset = search_toolset_creator(bot, r.bot_id) + vdb_toolset_creator(bot, r.bot_id, r.session_id)

    # Step 0: tracing
    # input tracing
    messages = [msg for msg in r.history if msg["role"] != "system"]
    last_user_msg = next(
        (m for m in messages[::-1] if m["role"] == "user"),
        None,
    )
    langfuse_context.update_current_observation(input=last_user_msg["content"])

    kwargs = {"tools": toolset, "system": bot.system_prompt}
    # metadata tracing
    langfuse_context.update_current_trace(
        session_id=r.session_id,
        metadata={"bot_id": r.bot_id} | kwargs,
    )
    yield "  \n"

    # Step 1: send initial message to the model
    response: AnthropicStream = chat_stream(messages, bot.chat_model, **kwargs)
    yield from anthropic_tools_stream(messages, response, bot, **kwargs)


@observe(capture_input=False, capture_output=False, as_type="generation")
def anthropic_tools_stream(
    messages: list[dict],
    response: AnthropicStream,
    bot: BotRequest,
    **kwargs: dict,
) -> Generator:
    tools_used = 0
    usage = {"input": 0, "output": 0}
    all_sources = []
    while tools_used < MAX_NUM_TOOLS:
        # Step 2: see if the model wants to call any functions
        current_tool_call = None
        current_text = None
        tool_call_id = None
        tool_msg = {"role": "assistant", "content": []}

        for chunk in response:
            if chunk.type == "content_block_start":
                if chunk.content_block.type == "tool_use":
                    current_tool_call = {
                        "name": chunk.content_block.name,
                        "input": "",
                    }
                    tool_call_id = chunk.content_block.id
                elif chunk.content_block.type == "text":
                    current_text = {"type": "text", "text": ""}
            elif chunk.type == "content_block_delta":
                if chunk.delta.type == "text_delta":
                    current_text["text"] += chunk.delta.text
                    yield chunk.delta.text
                elif chunk.delta.type == "input_json_delta" and current_tool_call:
                    current_tool_call["input"] += chunk.delta.partial_json
            elif chunk.type == "content_block_stop":
                if current_text:
                    tool_msg["content"].append(current_text)
                    current_text = None
                if current_tool_call:
                    # Step 3: call the function for the model
                    function_args = json.loads(current_tool_call["input"])
                    function_name = current_tool_call["name"]
                    yield f"  \nRunning {function_name} tool with the following arguments: {function_args}"

                    vdb_tool = find_vdb_tool(bot, function_name)
                    search_tool = find_search_tool(bot, function_name)
                    if vdb_tool:
                        tool_response, sources = run_vdb_tool(vdb_tool, function_args)
                    elif search_tool:
                        tool_response, sources = run_search_tool(search_tool, function_args)
                    else:
                        tool_response = "error: unable to identify tool"
                    # add sources from this tool call to the overall source list
                    if sources:
                        yield "Sources found: \n"
                        for i, src in enumerate(sources, start=len(all_sources) + 1):
                            yield f"{i}. {src} \n"
                        yield "\n\n"
                        all_sources += sources
                    # tool use message
                    current_tool_call["id"] = tool_call_id
                    current_tool_call["type"] = "tool_use"
                    current_tool_call["input"] = function_args
                    tool_msg["content"].append(current_tool_call)
                    messages.append(tool_msg)
                    # tool response message
                    tool_response_msg = {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": tool_response,
                    }
                    messages.append({"role": "user", "content": [tool_response_msg]})

                    tools_used += 1
                    current_tool_call = None
                    tool_msg = {"role": "assistant", "content": []}
                    break
            elif chunk.type == "message_delta" and \
            chunk.delta.stop_reason == "end_turn":
                usage["output"] += chunk.usage.output_tokens
                break
            elif chunk.type == "message_start":
                usage["input"] += chunk.message.usage.input_tokens
                usage["output"] += chunk.message.usage.output_tokens

        if tool_call_id:
            # append the source list as a system message
            source_list = "\n".join([f"{i}. {src}" for i, src in enumerate(all_sources, start=1)])
            messages.append({"role": "system", "content": "**Sources**:\n" + source_list})
            # Step 4: Send function results to the model and get a new response
            response = chat_stream(messages, bot.chat_model, **kwargs)
        else:
            usage["total"] = usage["input"] + usage["output"]
            langfuse_context.update_current_observation(
                model=bot.chat_model.model,
                usage=usage,
            )
            return
