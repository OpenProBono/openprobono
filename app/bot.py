"""Defines the bot engines. The meaty stuff."""
import json
from typing import Generator

import openai
from anthropic import Anthropic
from anthropic import Stream as AnthropicStream
from anthropic.types import Message as AnthropicMessage
from anthropic.types.tool_use_block import ToolUseBlock
from langfuse.decorators import langfuse_context, observe
from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)

from app.chat_models import chat
from app.models import BotRequest, ChatRequest
from app.moderation import moderate
from app.prompts import MAX_NUM_TOOLS
from app.search_tools import run_search_tool, search_toolset_creator
from app.vdb_tools import (
    run_vdb_tool,
    session_data_toolset_creator,
    vdb_toolset_creator,
)

openai.log = "debug"

def stream_openai_response(response: ChatCompletion):
    full_delta_dict_collection = []
    no_yield = False
    for chunk in response:
        if(chunk.choices[0].delta.tool_calls or no_yield):
            no_yield = True
            full_delta_dict_collection.append(chunk.choices[0].delta.to_dict())
        else:
            chunk_content = chunk.choices[0].delta.content
            if(chunk_content):
                yield chunk.choices[0].delta.content

    tool_calls, current_dict = [], {}
    if(no_yield):
        current_dict = full_delta_dict_collection[0]
        for i in range(1, len(full_delta_dict_collection)):
            merge_dicts_stream_openai_completion(current_dict, full_delta_dict_collection[i])

        tool_calls = [ChatCompletionMessageToolCall.model_validate(tool_call)
                    for tool_call in current_dict["tool_calls"]]
    return tool_calls, current_dict


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
    elif moderate(r.history[-1]["content"].strip()):
        yield (
            "I'm sorry, I can't help you with that. "
            "Please modify your message and try again."
        )
    else:
        client = OpenAI()
        messages = r.history
        yield "  \n"

        #vdb tool for user uploaded files
        session_data_toolset = session_data_toolset_creator(r.session_id)
        if session_data_toolset:
            bot.vdb_tools.append(session_data_toolset)

        toolset = search_toolset_creator(bot) + vdb_toolset_creator(bot)

        kwargs = {
            "client": client,
            "tools": toolset,
            "tool_choice": "auto",  # auto is default, but we'll be explicit
            "temperature": 0,
            "stream": True,
        }

        # response is a ChatCompletion object
        response: ChatCompletion = chat(messages, bot.chat_model, **kwargs)
        tool_calls, current_dict = yield from stream_openai_response(response)
        # Step 2: check if the model wanted to call a function
        if tool_calls:
            #sometimes doesnt capture the role with streaming + multiple tool calls
            if("role" not in current_dict): current_dict["role"] = "assistant" 
            if("content" not in current_dict): current_dict["content"] = None

            messages.append(current_dict)
            yield "  \n"
            yield from openai_tools_stream(messages, tool_calls, bot, **kwargs)

def openai_tools_stream(
    messages: list[dict],
    tool_calls: list[ChatCompletionMessageToolCall],
    bot: BotRequest,
    **kwargs: dict,
):
    """Handle tool calls in the conversation for OpenAI engine.

    Parameters
    ----------
    messages : list[dict]
        The conversation messages
    tool_calls : list[ChatCompletionMessageToolCall]
        List of tool calls
    bot : BotRequest
        BotRequest object

    Returns
    -------
    messages : list[dict]
        The updated conversation messages with tool responses appended

    """
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
            yield f"  \nRunning {function_name} tool with the following arguments: {function_args}  \n"
            if vdb_tool:
                tool_response = run_vdb_tool(
                    vdb_tool,
                    function_args,
                )
            elif search_tool:
                tool_response = run_search_tool(
                    search_tool,
                    function_args,
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
        yield "  \nAnalyzing tool results  \n"
        response = chat(messages, bot.chat_model, **kwargs)
        tool_calls, current_dict = yield from stream_openai_response(response)


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

    client = OpenAI()
    messages = r.history
    #vdb tool for user uploaded files
    session_data_toolset = session_data_toolset_creator(r.session_id)
    if session_data_toolset:
        bot.vdb_tools.append(session_data_toolset)

    toolset = search_toolset_creator(bot) + vdb_toolset_creator(bot)

    kwargs = {
        "tools": toolset,
        "tool_choice": "auto",  # auto is default, but we'll be explicit
        "temperature": 0,
    }

    # tracing
    last_user_msg = next(
        (m for m in messages[::-1] if m["role"] == "user"),
        None,
    )
    langfuse_context.update_current_observation(input=last_user_msg)
    langfuse_context.update_current_trace(
        session_id=r.session_id,
        metadata={"bot_id": r.bot_id} | kwargs,
    )

    kwargs["client"] = client

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
    while tool_calls and tools_used < MAX_NUM_TOOLS:
        messages.append(response_message.model_dump())
        # TODO: run tool calls in parallel
        for tool_call in tool_calls:
            print("RUN TOOL")
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
                tool_response = run_vdb_tool(vdb_tool, function_args)
            elif search_tool:
                tool_response = run_search_tool(search_tool, function_args)
            else:
                tool_response = "error: unable to run tool"
            # extend conversation with function response
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": tool_response,
            })
            tools_used += 1
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
    client = Anthropic()
    # The Messages API accepts a top-level `system` parameter,
    # not "system" as an input message role
    messages = [msg for msg in r.history if msg["role"] != "system"]
    #vdb tool for user uploaded files
    session_data_toolset = session_data_toolset_creator(r.session_id)
    if session_data_toolset:
        bot.vdb_tools.append(session_data_toolset)

    toolset = search_toolset_creator(bot) + vdb_toolset_creator(bot)

    kwargs = {
        "tools": toolset,
        "system": bot.system_prompt,
        "temperature": 0,
    }

    # Step 0: tracing
    last_user_msg = next(
        (m for m in messages[::-1] if m["role"] == "user"),
        None,
    )
    langfuse_context.update_current_observation(input=last_user_msg)
    langfuse_context.update_current_trace(
        session_id=r.session_id,
        metadata={"bot_id": r.bot_id} | kwargs,
    )
    # after tracing add client to kwargs
    kwargs["client"] = client

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
    tool_calls: list[ToolUseBlock] = [
        msg for msg in response.content if msg.type == "tool_use"
    ]
    if not tool_calls:
        return "\n".join([
            block.text for block in response.content if block.type == "text"
        ])

    tools_used = 0
    while tool_calls and tools_used < MAX_NUM_TOOLS:
        # add message before tool calls so the last message isn't duplicated in db
        messages.append({"role": response.role, "content": response.content})
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
                tool_response = run_vdb_tool(vdb_tool, tool_call.input)
            elif search_tool:
                tool_response = run_search_tool(search_tool, tool_call.input)
            else:
                tool_response = "error: unable to identify tool"
                tool_response_msg["is_error"] = True
            tool_response_msg["content"] = tool_response
            # extend conversation with function response
            messages.append({
                "role": "user",
                "content": [tool_response_msg],
            })
            tools_used += 1
        # Step 4: send info for each function call and function response to the model
        # get a new response from the model where it can see the function response
        response = chat(messages, bot.chat_model, **kwargs)
        tool_calls = [msg for msg in response.content if msg.type == "tool_use"]
    return "\n".join([
        block.text for block in response.content if block.type == "text"
    ])

def anthropic_bot_stream(r: ChatRequest, bot: BotRequest) -> Generator:
    if r.history[-1]["content"].strip() == "":
        return "Hi, how can I assist you today?"
    elif moderate(r.history[-1]["content"].strip()):
        return (
            "I'm sorry, I can't help you with that. "
            "Please modify your message and try again."
        )
    client = Anthropic()
    messages = r.history
    yield "  \n"
    messages = [msg for msg in r.history if msg["role"] != "system"]

    #vdb tool for user uploaded files
    session_data_toolset = session_data_toolset_creator(r.session_id)
    if session_data_toolset:
        bot.vdb_tools.append(session_data_toolset)

    toolset = search_toolset_creator(bot) + vdb_toolset_creator(bot)

    kwargs = {
        "tools": toolset,
        "system": bot.system_prompt,
        "temperature": 0,
    }

    # after tracing add client to kwargs
    kwargs["client"] = client

    # Step 1: send the conversation and available functions to the model
    response: AnthropicStream = chat(messages, bot.chat_model, **kwargs)
    yield from anthropic_tools_stream(messages, response, bot, **kwargs)

def anthropic_tools_stream(
    messages: list[dict],
    response: AnthropicStream,
    bot: BotRequest,
    **kwargs: dict,
) -> Generator:
    tools_used = 0
    while tools_used < MAX_NUM_TOOLS:
        current_tool_call = None
        tool_call_id = None

        for chunk in response:
            print(chunk)
            continue
            if chunk.type == "content_block_start":
                if chunk.content_block.type == "tool_use":
                    current_tool_call = {
                        "name": chunk.content_block.name,
                        "input": "",
                    }
                    tool_call_id = chunk.content_block.id
            elif chunk.type == "content_block_delta":
                if chunk.delta.type == "text_delta":
                    yield chunk.delta.text
                elif chunk.delta.type == "input_json_delta" and current_tool_call:
                    current_tool_call["input"] += chunk.delta.partial_json
            elif chunk.type == "content_block_stop":
                if current_tool_call:
                    function_args = json.loads(current_tool_call["input"])
                    function_name = current_tool_call["name"]

                    yield f"  \nRunning {function_name} tool with the following arguments: {function_args}"

                    vdb_tool = next(
                        (t for t in bot.vdb_tools if function_name == t.name),
                        None,
                    )
                    search_tool = next(
                        (t for t in bot.search_tools if function_name == t.name),
                        None,
                    )
                    if vdb_tool:
                        tool_response = run_vdb_tool(vdb_tool, function_args)
                    elif search_tool:
                        tool_response = run_search_tool(search_tool, function_args)
                    else:
                        tool_response = "error: unable to identify tool"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": current_tool_call["name"],
                        "content": tool_response,
                    })

                    tools_used += 1
                    current_tool_call = None
                    break
            elif chunk.type == "message_delta" and \
            chunk.delta.stop_reason == "end_turn":
                break

        # get a new response from the model where it can see the function response
        if tool_call_id:
            yield "  \nAnalyzing tool results  \n"
            response = chat(messages, bot.chat_model, **kwargs)
        else:
            return
