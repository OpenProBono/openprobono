"""Defines the bot engines. The meaty stuff."""
import json
from typing import TYPE_CHECKING

import langchain
from anthropic import Anthropic
from anthropic.types.beta.tools import ToolsBetaContentBlock
from langfuse.decorators import observe
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall

import app.chat_models as chat_models
from app.models import BotRequest, ChatRequest
from app.prompts import (
    COMBINE_TOOL_OUTPUTS_TEMPLATE,
    NEW_TEST_TEMPLATE,
    MAX_NUM_TOOLS,
    MULTIPLE_TOOLS_PROMPT,
)
from app.search_tools import (
    run_search_tool,
    search_toolset_creator,
)
from app.vdb_tools import (
    run_vdb_tool,
    vdb_toolset_creator,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion


langchain.debug = True
openai.log = "debug"

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

@observe(capture_input=True)
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
    if chat_models.moderate(r.history[-1]["content"].strip()):
        return (
            "I'm sorry, I can't help you with that. "
            "Please modify your message and try again."
        )

    client = OpenAI()
    messages = r.history
    messages.append({"role": "system", "content": bot.system_prompt})
    toolset = search_toolset_creator(bot) + vdb_toolset_creator(bot)
    kwargs = {
        "client": client,
        "tools": toolset,
        "tool_choice": "auto",  # auto is default, but we'll be explicit
        #"session_id": r.session_id, 
        "temperature": 0,
    }

    # response is a ChatCompletion object
    print("CALL LLM INITIAL")
    response: ChatCompletion = chat_models.chat(messages, bot.chat_model, **kwargs)
            
    print(response.usage.to_dict())
    print("tokens used^^^")
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
        print("CALL LLM WITH TOOL RESPONSES")
        response = chat_models.chat(messages, bot.chat_model, **kwargs)
        print(response.usage.to_dict())
        print("tokens used^^^")
        response_message = response.choices[0].message
        messages.append(response_message)
        tool_calls = response_message.tool_calls
    return response_message

# @observe(capture_input=False)
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
    # langfuse_prompt_msg = [{"role": "system", "content": kwargs["system"]}]
    # langfuse_context.update_current_observation(input={
    #     "input": messages[-1]["content"],
    #     "prompt": langfuse_prompt_msg,
    # })
    # langfuse_context.update_current_trace(
    #     session_id=r.session_id,
    #     metadata={"bot_id": r.bot_id, "engine": bot.chat_model.engine},
    # )

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
