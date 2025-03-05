"""Load messages and chat with chat models."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import google.generativeai as genai
import requests
from anthropic import Anthropic
from anthropic import Stream as AnthropicStream
from langfuse.decorators import langfuse_context, observe
from openai import OpenAI
from openai import Stream as OpenAIStream

from app.models import ChatModelParams, EngineEnum, HiveModelEnum
from app.prompts import HIVE_QA_PROMPT

if TYPE_CHECKING:
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import RawMessageStreamEvent
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

HIVE_TASK_URL = "https://api.thehive.ai/api/v2/task/sync"
MAX_TOKENS = 1000
TEMPERATURE = 0
TOP_P = 0.95
SEED = 0
TOOL_CHOICE = "auto"
NOT_GIVEN = "NOT_GIVEN"
ANTHROPIC_CLIENT = Anthropic()
OPENAI_CLIENT = OpenAI()

def chat(
    messages: list[dict],
    chatmodel: ChatModelParams,
    **kwargs: dict,
) -> tuple[str, list[str]] | ChatCompletion | AnthropicMessage | str:
    """Chat with an LLM.

    Parameters
    ----------
    messages : list[dict]
        The conversation history formatted for the given chat model.
    chatmodel : ChatModelParams
        The chat model to use for the conversation.
    kwargs : dict
        Keyword arguments for the given chat model.

    Returns
    -------
    tuple[str, list[str]] | ChatCompletion | AnthropicMessage | str
        The response from the LLM. Depends on engine.

    """
    match chatmodel.engine:
        case EngineEnum.openai:
            return chat_openai(messages, chatmodel.model, **kwargs)
        case EngineEnum.anthropic:
            return chat_anthropic(messages, chatmodel.model, **kwargs)
        case EngineEnum.hive:
            return chat_hive(messages, chatmodel.model, **kwargs)
        case EngineEnum.google:
            return chat_gemini(messages, chatmodel.model, **kwargs)


def chat_stream(
    messages: list[dict],
    chatmodel: ChatModelParams,
    **kwargs: dict,
) -> OpenAIStream[ChatCompletionChunk] | AnthropicStream[RawMessageStreamEvent]:
    """Chat with an LLM with streaming enabled.

    Parameters
    ----------
    messages : list[dict]
        The conversation history formatted for the given chat model.
    chatmodel : ChatModelParams
        The chat model to use for the conversation.
    kwargs : dict
        Keyword arguments for the given chat model.

    Returns
    -------
    Stream[ChatCompletionChunk] | Stream[RawMessageStreamEvent]
        The response from the LLM. Depends on engine.

    """
    match chatmodel.engine:
        case EngineEnum.openai:
            return chat_stream_openai(messages, chatmodel.model, **kwargs)
        case EngineEnum.anthropic:
            return chat_stream_anthropic(messages, chatmodel.model, **kwargs)


def chat_str(messages: list[dict], chatmodel: ChatModelParams, **kwargs: dict) -> str:
    """Chat with an LLM. Returns a string instead of a full response object.

    Parameters
    ----------
    messages : list[dict]
        The conversation history.
    chatmodel : ChatModelParams
        The chat model to use for the conversation.
    kwargs : dict
        Keyword arguments for the given chat model.

    Returns
    -------
    str
        A string response from the LLM. Depends on engine.

    """
    match chatmodel.engine:
        case EngineEnum.openai:
            return chat_str_openai(messages, chatmodel.model, **kwargs)
        case EngineEnum.anthropic:
            return chat_str_anthropic(messages, chatmodel.model, **kwargs)
        case EngineEnum.hive:
            return chat_str_hive(messages, chatmodel.model, **kwargs)
        case EngineEnum.google:
            return chat_gemini(messages, chatmodel.model, **kwargs)
    raise ValueError(chatmodel)


@observe(as_type="generation")
def chat_hive(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> tuple[str, list[str]]:
    """Chat with a Hive LLM.

    Parameters
    ----------
    messages : list[dict]
        The conversation history
    model : str
        The name of the model.
    kwargs : dict
        Keyword arguments for the given chat model.

    Returns
    -------
    tuple[str, list[str]]
        message, chunks used for augmentation (empty if no RAG)

    """
    use_embedding = kwargs.get("use_embedding", False)
    system = kwargs.get("system", HIVE_QA_PROMPT)
    max_tokens = kwargs.get("max_tokens", MAX_TOKENS)
    temperature = kwargs.get("temperature", 0.0)
    top_p = kwargs.get("top_p", 0.95)
    if use_embedding:
        key = "HIVE_7B_NORAG" if model == HiveModelEnum.hive_7b else "HIVE_70B_NORAG"
    elif model == HiveModelEnum.hive_7b:
        key = "HIVE_7B_API_KEY"
    else:
        key = "HIVE_70B_API_KEY"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Token {os.environ[key]}",
        "Content-Type": "application/json",
    }
    data = {
        "text_data": messages[-1]["content"],
        "options": {
            "max_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "system_prompt": system,
            "roles": {
                "user": "user",
                "model": "assistant",
            },
            "prompt_history": messages[:-1],
        },
    }
    response = requests.post(HIVE_TASK_URL, headers=headers, json=data, timeout=90)
    response_json = response.json()
    output = response_json["status"][0]["response"]["output"][0]
    message = output["choices"][0]["message"]
    chunks = output["augmentations"]
    return message, chunks


def chat_str_hive(messages: list[dict], model: str, **kwargs: dict) -> str:
    """Chat with an LLM using the hive engine and get a string response."""
    text, _ = chat_hive(messages, model, **kwargs)
    return text


def set_kwargs_openai(kwargs: dict) -> None:
    """Set default values for openai.Completion API call."""
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = MAX_TOKENS
    if "temperature" not in kwargs:
        kwargs["temperature"] = TEMPERATURE
    if "seed" not in kwargs:
        kwargs["seed"] = SEED
    if "tools" in kwargs and "tool_choice" not in kwargs:
        kwargs["tool_choice"] = TOOL_CHOICE


@observe(as_type="generation")
def chat_openai(messages: list[dict], model: str, **kwargs: dict) -> ChatCompletion:
    """Chat with an LLM using the openai engine.

    Parameters
    ----------
    messages : list[dict]
        The conversation history.
    model : str
        The name of the OpenAI LLM to use for conversation.
    kwargs : dict
        Keyword arguments for the LLM.

    Returns
    -------
    ChatCompletion
        The response from the LLM.

    """
    set_kwargs_openai(kwargs)
    response: ChatCompletion = OPENAI_CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    usage = {
        "input": response.usage.prompt_tokens,
        "output": response.usage.completion_tokens,
        "total": response.usage.total_tokens,
    }
    langfuse_context.update_current_observation(
        input=messages,
        output=response.choices[0].message,
        metadata=kwargs,
        model=model,
        usage=usage,
    )
    return response


@observe(capture_input=False, capture_output=False)
def chat_stream_openai(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> OpenAIStream[ChatCompletionChunk]:
    """Chat with an LLM using the openai engine with streaming enabled.

    Parameters
    ----------
    messages : list[dict]
        The conversation history.
    model : str
        The name of the OpenAI LLM to use for conversation.
    kwargs : dict
        Keyword arguments for the LLM.

    Returns
    -------
    Stream[ChatCompletionChunk]
        The response from the LLM.

    """
    set_kwargs_openai(kwargs)
    return OPENAI_CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        **kwargs,
    )


def chat_str_openai(messages: list[dict], model: str, **kwargs: dict) -> str:
    """Chat with an LLM using the anthropic engine and get a string response."""
    response = chat_openai(messages, model, **kwargs)
    return response.choices[0].message.content.strip()


def set_kwargs_anthropic(kwargs: dict) -> None:
    """Set default values for anthropic.Message API call."""
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = MAX_TOKENS
    if "temperature" not in kwargs:
        kwargs["temperature"] = TEMPERATURE


@observe(as_type="generation")
def chat_anthropic(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> AnthropicMessage:
    """Chat with an LLM using the anthropic engine.

    Parameters
    ----------
    messages : list[dict]
        The conversation history.
    model : str
        The name of the anthropic LLM to use for conversation.
    kwargs : dict
        Keyword arguments for the LLM.

    Returns
    -------
    Message
        The response from the LLM.

    """
    set_kwargs_anthropic(kwargs)
    response: AnthropicMessage = ANTHROPIC_CLIENT.messages.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    # report input, output, model, usage to langfuse
    usage = {
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens,
        "total": response.usage.input_tokens + response.usage.output_tokens,
    }
    langfuse_context.update_current_observation(
        input=messages,
        model=model,
        output=response.content,
        metadata=kwargs,
        usage=usage,
    )
    return response


@observe(capture_input=False, capture_output=False)
def chat_stream_anthropic(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> AnthropicStream[RawMessageStreamEvent]:
    """Chat with an LLM using the anthropic engine.

    Parameters
    ----------
    messages : list[dict]
        The conversation history.
    model : str
        The name of the anthropic LLM to use for conversation.
    kwargs : dict
        Keyword arguments for the LLM.

    Returns
    -------
    Stream[RawMessageStreamEvent]
        The response from the LLM.

    """
    set_kwargs_anthropic(kwargs)
    return ANTHROPIC_CLIENT.messages.create(
        model=model,
        messages=messages,
        stream=True,
        **kwargs,
    )


def chat_str_anthropic(messages: list[dict], model: str, **kwargs: dict) -> str:
    """Chat with an LLM using the anthropic engine and get a string response."""
    response = chat_anthropic(messages, model, **kwargs)
    return "\n".join([
        block.text for block in response.content if block.type == "text"
    ])


@observe(as_type="generation")
def chat_gemini(messages: list[dict], model: str, **kwargs: dict) -> str:
    """Chat with a Gemini LLM.

    Parameters
    ----------
    messages : list[dict]
        The conversation history.
    model : str
        The name of the model.
    kwargs : dict
        Keyword arguments for the LLM.

    Returns
    -------
    str
        The response from the LLM

    """
    messages = [{"role":msg["role"], "parts":[msg["content"]]} for msg in messages]
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    # Create the model
    # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
    set_kwargs_gemini(kwargs)
    generation_config = {"response_mime_type": "text/plain"} | kwargs

    llm = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        # safety_settings = Adjust safety settings
        # See https://ai.google.dev/gemini-api/docs/safety-settings
    )

    chat_session = llm.start_chat(history=messages)
    response_text = []
    for content in messages[-1]["parts"]:
        response = chat_session.send_message(content)
        response_text.append(response.text.strip())
    return "\n".join(response_text)


@observe(as_type="generation")
def chat_single_gemini(
    message: str,
    model: str,
    **kwargs: dict,
) -> str:
    """Chat with a Gemini LLM.

    Parameters
    ----------
    message : str
        The single message
    model : str
        The name of the model.
    kwargs : dict
        Keyword arguments for the LLM.

    Returns
    -------
    str
        The response from the LLM

    """
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    # Create the model
    # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
    set_kwargs_gemini(kwargs)
    generation_config = {"response_mime_type": "text/plain"} | kwargs

    llm = genai.GenerativeModel(
        model_name=model,
        # generation_config=generation_config,
        # safety_settings = Adjust safety settings
        # See https://ai.google.dev/gemini-api/docs/safety-settings
    )

    response_text = llm.generate_content(message).text.strip()
    return response_text

def set_kwargs_gemini(kwargs: dict) -> None:
    """Set default values for genai.ChatSession API call."""
    if "max_output_tokens" not in kwargs:
        kwargs["max_output_tokens"] = MAX_TOKENS
    if "temperature" not in kwargs:
        kwargs["temperature"] = TEMPERATURE
    if "top_p" not in kwargs:
        kwargs["top_p"] = TOP_P
