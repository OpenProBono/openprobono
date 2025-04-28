"""Load messages and chat with chat models."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import requests
from anthropic import Anthropic
from anthropic import Stream as AnthropicStream
from google.genai import Client
from google.genai.types import GenerateContentConfig
from langfuse.decorators import langfuse_context, observe
from openai import OpenAI
from openai import Stream as OpenAIStream

from app.logger import setup_logger
from app.models import ChatModelParams, EngineEnum, HiveModelEnum, OpenAIModelEnum
from app.prompts import HIVE_QA_PROMPT

if TYPE_CHECKING:
    from collections.abc import Iterator

    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import RawMessageStreamEvent
    from google.genai.types import GenerateContentResponse
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
GOOGLE_CLIENT = Client(api_key=os.environ["GEMINI_API_KEY"])
OPENAI_REASONING_MODELS = {
    OpenAIModelEnum.o1,
    OpenAIModelEnum.o1_mini,
    OpenAIModelEnum.o1_preview,
    OpenAIModelEnum.o3,
    OpenAIModelEnum.o3_mini,
    OpenAIModelEnum.o4_mini,
}

logger = setup_logger()

def chat(
    messages: list[dict],
    chatmodel: ChatModelParams,
    **kwargs: dict,
) -> ChatCompletion | AnthropicMessage | GenerateContentResponse | tuple[str, list[str]]:
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
    ChatCompletion | AnthropicMessage | GenerateContentResponse | tuple[str, list[str]]
        The response from the LLM. Depends on engine.

    """
    match chatmodel.engine:
        case EngineEnum.openai:
            return chat_openai(messages, chatmodel.model, **kwargs)
        case EngineEnum.anthropic:
            return chat_anthropic(messages, chatmodel.model, **kwargs)
        case EngineEnum.google:
            return chat_google(messages, chatmodel.model, **kwargs)
        case EngineEnum.hive:
            return chat_hive(messages, chatmodel.model, **kwargs)


def chat_stream(
    messages: list[dict],
    chatmodel: ChatModelParams,
    **kwargs: dict,
) -> OpenAIStream[ChatCompletionChunk] | AnthropicStream[RawMessageStreamEvent] | Iterator[GenerateContentResponse]:
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
    Stream[ChatCompletionChunk] | Stream[RawMessageStreamEvent] | Iterator[GenerateContentResponse]
        The response chunks from the LLM. Depends on engine.

    """
    match chatmodel.engine:
        case EngineEnum.openai:
            return chat_stream_openai(messages, chatmodel.model, **kwargs)
        case EngineEnum.anthropic:
            return chat_stream_anthropic(messages, chatmodel.model, **kwargs)
        case EngineEnum.google:
            return chat_stream_google(messages, chatmodel.model, **kwargs)


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
            return chat_str_google(messages, chatmodel.model, **kwargs)
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


def set_kwargs_openai(kwargs: dict, model: str) -> None:
    """Set default values for openai.Completion API call."""
    if "max_tokens" not in kwargs:
        key = "max_tokens"
        if model in OPENAI_REASONING_MODELS:
            key = "max_completion_tokens"
        kwargs[key] = MAX_TOKENS
    if "temperature" not in kwargs and model not in OPENAI_REASONING_MODELS:
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
    set_kwargs_openai(kwargs, model)
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
    set_kwargs_openai(kwargs, model)
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
    """Chat with an LLM using the openai engine and get a string response."""
    response = chat_anthropic(messages, model, **kwargs)
    return "\n".join([
        block.text for block in response.content if block.type == "text"
    ])


@observe(as_type="generation")
def chat_google(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> GenerateContentResponse:
    """Chat with a Google LLM.

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
    GenerateContentResponse
        The response from the LLM

    """
    formatted_messages = format_messages_for_google(messages)
    set_kwargs_google(kwargs)
    config = GenerateContentConfig(response_mime_type="text/plain", **kwargs)
    response = GOOGLE_CLIENT.models.generate_content(
        model=model,
        contents=formatted_messages,
        config=config,
    )
    usage = {
        "input": response.usage_metadata.prompt_token_count,
        "output": response.usage_metadata.candidates_token_count,
        "total": response.usage_metadata.total_token_count,
    }
    langfuse_context.update_current_observation(
        input=messages,
        output=response.text,
        metadata=kwargs,
        model=model,
        usage=usage,
    )
    return response


@observe(capture_input=False, capture_output=False)
def chat_stream_google(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> Iterator[GenerateContentResponse]:
    """Chat with a Gemini LLM with streaming.

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
    Iterator[GenerateContentResponse]
        The streamed response from the LLM

    """
    formatted_messages = format_messages_for_google(messages)
    set_kwargs_google(kwargs)
    config = GenerateContentConfig(response_mime_type="text/plain", **kwargs)
    return GOOGLE_CLIENT.models.generate_content_stream(
        model=model,
        contents=formatted_messages,
        config=config,
    )


def chat_str_google(messages: list[dict], model: str, **kwargs: dict) -> str:
    """Chat with an LLM using the openai engine and get a string response."""
    response = chat_google(messages, model, **kwargs)
    return response.text


def format_messages_for_google(messages: list[dict]) -> list[dict]:
    """Format messages for the Gemini API.

    Parameters
    ----------
    messages : list[dict]
        The conversation history in standard format.

    Returns
    -------
    list[dict]
        The conversation history formatted for Gemini.

    """
    formatted_messages = []
    logger.info(messages)
    for msg in messages:
        if "content" in msg:
            # Simple text message
            formatted_messages.append({
                "role": "model" if msg["role"] == "assistant" else "user",
                "parts": [{"text": msg["content"]}],
            })
        elif msg["role"] == "tool_call":
            formatted_messages.append({
                "role": "model",
                "parts": [{
                    "function_call": {
                        "name": msg["name"],
                        "args": msg["args"],
                    },
                }],
            })
        elif msg["role"] == "tool": # tool result
            formatted_messages.append({
                "role": "model",
                "parts": [{
                    "function_response": {
                        "name": msg["name"],
                        "response": msg["content"],
                    },
                }],
            })
    return formatted_messages


def set_kwargs_google(kwargs: dict) -> None:
    """Set default values for genai.ChatSession API call."""
    if "max_output_tokens" not in kwargs:
        kwargs["max_output_tokens"] = MAX_TOKENS
    if "temperature" not in kwargs:
        kwargs["temperature"] = TEMPERATURE
    if "top_p" not in kwargs:
        kwargs["top_p"] = TOP_P
    if "seed" not in kwargs:
        kwargs["seed"] = SEED
