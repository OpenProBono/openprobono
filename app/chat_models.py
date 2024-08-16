"""Load messages and chat with chat models."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

import anthropic
import google.generativeai as genai
import requests
from langchain_openai import ChatOpenAI
from langfuse.decorators import langfuse_context, observe
from openai import OpenAI, Stream

from app.models import ChatModelParams, EngineEnum, HiveModelEnum
from app.prompts import HIVE_QA_PROMPT

if TYPE_CHECKING:
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import RawMessageStreamEvent
    from langchain.llms.base import BaseLanguageModel
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

HIVE_TASK_URL = "https://api.thehive.ai/api/v2/task/sync"
MAX_TOKENS = 1000
NOT_GIVEN = "NOT_GIVEN"

def chat(
    messages: list[dict],
    chatmodel: ChatModelParams,
    **kwargs: dict,
) -> tuple[str, list[str]] | ChatCompletion | Stream[ChatCompletionChunk] |\
    AnthropicMessage | Stream[RawMessageStreamEvent] | str:
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
    tuple[str, list[str]] | ChatCompletion | Stream[ChatCompletionChunk] |
    AnthropicMessage | Stream[RawMessageStreamEvent] | str

        The response from the LLM. Depends on engine and if streaming is enabled.

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


def chat_str_fn(chatmodel: ChatModelParams) -> Callable:
    """Get a chat function that returns strings for the given chat model.

    Parameters
    ----------
    chatmodel : ChatModelParams
        The chat model to use for the conversation.

    Returns
    -------
    Callable
        A function that returns string responses from the chat model.

    Raises
    ------
    ValueError
        If the given chat model is not supported.

    """
    match chatmodel.engine:
        case EngineEnum.openai:
            return chat_str_openai
        case EngineEnum.anthropic:
            return chat_str_anthropic
        case EngineEnum.hive:
            return chat_str_hive
        case EngineEnum.google:
            return chat_gemini
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


@observe(as_type="generation")
def chat_str_hive(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> str:
    text, _ = chat_hive(messages, model, **kwargs)
    return text


@observe(as_type="generation")
def chat_openai(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> ChatCompletion | Stream[ChatCompletionChunk]:
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
    client = kwargs.get("client", OpenAI())
    max_tokens = kwargs.get("max_tokens", MAX_TOKENS)
    temperature = kwargs.get("temperature", 0.0)
    seed = kwargs.get("seed", 0)
    tools = kwargs.get("tools", NOT_GIVEN)
    tool_choice = kwargs.get("tool_choice", NOT_GIVEN)
    stream = kwargs.get("stream", False)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        tools=tools,
        tool_choice=tool_choice,
        stream=stream,
    )
    if not isinstance(response, Stream):
        usage = {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        }
        langfuse_context.update_current_observation(
            input=messages,
            output=response.choices[0].message,
            model=model,
            usage=usage,
        )
    return response


@observe(as_type="generation")
def chat_str_openai(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> str:
    response = chat_openai(messages, model, **kwargs)
    return response.choices[0].message.content.strip()


@observe(as_type="generation")
def chat_anthropic(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> AnthropicMessage | Stream[RawMessageStreamEvent]:
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
    AnthropicMessage | Stream[RawMessageStreamEvent]
        The response from the LLM.

    """
    client = kwargs.get("client", anthropic.Anthropic())
    max_tokens = kwargs.get("max_tokens", MAX_TOKENS)
    temperature = kwargs.get("temperature", 0.0)
    tools = kwargs.get("tools", NOT_GIVEN)
    system = kwargs.get("system", NOT_GIVEN)
    stream = kwargs.get("stream", False)
    response = client.messages.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
        system=system,
        stream=stream,
    )
    # report input, output, model, usage to langfuse
    if not isinstance(response, Stream):
        usage = {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
            "total": response.usage.input_tokens + response.usage.output_tokens,
        }
        langfuse_context.update_current_observation(
            input=messages,
            model=model,
            output=response.content,
            usage=usage,
        )
    return response


@observe(as_type="generation")
def chat_str_anthropic(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> str:
    response = chat_anthropic(messages, model, **kwargs)
    return "\n".join([
        block.text for block in response.content if block.type == "text"
    ])


@observe(as_type="generation")
def chat_gemini(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> str:
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
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    # Create the model
    # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    temperature = kwargs.pop("temperature", 0.0)
    top_p = kwargs.pop("top_p", 0.95)
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 64,
        "max_output_tokens": max_tokens,
        "response_mime_type": "text/plain",
    }

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


def get_langchain_chat_model(model: str, **kwargs: dict) -> BaseLanguageModel:
    """Load a LangChain BaseLanguageModel.

    Currently only supports OpenAI models.

    Parameters
    ----------
    model : str
        The name of the LLM to load.
    kwargs : dict
        For the LLM. By default, temperature = 0 and max_tokens = 1000.

    Returns
    -------
    BaseLanguageModel
        The loaded LLM.

    """
    temperature = kwargs.pop("temperature", 0.0)
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
