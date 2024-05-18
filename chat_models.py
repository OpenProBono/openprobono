"""Load messages and chat with chat models."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anthropic
import requests
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import OpenAI

from models import (
    AnthropicChatModel,
    ChatModelParams,
    EngineEnum,
    HiveChatModel,
    OpenAIModerationModel,
)
from prompts import HIVE_QA_PROMPT, MODERATION_PROMPT

if TYPE_CHECKING:
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types.beta.tools import ToolsBetaMessage
    from langchain_core.documents import Document as LCDocument
    from openai.types.chat import ChatCompletion
    from unstructured.documents.elements import Element

HIVE_TASK_URL = "https://api.thehive.ai/api/v1/task/sync"
MAX_TOKENS = 1000

def messages(
    history: list[tuple[str | None, str | None]],
    engine: EngineEnum,
) -> list[dict] | list[BaseMessage]:
    match engine:
        case EngineEnum.openai | EngineEnum.anthropic | EngineEnum.hive:
            return messages_dicts(history)
        case EngineEnum.langchain:
            return messages_langchain(history)
    raise ValueError(engine)

def messages_dicts(
    history: list[tuple[str | None, str | None]],
) -> list[dict]:
    messages = []
    for tup in history:
        if tup[0]:
            messages.append({"role": "user", "content": tup[0]})
        if tup[1]:
            messages.append({"role": "assistant", "content": tup[1]})
    return messages

def messages_langchain(
        history: list[tuple[str | None, str | None]],
) -> list[BaseMessage]:
    messages = []
    for tup in history[1:len(history) - 1]:
        if tup[0]:
            messages.append(HumanMessage(content=tup[0]))
        if tup[1]:
            messages.append(AIMessage(content=tup[1]))
    return messages

def chat(
    messages: list,
    chatmodel: ChatModelParams,
    **kwargs: dict,
) -> (tuple[str, list[str]] | ChatCompletion | AnthropicMessage | ToolsBetaMessage):
    match chatmodel.engine:
        case EngineEnum.hive:
            return chat_hive(messages, chatmodel.model, **kwargs)
        case EngineEnum.openai:
            return chat_openai(messages, chatmodel.model, **kwargs)
        case EngineEnum.anthropic:
            return chat_anthropic(messages, chatmodel.model, **kwargs)
        case EngineEnum.langchain:
            msg = "langchain chat function must be implemented manually"
    raise ValueError(msg)

def chat_hive(
    messages: list,
    model: str,
    **kwargs: dict,
) -> tuple[str, list[str]]:
    key = "HIVE_7B_API_KEY" if model == HiveChatModel.HIVE_7B else "HIVE_70B_API_KEY"
    system = kwargs.pop("system", HIVE_QA_PROMPT)
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    temperature = kwargs.pop("temperature", 0.0)
    top_p = kwargs.pop("top_p", 0.95)
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
    response = requests.post(HIVE_TASK_URL, headers=headers, json=data, timeout=30)
    response_json = response.json()
    output = response_json["status"][0]["response"]["output"][0]
    message = output["choices"][0]["message"]
    chunks = output["augmentations"]
    return message, chunks

def chat_openai(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> ChatCompletion:
    client = kwargs.pop("client", OpenAI())
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    temperature = kwargs.pop("temperature", 0.0)
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )

@observe(as_type="generation")
def chat_anthropic(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> AnthropicMessage | ToolsBetaMessage:
    client = kwargs.pop("client", anthropic.Anthropic())
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    tools = kwargs.get("tools", [])
    temperature = kwargs.pop("temperature", 0.0)
    endpoint = client.beta.tools.messages if tools else client.messages
    response: AnthropicMessage | ToolsBetaMessage = endpoint.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
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
        usage=usage,
    )
    return response

def moderate(
    message: str,
    engine: EngineEnum,
    model: str,
    client: OpenAI | anthropic.Anthropic | None = None,
) -> bool:
    """Moderates the message using the specified engine and model.

    Parameters
    ----------
    message : str
        The message to be moderated.
    engine : EngineEnum:
        The moderation engine to use.
    model : str
        The moderation model to use.
    client : OpenAI | anthropic.Anthropic, optional
        The client to use for the moderation request. If not specified,
        one will be created.

    Returns
    -------
    bool
        True if flagged; False otherwise.

    """
    match engine:
        case EngineEnum.openai:
            return moderate_openai(message, model, client)
        case EngineEnum.anthropic:
            return moderate_anthropic(message, model, client)
    msg = f"Unsupported engine: {engine}"
    raise ValueError(msg)

def moderate_openai(
    message: str,
    model: str = OpenAIModerationModel.LATEST.value,
    client: OpenAI | None = None,
) -> bool:
    """Moderates the message using OpenAI's Moderation API.

    Parameters
    ----------
    message : str
        The message to be moderated.
    model : str, optional
        The model to use for moderation, by default `text-moderation-latest`.
    client : OpenAI, optional
        The client to use, by default None.

    Returns
    -------
    bool
        True if flagged; False otherwise.

    """
    client = OpenAI() if client is None else client
    response = client.moderations.create(model=model, input=message)
    return response.results[0].flagged

def moderate_anthropic(
    message: str,
    model: str = AnthropicChatModel.CLAUDE_3_HAIKU.value,
    client: anthropic.Anthropic | None = None,
) -> bool:
    """Moderates the message using an Anthropic model.

    Parameters
    ----------
    message : str
        The message to be moderated.
    model : str, optional
        The model to use for moderation, by default `claude-3-haiku-20240307`.
    client : Anthropic, optional
        The client to use, by default None.

    Returns
    -------
    bool
        True if flagged; False otherwise.

    """
    client = anthropic.Anthropic() if client is None else client
    moderation_msg = {
        "role": "user",
        "content": MODERATION_PROMPT.format(user_input=message),
    }
    response = client.messages.create(
        model=model,
        max_tokens=10,
        temperature=0,
        messages=[moderation_msg],
    )
    return "Y" in response.content[-1].text.strip()
