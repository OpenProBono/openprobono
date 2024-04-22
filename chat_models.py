from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anthropic
import openai
import openai.resources
import requests
import torch
import transformers
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from models import EngineEnum
from prompts import HIVE_QA_PROMPT

if TYPE_CHECKING:
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types.beta.tools import ToolsBetaMessage

HIVE_TASK_URL = "https://api.thehive.ai/api/v1/task/sync"

HIVE_7B = "hive-7b"
HIVE_70B = "hive-70b"

GPT_3_5 = "gpt-3.5-turbo-0125"
GPT_4 = "gpt-4"
GPT_4_TURBO = "gpt-4-turbo-preview"

ANTHROPIC_MSG_URL = "https://api.anthropic.com/v1/messages"

CLAUDE_3_OPUS = "claude-3-opus-20240229"
CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

LLAMA_3_70B_INSTRUCT = "meta-llama/Meta-Llama-3-70B-Instruct"
LLAMA_3_8B_INSTRUCT = "meta-llama/Meta-Llama-3-8B-Instruct"

MAX_TOKENS = 1000

class ChatModelParams:
    """Define a chat model for RAG."""

    def __init__(self: ChatModelParams, engine: EngineEnum, model: str) -> None:
        """Define parameters for a chat model.

        Parameters
        ----------
        engine : str
            The API/Framework on which the model runs
        model : str
            The name of the model

        """
        self.engine = engine
        self.model = model

def messages(history: list[tuple[str | None, str | None]], engine: EngineEnum) -> list:
    match engine:
        case EngineEnum.openai | EngineEnum.anthropic | EngineEnum.huggingface:
            return messages_dicts(history)
        case EngineEnum.hive:
            return messages_hive(history)
        case EngineEnum.langchain:
            return messages_langchain(history)
    return []

def messages_hive(history: list[tuple]):
    pass

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
    temperature: float,
    **kwargs: dict,
):
    match chatmodel.engine:
        case EngineEnum.hive:
            return chat_hive(messages[-1], chatmodel.model, temperature, **kwargs)
        case EngineEnum.openai:
            return chat_openai(messages, chatmodel.model, temperature, **kwargs)
        case EngineEnum.anthropic:
            return chat_anthropic(messages, chatmodel.model, temperature, **kwargs)

def chat_hive(
    message: str,
    model: str,
    temperature: float,
    **kwargs: dict,
):
    key = "HIVE_7B_API_KEY" if model == HIVE_7B else "HIVE_70B_API_KEY"
    system = kwargs.pop("system", HIVE_QA_PROMPT)
    headers = {
        "Accept": "application/json",
        "Authorization": f"Token {os.environ[key]}",
        "Content-Type": "application/json",
    }
    data = {
        "text_data": message,
        "options": {
            "max_tokens": 4096,
            "top_p": 0.95,
            "temperature": temperature,
            "system_prompt": system,
            "roles": {
                "user": "User",
                "model": "Assistant",
            },
        },
    }
    response = requests.post(HIVE_TASK_URL, headers=headers, json=data, timeout=10)
    response_json = response.json()
    return response_json["status"][0]["response"]["output"][0]["choices"][0]["message"]

def chat_openai(
    messages: list[dict],
    model: str,
    temperature: float,
    **kwargs: dict,
):
    client = kwargs.pop("client", openai.OpenAI())
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )

def chat_anthropic(
    messages: list[dict],
    model: str,
    temperature: float,
    **kwargs: dict,
) -> AnthropicMessage | ToolsBetaMessage:
    client = kwargs.pop("client", anthropic.Anthropic())
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    tools = kwargs.get("tools", [])
    endpoint = client.beta.tools.messages if tools else client.messages
    return endpoint.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )

def chat_llama(
    messages: list[dict],
    model: str,
    temperature: float,
    **kwargs: dict,
):
    temperature = min([1, max([0.01, temperature])])
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="mps",
    )
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    outputs = pipeline(
        prompt,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    return outputs[0]["generated_text"][len(prompt):]
