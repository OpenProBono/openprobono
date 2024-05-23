"""Load messages and chat with chat models."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anthropic
import requests
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import OpenAI
from unstructured.documents.elements import Element

from models import (
    AnthropicModelEnum,
    ChatModelParams,
    EngineEnum,
    HiveModelEnum,
    OpenAIModelEnum,
    SummaryMethodEnum,
)
from prompts import (
    HIVE_QA_PROMPT,
    MODERATION_PROMPT,
    SUMMARY_MAP_PROMPT,
    SUMMARY_PROMPT,
    SUMMARY_REFINE_PROMPT,
)

if TYPE_CHECKING:
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types.beta.tools import ToolsBetaMessage
    from langchain.llms.base import BaseLanguageModel
    from openai.types.chat import ChatCompletion

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
    for tup in history:
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

@observe(as_type="generation")
def chat_hive(
    messages: list,
    model: str,
    **kwargs: dict,
) -> tuple[str, list[str]]:
    key = "HIVE_7B_API_KEY" if model == HiveModelEnum.hive_7b else "HIVE_70B_API_KEY"
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
    chatmodel: ChatModelParams | None = None,
    client: OpenAI | anthropic.Anthropic | None = None,
) -> bool:
    """Moderate a message using the specified engine and model.

    Parameters
    ----------
    message : str
        The message to be moderated.
    chatmodel : ChatModelParams, optional
        The engine and model to use for moderation, by default
        openai and text-moderation-latest.
    client : OpenAI | anthropic.Anthropic, optional
        The client to use for the moderation request. If not specified,
        one will be created.

    Returns
    -------
    bool
        True if flagged; False otherwise.

    Raises
    ------
    ValueError
        If chatmodel.engine is not `openai` or `anthropic`.

    """
    if chatmodel is None:
        chatmodel = ChatModelParams(EngineEnum.openai, OpenAIModelEnum.mod_latest)
    match chatmodel.engine:
        case EngineEnum.openai:
            return moderate_openai(message, chatmodel.model, client)
        case EngineEnum.anthropic:
            return moderate_anthropic(message, chatmodel.model, client)
    msg = f"Unsupported engine: {chatmodel.engine}"
    raise ValueError(msg)

def moderate_openai(
    message: str,
    model: str = OpenAIModelEnum.mod_latest,
    client: OpenAI | None = None,
) -> bool:
    """Moderate a message using OpenAI's Moderation API.

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
    model: str = AnthropicModelEnum.claude_3_haiku,
    client: anthropic.Anthropic | None = None,
) -> bool:
    """Moderate a message using an Anthropic model.

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

def get_summary_message(
    documents: list[str | Element],
    method: SummaryMethodEnum,
    chatmodel: ChatModelParams,
    **kwargs: dict,
) -> dict:
    """Get a prompt to summarize documents using the specified model and method.

    Parameters
    ----------
    documents : list[str  |  Element]
       The list of documents to summarize.
    method : str
        The summarization method, must be `stuffing`, `map_reduce`, or `refine`.
    chatmodel : ChatModelParams
        The engine and model to use for summarization.
    kwargs : dict, optional
        For the LLM. By default, temperature = 0 and max_tokens = 1000.

    Returns
    -------
    dict
        The final prompt to get a summary of all the documents.

    Raises
    ------
    ValueError
        If chatmodel.engine is not `openai` or `anthropic`.
    ValueError
        if method is not `stuffing`, `map_reduce`, or `refine`.

    """
    match chatmodel.engine:
        case EngineEnum.openai:
            chat_fn = chat_openai
        case EngineEnum.anthropic:
            chat_fn = chat_anthropic
        case _:
            raise ValueError(chatmodel.engine)
    # convert documents to text if elements were given
    if isinstance(documents[0], Element):
        documents = [doc.text for doc in documents]
    # summarize by method
    match method:
        case SummaryMethodEnum.stuffing:
            # concatenate the documents into a single document
            text = "\n".join(documents)
            prompt = SUMMARY_PROMPT.format(text=text)
            msg = {"role":"user", "content":prompt}
        case SummaryMethodEnum.map_reduce:
            # summarize each document (map) and concatenate the summaries (reduce)
            summaries = []
            for doc in documents:
                prompt = SUMMARY_MAP_PROMPT.format(text=doc)
                doc_msg = {"role":"user", "content":prompt}
                # response is ChatCompletion if openai or AnthropicMessage if anthropic
                response = chat_fn([doc_msg], chatmodel.model, **kwargs)
                if chatmodel.engine == EngineEnum.openai:
                    summary = response.choices[0].message.content
                else:
                    summary = "\n".join([
                        block.text for block in response.content if block.type == "text"
                    ])
                summaries.append(summary)
            concat_summary = "\n".join(summaries)
            prompt = SUMMARY_PROMPT.format(text=concat_summary)
            msg = {"role":"user", "content":prompt}
        case SummaryMethodEnum.refine:
            # summarize each document with the previous summary as context
            # and concatenate them
            summaries = []
            for i, doc in enumerate(documents):
                if i == 0:
                    prompt = SUMMARY_REFINE_PROMPT.format(context="", text=doc)
                else:
                    prompt = SUMMARY_REFINE_PROMPT.format(
                        context=summaries[i - 1],
                        text=doc,
                    )
                doc_msg = {"role":"user", "content":prompt}
                response = chat_fn([doc_msg], chatmodel.model, **kwargs)
                if chatmodel.engine == EngineEnum.openai:
                    summary = response.choices[0].message.content
                else:
                    summary = "\n".join([
                        block.text for block in response.content if block.type == "text"
                    ])
                summaries.append(summary)
            concat_summary = "\n".join(summaries)
            prompt = SUMMARY_PROMPT.format(text=concat_summary)
            msg = {"role":"user", "content":prompt}
        case _:
            raise ValueError(method)
    return msg

@observe(capture_input=False)
def summarize(
    documents: list[str | Element | LCDocument],
    method: str = SummaryMethodEnum.stuffing,
    chatmodel: ChatModelParams | None = None,
    **kwargs: dict,
) -> str:
    """Summarize text using the specified model and method.

    Parameters
    ----------
    documents : list[str  |  Element  |  LCDocument]
        The list of documents to summarize.
    method : str, optional
        The summarization method, by default `stuffing`.
    chatmodel : ChatModelParams, optional
        The engine and model to use for summarization, by default `langchain` and
        `gpt-3.5-turbo-0125`.
    kwargs : dict, optional
        For the LLM. By default, temperature = 0 and max_tokens = 1000.

    Returns
    -------
    str
        The summarized text.

    Raises
    ------
    ValueError
        If chatmodel.engine is not `openai`, `anthropic`, or `langchain`.

    """
    # hard limit on the number of tokens to be summarized, for cost and rate limits
    max_summary_chunks = 200
    max_summary_tokens = 150000
    chatmodel = ChatModelParams() if chatmodel is None else chatmodel
    from encoders import token_count
    tokens = 0
    # need an accurate tokenizer for anthropic models, so use gpt_3_5 for now
    if chatmodel.engine == EngineEnum.anthropic:
        model = OpenAIModelEnum.gpt_3_5
    else:
        model = chatmodel.model
    # count tokens to find the number of documents to summarize
    for i, doc in enumerate(documents, start=1):
        if isinstance(doc, str):
            tokens += token_count(doc, model)
        elif isinstance(doc, Element):
            tokens += token_count(doc.text, model)
        elif isinstance(doc, LCDocument):
            tokens += token_count(doc.page_content, model)
        if tokens > max_summary_tokens:
            max_summary_chunks = i
            break
    documents = documents[:max_summary_chunks]
    langfuse_context.update_current_observation(
        input={"method":method, "num_docs":len(documents), **kwargs},
    )
    match chatmodel.engine:
        case EngineEnum.openai:
            client = kwargs.pop("client", OpenAI())
            msg = get_summary_message(
                documents,
                method,
                chatmodel,
                client=client,
                **kwargs,
            )
            response = chat_openai([msg], chatmodel.model, client=client, **kwargs)
            return response.choices[0].message.content.strip()
        case EngineEnum.anthropic:
            client = kwargs.pop("client", anthropic.Anthropic())
            msg = get_summary_message(
                documents,
                method,
                chatmodel,
                client=client,
                **kwargs,
            )
            response = chat_anthropic([msg], chatmodel.model, client=client, **kwargs)
            return "\n".join([
                block.text for block in response.content if block.type == "text"
            ])
        case EngineEnum.langchain:
            return summarize_langchain(documents, method, chatmodel.model, **kwargs)
    raise ValueError(chatmodel.engine)

def summarize_langchain(
    documents: list[str | LCDocument],
    method: str,
    model: str,
    **kwargs: dict,
) -> str:
    """Summarize the documents using a chain.

    Parameters
    ----------
    documents : list[str | LCDocument]
        The list of documents to summarize.
    method : str
        The summarization method.
    model : str
        The model to use for summarization.
    kwargs : dict, optional
        For the LLM. By default, temperature = 0 and max_tokens = 1000.

    Returns
    -------
    str
        The summarized text.

    """
    # convert to Documents if strs were given
    if isinstance(documents[0], str):
        documents = [LCDocument(page_content=doc) for doc in documents]
    chain_type = method if method != SummaryMethodEnum.stuffing else "stuff"
    chain = load_summarize_chain(
        get_langchain_chat_model(model, **kwargs),
        chain_type=chain_type,
    )
    result = chain.invoke({"input_documents": documents})
    return result["output_text"].strip()

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
