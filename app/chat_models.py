"""Load messages and chat with chat models."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import anthropic
import google.generativeai as genai
import requests
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document as LCDocument
from langchain_openai import ChatOpenAI
from langfuse.decorators import langfuse_context, observe
from openai import OpenAI
from unstructured.documents.elements import Element

from app.encoders import token_count
from app.models import (
    AnthropicModelEnum,
    ChatModelParams,
    EngineEnum,
    HiveModelEnum,
    OpenAIModelEnum,
    SummaryMethodEnum,
)
from app.prompts import (
    HIVE_QA_PROMPT,
    MODERATION_PROMPT,
)
from app.summarization import (
    summarize_map_reduce_msg,
    summarize_refine_msg,
    summarize_stuffing_msg,
)

if TYPE_CHECKING:
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types.beta.tools import ToolsBetaMessage
    from langchain.llms.base import BaseLanguageModel
    from openai.types.chat import ChatCompletion

HIVE_TASK_URL = "https://api.thehive.ai/api/v2/task/sync"
MAX_TOKENS = 1000

def messages(
    history: list[tuple[str | None, str | None]],
    engine: EngineEnum,
) -> list[dict]:
    """Convert conversation history into the right format for the given engine.

    Parameters
    ----------
    history : list[tuple[str  |  None, str  |  None]]
        The conversation history to convert.
    engine : EngineEnum
        The engine to use for the conversation.

    Returns
    -------
    list[dict]
        The converted conversation history.

    Raises
    ------
    ValueError
        If engine is not `openai`, `anthropic`, `hive`, or `langchain`.

    """
    match engine:
        case EngineEnum.openai | EngineEnum.anthropic | EngineEnum.hive:
            return messages_dicts(history)
        case EngineEnum.google:
            return messages_gemini(history)
    raise ValueError(engine)


def messages_dicts(
    history: list[tuple[str | None, str | None]],
) -> list[dict]:
    """Convert conversation history into dictionary format.

    Parameters
    ----------
    history : list[tuple[str  |  None, str  |  None]]
        The original conversation history.

    Returns
    -------
    list[dict]
        The converted conversation history.

    """
    messages = []
    for tup in history:
        if tup[0]:
            messages.append({"role": "user", "content": tup[0]})
        if tup[1]:
            messages.append({"role": "assistant", "content": tup[1]})
    return messages


def messages_gemini(
    history: list[tuple[str | None, str | None]],
) -> list[dict]:
    """Convert conversation history into Gemini API dictionary format.

    Parameters
    ----------
    history : list[tuple[str  |  None, str  |  None]]
        The original conversation history.

    Returns
    -------
    list[dict]
        The converted conversation history.

    """
    messages = []
    for tup in history:
        if tup[0]:
            messages.append({"role": "user", "parts": [tup[0]]})
        if tup[1]:
            messages.append({"role": "model", "parts": [tup[1]]})
    return messages


def chat(
    messages: list[dict],
    chatmodel: ChatModelParams,
    **kwargs: dict,
) -> tuple[str, list[str]] | ChatCompletion | AnthropicMessage | ToolsBetaMessage:
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
    tuple[str, list[str]] | ChatCompletion | AnthropicMessage | ToolsBetaMessage
        The response from the LLM.

    Raises
    ------
    ValueError
        If the given chat model is not supported.

    """
    match chatmodel.engine:
        case EngineEnum.openai:
            return chat_openai(messages, chatmodel.model, **kwargs)
        case EngineEnum.hive:
            return chat_hive(messages, chatmodel.model, **kwargs)
        case EngineEnum.anthropic:
            return chat_anthropic(messages, chatmodel.model, **kwargs)


@observe(as_type="generation")
def chat_hive(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> tuple[str, list[str]]:
    use_embedding = kwargs.pop("use_embedding", False)
    system = kwargs.pop("system", HIVE_QA_PROMPT)
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    temperature = kwargs.pop("temperature", 0.0)
    top_p = kwargs.pop("top_p", 0.95)
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
def chat_openai(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> ChatCompletion:
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
    client = kwargs.pop("client", OpenAI())
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    temperature = kwargs.pop("temperature", 0.0)
    seed = kwargs.pop("seed", 0)
    response: ChatCompletion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
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
        model=model,
        usage=usage,
    )
    return response


@observe(as_type="generation")
def chat_anthropic(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> AnthropicMessage | ToolsBetaMessage:
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
    AnthropicMessage | ToolsBetaMessage
        The response from the LLM.

    """
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


@observe(as_type="generation")
def chat_gemini(
    messages: list[dict],
    model: str,
    **kwargs: dict,
) -> str:
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
        chatmodel = ChatModelParams(
            engine=EngineEnum.openai,
            model=OpenAIModelEnum.mod_latest,
        )
    match chatmodel.engine:
        case EngineEnum.openai:
            return moderate_openai(message, chatmodel.model, client)
        case EngineEnum.anthropic:
            return moderate_anthropic(message, chatmodel.model, client)
    msg = f"Unsupported engine: {chatmodel.engine}"
    raise ValueError(msg)


@observe()
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


@observe()
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
            msg = summarize_stuffing_msg(documents)
        case SummaryMethodEnum.map_reduce:
            msg = summarize_map_reduce_msg(documents, chatmodel, chat_fn, **kwargs)
        case SummaryMethodEnum.refine:
            msg = summarize_refine_msg(documents, chatmodel, chat_fn, **kwargs)
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
    raise ValueError(chatmodel.engine)


@observe(capture_input=False)
def summarize_langchain(
    documents: list[str | Element | LCDocument],
    model: str,
    **kwargs: dict,
) -> str:
    """Summarize the documents using a chain.

    Parameters
    ----------
    documents : list[str | Element | LCDocument]
        The list of documents to summarize.
    model : str
        The model to use for summarization.
    kwargs : dict, optional
        For the LLM. By default, temperature = 0 and max_tokens = 1000.

    Returns
    -------
    str
        The summarized text.

    """
    max_chunk_indexes = documents_max_tokens_index(documents, 120000)
    # convert to Documents if strs were given
    if isinstance(documents[0], str):
        documents = [LCDocument(page_content=doc) for doc in documents]
    chain_type = "stuff"
    chain = load_summarize_chain(
        get_langchain_chat_model(model, **kwargs),
        chain_type=chain_type,
    )
    # get the langchain handler for the current trace
    langfuse_handler = langfuse_context.get_current_langchain_handler()
    # if we need to summarize groups of documents and then summarize those groups
    # because context is too long
    if len(max_chunk_indexes) > 1:
        # summarize each group of documents
        last = 0
        summaries = []
        for max_chunk_index in max_chunk_indexes:
            result = chain.invoke({"input_documents": documents[last: max_chunk_index]}, config={"callbacks": [langfuse_handler]})
            summaries.append(result["output_text"].strip())
            last = max_chunk_index
        # summarize the document group summaries
        documents = [LCDocument(page_content=doc) for doc in summaries]
    result = chain.invoke({"input_documents": documents}, config={"callbacks": [langfuse_handler]})
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


def documents_max_tokens_index(documents: list[str | Element | LCDocument], max_tokens: int) -> list[int]:
    # hard limit on the number of tokens to be summarized, for cost and rate limits
    tokens = 0
    # count tokens to find the number of documents to summarize
    # need an accurate tokenizer for anthropic models, so use OpenAI's for now
    embedding_model = OpenAIModelEnum.embed_small
    indices = []
    for i, doc in enumerate(documents, start=1):
        if isinstance(doc, str):
            tokens += token_count(doc, embedding_model)
        elif isinstance(doc, Element):
            tokens += token_count(doc.text, embedding_model)
        elif isinstance(doc, LCDocument):
            tokens += token_count(doc.page_content, embedding_model)
        if tokens > max_tokens:
            tokens = 0
            indices.append(i)
    indices.append(len(documents))
    return indices
