"""Summarization functions."""
from __future__ import annotations

from typing import Callable

from models import ChatModelParams, EngineEnum
from prompts import SUMMARY_MAP_PROMPT, SUMMARY_PROMPT, SUMMARY_REFINE_PROMPT


def summarize_stuffing_msg(documents: list[str]) -> dict[str, str]:
    """Concatenate the documents into a single document.

    Parameters
    ----------
    documents : list[str]
        The documents to concatenate.

    Returns
    -------
    dict : dict[str, str]
        The summary message.

    """
    text = "\n".join(documents)
    prompt = SUMMARY_PROMPT.format(text=text)
    return {"role":"user", "content":prompt}

def summarize_map_reduce_msg(
    documents: list[str],
    chatmodel: ChatModelParams,
    chat_fn: Callable,
    **kwargs: dict,
) -> dict[str, str]:
    """Summarize each document (map) and concatenate the summaries (reduce).

    Parameters
    ----------
    documents : list[str]
        The documents to summarize.
    chatmodel : ChatModelParams
        The chat model parameters.
    chat_fn : Callable
        The chat function to call.
    kwargs : dict, optional
        Keyword arguments for the chat function.

    Returns
    -------
    dict[str, str]
        The summary message.

    """
    summaries = []
    for doc in documents:
        prompt = SUMMARY_MAP_PROMPT.format(text=doc)
        doc_msg = {"role":"user", "content":prompt}
        # response is ChatCompletion if openai or AnthropicMessage if anthropic
        if chatmodel.engine == EngineEnum.openai:
            response = chat_fn([doc_msg], chatmodel.model, **kwargs)
            summary = response.choices[0].message.content
        else:
            summary = "\n".join([
                block.text for block in response.content if block.type == "text"
            ])
        summaries.append(summary)
    concat_summary = "\n".join(summaries)
    prompt = SUMMARY_PROMPT.format(text=concat_summary)
    return {"role":"user", "content":prompt}

def summarize_refine_msg(
    documents: list[str],
    chatmodel: ChatModelParams,
    chat_fn: Callable,
    **kwargs: dict,
) -> dict[str, str]:
    """Summarize each document with the previous summary as context and concatenate.

    Parameters
    ----------
    documents : list[str]
        The documents to summarize.
    chatmodel : ChatModelParams
        The chat model parameters.
    chat_fn : Callable
        The chat function to call.
    kwargs : dict, optional
        Keyword arguments for the chat function.

    Returns
    -------
    dict[str, str]
        The summary message.

    """
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
    return {"role":"user", "content":prompt}
