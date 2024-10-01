"""Summarization functions."""
from __future__ import annotations

from langfuse.decorators import langfuse_context, observe

from app.chat_models import chat_single_gemini, chat_str
from app.encoders import max_token_indices
from app.models import ChatModelParams, EngineEnum, GoogleModelEnum, OpenAIModelEnum, SummaryMethodEnum
from app.prompts import (
    OPINION_SUMMARY_MAP_PROMPT,
    OPINION_SUMMARY_REDUCE_PROMPT,
    SUMMARY_MAP_PROMPT,
    SUMMARY_PROMPT,
    SUMMARY_REFINE_PROMPT,
)


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

def summarize_stuff_reduce_msg(
    documents: list[str],
    chat_model: ChatModelParams,
    **kwargs: dict,
) -> dict[str, str]:
    """Combine stuffing and map-reduce.

    Concatenate (stuff) documents to be just within the LLMs context window,
    summarize the concatenated documents,
    and then reduce to a single summary.

    Parameters
    ----------
    documents : list[str]
        The documents to summarize.
    chat_model : ChatModelParams
        The chat model parameters.
    kwargs : dict, optional
        Keyword arguments for the chat function.

    Returns
    -------
    dict[str, str]
        The summary message.

    """
    # prepare token-maximized document group messages
    max_indices = max_token_indices(documents, chat_model.model)
    msgs = []
    start_idx = 0
    for max_idx in max_indices:
        concatted_docs = "\n\n".join(documents[start_idx:max_idx])
        msgs.append({
            "role": "user",
            "content": SUMMARY_PROMPT.format(text=concatted_docs),
        })
        start_idx = max_idx
    if len(msgs) == 1:
        # the documents fit into a single message, return it
        return msgs[0]
    summaries = []
    # get the document group summaries
    for msg in msgs:
        summary = chat_str([msg], chat_model, **kwargs)
        summaries.append(summary)
    # return a combined summary
    concatted_summaries = "\n\n".join(summaries)
    return {
        "role": "user",
        "content": SUMMARY_PROMPT.format(text=concatted_summaries),
    }

def summarize_map_reduce_msg(
    documents: list[str],
    chat_model: ChatModelParams,
    **kwargs: dict,
) -> dict[str, str]:
    """Summarize each document (map) and concatenate the summaries (reduce).

    Parameters
    ----------
    documents : list[str]
        The documents to summarize.
    chat_model : ChatModelParams
        The chat model parameters.
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
        summary = chat_str([doc_msg], chat_model, **kwargs)
        summaries.append(summary)
    concat_summary = "\n".join(summaries)
    prompt = SUMMARY_PROMPT.format(text=concat_summary)
    return {"role": "user", "content": prompt}

def summarize_refine_msg(
    documents: list[str],
    chat_model: ChatModelParams,
    **kwargs: dict,
) -> dict[str, str]:
    """Summarize each document with the previous summary as context and concatenate.

    Parameters
    ----------
    documents : list[str]
        The documents to summarize.
    chat_model : ChatModelParams
        The chat model parameters.
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
        summary = chat_str([doc_msg], chat_model, **kwargs)
        summaries.append(summary)
    concat_summary = "\n\n".join(summaries)
    prompt = SUMMARY_PROMPT.format(text=concat_summary)
    return {"role": "user", "content": prompt}

@observe(capture_input=False, capture_output=False)
def summarize_opinion(
    documents: list[str],
    chat_model: ChatModelParams | None = None,
    **kwargs: dict,
) -> str:
    """Summarize a judicial opinion with a custom prompt.

    Parameters
    ----------
    documents : list[str]
        A chunked opinion
    chat_model : ChatModelParams | None, optional
        An LLM and engine to use, by default None
    **kwargs : dict
        Keyword arguments for the chat function.

    Returns
    -------
    str
        A summary

    """
    prompt_msg = {"role": "system", "content": OPINION_SUMMARY_MAP_PROMPT}
    # prepare token-maximized document group messages
    max_indices = max_token_indices(documents, chat_model.model)
    msgs = []
    start_idx = 0
    for max_idx in max_indices:
        msgs.append({
            "role": "user",
            "content": "\n\n".join(documents[start_idx:max_idx]),
        })
        start_idx = max_idx
    summaries = []
    # get the document group summaries
    for msg in msgs:
        summary = chat_str([prompt_msg, msg], chat_model, **kwargs)
        summaries.append(summary)
    if len(summaries) == 1:
        # the opinion fit into a single message, return the summary
        return summaries[0]
    # return a combined summary
    prompt_msg["content"] = OPINION_SUMMARY_REDUCE_PROMPT
    partial_summaries_msg = {"role":"user", "content": "\n\n".join(summaries)}
    return chat_str(
        [prompt_msg, partial_summaries_msg],
        chat_model,
        **kwargs,
    )

@observe(capture_input=False)
def get_summary_message(
    documents: list[str],
    method: SummaryMethodEnum,
    chat_model: ChatModelParams,
    **kwargs: dict,
) -> dict:
    """Get a prompt to summarize documents using the specified model and method.

    Parameters
    ----------
    documents : list[str]
       The list of documents to summarize.
    method : str
        The summarization method, must be `stuff_reduce`, `stuffing`, `map_reduce`,
        or `refine`.
    chat_model : ChatModelParams
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
        if method is not `stuff_reduce`, `stuffing`, `map_reduce`, or `refine`.

    """
    # summarize by method
    match method:
        case SummaryMethodEnum.stuff_reduce:
            msg = summarize_stuff_reduce_msg(documents, chat_model, **kwargs)
        case SummaryMethodEnum.stuffing:
            msg = summarize_stuffing_msg(documents)
        case SummaryMethodEnum.map_reduce:
            msg = summarize_map_reduce_msg(documents, chat_model, **kwargs)
        case SummaryMethodEnum.refine:
            msg = summarize_refine_msg(documents, chat_model, **kwargs)
        case _:
            raise ValueError(method)
    return msg

@observe()
def summarize_gemini_full(docs: list[str]) -> str:
    """Summarize text using Google's Gemini model without chunking.

    Parameters
    ----------
    docs: list[str]
        list of strings that is the documents

    Returns
    -------
    str
        The summarized text

    """
    fulltext = ""
    for text in docs:
        fulltext += text
        fulltext += "\n"
    chat_model = ChatModelParams(engine=EngineEnum.google, model=GoogleModelEnum.gemini_1_5_flash)
    return chat_single_gemini(SUMMARY_PROMPT.format(text=fulltext), chat_model.model)


@observe(capture_input=False)
def summarize(
    documents: list[str],
    method: str = SummaryMethodEnum.stuff_reduce,
    chat_model: ChatModelParams | None = None,
    **kwargs: dict,
) -> str:
    """Summarize text using the specified model and method.

    Parameters
    ----------
    documents : list[str]
        The list of documents to summarize.
    method : str, optional
        The summarization method, by default `stuff_reduce`.
    chat_model : ChatModelParams, optional
        The engine and model to use for summarization, by default `openai` and
        `gpt-4o`.
    kwargs : dict, optional
        For the LLM. By default, temperature = 0 and max_tokens = 1000.

    Returns
    -------
    str
        The summarized text.

    """
    # if(method != SummaryMethodEnum.gemini_full):
    #     print("in method not gemini")
    if chat_model is None:
        chat_model = ChatModelParams(model=OpenAIModelEnum.gpt_4o)
    langfuse_context.update_current_observation(
        input={"method": method, "chat_model": chat_model},
        metadata=kwargs,
    )
    msg = get_summary_message(documents, method, chat_model, **kwargs)
    return chat_str([msg], chat_model, **kwargs)
