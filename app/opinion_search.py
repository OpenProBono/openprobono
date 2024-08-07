"""Search court opinions using CAP and courtlistener data."""
from __future__ import annotations

import time

from langfuse.decorators import langfuse_context, observe

from app.chat_models import summarize_langchain
from app.courtlistener import courtlistener_collection, courtlistener_search
from app.milvusdb import MAX_K, get_expr, query_iterator, upsert_expr_json
from app.models import OpenAIModelEnum


@observe(capture_output=False)
def opinion_search(
    query: str,
    k: int = 10,
    jurisdictions: list[str] | None = None,
    keyword_query: str | None = None,
    after_date: str | None = None,
    before_date: str | None = None,
) -> list[dict]:
    """Search CAP and courtlistener collections for relevant opinions.

    Parameters
    ----------
    query : str
        The users semantic query
    k : int, optional
        The number of results to return, by default 10
    jurisdictions : str | None, optional
        The jurisdictions to filter by, by default None
    keyword_query: str | None, optional
        The users keyword query, by default None
    after_date : str | None, optional
        The after date to filter by, by default None
    before_date : str | None, optional
        The before date to filter by, by default None

    Returns
    -------
    list[dict]
        A list of dicts containing the results from the search query

    """
    start = time.time()
    # get courtlistener results
    cl_result = courtlistener_search(
        query,
        k,
        jurisdictions,
        keyword_query,
        after_date,
        before_date,
    )
    cl_hits = cl_result["result"]
    langfuse_context.update_current_observation(
        output=[hit["entity"]["metadata"]["id"] for hit in cl_hits],
    )
    end = time.time()
    print(f"opinion search time: {end - start!s}")
    return cl_hits


@observe()
def summarize_opinion(opinion_id: int) -> str:
    start = time.time()
    res = get_expr(courtlistener_collection, f"metadata['id']=={opinion_id}")
    hits = res["result"]
    hits = sorted(hits, key= lambda x: x["metadata"]["id"])
    texts = [hit["text"] for hit in hits]
    summary = summarize_langchain(texts, OpenAIModelEnum.gpt_4o)
    for hit in hits:
        hit["metadata"]["ai_summary"] = summary
    # save the summary to Milvus for future searches
    upsert_expr_json(courtlistener_collection, f"metadata['id']=={opinion_id}", hits)
    end = time.time()
    print(f"summarization time: {end - start!s}")
    return summary

def count_opinions() -> int:
    # start = time.time()
    # q_iter = query_iterator(courtlistener_collection, "", ["opinion_id"], MAX_K)
    # opinions = set()
    # res = q_iter.next()
    # while len(res) > 0:
    #     for hit in res:
    #         if hit["opinion_id"] not in opinions:
    #             opinions.add(hit["opinion_id"])
    #     res = q_iter.next()
    # q_iter.close()
    # end = time.time()
    # print(f"opinion count time: {end - start!s}")
    return 623662
