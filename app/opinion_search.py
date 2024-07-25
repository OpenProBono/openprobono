"""Search court opinions using CAP and courtlistener data."""
from __future__ import annotations

import time

from langfuse.decorators import langfuse_context, observe

from app.chat_models import summarize_langchain
from app.courtlistener import courtlistener_collection, courtlistener_search
from app.milvusdb import collection_iterator, get_expr, upsert_expr_json
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
    print("opinion search time: " + str(end - start))
    return cl_hits


@observe()
def summarize_opinion(opinion_id: int) -> str:
    res = get_expr("courtlistener", f"metadata['id']=={opinion_id}")
    hits = res["result"]
    hits = sorted(hits, key= lambda x: x["metadata"]["id"])
    texts = [hit["text"] for hit in hits]
    summary = summarize_langchain(texts, OpenAIModelEnum.gpt_4o)
    for hit in hits:
        hit["metadata"]["ai_summary"] = summary
    # save the summary to Milvus for future searches
    upsert_expr_json(courtlistener_collection, f"metadata['id']=={opinion_id}", hits)
    return summary

def count_opinions() -> int:
    coll_iter = collection_iterator(courtlistener_collection, "", ["metadata"], 100)
    opinions = set()
    res = coll_iter.next()
    start = time.time()
    while len(res) > 0:
        for hit in res:
            if hit["metadata"]["id"] not in opinions:
                opinions.add(hit["metadata"]["id"])
        res = coll_iter.next()
    coll_iter.close()
    end = time.time()
    print("summarization time: " + str(end - start))
    return len(opinions)
