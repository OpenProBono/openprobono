"""Search court opinions using CAP and courtlistener data."""
from __future__ import annotations

from langfuse.decorators import observe

from app.courtlistener import courtlistener_collection, courtlistener_query
from app.milvusdb import get_expr, upsert_expr
from app.models import ChatModelParams, OpinionSearchRequest
from app.summarization import summarize_opinion


@observe(capture_output=False)
def opinion_search(request: OpinionSearchRequest) -> list[dict]:
    """Search CAP and courtlistener collections for relevant opinions.

    Parameters
    ----------
    request : OpinionSearchRequest
        The opinion search request object

    Returns
    -------
    list[dict]
        A list of dicts containing the results from the search query

    """
    # get courtlistener results
    cl_result = courtlistener_query(request)
    return cl_result["result"]


@observe()
def add_opinion_summary(opinion_id: int) -> str:
    """Summarize an opinion and update its entries in Milvus.

    Parameters
    ----------
    opinion_id : int
        The opinion_id of chunks in Milvus to summarize

    Returns
    -------
    str
        The opinion summary

    """
    res = get_expr(courtlistener_collection, f"opinion_id=={opinion_id}")
    hits = res["result"]
    hits = sorted(hits, key=lambda x: x["pk"])
    texts = [hit["text"] for hit in hits]
    summary = summarize_opinion(texts, ChatModelParams(model="gpt-4o"))
    for hit in hits:
        hit["metadata"]["ai_summary"] = summary
        del hit["pk"]
    # save the summary to Milvus for future searches
    upsert_expr(courtlistener_collection, f"opinion_id=={opinion_id}", hits)
    return summary


def count_opinions() -> int:
    # hardcoded for now
    return 1248514
