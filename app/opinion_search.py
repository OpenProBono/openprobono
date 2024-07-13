"""Search court opinions using CAP and courtlistener data."""
from __future__ import annotations

from langfuse.decorators import langfuse_context, observe

from app.cap import cap
from app.chat_models import summarize_langchain
from app.courtlistener import courtlistener_collection, courtlistener_search
from app.milvusdb import collection_iterator, fields_to_json, get_expr, upsert_expr_json
from app.models import OpenAIModelEnum


def cap_to_courtlistener(hit: dict) -> None:
    """Convert CAP field names to courtlistener format."""
    match hit["entity"]["jurisdiction_name"]:
        case "Ark.":
            hit["entity"]["court_id"] = "ar"
        case "Ill.":
            hit["entity"]["court_id"] = "il"
        case "N.C.":
            hit["entity"]["court_id"] = "nc"
        case "N.M.":
            hit["entity"]["court_id"] = "nm"
    del hit["entity"]["jurisdiction_name"]
    hit["entity"]["date_filed"] = hit["entity"].pop("decision_date")
    hit["entity"]["case_name"] = hit["entity"].pop("case_name_abbreviation")
    hit["entity"]["author_name"] = hit["entity"].pop("opinion_author")

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
    # get CAP results
    cap_hits = []
    if jurisdictions:
        cap_hits = cap(query, k, jurisdictions, keyword_query, after_date, before_date)
        cap_hits = cap_hits.pop("result", [])

    # get courtlistener results
    cl_result = courtlistener_search(
        query,
        k,
        jurisdictions,
        keyword_query,
        after_date,
        before_date,
    )
    cl_hits: list = cl_result["result"]

    # if there are no results it was probably a bad query date range
    if not cap_hits and not cl_hits:
        return []

    # get k closest results from either tool; merge two sorted lists to size k
    hits = []
    while len(hits) < k and len(cap_hits) > 0 and len(cl_hits) > 0:
        if cap_hits[0]["distance"] < cl_hits[0]["distance"]:
            h = cap_hits.pop(0)
            h["source"] = "cap"
            cap_to_courtlistener(h)
            hits.append(fields_to_json(h))
        else:
            h = cl_hits.pop(0)
            h["source"] = "courtlistener"
            hits.append(h)
    while len(hits) < k and len(cap_hits) > 0:
        h = cap_hits.pop(0)
        h["source"] = "cap"
        cap_to_courtlistener(h)
        hits.append(fields_to_json(h))
    while len(hits) < k and len(cl_hits) > 0:
        h = cl_hits.pop(0)
        h["source"] = "courtlistener"
        hits.append(h)

    langfuse_context.update_current_observation(
        output=[hit["entity"]["metadata"]["id"] for hit in hits],
    )
    return hits


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
    upsert_expr_json("courtlistener", f"metadata['id']=={opinion_id}", hits)
    return summary

def count_opinions() -> int:
    q_iter = collection_iterator(courtlistener_collection, "", ["metadata"], 100)
    opinions = set()
    res = q_iter.next()
    while len(res) > 0:
        for hit in res:
            if hit["metadata"]["id"] not in opinions:
                opinions.add(hit["metadata"]["id"])
        res = q_iter.next()
    q_iter.close()
    return len(opinions)
