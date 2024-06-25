"""Search court opinions using CAP and courtlistener data."""
from __future__ import annotations

from langfuse.decorators import observe

from app.cap import cap
from app.courtlistener import courtlistener_search
from app.milvusdb import fields_to_json


def cap_to_courtlistener(hit: dict, jurisdiction: str) -> None:
    """Convert CAP field names to courtlistener format."""
    del hit["entity"]["jurisdiction_name"]
    hit["entity"]["court_id"] = jurisdiction
    hit["entity"]["date_filed"] = hit["entity"].pop("decision_date")
    hit["entity"]["case_name"] = hit["entity"].pop("case_name_abbreviation")
    hit["entity"]["author_name"] = hit["entity"].pop("opinion_author")

@observe()
def opinion_search(
    query: str,
    k: int = 10,
    jurisdiction: str | None = None,
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
    jurisdiction : str | None, optional
        The jurisdiction to filter by, by default None
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
    if jurisdiction == "ar":
        cap_hits = cap(query, k, "Ark.", keyword_query, after_date, before_date)
    elif jurisdiction == "il":
        cap_hits = cap(query, k, "Ill.", keyword_query, after_date, before_date)
    elif jurisdiction == "nc":
        cap_hits = cap(query, k, "N.C.", keyword_query, after_date, before_date)
    elif jurisdiction == "nm":
        cap_hits = cap(query, k, "N.M.", keyword_query, after_date, before_date)
    cap_hits = cap_hits["result"] if cap_hits else cap_hits
    # get courtlistener results
    cl_result = courtlistener_search(
        query,
        k,
        jurisdiction,
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
            cap_to_courtlistener(h, jurisdiction)
            hits.append(fields_to_json(h))
        else:
            h = cl_hits.pop(0)
            h["source"] = "courtlistener"
            hits.append(h)
    while len(hits) < k and len(cap_hits) > 0:
        h = cap_hits.pop(0)
        h["source"] = "cap"
        cap_to_courtlistener(h, jurisdiction)
        hits.append(fields_to_json(h))
    while len(hits) < k and len(cl_hits) > 0:
        h = cl_hits.pop(0)
        h["source"] = "courtlistener"
        hits.append(h)
    return hits
