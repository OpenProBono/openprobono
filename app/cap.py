"""A module for interacting with the Caselaw Access Project collection."""
from __future__ import annotations

from langfuse.decorators import observe

from app.milvusdb import fuzzy_keyword_query, query

cap_collection = "CAP"
cap_tool_args = {
    "jurisdiction": {
        "type": "string",
        "description": (
            "The jurisdiction to query, must be one of: 'Ark.', "
            "'Ill.', 'N.C.', 'N.M."
        ),
    },
    "after-date": {
        "type": "string",
        "description": (
            "The after date for the query date range in YYYY-MM-DD format."
        ),
    },
    "before-date": {
        "type": "string",
        "description": (
            "The before date for the query date range in YYYY-MM-DD format."
        ),
    },
}

@observe()
def cap(
    q: str,
    k: int,
    jurisdictions: list[str],
    keyword_q: str | None = None,
    after_date: str | None = None,
    before_date: str | None = None,
) -> dict:
    """Query CAP data.

    Parameters
    ----------
    q : str
        The query text
    k : int
        How many chunks to return
    jurisdictions : list[str]
        Valid jurisdictions: "Ark.", "Ill.", "N.C.", "N.M."
    keyword_q : str | None, optional
        The keyword query text, by default None
    after_date : str | None, optional
        The after date for the query date range in YYYY-MM-DD format, by default None
    before_date : str | None, optional
        The before date for the query date range in YYYY-MM-DD format, by default None

    Returns
    -------
    dict
        Contains `message`, `result` list if successful

    """
    valid_jurisdics = []
    num_jurisdics = 4
    if "ar" in jurisdictions:
        valid_jurisdics.append("Ark.")
    if "il" in jurisdictions:
        valid_jurisdics.append("Ill.")
    if "nc" in jurisdictions:
        valid_jurisdics.append("N.C.")
    if "nm" in jurisdictions:
        valid_jurisdics.append("N.M.")
    if not valid_jurisdics:
        return {"message": "Failure: no valid jurisdictions were found"}
    collection_name = "CAP"
    if len(valid_jurisdics) == num_jurisdics:
        # dont need to filter by jurisdiction
        expr = ""
    else:
        expr =  f"jurisdiction_name in {valid_jurisdics}"
    if after_date:
        expr += (" and " if expr else "") + f"decision_date>'{after_date}'"
    if before_date:
        expr += (" and " if expr else "") + f"decision_date<'{before_date}'"
    if keyword_q:
        keyword_q = fuzzy_keyword_query(keyword_q)
        expr += (" and " if expr else "") + f"text like '% {keyword_q} %'"
    return query(collection_name, q, k, expr)
