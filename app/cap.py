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
    jurisdiction: str,
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
    jurisdiction : str
        Must be one of: "Ark.", "Ill.", "N.C.", "N.M."
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
    collection_name = "CAP"
    expr = ""
    if jurisdiction:
        expr += f"jurisdiction_name=='{jurisdiction}'"
    if after_date:
        expr += (" and " if expr else "") + f"decision_date>'{after_date}'"
    if before_date:
        expr += (" and " if expr else "") + f"decision_date<'{before_date}'"
    if keyword_q:
        keyword_q = fuzzy_keyword_query(keyword_q)
        expr += (" and " if expr else "") + f"text like '%{keyword_q}%'"
    return query(collection_name, q, k, expr)
