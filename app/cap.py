"""A module for interacting with the Caselaw Access Project collection."""
from __future__ import annotations

from app.milvusdb import query

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


def cap(
    q: str,
    k: int,
    jurisdiction: str,
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
        if expr:
            expr += " and "
        expr += f"decision_date>'{after_date}'"
    if before_date:
        if expr:
            expr += " and "
        expr += f"decision_date<'{before_date}'"
    return query(collection_name, q, k, expr)
