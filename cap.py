"""A module for interacting with the Caselaw Access Project collection."""
from __future__ import annotations

from milvusdb import query

cap_collection = "CAP"
cap_tool_args = {
    "jurisdiction": {
        "type": "string",
        "description": (
            "The jurisdiction to query, must be one of: 'Ark.', "
            "'Ill.', 'N.C.', 'N.M."
        ),
    },
    "from-date": {
        "type": "string",
        "description": (
            "The start date for the query date range in YYYY-MM-DD format."
        ),
    },
    "to-date": {
        "type": "string",
        "description": (
            "The end date for the query date range in YYYY-MM-DD format."
        ),
    },
}


def cap(
    q: str,
    k: int,
    jurisdiction: str,
    from_date: str | None = None,
    to_date: str | None = None,
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
    from_date : str | None, optional
        The start date for the query date range in YYYY-MM-DD format, by default None
    to_date : str | None, optional
        The end date for the query date range in YYYY-MM-DD format, by default None

    Returns
    -------
    dict
        Contains `message`, `result` list if successful

    """
    collection_name = "CAP"
    expr = ""
    if jurisdiction:
        expr += f"jurisdiction_name=='{jurisdiction}'"
    if from_date:
        if expr:
            expr += " and "
        expr += f"decision_date>='{from_date}'"
    if to_date:
        if expr:
            expr += " and "
        expr += f" and decision_date<='{to_date}'"
    return query(collection_name, q, k, expr)
