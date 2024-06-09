"""A module for interacting with the CourtListener API. Written by Arman Aydemir."""
import os

import requests
from langchain.agents import Tool

from app.milvusdb import query, upload_courtlistener
from app.models import SearchTool

courtlistener_token = os.environ["COURTLISTENER_API_KEY"]
courtlistener_header = {"Authorization": "Token " + courtlistener_token}
base_url = "https://www.courtlistener.com"
search_url = base_url + "/api/rest/v3/search/?q="
opinion_url = base_url + "/api/rest/v3/opinions/?id="
cluster_url = base_url + "/api/rest/v3/clusters/?id="
docket_url = base_url + "/api/rest/v3/dockets/?id="

courtlistener_collection = "courtlistener"

courtlistener_timeout = 30 #seconds

def search(q: str) -> dict:
    """Call the general search api from courtlistener.

    Parameters
    ----------
    q : str
        the query

    Returns
    -------
    dict
        dict containing the results

    """
    response = requests.get(search_url + q,
                            headers=courtlistener_header,
                            timeout=courtlistener_timeout)
    return response.json()

def get_opinion(result:dict) -> dict:
    """Get the full opinion info for a search result from search().

    Parameters
    ----------
    result : dict
        a single result from search()

    Returns
    -------
    dict
        dict containing the Opinion info

    """
    # grabbing the opinion id from the abs url
    opinion_id = result["absolute_url"].split("/")[2]

    response = requests.get(opinion_url + opinion_id,
                            headers=courtlistener_header,
                            timeout=courtlistener_timeout)

    op = response.json()["results"][0]  # the actual opinion

    # getting the text, in the best format possible
    op["text"] = op["html_with_citations"]
    # these are backup formats
    backups = ["html", "plain_text", "html_lawbox", "html_columbia"]
    b_index = 0
    while op["text"] == "" and b_index < len(backups):
        op["text"] = op[backups[b_index]]
        b_index += 1
    return op


def get_cluster(result: dict) -> dict:
    """Get the full cluster info for a search result from search().

    Parameters
    ----------
    result : dict
        A single result from search()

    Returns
    -------
    dict
        dict containing the Cluster info

    """
    cid = str(result["cluster_id"])
    response = requests.get(cluster_url + cid,
                            headers=courtlistener_header,
                            timeout=courtlistener_timeout)

    return response.json()["results"][0]

def get_docket(result: dict) -> dict:
    """Get the full docket info for a search result from search().

    Parameters
    ----------
    result : dict
        A single result from search()

    Returns
    -------
    dict
        dict containing the Docket info

    """
    docket_id = str(result["docket_id"])
    response = requests.get(docket_url + docket_id,
                            headers=courtlistener_header,
                            timeout=courtlistener_timeout)

    return response.json()["results"][0]


# TODO: Need to parallelize this
def courtlistener_search(q: str, k: int = 3) -> dict:
    """Search courtlistener for a query.

    Search, get opinion text, upload the opinion data to milvus, and query it.

    Parameters
    ----------
    q : str
        The query
    k : int, optional
        The number of results to return, by default 3

    Returns
    -------
    dict
        the response with relevant info from courtlistener

    """
    for result in search(q)["results"][:k]:
        oo = get_opinion(result)
        upload_courtlistener(courtlistener_collection, oo)

    return query(courtlistener_collection, q)


def courtlistener_tool_creator(t: SearchTool) -> Tool:
    """Create the courtlistener tool for agents to call.

    Parameters
    ----------
    t : SearchTool
        The search tool definition (we only use the name and prompt for now)

    Returns
    -------
    Tool
        The Tool object for the courtlistener search

    """
    def query_tool(q: str) -> dict:
        return courtlistener_search(q)

    async def async_query_tool(q: str) -> dict:
        return courtlistener_search(q)

    name = t.name
    prompt = t.prompt

    tool_func = lambda q: query_tool(q)
    co_func = lambda q: async_query_tool(q)
    return Tool(
        name=name,
        func=tool_func,
        coroutine=co_func,
        description=prompt,
    )
