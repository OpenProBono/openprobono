import os
import requests
from langchain.agents import Tool
from langfuse.decorators import observe
from milvusdb import collection_upload_str, query
from models import SearchTool

courtlistener_token = os.environ["COURTLISTENER_API_KEY"]
courtlistener_header = {"Authorization": "Token " + courtlistener_token}
base_url = "https://www.courtlistener.com"
search_url = base_url + "/api/rest/v3/search/?q="
opinion_url = base_url + "/api/rest/v3/opinions/?id="
cluster_url = base_url + "/api/rest/v3/clusters/?id="
docket_url = base_url + "/api/rest/v3/dockets/?id="

courtlistener_collection = "courtlistener"


def search(q: str) -> dict:
    """
    Calls the general search api from courtlistener
    Args:
        q: the query

    Returns: dict containing the results

    """
    response = requests.get(search_url + q, headers=courtlistener_header)
    return response.json()


def get_opinion(result:dict) -> dict:
    """
    This gets the full opinion info for a search result from search()
    Also defines "text" to be the best format of the opinion text given, so it can be used by the llm
    Args:
        result: A single result from search()

    Returns: dict containing the Opinion info

    """
    id = result['absolute_url'].split("/")[2]  # grabbing the opinion id from the abs url
    response = requests.get(opinion_url + id, headers=courtlistener_header)

    op = response.json()["results"][0]  # the actual opinion

    op["text"] = op["html_with_citations"]  # getting the text, in the best format possible
    backups = ["html", "plain_text", "html_lawbox", "html_columbia"]  # these are backup formats
    b_index = 0
    while op["text"] == "" and b_index < len(backups):
        op["text"] = op[backups[b_index]]
        b_index += 1
    return op


def get_cluster(result: dict) -> dict:
    """
    This gets the full cluster info for a search result from search()
    Args:
        result: A single result from search()

    Returns: dict containing the Cluster info

    """
    cid = str(result["cluster_id"])
    response = requests.get(cluster_url + cid, headers=courtlistener_header)

    return response.json()["results"][0]


def get_docket(result: dict) -> dict:
    """
`   This gets the full docket info for a search result from search()
    Args:
        result: A single result from search()

    Returns: dict containing the Docket info

    """
    id = str(result["docket_id"])
    response = requests.get(docket_url + id, headers=courtlistener_header)

    return response.json()["results"][0]


# TODO: Need to parallelize this
@observe()
def courtlistener_search(q: str, k: int = 3) -> dict:
    """
    This is the actual courtlistener search implemented for agents.
    For now, we only grab the full opinion data.
    Then we embed the text into our milvus and query it
    Args:
        q: the query passed to us by the agent
        k: the number of results to return
    Returns:
        the response with relevant info from courtlistener
    """
    for result in search(q)["results"][:k]:
        oo = get_opinion(result)
        collection_upload_str(oo["text"], courtlistener_collection, oo["absolute_url"])

    return query(courtlistener_collection, q)


def courtlistener_query_tool(t: SearchTool) -> Tool:
    """
    This is what creates the courtlistener tool for agents to call
    Args:
        t (SearchTool): the search tool definition

    Returns:
        Tool
    """
    def query_tool(q: str):
        return courtlistener_search(q)

    async def async_query_tool(q: str):
        return courtlistener_search(q)

    name = t.name
    prompt = t.prompt

    tool_func = lambda q: query_tool(q)
    co_func = lambda q: async_query_tool(q)
    return Tool(
        name=name,
        func=tool_func,
        coroutine=co_func,
        description=prompt
    )
