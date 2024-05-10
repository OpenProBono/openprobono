"""The search api functions and search toolset creation. Written by Arman Aydemir."""
import os

import requests
from langchain.agents import Tool
from serpapi.google_search import GoogleSearch

from courtlistener import courtlistener_search, courtlistener_tool_creator
from milvusdb import qa, query, scrape
from models import BotRequest, EngineEnum, SearchMethodEnum, SearchTool

GoogleSearch.SERP_API_KEY = os.environ["SERPAPI_KEY"]

COURTROOM5_SEARCH_CX_KEY = "05be7e1be45d04eda"

search_collection = "search_collection"

def filtered_search(results: dict) -> dict:
    """Filter search results returned by serpapi to only include relevant results.

    Parameters
    ----------
    results : dict
         the results from serpapi search

    Returns
    -------
    dict
        filtered results

    """
    new_dict = {}
    if "sports_results" in results:
        new_dict["sports_results"] = results["sports_results"]
    if "organic_results" in results:
        new_dict["organic_results"] = results["organic_results"]
    return new_dict


def dynamic_serpapi_tool(qr: str, prf: str, num_results: int = 3) -> dict:
    """Upgraded serpapi tool, scrape the websites and embed them to query whole pages.

    Parameters
    ----------
    qr : str
        the query
    prf : str
        the prefix given by tool (used for whitelists)
    num_results : int, optional
        number of results to return, by default 5

    Returns
    -------
    dict
        result of the query on the embeddings uploaded to the search collection

    """
    response = filtered_search(
        GoogleSearch({
            "q": prf + " " + qr,
            "num": num_results,
        }).get_dict())
    for result in response["organic_results"]:
        scrape(result["link"], [], [], search_collection)
    return query(search_collection, qr)

def google_search_tool(qr: str, prf: str, max_len: int = 6400) -> str:
    """Query the google search api.

    Parameters
    ----------
    qr : str
        the query itself
    prf : str
        the prefix given by the tool (used for whitelists)
    max_len : int, optional
        maximum length of response text, by default 6400

    Returns
    -------
    str
        the search results

    """
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    params = {
        "key": os.environ["GOOGLE_SEARCH_API_KEY"],
        "cx": os.environ["GOOGLE_SEARCH_API_CX"],
        "q": prf + " " + qr,
    }
    return str(
        requests.get("https://www.googleapis.com/customsearch/v1",
                     params=params,
                     headers=headers, timeout=30).json())[0:max_len]

def courtroom5_search_tool(qr: str, prf: str, max_len: int = 6400) -> str:
    """Query the custom courtroom5 google search api.

    Whitelisted sites defined by search cx key.

    Parameters
    ----------
    qr : str
        the query itself
    prf : str
        the prefix given by the tool (whitelisted sites defined by search cx key)
    max_len : int, optional
        maximum length of response text, by default 6400

    Returns
    -------
    str
        the search results

    """
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    params = {
        "key": os.environ["GOOGLE_SEARCH_API_KEY"],
        "cx": COURTROOM5_SEARCH_CX_KEY,
        "q": prf + " " + qr,
    }
    return str(
        requests.get("https://www.googleapis.com/customsearch/v1",
                     params=params,
                     headers=headers, timeout=30).json())[0:max_len]

# Implement this for regular programatic google search as well.
def dynamic_courtroom5_search_tool(qr: str, prf: str) -> str:
    """Query the custom courtroom5 google search api, scrape the sites and embed them.

    Whitelisted sites defined by search cx key.

    Parameters
    ----------
    qr : str
        the query itself
    prf : str
        the prefix given by the tool

    Returns
    -------
    str
        the search results

    """
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    params = {
        "key": os.environ["GOOGLE_SEARCH_API_KEY"],
        "cx": COURTROOM5_SEARCH_CX_KEY,
        "q": prf + " " + qr,
    }
    response = requests.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params=params,
                    headers=headers,
                    timeout=30,
                ).json()
    for result in response["items"]:
        scrape(result["link"], [], [], search_collection)
    return query(search_collection, qr)

def serpapi_tool(qr: str, prf: str, num_results: int = 5) -> dict:
    """Query the serpapi search api.

    Parameters
    ----------
    qr : str
        the query
    prf : str
        prefix defined by tool (used for whitelist)
    num_results : int, optional
        number of results to return, by default 5

    Returns
    -------
    dict
        the dict of results

    """
    results = filtered_search(
        GoogleSearch({
            "q": prf + " " + qr,
            "num": num_results,
        }).get_dict())
    return results


def dynamic_serpapi_tool_creator(t: SearchTool) -> Tool:
    """Generate the dynamic serpapi tool to give to agents.

    Parameters
    ----------
    t : SearchTool
       The SearchTool object which describes the tool

    Returns
    -------
    Tool
       The tool created to be used by agents

    """
    name = t.name
    prompt = t.prompt
    prf = t.prefix

    async def async_search_tool(qr: str, prf: str) -> dict:
        return dynamic_serpapi_tool(qr, prf)

    tool_func = lambda qr: dynamic_serpapi_tool(qr, prf)  # noqa: E731
    co_func = lambda qr: async_search_tool(qr, prf)  # noqa: E731
    return Tool(name=name,
                func=tool_func,
                coroutine=co_func,
                description=prompt)


def search_tool_creator(t: SearchTool) -> Tool:
    """Create a google search api tool for agents to use.

    Parameters
    ----------
    t : SearchTool
        The SearchTool object which describes the tool

    Returns
    -------
    Tool
        The tool created to be used by agents

    """
    name = t.name
    prompt = t.prompt
    txt = t.prefix

    async def async_search_tool(qr: str, txt: str) -> str:
        return google_search_tool(qr, txt)

    tool_func = lambda qr: google_search_tool(qr, txt)  # noqa: E731
    co_func = lambda qr: async_search_tool(qr, txt)  # noqa: E731

    return Tool(name=name,
                func=tool_func,
                coroutine=co_func,
                description=prompt)

def courtroom5_tool_creator(t: SearchTool) -> Tool:
    """Create a custom courtroom5 search api tool for agents to use.

    Parameters
    ----------
    t : SearchTool
        The SearchTool object which describes the tool

    Returns
    -------
    Tool
        The tool created to be used by agents

    """
    name = t.name
    prompt = t.prompt
    txt = t.prefix

    async def async_search_tool(qr: str, txt: str) -> str:
        return courtroom5_search_tool(qr, txt)

    tool_func = lambda qr: courtroom5_search_tool(qr, txt)  # noqa: E731
    co_func = lambda qr: async_search_tool(qr, txt)  # noqa: E731

    return Tool(name=name,
                func=tool_func,
                coroutine=co_func,
                description=prompt)


def dynamic_courtroom5_tool_creator(t: SearchTool) -> Tool:
    """Create a custom courtroom5 search api tool for agents to use.

    Parameters
    ----------
    t : SearchTool
        The SearchTool object which describes the tool

    Returns
    -------
    Tool
        The tool created to be used by agents

    """
    name = t.name
    prompt = t.prompt
    txt = t.prefix

    async def async_search_tool(qr: str, txt: str) -> str:
        return dynamic_courtroom5_search_tool(qr, txt)

    tool_func = lambda qr: dynamic_courtroom5_search_tool(qr, txt)  # noqa: E731
    co_func = lambda qr: async_search_tool(qr, txt)  # noqa: E731

    return Tool(name=name,
                func=tool_func,
                coroutine=co_func,
                description=prompt)



def serpapi_tool_creator(t: SearchTool) -> Tool:
    """Create a serpapi tool for agents to use.

    Parameters
    ----------
    t : SearchTool
        The SearchTool object which describes the tool

    Returns
    -------
    Tool
        The tool created to be used by agents

    """
    name = t.name
    prompt = t.prompt
    txt = t.prefix

    async def async_search_tool(qr: str, txt: str, prompt: str) -> dict:
        return serpapi_tool(qr, txt)

    tool_func = lambda qr: serpapi_tool(qr, txt, prompt)  # noqa: E731
    co_func = lambda qr: async_search_tool(qr, txt, prompt)  # noqa: E731
    return Tool(name=name,
                func=tool_func,
                coroutine=co_func,
                description=prompt)


def openai_tool(t: SearchTool) -> dict:
    """Create a tool for openai agents to use.

    Parameters
    ----------
    t : SearchTool
        The SearchTool object which describes the tool

    Returns
    -------
    dict
        The description of tool created to be used by agents

    """
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.prompt,
            "parameters": {
                "type": "object",
                "properties": {
                    "qr": {
                        "type": "string",
                        "description": "the search text",
                    },
                },
                "required": ["qr"],
            },
        },
    }

def anthropic_tool(t: SearchTool) -> dict:
    return {
        "name": t.name,
        "description": t.prompt,
        "input_schema": {
            "type": "object",
            "properties": {
                "qr": {
                    "type": "string",
                    "description": "The search text",
                },
            },
            "required": ["qr"],
        },
    }


def search_openai_tool(tool: SearchTool, function_args) -> str:
    """Create a search tool for an openai agent.

    Parameters
    ----------
    tool : SearchTool
        The SearchTool object which describes the tool
    function_args : _type_
        The arguments to pass to the function

    Returns
    -------
    str
        The response from the search tool

    """
    function_response = None
    prf = tool.prefix
    qr = function_args.get("qr")
    match tool.method:
        case SearchMethodEnum.serpapi:
            function_response = serpapi_tool(qr, prf)
        case SearchMethodEnum.dynamic_serpapi:
            function_response = dynamic_serpapi_tool(qr, prf)
        case SearchMethodEnum.google:
            function_response = google_search_tool(qr, prf)
        case SearchMethodEnum.courtlistener:
            function_response = courtlistener_search(qr)
    return str(function_response)

def search_anthropic_tool(tool: SearchTool, function_args: dict) -> str:
    function_response = None
    prompt = tool.prompt
    prf = tool.prefix
    qr = function_args["qr"]
    match tool.method:
        case SearchMethodEnum.serpapi:
            function_response = serpapi_tool(qr, prf)
        case SearchMethodEnum.dynamic_serpapi:
            function_response = dynamic_serpapi_tool(qr, prf)
        case SearchMethodEnum.google:
            function_response = google_search_tool(qr, prf)
        case SearchMethodEnum.courtlistener:
            function_response = courtlistener_search(qr)
    return str(function_response)


def search_toolset_creator(bot: BotRequest):
    toolset = []
    for t in bot.search_tools:
        match bot.chat_model.engine:
            case EngineEnum.langchain:
                match t.method:
                    case SearchMethodEnum.serpapi:
                        toolset.append(serpapi_tool_creator(t))
                    case SearchMethodEnum.dynamic_serpapi:
                        toolset.append(dynamic_serpapi_tool_creator(t))
                    case SearchMethodEnum.google:
                        toolset.append(search_tool_creator(t))
                    case SearchMethodEnum.courtlistener:
                        toolset.append(courtlistener_query_tool(t))
            case EngineEnum.openai:
                toolset.append(openai_tool(t))
            case EngineEnum.anthropic:
                toolset.append(anthropic_tool(t))
    return toolset
