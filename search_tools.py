"""The search api functions and search toolset creation. Written by Arman Aydemir."""
import os

import requests
from langchain.agents import Tool
from serpapi.google_search import GoogleSearch

from courtlistener import courtlistener_tool_creator, courtlistener_search
from milvusdb import query, scrape
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


def dynamic_serpapi_tool(qr: str, prf: str, num_results: int = 5) -> dict:
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
        "cx": COURTROOM5_SEARCH_CX_KEY,
        "q": prf + " " + qr,
    }
    return str(
        requests.get("https://www.googleapis.com/customsearch/v1",
                     params=params,
                     headers=headers, timeout=30).json())[0:max_len]

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
    return filtered_search(
        GoogleSearch({
            "q": prf + " " + qr,
            "num": num_results,
        }).get_dict())


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
        return serpapi_tool(qr, txt, prompt)

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
    name = t.name
    prompt = t.prompt
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": prompt,
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
    if (tool.method == SearchMethodEnum.serpapi):
        function_response = serpapi_tool(qr, prf)
    elif (tool.method == SearchMethodEnum.dynamic_serpapi):
        function_response = dynamic_serpapi_tool(qr, prf)
    elif (tool.method == SearchMethodEnum.google):
        function_response = google_search_tool(qr, prf)
    elif (tool.method == SearchMethodEnum.courtlistener):
        function_response = courtlistener_search(qr)
    elif (tool.method == SearchMethodEnum.courtroom5):
        function_response = courtroom5_search_tool(qr, prf)
    return str(function_response)


def search_toolset_creator(bot: BotRequest) -> list[Tool]:
    """Create the toolset for the bot to use.

    Account for the different search methods and engines.

    Parameters
    ----------
    bot : BotRequest
        The BotRequest object which includes list of SearchTools

    Returns
    -------
    list[Tool]
        The list of tools (toolset) created for the bot

    """
    toolset = []
    for t in bot.search_tools:
        if (bot.engine == EngineEnum.langchain):
            if t.method == SearchMethodEnum.serpapi:
                toolset.append(serpapi_tool_creator(t))
            elif t.method == SearchMethodEnum.dynamic_serpapi:
                toolset.append(dynamic_serpapi_tool_creator(t))
            elif t.method == SearchMethodEnum.google:
                toolset.append(search_tool_creator(t))
            elif t.method == SearchMethodEnum.courtlistener:
                toolset.append(courtlistener_tool_creator(t))
            elif t.method == SearchMethodEnum.courtroom5:
                toolset.append(courtroom5_tool_creator(t))
        elif bot.engine == EngineEnum.openai:
            toolset.append(openai_tool(t))
    return toolset
