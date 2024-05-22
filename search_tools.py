import os

import requests
from langchain.agents import Tool
from langfuse.decorators import observe
from serpapi.google_search import GoogleSearch

from courtlistener import courtlistener_query_tool, courtlistener_search
from milvusdb import query, upload_site
from models import BotRequest, EngineEnum, SearchMethodEnum, SearchTool

GoogleSearch.SERP_API_KEY = os.environ["SERPAPI_KEY"]

search_collection = "search_collection"


def filtered_search(results: dict) -> dict:
    """
    Filter search results returned by serpapi to only include relevant results
    Args:
        results: the results from serpapi search

    Returns:
        filtered results
    """
    new_dict = {}
    if 'sports_results' in results:
        new_dict['sports_results'] = results['sports_results']
    if 'organic_results' in results:
        new_dict['organic_results'] = results['organic_results']
    return new_dict


@observe()
def dynamic_serpapi_tool(qr: str, prf: str, num_results: int = 5) -> dict:
    """
    Upgraded serpapi tool which scrapes the returned websites and embeds them to query whole pages
    Args:
        qr: the query
        prf: the prefix given by tool (used for whitelists)
        num_results: number of results to return

    Returns:
        result of the query on the embeddings which were uploaded to the search collection
    """
    response = filtered_search(
        GoogleSearch({
            'q': prf + " " + qr,
            'num': num_results
        }).get_dict())
    for result in response["organic_results"]:
        upload_site(search_collection, result["link"])
    return query(search_collection, qr)


@observe()
def google_search_tool(qr: str, prf: str, max_len: int = 6400) -> str:
    """
    Queries the google search api
    Args:
        qr: the query itself
        prf: the prefix given by the tool (used for whitelists)
        max_len: maximum length of response text

    Returns
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
        requests.get('https://www.googleapis.com/customsearch/v1',
                     params=params,
                     headers=headers).json())[0:max_len]


@observe()
def serpapi_tool(qr: str, prf: str, num_results: int = 5) -> dict:
    """
    Queries the serpapi
    Args:
        qr: the query
        prf: prefix defined by tool
        num_results: number of results to return (default 5)

    Returns:
        the dict of results
    """
    return filtered_search(
        GoogleSearch({
            "q": prf + " " + qr,
            "num": num_results,
        }).get_dict())


def dynamic_serpapi_tool_creator(t: SearchTool) -> Tool:
    name = t.name
    prompt = t.prompt
    prf = t.prefix

    async def async_search_tool(qr, txt):
        return dynamic_serpapi_tool(qr, txt)

    tool_func = lambda qr: dynamic_serpapi_tool(qr, prf)
    co_func = lambda qr: async_search_tool(qr, prf)
    return Tool(name=name,
                func=tool_func,
                coroutine=co_func,
                description=prompt)


def search_tool_creator(t: SearchTool) -> Tool:
    name = t.name
    prompt = t.prompt
    txt = t.prefix

    async def async_search_tool(qr, txt, prompt):
        return google_search_tool(qr, txt)

    tool_func = lambda qr: google_search_tool(qr, txt)
    co_func = lambda qr: async_search_tool(qr, txt, prompt)

    return Tool(name=name,
                func=tool_func,
                coroutine=co_func,
                description=prompt)


def serpapi_tool_creator(t: SearchTool) -> Tool:
    name = t.name
    prompt = t.prompt
    txt = t.prefix

    async def async_search_tool(qr, txt):
        return serpapi_tool(qr, txt)

    tool_func = lambda qr: serpapi_tool(qr, txt)
    co_func = lambda qr: async_search_tool(qr, txt)
    return Tool(name=name,
                func=tool_func,
                coroutine=co_func,
                description=prompt)


def openai_tool(t: SearchTool) -> dict:
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
