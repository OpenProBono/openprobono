import os

import requests
from langchain.agents import Tool
from serpapi.google_search import GoogleSearch
from courtlistener import courtlistener_search, courlistener_query_tool
from milvusdb import query, scrape

from models import BotRequest, EngineEnum, SearchTool, SearchMethodEnum

GoogleSearch.SERP_API_KEY = os.environ["SERPAPI_KEY"]

search_collection = "search_collection"

#Filter search results retured by serpapi to only include relavant results
def filtered_search(results):
    new_dict = {}
    if('sports_results' in results):
        new_dict['sports_results'] = results['sports_results']
    if('organic_results' in results):
        new_dict['organic_results'] = results['organic_results']
        return new_dict

def dynamic_serpapi_tool(qr, txt, prompt):
    response = filtered_search(GoogleSearch({
        'q': txt + " " + qr,
        'num': 5
        }).get_dict())
    for result in response["organic_results"]:
        scrape(result["link"], [], [], search_collection)
    return query(search_collection, qr)

def google_search_tool(qr, txt, prompt):
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
        }
        params = {
            'key': os.environ["GOOGLE_SEARCH_API_KEY"],
            'cx': os.environ["GOOGLE_SEARCH_API_CX"],
            'q': txt + " " + qr,
        }
        return str(requests.get('https://www.googleapis.com/customsearch/v1', params=params, headers=headers).json())[0:6400]

def serpapi_tool(qr, txt, prompt):
    return filtered_search(GoogleSearch({
        'q': txt + " " + qr,
        'num': 5
        }).get_dict())

def dynamic_serpapi_tool_creator(t: SearchTool):
    name = t.name
    prompt = t.prompt
    txt = t.prefix
    async def async_search_tool(qr, txt, prompt):
        return dynamic_serpapi_tool(qr, txt, prompt)

    tool_func = lambda qr: dynamic_serpapi_tool(qr, txt, prompt)
    co_func = lambda qr: async_search_tool(qr, txt, prompt)
    return Tool(
                name = name,
                func = tool_func,
                coroutine = co_func,
                description = prompt
            )

def search_tool_creator(t: SearchTool):
    name = t.name
    prompt = t.prompt
    txt = t.prefix
    async def async_search_tool(qr, txt, prompt):
        return google_search_tool(qr, txt, prompt)

    tool_func = lambda qr: google_search_tool(qr, txt, prompt)
    co_func = lambda qr: async_search_tool(qr, txt, prompt)

    return Tool(
                name = name,
                func = tool_func,
                coroutine = co_func,
                description = prompt
            )

def serpapi_tool_creator(t: SearchTool):
    name = t.name
    prompt = t.prompt
    txt = t.prefix
    async def async_search_tool(qr, txt, prompt):
        return serpapi_tool(qr, txt, prompt)

    tool_func = lambda qr: serpapi_tool(qr, txt, prompt)
    co_func = lambda qr: async_search_tool(qr, txt, prompt)
    return Tool(
                name = name,
                func = tool_func,
                coroutine = co_func,
                description = prompt
            )

def openai_tool(t: SearchTool):
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
                    "qr": {"type": "string", "description": "the search text"},
                },
                "required": ["qr"],
            },
        },
    }

def search_openai_tool(tool: SearchTool, function_args):
    function_response = None
    prompt = tool.prompt
    prf = tool.prefix
    qr = function_args.get("qr")
    if(tool.method == SearchMethodEnum.serpapi):
        function_response = serpapi_tool(qr, prf, prompt)
    elif(tool.method == SearchMethodEnum.dynamic_serpapi):
        function_response = dynamic_serpapi_tool(qr, prf, prompt)
    elif(tool.method == SearchMethodEnum.google):
        function_response = google_search_tool(qr, prf, prompt)
    elif(tool.method == SearchMethodEnum.courtlistener):
        function_response = courtlistener_search(qr)
    return str(function_response)

def search_toolset_creator(bot: BotRequest):
    toolset = []
    for t in bot.search_tools:
        if(bot.engine == EngineEnum.langchain):
            if t.method == SearchMethodEnum.serpapi:
                toolset.append(serpapi_tool_creator(t))
            elif t.method == SearchMethodEnum.dynamic_serpapi:
                toolset.append(dynamic_serpapi_tool_creator(t))
            elif t.method == SearchMethodEnum.google:
                toolset.append(search_tool_creator(t))
            elif t.method == SearchMethodEnum.courtlistener:
                toolset.append(courlistener_query_tool(t))
        elif bot.engine == EngineEnum.openai:
            toolset.append(openai_tool(t))
    return toolset
