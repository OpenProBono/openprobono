import os
from urllib import response

import requests
from langchain.agents import Tool
from serpapi.google_search import GoogleSearch
from courtlistener import courtlistener_search, courlistener_query_tool
from milvusdb import create_collection, query, scrape

from models import BotRequest

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

def dynamic_serpapi_tool_creator(name, txt, prompt):
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

def search_tool_creator(name, txt, prompt):
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

def serpapi_tool_creator(name, txt, prompt):
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

def openai_tool(name: str, prompt: str):
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

def search_openai_tool(tool: dict, function_args):
    function_response = None
    prompt = tool["prompt"]
    txt = tool["txt"]
    qr = function_args.get("qr")
    if "serpapi" in tool["name"]:
        function_response = dynamic_serpapi_tool(qr, txt, prompt)
    elif("court" in tool["name"]):
        function_response = courtlistener_search(qr)
    else:
        function_response = google_search_tool(qr, txt, prompt)
    return str(function_response)

def search_toolset_creator(bot: BotRequest):
    toolset = []
    for t in bot.search_tools:
        if bot.engine == 'langchain':
            if "serpapi" in t["name"]:
                toolset.append(dynamic_serpapi_tool_creator(t['name'], t['txt'], t['prompt']))
            elif("court" in t["name"]):
                toolset.append(courlistener_query_tool(t['name'], t['txt'], t['prompt']))
            else:
                toolset.append(search_tool_creator(t['name'], t['txt'], t['prompt']))
        elif bot.engine == 'openai':
            toolset.append(openai_tool(t['name'], t['prompt']))
    return toolset
