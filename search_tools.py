import os
from urllib import response

import requests
from langchain.agents import Tool
from serpapi.google_search import GoogleSearch
from courtlistener import courlistener_query_tool
from milvusdb import create_collection, query, scrape

from models import BotRequest

GoogleSearch.SERP_API_KEY = os.environ["SERPAPI_KEY"]

search_collection = "search_collection"

def dynamic_serpapi_tool_creator(name, txt, prompt):
    def filtered_search(results):
        new_dict = {}
        if('sports_results' in results):
            new_dict['sports_results'] = results['sports_results']
        if('organic_results' in results):
            new_dict['organic_results'] = results['organic_results']
            return new_dict

    def search_tool(qr, txt, prompt):
        response = filtered_search(GoogleSearch({
            'q': txt + " " + qr,
            'num': 5
            }).get_dict())
        for result in response["organic_results"]:
            scrape(result["link"], [], [], search_collection)
        return query(search_collection, qr)
    
    async def async_search_tool(qr, txt, prompt):
        return search_tool(qr, txt, prompt)

    tool_func = lambda qr: search_tool(qr, txt, prompt)
    co_func = lambda qr: async_search_tool(qr, txt, prompt)
    return Tool(
                name = name,
                func = tool_func,
                coroutine = co_func,
                description = prompt
            )

def search_tool_creator(name, txt, prompt):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    def search_tool(qr, txt, prompt):
        params = {
            'key': os.environ["GOOGLE_SEARCH_API_KEY"],
            'cx': os.environ["GOOGLE_SEARCH_API_CX"],
            'q': txt + " " + qr,
        }
        return str(requests.get('https://www.googleapis.com/customsearch/v1', params=params, headers=headers).json())[0:6400]
    
    async def async_search_tool(qr, txt, prompt):
        return search_tool(qr, txt, prompt)

    tool_func = lambda qr: search_tool(qr, txt, prompt)
    co_func = lambda qr: async_search_tool(qr, txt, prompt)

    return Tool(
                name = name,
                func = tool_func,
                coroutine = co_func,
                description = prompt
            )

def serpapi_tool_creator(name, txt, prompt):
    #Filter search results retured by serpapi to only include relavant results
    def filtered_search(results):
        new_dict = {}
        if('sports_results' in results):
            new_dict['sports_results'] = results['sports_results']
        if('organic_results' in results):
            new_dict['organic_results'] = results['organic_results']
            return new_dict

    def search_tool(qr, txt, prompt):
        return filtered_search(GoogleSearch({
            'q': txt + " " + qr,
            'num': 5
            }).get_dict())
    
    async def async_search_tool(qr, txt, prompt):
        return search_tool(qr, txt, prompt)

    tool_func = lambda qr: search_tool(qr, txt, prompt)
    co_func = lambda qr: async_search_tool(qr, txt, prompt)
    return Tool(
                name = name,
                func = tool_func,
                coroutine = co_func,
                description = prompt
            )

def search_toolset_creator(bot: BotRequest):
    create_collection(search_collection, "Database of search results from the search tools.") #temp
    toolset = []
    for t in bot.search_tools:
        if "serpapi" in t["name"]:
            toolset.append(dynamic_serpapi_tool_creator(t['name'], t['txt'], t['prompt']))
        elif("court" in t["name"]):
            toolset.append(courlistener_query_tool(t['name'], t['txt'], t['prompt']))
        else:
            toolset.append(search_tool_creator(t['name'], t['txt'], t['prompt']))
    return toolset
