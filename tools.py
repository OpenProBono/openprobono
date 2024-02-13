import os
from json import loads
from os import environ
from typing import Type

import firebase_admin
import requests
from firebase_admin import credentials, firestore
from langchain.agents import Tool
from langchain.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, Field
from serpapi.google_search import GoogleSearch

GoogleSearch.SERP_API_KEY = os.environ["SERPAPI_KEY"]
# def check_api_keys():
#     if("SERPAPI_KEY" in os.environ.keys() and "GOOGLE_SEARCH_API_KEY" in os.environ.keys() and "GOOGLE_SEARCH_API_CX" in os.environ.keys()):
#         GoogleSearch.SERP_API_KEY = os.environ["SERPAPI_KEY"]
#         return True
#     firebase_config = loads(environ["Firebase"])
#     cred = credentials.Certificate(firebase_config)
#     firebase_admin.initialize_app(cred)
#     db = firestore.client()

#     os.environ["SERPAPI_KEY"] = db.collection("third_party_api_keys").document("serpapi").get().to_dict()['key']
#     GoogleSearch.SERP_API_KEY = os.environ["SERPAPI_KEY"]

#     os.environ["GOOGLE_SEARCH_API_KEY"] = db.collection("third_party_api_keys").document("google_search").get().to_dict()['key']

#     os.environ["GOOGLE_SEARCH_API_CX"] = db.collection("third_party_api_keys").document("google_search").get().to_dict()['cx']
#     return True

class SearchToolSchema(BaseModel):
    query: str = Field(default="", description="The search query")

def search_tool_creator(name, txt, prompt):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    def search_tool(qr, txt, prompt):
        data = {"search": txt + " " + qr, 'prompt': prompt, 'timestamp': firestore.SERVER_TIMESTAMP}
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
    args_schema: Type[SearchToolSchema] = SearchToolSchema()
    return StructuredTool(
                args_schema = args_schema,
                name = name,
                func = tool_func,
                coroutine = co_func,
                description = prompt
            )

def search_toolset_creator(r):
    # check_api_keys()
    toolset = []
    for t in r.tools:
        toolset.append(search_tool_creator(t['name'], t['txt'], t['prompt']))
    return toolset

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
    return StructuredTool(
                args_schema=SearchToolSchema(),
                name = name,
                func = tool_func,
                coroutine = co_func,
                description = prompt
            )

def serpapi_toolset_creator(r):
    # check_api_keys()
    toolset = []
    for t in r.tools:
        toolset.append(serpapi_tool_creator(t['name'], t['txt'], t['prompt']))
    return toolset