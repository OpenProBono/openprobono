import os

import requests
from langchain.agents import Tool
from serpapi.google_search import GoogleSearch
import milvusdb
from pydantic import BaseModel
from json import loads

class BotTool(BaseModel):
    name: str
    params: dict[str, object]

GoogleSearch.SERP_API_KEY = os.environ["SERPAPI_KEY"]

def search_tool_creator(tool: BotTool):
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

    tool_func = lambda qr: search_tool(qr, tool.params["txt"], tool.params["prompt"])
    co_func = lambda qr: async_search_tool(qr, tool.params["txt"], tool.params["prompt"])

    return Tool(
                name = tool.name,
                func = tool_func,
                coroutine = co_func,
                description = tool.params["prompt"]
            )

def serpapi_tool_creator(tool: BotTool):
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

    tool_func = lambda qr: search_tool(qr, tool.params["txt"], tool.params["prompt"])
    co_func = lambda qr: async_search_tool(qr, tool.params["txt"], tool.params["prompt"])
    return Tool(
                name = tool.name,
                func = tool_func,
                coroutine = co_func,
                description = tool.params["prompt"]
            )

def vdb_qa_tool(tool: BotTool):
    def vdb_tool(tool: BotTool, question: str):
        return milvusdb.qa(tool.params["database_name"], question, tool.params["k"])

    async def async_vdb_tool(tool: BotTool, question: str):
        return vdb_tool(tool, question)
    
    tool_func = lambda q: vdb_tool(tool, q)
    co_func = lambda q: async_vdb_tool(tool, q)

    return Tool(
        name = tool.name,
        func = tool_func,
        coroutine = co_func,
        description = f"""Tool used to answer questions using the {tool.params["k"]} 
                        most relevant text chunks from a vector database named {tool.params["database_name"]}."""
    )

def vdb_query_tool(tool: BotTool):
    def vdb_tool(tool: BotTool, query: str):
        return milvusdb.query(tool.params["database_name"], query, tool.params["k"])

    async def async_vdb_tool(tool: BotTool, query: str):
        return vdb_tool(tool, query)
    
    tool_func = lambda q: vdb_tool(tool, q)
    co_func = lambda q: async_vdb_tool(tool, q)
    return Tool(
        name = tool.name,
        func = tool_func,
        coroutine = co_func,
        description = f"""Tool used to query a vector database named {tool.params["database_name"]} 
                        and return the {tool.params["k"]} most relevant text chunks."""
    )

def toolset_creator(tools: list[BotTool]):
    toolset = []
    for t in tools:
        if t.name == "serpapi":
            toolset.append(serpapi_tool_creator(t))
        elif t.name == "google_search":
            toolset.append(search_tool_creator(t))
        elif t.name == "vectorstore-qa":
            toolset.append(vdb_qa_tool(t))
        elif t.name == "vectorstore-query":
            toolset.append(vdb_query_tool(t))
    return toolset
