"""The search api functions and search toolset creation. Written by Arman Aydemir."""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context

import requests
from langfuse.decorators import observe
from serpapi.google_search import GoogleSearch

from app.courtlistener import courtlistener_query, courtlistener_tool_args
from app.milvusdb import query, source_exists, upload_site
from app.models import (
    BotRequest,
    EngineEnum,
    OpinionSearchRequest,
    SearchMethodEnum,
    SearchTool,
)
from app.prompts import FILTERED_CASELAW_PROMPT

GoogleSearch.SERP_API_KEY = os.environ["SERPAPI_KEY"]

COURTROOM5_SEARCH_CX_KEY = "05be7e1be45d04eda"

search_collection = "search_collection_vj1"

def filtered_search(results: dict) -> tuple[dict, list[str]]:
    """Filter search results returned by serpapi to only include relevant results.

    Parameters
    ----------
    results : dict
         the results from serpapi search

    Returns
    -------
    tuple[dict, list[str]]
        filtered results, sources

    """
    new_dict = {}
    sources = []
    if "sports_results" in results:
        new_dict["sports_results"] = results["sports_results"]
        sources += [result["link"] for result in results["sports_results"]]
    if "organic_results" in results:
        new_dict["organic_results"] = results["organic_results"]
        sources += [result["link"] for result in results["organic_results"]]
    return new_dict, sources


@observe(capture_output=False)
def process_site(result: dict, bot_id: str, tool_name: str) -> None:
    try:
        if(not source_exists(search_collection, result["link"], bot_id, tool_name)):
            print("Uploading site: " + result["link"])
            upload_site(search_collection, result["link"], bot_id=bot_id, tool_name=tool_name)
    except Exception as error:
        print("Warning: Failed to upload site for dynamic serpapi: " + result["link"])
        print("The error was: " + str(error))

@observe()
def dynamic_serpapi_tool(
    qr: str,
    prf: str,
    num_results: int = 3,
    k: int = 3,
    bot_id: str = "",
    tool_name: str = "",
) -> tuple[dict, list[str]]:
    """Upgraded serpapi tool, scrape the websites and embed them to query whole pages.

    Parameters
    ----------
    qr : str
        the query
    prf : str
        the prefix given by tool (used for whitelists)
    num_results : int, optional
        number of results to return, by default 3
    k : int, optional
        number of chunks to return from milvus, by default 3
    bot_id : str, optional
        the ID of the bot using this tool, by default ""
    tool_name : str, optional
        the name of the search tool, by default ""

    Returns
    -------
    tuple[dict, list[str]]
        results
        (the result of the query on the embeddings uploaded to the search collection),
        sources

    """
    response, _ = filtered_search(
        GoogleSearch({
            "q": prf + " " + qr,
            "num": num_results,
        }).get_dict())

    with ThreadPoolExecutor() as executor:
        futures = []
        for result in response["organic_results"]:
            ctx = copy_context()
            def task(r=result, context=ctx):  # noqa: ANN001, ANN202
                return context.run(process_site, r, bot_id, tool_name)
            futures.append(executor.submit(task))

        for future in as_completed(futures):
            _ = future.result()
    filter_expr = f"metadata['bot_id'] == '{bot_id}' && metadata['tool_name'] == '{tool_name}'"
    res = query(search_collection, qr, k=k, expr=filter_expr)
    # get the list of sources from the query results
    urls = [item["entity"]["metadata"]["url"] for item in res["result"]]
    return res, urls

@observe()
def google_search_tool(qr: str, prf: str, k: int = 4) -> tuple[dict, list[str]]:
    """Query the google search api.

    Parameters
    ----------
    qr : str
        the query itself
    prf : str
        the prefix given by the tool (used for whitelists)
    k : int, optional
        maximum number of results, by default 4

    Returns
    -------
    tuple[dict, list[str]]
        search results, sources

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
    site = "https://www.googleapis.com/customsearch/v1"
    res = requests.get(site, params=params, headers=headers, timeout=30).json()
    urls = [item["link"] for item in res["items"][:k]]
    return {"items": res["items"][:k]}, urls


@observe()
def courtroom5_search_tool(qr: str, prf: str = "", k: int = 4) -> tuple[dict, list[str]]:
    """Query the custom courtroom5 google search api.

    Whitelisted sites defined by search cx key.

    Parameters
    ----------
    qr : str
        the query itself
    prf : str
        the prefix given by the tool (whitelisted sites defined by search cx key)
    k : int, optional
        maximum number of results, by default 4

    Returns
    -------
    tuple[dict, list[str]]
        search results, sources

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
    site = "https://www.googleapis.com/customsearch/v1"
    res = requests.get(site, params=params, headers=headers, timeout=30).json()
    urls = [item["link"] for item in res["items"][:k]]
    return {"items": res["items"][:k]}, urls

# Implement this for regular programatic google search as well.
@observe()
def dynamic_courtroom5_search_tool(qr: str, prf: str="", bot_id: str = "", tool_name: str = "") -> tuple[dict, list[str]]:
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
    tuple[dict, list[str]]
        search results, sources

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

    def process_site(result: dict) -> None:
        try:
            if(not source_exists(search_collection, result["link"], bot_id, tool_name)):
                print("Uploading site: " + result["link"])
                upload_site(search_collection, result["link"], bot_id=bot_id, tool_name=tool_name)
        except Exception as error:
            print("Warning: Failed to upload site for dynamic serpapi: " + result["link"])
            print("The error was: " + str(error))

    for result in response["items"]:
        process_site(result)
    filter_expr = f"metadata['bot_id'] == '{bot_id}' && metadata['tool_name'] == '{tool_name}'"
    res = query(search_collection, qr, expr=filter_expr)
    # get the list of sources from the query results
    urls = [item["entity"]["metadata"]["url"] for item in res["result"]]
    return res, urls


@observe()
def serpapi_tool(qr: str, prf: str, num_results: int = 5) -> tuple[dict, list[str]]:
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
    tuple[dict, list[str]]
        results, sources

    """
    return filtered_search(
        GoogleSearch({
            "q": prf + " " + qr,
            "num": num_results,
        }).get_dict())

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
    body = {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.prompt,
            "parameters": {
                "type": "object",
                "properties": {
                    "qr": {
                        "type": "string",
                        "description": "The search text",
                    },
                },
                "required": ["qr"],
            },
        },
    }
    if t.method == SearchMethodEnum.courtlistener:
        # arg definitions
        body["function"]["parameters"]["properties"].update(courtlistener_tool_args)
        # modify query text for semantic + keyword queries
        body["function"]["parameters"]["properties"]["qr"]["description"] = (
            "A semantic query to search for general concepts and terms."
        )
        # default tool definition
        if not t.prompt:
            body["function"]["description"] = FILTERED_CASELAW_PROMPT
    return body

def anthropic_tool(t: SearchTool) -> dict:
    """Create a tool for anthropic agents to use.

    Parameters
    ----------
    t : SearchTool
        The SearchTool object which describes the tool

    Returns
    -------
    dict
        The description of tool created to be used by agents

    """
    body = {
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
    if t.method == SearchMethodEnum.courtlistener:
        # add courtlistener arg definitions
        body["input_schema"]["properties"].update(courtlistener_tool_args)
        # modify query text for semantic + keyword queries
        body["input_schema"]["properties"]["qr"]["description"] = (
            "A semantic query to search for general concepts and terms."
        )
        # default tool definition
        if not t.prompt:
            body["description"] = FILTERED_CASELAW_PROMPT
    return body


def run_search_tool(tool: SearchTool, function_args: dict) -> tuple[dict, list[str]]:
    """Create a search tool for an openai agent.

    Parameters
    ----------
    tool : SearchTool
        The SearchTool object which describes the tool
    function_args : dict
        The arguments to pass to the function

    Returns
    -------
    tuple[dict, list[str]]
        response, sources

    """
    function_response, sources = None, []
    prf = tool.prefix
    qr = function_args["qr"]
    match tool.method:
        case SearchMethodEnum.serpapi:
            function_response, sources = serpapi_tool(qr, prf)
        case SearchMethodEnum.dynamic_serpapi:
            function_response, sources = dynamic_serpapi_tool(qr, prf, bot_id=tool.bot_id, tool_name=tool.name)
        case SearchMethodEnum.google:
            function_response, sources = google_search_tool(qr, prf)
        case SearchMethodEnum.courtlistener:
            tool_jurisdictions = None
            tool_kw_query = None
            tool_after_date = None
            tool_before_date = None
            if "jurisdictions" in function_args:
                tool_jurisdictions = [jurisdiction.lower() for jurisdiction in function_args["jurisdictions"]]
            if "keyword-qr" in function_args:
                tool_kw_query = function_args["keyword-qr"]
            if "after-date" in function_args:
                tool_after_date = function_args["after-date"]
            if "before-date" in function_args:
                tool_before_date = function_args["before-date"]
            request = OpinionSearchRequest(
                query=qr,
                jurisdictions=tool_jurisdictions,
                keyword_query=tool_kw_query,
                after_date=tool_after_date,
                before_date=tool_before_date,
            )
            function_response, sources = courtlistener_query(request)
        case SearchMethodEnum.courtroom5:
            function_response, sources = courtroom5_search_tool(qr, prf)
        case SearchMethodEnum.dynamic_courtroom5:
            function_response, sources = dynamic_courtroom5_search_tool(qr, prf, bot_id=tool.bot_id, tool_name=tool.name)
    return function_response, sources

def search_toolset_creator(bot: BotRequest, bot_id: str) -> list:
    """Create a search toolset for the bot from all the search tools.

    Parameters
    ----------
    bot : BotRequest
        Bot object
    bot_id: str
        the bot id string

    Returns
    -------
    list
        The list of search tools formatted for the bot engine

    """
    toolset = []
    for t in bot.search_tools:
        t.bot_id = bot_id
        match bot.chat_model.engine:
            case EngineEnum.openai:
                toolset.append(openai_tool(t))
            case EngineEnum.anthropic:
                toolset.append(anthropic_tool(t))
    return toolset


def find_search_tool(bot: BotRequest, tool_name: str) -> SearchTool | None:
    """Find the search tool with the given name.

    Parameters
    ----------
    bot : BotRequest
        The bot
    tool_name : str
        The tool/function name

    Returns
    -------
    SearchTool | None
        The matching tool or None if not found

    """
    return next(
        (t for t in bot.search_tools if tool_name == t.name),
        None,
    )


def format_search_tool_results(tool_output: dict, tool: SearchTool) -> list[str]:
    formatted_results = []

    if tool_output["message"] != "Success" or "result" not in tool_output:
        return formatted_results

    for i, result in enumerate(tool_output["result"], start=1):
        entity = result["entity"]
        metadata = entity["metadata"]

        # Format the text, truncating if necessary
        text = entity["text"][:500] + "..." if len(entity["text"]) > 500 else entity["text"]

        # Select important metadata based on tool type
        if tool.method == SearchMethodEnum.courtlistener:
            important_metadata = {
                "Case Name": metadata.get("case_name", "N/A"),
                "Date Filed": metadata.get("date_filed", "N/A"),
                "Court": metadata.get("court_name", "N/A"),
                "Author": metadata.get("author_name", "N/A"),
                "Precedential Status": metadata.get("precedential_status", "N/A"),
            }
            title = metadata.get("case_name", "Unknown Opinion")
        elif tool.method == SearchMethodEnum.dynamic_serpapi:
            important_metadata = {
                "URL": metadata.get("url", "N/A"),
                "AI Summary": metadata.get("ai_summary", "N/A"),
                "Page Number": metadata.get("page_number", "N/A"),
            }
            title = metadata.get("url", "Unknown URL")
        else:
            important_metadata = {k: v for k, v in metadata.items() if v is not None}
            title = "Unknown Source"

        # Format the knowledge card
        card = f"{i}. {title} (Distance: {result['distance']:.4f})\n"
        card += "-" * 40 + "\n"
        for key, value in important_metadata.items():
            card += f"{key}: {value}\n"
        card += "-" * 40 + "\n"
        card += f"Text:\n{text}\n"
        card += "=" * 40 + "\n"

        formatted_results.append(card)

    return formatted_results