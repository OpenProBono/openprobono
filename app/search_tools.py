"""The search api functions and search toolset creation. Written by Arman Aydemir."""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from threading import Lock

import requests
from langfuse.decorators import langfuse_context, observe
from serpapi.google_search import GoogleSearch

from app.courtlistener import courtlistener_query, courtlistener_tool_args
from app.logger import setup_logger
from app.milvusdb import query, source_exists, upload_site
from app.models import (
    BotRequest,
    EngineEnum,
    OpinionSearchRequest,
    SearchMethodEnum,
    SearchTool,
)
from app.prompts import FILTERED_CASELAW_PROMPT

logger = setup_logger()

GoogleSearch.SERP_API_KEY = os.environ["SERPAPI_KEY"]

COURTROOM5_SEARCH_CX_KEY = "05be7e1be45d04eda"

search_collection = "search_collection_vj1"

# Global lock for URL processing
url_locks = {}
url_lock = Lock()

# Set to keep track of failed URLs
failed_urls = set()
failed_urls_lock = Lock()

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


@observe(capture_input=False, capture_output=False)
def process_site(result: dict, bot_id: str, tool: SearchTool) -> None:
    url = result["link"]
    num_attempts = 10

    langfuse_context.update_current_observation(input=url)

    with url_lock:
        if url not in url_locks:
            url_locks[url] = Lock()

    with url_locks[url]:
        # Check if URL has already failed before processing
        with failed_urls_lock:
            if url in failed_urls:
                logger.info("Skipping previously failed URL: %s", url)
                return
        try:
            if not source_exists(search_collection, result, bot_id, tool.name):
                logger.info("Uploading site: %s", url)
                res = upload_site(search_collection, result, tool)
                if res["message"] == "Success":
                    # check to ensure site appears in collection before releasing URL lock
                    attempt = 0
                    while not source_exists(search_collection, result, bot_id, tool.name) and attempt < num_attempts:
                        attempt += 1
                    if attempt == num_attempts:
                        logger.error("Site not found in collection, add to failed URLs: %s", url)
                        with failed_urls_lock:
                            failed_urls.add(url)
                else:
                    with failed_urls_lock:
                        failed_urls.add(url)
            else:
                logger.info("Site already uploaded: %s", url)
        except Exception:
            logger.exception("Failed to upload site for dynamic serpapi: %s", url)
            with failed_urls_lock:
                failed_urls.add(url)


@observe(capture_input=False, capture_output=False)
def dynamic_serpapi_tool(
    qr: str,
    prf: str,
    tool: SearchTool,
    num_results: int = 5,
    k: int = 3,
) -> dict:
    """Upgraded serpapi tool, scrape the websites and embed them to query whole pages.

    Parameters
    ----------
    qr : str
        the query
    prf : str
        the prefix given by tool (used for whitelists)
    tool : SearchTool
        the tool using this search method
    num_results : int, optional
        number of results to return, by default 3
    k : int, optional
        number of chunks to return from milvus, by default 3

    Returns
    -------
    dict
        result of the query on the embeddings uploaded to the search collection

    """
    bot_id = tool.bot_id
    tool_name = tool.name
    # tracing
    langfuse_context.update_current_observation(
        input={
            "query": qr,
            "prefix": prf,
            "num_results": num_results,
            "k": k,
            "bot_id": bot_id,
            "tool_name": tool_name,
        },
    )

    response = filtered_search(
        GoogleSearch({
            "q": prf + " " + qr,
            "num": num_results,
        }).get_dict())

    with ThreadPoolExecutor() as executor:
        futures = []
        if("organic_results" not in response):
            return {"message": "No results found."}
        for result in response["organic_results"]:
            ctx = copy_context()
            def task(r=result, b=bot_id, t=tool, context=ctx):  # noqa: ANN001, ANN202
                return context.run(process_site, r, b, t)
            futures.append(executor.submit(task))

        for future in as_completed(futures):
            _ = future.result()

    filter_expr = f"json_contains(metadata['bot_and_tool_id'], '{bot_id + tool_name}')"
    if tool.jurisdictions:
        filter_expr += f" and ARRAY_CONTAINS_ANY(metadata['jurisdictions'], {tool.jurisdictions})"
    res = query(search_collection, qr, k=k, expr=filter_expr)
    if "result" in res:
        pks = [str(hit["id"]) for hit in res["result"]]
        langfuse_context.update_current_observation(output=pks)
    return res


@observe()
def google_search_tool(qr: str, prf: str, k: int = 4) -> dict:
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
    dict
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
    res = requests.get(site, params=params, headers=headers, timeout=15).json()
    return res["items"][:k]


@observe()
def courtroom5_search_tool(qr: str, prf: str = "", k: int = 4) -> dict:
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
    dict
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
    res = requests.get(site, params=params, headers=headers, timeout=15).json()
    return res["items"][:k]


# Implement this for regular programatic google search as well.
@observe()
def dynamic_courtroom5_search_tool(qr: str, tool: SearchTool, prf: str="") -> dict:
    """Query the custom courtroom5 google search api, scrape the sites and embed them.

    Whitelisted sites defined by search cx key.

    Parameters
    ----------
    qr : str
        the query itself
    tool : SearchTool
        the tool using this search method
    prf : str
        the prefix given by the tool, by default empty string

    Returns
    -------
    dict
        search results

    """
    bot_id = tool.bot_id
    tool_name = tool.name
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
            if(not source_exists(search_collection, result, bot_id, tool_name)):
                logger.info("Uploading site: %s", result["link"])
                upload_site(search_collection, result, tool)
        except Exception:
            logger.exception("Warning: Failed to upload site for dynamic serpapi: %s", result["link"])

    for result in response["items"]:
        process_site(result)
    filter_expr = f"json_contains(metadata['bot_and_tool_id'], '{bot_id + tool_name}')"
    return query(search_collection, qr, expr=filter_expr)


@observe()
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
        results

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
    if t.method == SearchMethodEnum.dynamic_serpapi:
        # add jurisdictions argument
        body["function"]["parameters"]["properties"].update({
            "jurisdictions": courtlistener_tool_args["jurisdictions"],
        })
        # modify query text to include jurisdiction in both args
        body["function"]["parameters"]["properties"]["qr"]["description"] = (
            "The search text. Include the jurisdiction here as well, if provided."
        )
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


def run_search_tool(tool: SearchTool, function_args: dict) -> dict:
    """Create a search tool for an openai agent.

    Parameters
    ----------
    tool : SearchTool
        The SearchTool object which describes the tool
    function_args : dict
        The arguments to pass to the function

    Returns
    -------
    dict
        The response from the search tool

    """
    function_response = None
    prf = tool.prefix
    qr = function_args["qr"]
    match tool.method:
        case SearchMethodEnum.serpapi:
            function_response = serpapi_tool(qr, prf)
        case SearchMethodEnum.dynamic_serpapi:
            if "jurisdictions" in function_args:
                tool.jurisdictions = [j.upper() for j in function_args["jurisdictions"]]
            function_response = dynamic_serpapi_tool(qr, prf, tool)
        case SearchMethodEnum.google:
            function_response = google_search_tool(qr, prf)
        case SearchMethodEnum.courtlistener:
            tool_jurisdictions = None
            tool_kw_query = None
            tool_after_date = None
            tool_before_date = None
            if "jurisdictions" in function_args:
                tool_jurisdictions = [
                    jurisdiction.lower()
                    for jurisdiction in function_args["jurisdictions"]
                ]
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
            function_response = courtlistener_query(request)
        case SearchMethodEnum.courtroom5:
            function_response = courtroom5_search_tool(qr, prf)
        case SearchMethodEnum.dynamic_courtroom5:
            function_response = dynamic_courtroom5_search_tool(qr, prf, tool)
    return function_response

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


def format_search_tool_results(tool_output: dict, tool: SearchTool) -> list[dict]:
    formatted_results = []

    if tool_output["message"] != "Success" or "result" not in tool_output:
        logger.error("Unable to format search tool results: %s", tool.name)
        return formatted_results

    for result in tool_output["result"]:
        if "entity" in result:
            entity = result["entity"]
            # pks need to be strings to handle in JavaScript front end
            entity["pk"] = str(result["id"])
        if tool.method == SearchMethodEnum.courtlistener:
            entity_type = "opinion"
            entity_id = entity["opinion_id"]
        elif tool.method in (
            SearchMethodEnum.dynamic_serpapi,
            SearchMethodEnum.dynamic_courtroom5,
        ):
            entity_type = "url"
            entity_id = entity["metadata"]["url"]
        elif tool.method in (
            SearchMethodEnum.courtroom5,
            SearchMethodEnum.google,
            SearchMethodEnum.serpapi,
        ):
            entity_type = "url"
            entity_id = entity["link"]
        else:
            entity_type = "unknown"
            entity_id = "unknown"

        formatted_results.append({
            "type": entity_type,
            "entity": entity,
            "id": entity_id,
        })

    return formatted_results
