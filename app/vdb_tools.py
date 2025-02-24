"""Vector database functions and toolset creation."""
from __future__ import annotations

from datetime import UTC, datetime

from langfuse.decorators import observe
from pymilvus import Collection

from app.courtlistener import (
    courtlistener_collection,
    courtlistener_query,
    jurisdiction_codes,
)
from app.db import fetch_session
from app.logger import setup_logger
from app.milvusdb import SESSION_DATA, fuzzy_keyword_query, get_expr, query
from app.models import (
    BotRequest,
    EngineEnum,
    FetchSession,
    OpinionSearchRequest,
    SearchMethodEnum,
    VDBMethodEnum,
    VDBTool,
)
from app.prompts import VDB_QUERY_PROMPT, VDB_SOURCE_PROMPT
from app.search_tools import search_collection

logger = setup_logger()

def tool_prompt(tool: VDBTool) -> str:
    """Create a prompt for the given tool.

    Parameters
    ----------
    tool : VDBTool
        Tool parameters

    Returns
    -------
    str
        The tool prompt

    """
    # add default prompts
    match tool.method:
        case VDBMethodEnum.query:
            return VDB_QUERY_PROMPT.format(
                collection_name=tool.collection_name,
                k=tool.k,
                description=Collection(tool.collection_name).description,
            )
        case VDBMethodEnum.get_source:
            return VDB_SOURCE_PROMPT

def openai_tool(tool: VDBTool) -> dict:
    """Create a VDBTool definition for the OpenAI API.

    Parameters
    ----------
    tool : VDBTool
        Tool parameters

    Returns
    -------
    dict
        The tool definition

    """
    prompt = tool.prompt if tool.prompt else tool_prompt(tool)
    match tool.method:
        case VDBMethodEnum.query:
            property_name = "query"
            property_desc = "the query text for vector search"
        case VDBMethodEnum.get_source:
            property_name = "source_id"
            property_desc = (
                "The source identifier for the document to retrieve. "
                "Depending on the type of source, this may be an integer "
                "ID, filename, or URL."
            )
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": prompt,
            "parameters": {
                "type": "object",
                "properties": {
                    property_name: {
                        "type": "string",
                        "description": property_desc,
                    },
                },
                "required": [property_name],
            },
        },
    }


def anthropic_tool(tool: VDBTool) -> dict:
    """Create a VDBTool definition for the Anthropic API.

    Parameters
    ----------
    tool : VDBTool
        Tool parameters

    Returns
    -------
    dict
        The tool definition

    """
    prompt = tool.prompt if tool.prompt else tool_prompt(tool)
    match tool.method:
        case VDBMethodEnum.query:
            property_name = "query"
            property_desc = "the query text for vector search"
        case VDBMethodEnum.get_source:
            property_name = "source_id"
            property_desc = (
                "The source identifier for the document to retrieve. "
                "Depending on the type of source, this may be an integer "
                "ID, filename, or URL."
            )
    return {
        "name": tool.name,
        "description": prompt,
        "input_schema": {
            "type": "object",
            "properties": {
                property_name: {
                    "type": "string",
                    "description": property_desc,
                },
            },
            "required": [property_name],
        },
    }

@observe(capture_output=False)
def run_vdb_tool(t: VDBTool, function_args: dict) -> dict:
    """Run a tool on a vector database.

    Parameters
    ----------
    t : VDBTool
        Tool parameters
    function_args : dict
        Any arguments for the tool function

    Returns
    -------
    dict
        The response from the tool

    """
    function_response = None
    collection_name = t.collection_name
    k = t.k
    match t.method:
        case VDBMethodEnum.query:
            tool_query = function_args["query"]
            expr = ""
            if collection_name == courtlistener_collection:
                req = OpinionSearchRequest(**function_args)
                function_response = courtlistener_query(req)
            else:
                if "keyword_query" in function_args:
                    tool_keyword_query = function_args["keyword_query"]
                    keyword_query = fuzzy_keyword_query(tool_keyword_query)
                    expr += f"text like '% {keyword_query} %'"
                if "jurisdictions" in function_args:
                    valid_jurisdics = []
                    # look up each str in dictionary, append matches as lists
                    for juris in function_args["jurisdictions"]:
                        if juris.lower() in jurisdiction_codes:
                            valid_jurisdics += jurisdiction_codes[juris.lower()].split(" ")
                    # clear duplicate federal district jurisdictions if they exist
                    valid_jurisdics = list(set(valid_jurisdics))
                    expr = f"ARRAY_CONTAINS_ANY(metadata['jurisdictions'], {valid_jurisdics})"
                if "after_date" in function_args:
                    expr += (" and " if expr else "")
                    # convert YYYY-MM-DD to epoch time
                    after_date = datetime.strptime(
                        function_args["after_date"],
                        "%Y-%m-%d",
                    ).replace(tzinfo=UTC)
                    expr += f"metadata['timestamp']>{after_date.timestamp()}"
                if "before_date" in function_args:
                    expr += (" and " if expr else "")
                    # convert YYYY-MM-DD to epoch time
                    before_date = datetime.strptime(
                        function_args["before_date"],
                        "%Y-%m-%d",
                    ).replace(tzinfo=UTC)
                    expr += f"metadata['timestamp']<{before_date.timestamp()}"
                function_response = query(
                    collection_name,
                    tool_query,
                    k,
                    expr=expr,
                    session_id=t.session_id,
                )
        case VDBMethodEnum.get_source:
            tool_source = function_args["source_id"]
            if collection_name == SESSION_DATA:
                # source id is a filename
                expr = (
                    f"metadata['filename']=='{tool_source}' "
                    f"and metadata['session_id']=='{t.session_id}'"
                )
            elif collection_name == courtlistener_collection:
                # source id is an opinion id
                expr = f"source_id=={tool_source}"
            else:
                # probably search_collection, assume source id is a URL
                expr = f"metadata['url']=='{tool_source}'"
            function_response = get_expr(collection_name, expr)
    return function_response


def vdb_toolset_creator(bot: BotRequest, bot_id: str, session_id: str) -> list[VDBTool]:
    """Create a list of VDBTools from the current bot and session.

    Parameters
    ----------
    bot : BotRequest
        The bot definition
    bot_id: str
        The bot ID
    session_id : str
        The session ID to look for session files

    Returns
    -------
    list[VDBTool]
        The list of VDBTools

    """
    toolset = []
    # create source lookup tools for each search tool
    search_src_tools = []
    for t in bot.search_tools:
        if t.method == SearchMethodEnum.courtlistener:
            coll_name = courtlistener_collection
        else:
            coll_name = search_collection
        vdb_tool = VDBTool(
            name=t.name + "-get-source",
            collection_name=coll_name,
            method=VDBMethodEnum.get_source,
            bot_id=bot_id,
        )
        search_src_tools.append(vdb_tool)
    # create source lookup tools for each query tool
    query_src_tools = []
    for t in bot.vdb_tools:
        t.bot_id = bot_id
        src_tool = t
        src_tool.method = VDBMethodEnum.get_source
        src_tool.prompt = VDB_SOURCE_PROMPT
        query_src_tools.append(src_tool)
    # add the source lookup tools to vdb tool list so we can call them for LLM
    bot.vdb_tools += search_src_tools + query_src_tools
    # add the session query tool, if necessary
    session = FetchSession(api_key=bot.api_key, session_id=session_id)
    session_info = fetch_session(session)
    if session_info.file_count > 0:
        bot.vdb_tools += [
            VDBTool(
                name="session_data",
                collection_name=SESSION_DATA,
                k=5,
                prompt="Use to search user uploaded files. ALWAYS use this tool if its available.",
                session_id=session_id,
            ),
            VDBTool(
                name="session_data-get-source",
                collection_name=SESSION_DATA,
                method=VDBMethodEnum.get_source,
                bot_id=bot_id,
                prompt="This tool gets all of the text chunks comprising a user uploaded file in their original order. The source ID is always the filename.",
                session_id=session_id,
            ),
        ]
    match bot.chat_model.engine:
        case EngineEnum.openai:
            toolset += [openai_tool(t) for t in bot.vdb_tools]
        case EngineEnum.anthropic:
            toolset += [anthropic_tool(t) for t in bot.vdb_tools]
    return toolset


def find_vdb_tool(bot: BotRequest, tool_name: str) -> VDBTool | None:
    """Find the vdb tool with the given name.

    Parameters
    ----------
    bot : BotRequest
        The bot
    tool_name : str
        The tool/function name

    Returns
    -------
    VDBTool | None
        The matching tool or None if not found

    """
    return next(
        (t for t in bot.vdb_tools if tool_name == t.name),
        None,
    )


def format_vdb_tool_results(tool_output: dict, tool: VDBTool) -> list[dict]:
    """Format the output of a VDB tool.

    Parameters
    ----------
    tool_output : dict
        The output of the VDB tool
    tool : VDBTool
        The tool definition

    Returns
    -------
    list[dict]
        The formatted results

    """
    if tool_output["message"] != "Success" or "result" not in tool_output:
        logger.error("Unable to format VDB tool results: %s", tool.name)
        return []

    if tool.method == VDBMethodEnum.query:
        entities = []
        # query results contain 'distance', 'pk', and 'entity' keys
        for hit in tool_output["result"]:
            entity = hit["entity"]
            # pks need to be strings to handle in JavaScript front end
            entity["pk"] = str(hit["pk" if "pk" in hit else "id"])
            entity["distance"] = hit["distance"]
            entities.append(entity)
    else:
        # get expression results contain 'pk' and 'vector' keys
        entities = tool_output["result"]
        for hit in entities:
            if "vector" in hit:
                del hit["vector"]
            # pks need to be strings to handle in JavaScript front end
            hit["pk"] = str(hit["pk" if "pk" in hit else "id"])

    if tool.collection_name == courtlistener_collection:
        entity_type = "opinion"
        entity_id_key = "id"
    elif tool.collection_name in {search_collection, "search_collection_gemini"}:
        entity_type = "url"
        entity_id_key = "url"
    elif tool.collection_name == SESSION_DATA:
        entity_type = "file"
        entity_id_key = "filename"
    else:
        entity_type = "unknown"
        entity_id_key = "unknown"

    return [{
        "id": "unknown" if entity_type == "unknown" else hit["metadata"][entity_id_key],
        "type": entity_type,
        "entity": hit,
    } for hit in entities]
