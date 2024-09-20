"""Vector database functions and toolset creation."""
from __future__ import annotations

from pymilvus import Collection

from app.courtlistener import courtlistener_collection
from app.milvusdb import SESSION_DATA, check_session_data, get_expr, query
from app.models import BotRequest, EngineEnum, SearchMethodEnum, VDBMethodEnum, VDBTool
from app.prompts import VDB_QUERY_PROMPT, VDB_SOURCE_PROMPT
from app.search_tools import search_collection


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


def run_vdb_tool(t: VDBTool, function_args: dict) -> tuple[dict, list[str]]:
    """Run a tool on a vector database.

    Parameters
    ----------
    t : VDBTool
        Tool parameters
    function_args : dict
        Any arguments for the tool function

    Returns
    -------
    tuple[dict, list[str]]
        response, sources

    """
    function_response = None
    collection_name = t.collection_name
    k = t.k
    match t.method:
        case VDBMethodEnum.query:
            tool_query = function_args["query"]
            if collection_name == SESSION_DATA:
                function_response = query(
                    collection_name,
                    tool_query,
                    k,
                    session_id=t.session_id,
                )
                # session data uses filenames
                sources = [
                    hit["entity"]["metadata"]["filename"]
                    for hit in function_response["result"]
                ] if "result" in function_response else []
            else:
                function_response = query(collection_name, tool_query, k)
                # other collections use urls
                sources = [
                    hit["entity"]["metadata"]["url"]
                    for hit in function_response["result"]
                ] if "result" in function_response else []
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
                expr = f"opinion_id=={tool_source}"
            else:
                # probably search_collection, assume source id is a URL
                expr = f"metadata['url']=='{tool_source}'"
            res = get_expr(collection_name, expr)
            function_response = {
                "text": "\n".join([hit["text"] for hit in res["result"]]),
                # we're not currently doing anything that requires every chunks metadata
                # so just return the first instance
                "metadata": res["result"][0]["metadata"],
            }
            sources = [tool_source]
    return function_response, sources


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
    if session_id and check_session_data(session_id):
        bot.vdb_tools.append(VDBTool(
            name="session_data",
            collection_name=SESSION_DATA,
            k=5,
            prompt="Used to search user uploaded data. Only available if a user has uploaded a file.",
            session_id=session_id,
        ))
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


def format_vdb_tool_results(tool_output: dict, tool: VDBTool) -> list[str]:
    formatted_results = []

    if tool_output["message"] != "Success" or "result" not in tool_output:
        return formatted_results

    for i, result in enumerate(tool_output["result"], start=1):
        entity = result["entity"]
        metadata = entity["metadata"]

        # Format the text, truncating if necessary
        text = entity["text"][:500] + "..." if len(entity["text"]) > 500 else entity["text"]

        # Select important metadata based on tool type
        if tool.collection_name == courtlistener_collection:
            important_metadata = {
                "Case Name": metadata.get("case_name", "N/A"),
                "Date Filed": metadata.get("date_filed", "N/A"),
                "Court": metadata.get("court_name", "N/A"),
                "Author": metadata.get("author_name", "N/A"),
                "Precedential Status": metadata.get("precedential_status", "N/A"),
            }
            title = metadata.get("case_name", "Unknown Opinion")
        elif tool.collection_name == search_collection:
            important_metadata = {
                "URL": metadata.get("url", "N/A"),
                "AI Summary": metadata.get("ai_summary", "N/A"),
                "Page Number": metadata.get("page_number", "N/A"),
            }
            title = metadata.get("url", "Unknown URL")
        elif tool.collection_name == SESSION_DATA:
            important_metadata = {
                "Filename": metadata.get("filename", "N/A"),
                "Page Number": metadata.get("page_number", "N/A"),
            }
            title = metadata.get("filename", "Unknown File")
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
