"""Vector database functions and toolset creation."""
from __future__ import annotations

from pymilvus import Collection

from app.milvusdb import SESSION_DATA, check_session_data, query
from app.models import BotRequest, EngineEnum, VDBTool
from app.prompts import VDB_PROMPT


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
    return VDB_PROMPT.format(
        collection_name=tool.collection_name,
        k=tool.k,
        description=Collection(tool.collection_name).description,
    )


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
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": prompt,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the query text",
                    },
                },
                "required": ["query"],
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
    return {
        "name": tool.name,
        "description": prompt,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "the query text",
                },
            },
            "required": ["query"],
        },
    }


def run_vdb_tool(t: VDBTool, function_args: dict) -> str:
    """Run a tool on a vector database.

    Parameters
    ----------
    t : VDBTool
        Tool parameters
    function_args : dict
        Any arguments for the tool function

    Returns
    -------
    str
        The response from the tool function

    """
    function_response = None
    collection_name = t.collection_name
    k = t.k
    tool_query = function_args["query"]
    if collection_name == SESSION_DATA:
        function_response = query(collection_name, tool_query, k, session_id=t.session_id)
    else:
        function_response = query(collection_name, tool_query, k)
    return str(function_response)


def vdb_toolset_creator(bot: BotRequest) -> list[VDBTool]:
    """Create a list of VDBTools from the current chat model.

    Parameters
    ----------
    bot : BotRequest
        The bot definition

    Returns
    -------
    list[VDBTool]
        The list of VDBTools

    """
    toolset = []
    for t in bot.vdb_tools:
        if (bot.chat_model.engine == EngineEnum.openai):
            toolset.append(openai_tool(t))
        elif bot.chat_model.engine == EngineEnum.anthropic:
            toolset.append(anthropic_tool(t))
    return toolset


def session_data_toolset_creator(session_id: str | None) -> VDBTool | None:
    """Get a vdb tool for user uploaded files."""
    if(session_id and check_session_data(session_id)):
        return VDBTool(
            name="session_data",
            collection_name=SESSION_DATA,
            k=5,
            prompt="Used to search user uploaded data. Only available if a user has uploaded a file.",
            session_id=session_id,
        )
    else:
        return None


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
