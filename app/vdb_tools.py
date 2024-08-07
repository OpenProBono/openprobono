"""Vector database functions and toolset creation."""
from __future__ import annotations

from langchain.agents import Tool
from pymilvus import Collection

from app.cap import cap, cap_collection, cap_tool_args
from app.milvusdb import SESSION_DATA, query
from app.models import BotRequest, EngineEnum, VDBTool
from app.prompts import FILTERED_CASELAW_PROMPT, VDB_PROMPT


def cap_tool(tool: VDBTool) -> Tool:
    """Create a tool for filtered queries on the CAP collection.

    Parameters
    ----------
    tool : VDBTool
        Tool parameters

    Returns
    -------
    Tool
        The initialized tool

    """
    async def async_cap(
        tool: VDBTool,
        q: str,
        jurisdiction: str,
        after_date: str | None = None,
        before_date: str | None = None,
    ) -> dict:
        return cap(q, tool.k, jurisdiction, after_date, before_date)

    def tool_func(
        tool: VDBTool,
        q: str,
        jurisdiction: str,
        after_date: str | None = None,
        before_date: str | None = None,
    ) -> dict:
        return cap(q, tool.k, jurisdiction, after_date, before_date)
    def tool_co(
        tool: VDBTool,
        q: str,
        jurisdiction: str,
        after_date: str | None = None,
        before_date: str | None = None,
    ) -> dict:
        return async_cap(tool, q, jurisdiction, after_date, before_date)

    prompt = tool.prompt if tool.prompt else FILTERED_CASELAW_PROMPT
    return Tool(name=tool.name,
                func=tool_func,
                coroutine=tool_co,
                description=prompt)

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
    if tool.collection_name == cap_collection:
        return FILTERED_CASELAW_PROMPT
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
    body = {
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
    if tool.collection_name == cap_collection:
        body["function"]["parameters"]["properties"].update(cap_tool_args)
    return body


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
    body = {
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
    if tool.collection_name == cap_collection:
        body["function"]["parameters"]["properties"].update(cap_tool_args)
    return body


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
    if collection_name == cap_collection:
        tool_jurisdiction = function_args["jurisdiction"]
        tool_after_date, tool_before_date = None, None
        if "after-date" in function_args:
            tool_after_date = function_args["after-date"]
        if "before-date" in function_args:
            tool_before_date = function_args["before-date"]
        function_response = cap(
            tool_query,
            k,
            tool_jurisdiction,
            tool_after_date,
            tool_before_date,
        )
    elif collection_name == SESSION_DATA:
        session_id = function_args["session_id"]
        function_response = query(collection_name, tool_query, k, session_id=session_id)
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


# TODO: implement this for openai
# this is a unique type of search for files uploaded during the session (by the user), not defined by the bot
def session_query_tool(session_id: str, source_summaries: dict):

    def query_tool(q: str):
        return query(SESSION_DATA, q, session_id=session_id)

    async def async_query_tool(q: str):
        return query_tool(q)

    tool_func = lambda q: query_tool(q)
    co_func = lambda q: async_query_tool(q)
    return Tool(
        name="session_query_tool",
        func=tool_func,
        coroutine=co_func,
        description=
        "Tool used to query a vector database including information uploaded by the user and return the most relevant text chunks."
    )
