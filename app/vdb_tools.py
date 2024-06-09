from langchain.agents import Tool
from pymilvus import Collection

from app.milvusdb import SESSION_DATA, query
from app.models import BotRequest, EngineEnum, VDBTool


def get_tool_description(tool: VDBTool):
    if tool.prompt != "":
        return tool.prompt
    return (
        f"This tool queries a database named {tool.collection_name} "
        f"and returns the top {tool.k} results. "
        f"The database description is: {Collection(tool.collection_name).description}."
    )


def query_tool(tool: VDBTool):
    async def async_query(tool: VDBTool, q: str):
        return query(tool.collection_name, q, tool.k)

    tool_func = lambda q: query(tool.collection_name, q, tool.k)
    tool_co = lambda q: async_query(tool, q)

    return Tool(name="query-" + tool.collection_name,
                func=tool_func,
                coroutine=tool_co,
                description=get_tool_description(tool))


def openai_query_tool(tool: VDBTool):
    return {
        "type": "function",
        "function": {
            "name": "query-" + tool.collection_name,
            "description": get_tool_description(tool),
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


def anthropic_query_tool(tool: VDBTool):
    return {
        "name": "query-" + tool.collection_name,
        "description": get_tool_description(tool),
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


def run_vdb_tool(t: VDBTool, function_args, engine: EngineEnum):
    function_response = None
    collection_name = t.collection_name
    k = t.k
    if engine == EngineEnum.openai:
        tool_query = function_args.get("query")
    else: # anthropic
        tool_query = function_args["query"]
    function_response = query(collection_name, tool_query, k)
    return str(function_response)


def vdb_toolset_creator(bot: BotRequest):
    toolset = []
    for t in bot.vdb_tools:
        if (bot.chat_model.engine == EngineEnum.openai):
            toolset.append(openai_query_tool(t))
        elif bot.chat_model.engine == EngineEnum.anthropic:
            toolset.append(anthropic_query_tool(t))
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
