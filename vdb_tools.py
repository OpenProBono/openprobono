from langchain.agents import Tool
from pymilvus import Collection

from milvusdb import SESSION_DATA, qa, query
from models import BotRequest, EngineEnum, VDBMethodEnum, VDBTool


def qa_tool(tool: VDBTool):
    async def async_qa(tool: VDBTool, question: str):
        return qa(tool.collection_name, question, tool.k)

    tool_func = lambda q: qa(tool.collection_name, q, tool.k)
    tool_co = lambda q: async_qa(tool, q)

    if tool.prompt != "":
        description = tool.prompt
    else:
        description = f"""This tool answers questions by searching for the top {tool.k} results from a database named {tool.collection_name}."""
        description += f" The database description is: {Collection(tool.collection_name).description}."

    return Tool(name=tool.name,
                func=tool_func,
                coroutine=tool_co,
                description=description)


def query_tool(tool: VDBTool):
    async def async_query(tool: VDBTool, q: str):
        return query(tool.collection_name, q, tool.k)

    tool_func = lambda q: query(tool.collection_name, q, tool.k)
    tool_co = lambda q: async_query(tool, q)

    if tool.prompt != "":
        description = tool.prompt
    else:
        description = f"This tool queries a database named {tool.collection_name} and returns the top {tool['k']} results."
        description += f" The database description is: {Collection(tool.collection_name).description}."

    return Tool(name=tool.name,
                func=tool_func,
                coroutine=tool_co,
                description=description)

def openai_qa_tool(tool: VDBTool):
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.prompt,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "the question to answer",
                    },
                },
                "required": ["question"],
            },
        },
    }

def anthropic_qa_tool(tool: VDBTool):
    return {
        "name": tool.name,
        "description": tool.prompt,
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "the question to answer",
                },
            },
            "required": ["question"],
        },
    }


def openai_query_tool(tool: VDBTool):
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.prompt,
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
        "name": tool.name,
        "description": tool.prompt,
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


def vdb_openai_tool(t: VDBTool, function_args):
    function_response = None
    collection_name = t.collection_name
    k = t.k
    if (t.method == VDBMethodEnum.query):
        tool_query = function_args.get("query")
        function_response = query(collection_name, tool_query, k)
    elif (t.method == VDBMethodEnum.qa):
        tool_question = function_args.get("question")
        function_response = qa(collection_name, tool_question, k)
    return str(function_response)

def vdb_anthropic_tool(t: VDBTool, function_args: dict):
    function_response = None
    collection_name = t.collection_name
    k = t.k
    if (t.method == VDBMethodEnum.query):
        tool_query = function_args["query"]
        function_response = query(collection_name, tool_query, k)
    elif (t.method == VDBMethodEnum.qa):
        tool_question = function_args["question"]
        function_response = qa(collection_name, tool_question, k)
    return str(function_response)

def vdb_toolset_creator(bot: BotRequest):
    toolset = []
    for t in bot.vdb_tools:
        if (t.method == VDBMethodEnum.qa):
            if (bot.chat_model.engine == EngineEnum.langchain):
                toolset.append(qa_tool(t))
            elif (bot.chat_model.engine == EngineEnum.openai):
                toolset.append(openai_qa_tool(t))
            elif bot.chat_model.engine == EngineEnum.anthropic:
                toolset.append(anthropic_qa_tool(t))
        elif (t.method == VDBMethodEnum.query):
            if (bot.chat_model.engine == EngineEnum.langchain):
                toolset.append(query_tool(t))
            elif (bot.chat_model.engine == EngineEnum.openai):
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
