from milvusdb import qa, query, COLLECTIONS, SESSION_PDF, Collection
from langchain.agents import Tool

from models import BotRequest

def qa_tool(tool: dict):
    if tool["collection_name"] not in COLLECTIONS:
        raise ValueError(f"invalid collection_name {tool['collection_name']}")
    
    async def async_qa(tool: dict, question: str):
        return qa(tool["collection_name"], question, tool["k"])

    tool_func = lambda q: qa(tool["collection_name"], q, tool["k"])
    tool_co = lambda q: async_qa(tool, q)

    if "description" in tool:
        description = tool["description"]
    else:
        description = f"""This tool answers questions by searching for the top {tool["k"]} results from a database named {tool["collection_name"]}."""
        description += f" The database description is: {Collection(tool['collection_name']).description}."
    
    return Tool(
        name = f"{tool['name']}-{tool['collection_name']}",
        func = tool_func,
        coroutine = tool_co,
        description = description
    )

def query_tool(tool: dict):
    if tool["collection_name"] not in COLLECTIONS:
        raise ValueError(f"invalid collection_name {tool['collection_name']}")
    
    async def async_query(tool: dict, q: str):
        return query(tool["collection_name"], q, tool["k"])

    tool_func = lambda q: query(tool["collection_name"], q, tool["k"])
    tool_co = lambda q: async_query(tool, q)

    if 'description' in tool:
        description = tool["description"]
    else:
        description = f"This tool queries a database named {tool['collection_name']} and returns the top {tool['k']} results."
        description += f" The database description is: {Collection(tool['collection_name']).description}."

    return Tool(
        name = f"{tool['name']}-{tool['collection_name']}",
        func = tool_func,
        coroutine = tool_co,
        description = description
    )

def session_query_tool(session_id: str, source_summaries: dict):
    def query_tool(q: str):
        return query(SESSION_PDF, q, session_id=session_id)
    
    async def async_query_tool(q: str):
        return query_tool(q)
    
    tool_func = lambda q: query_tool(q)
    co_func = lambda q: async_query_tool(q)
    return Tool(
            name = "session_query_tool",
            func = tool_func,
            coroutine = co_func,
            description = f"Tool used to query a vector database including information uploaded by the user and return the most relevant text chunks." 
        )

def openai_qa_tool(tool: dict):
    return {
        "type": "function",
        "function": {
            "name": f"qa_{tool['collection_name']}",
            "description": f"This tool answers questions using a database named {tool['collection_name']}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "the question to answer"},
                },
                "required": ["question"],
            },
        },
    }

def openai_query_tool(tool: dict):
    return {
        "type": "function",
        "function": {
            "name": f"query_{tool['collection_name']}",
            "description": f"This tool queries a database named {tool['collection_name']} and returns the top {tool['k']} results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the query text"},
                },
                "required": ["query"],
            },
        },
    }

def vdb_openai_tool(tool: dict, function_args):
    function_response = None
    collection_name = tool["collection_name"]
    k = tool["k"]
    if 'query' in tool["name"]:
        tool_query = function_args.get("query")
        function_response = query(collection_name, tool_query, k)
    elif 'qa' in tool["name"]:
        tool_question = function_args.get("question")
        function_response = qa(collection_name, tool_question, k)
    return str(function_response)

def vdb_toolset_creator(bot: BotRequest):
    toolset = []
    for t in bot.vdb_tools:
        if "qa" in t["name"]:
            if bot.engine == 'langchain':
                toolset.append(qa_tool(t))
            elif bot.engine == 'openai':
                toolset.append(openai_qa_tool(t))
        elif "query" in t["name"]:
            if bot.engine == 'langchain':
                toolset.append(query_tool(t))
            elif bot.engine == 'openai':
                toolset.append(openai_query_tool(t))
    return toolset