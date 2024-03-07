from milvusdb import qa, query, COLLECTIONS, SESSION_PDF
from langchain.agents import Tool

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
        description = f"""Tool used to answer questions using the {tool["k"]} most relevant text chunks from a vector database named {tool["collection_name"]}."""
    
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
        description = f"Tool used to query a vector database named {tool['collection_name']} and return the {tool['k']} most relevant text chunks."

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
        description = f"Tool used to query a vector database including information about the San Diego Volunteer Lawyer Program and return the most relevant text chunks. You can use this tool to query for legal and procedural information as well." #this temporary change for testings
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
    if 'query' in tool["name"]:
        tool_query = function_args.get("query")
        collection_name = tool["collection_name"]
        k = tool["k"]
        function_response = query(collection_name, tool_query, k)
    elif 'qa' in tool["name"]:
        tool_question = function_args.get("question")
        collection_name = tool["collection_name"]
        k = tool["k"]
        function_response = qa(collection_name, tool_question, k)
    if function_response:
        return str(function_response)
    return "error: unable to run tool"

def vdb_toolset_creator(tools: list[dict]):
    toolset = []
    for t in tools:
        if "qa" in t["name"]:
            toolset.append(qa_tool(t))
        elif "query" in t["name"]:
            toolset.append(query_tool(t))
    return toolset

def vdb_openai_toolset_creator(tools: list[dict]):
    toolset = []
    for t in tools:
        if "qa" in t["name"]:
            toolset.append(openai_qa_tool(t))
        elif "query" in t["name"]:
            toolset.append(openai_query_tool(t))
    return toolset