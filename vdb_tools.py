from milvusdb import qa, query, COLLECTIONS, SESSION_PDF
from langchain.agents import Tool

def vdb_qa_tool(tool: dict):
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

def vdb_query_tool(tool: dict):
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

def vdb_openai_query_tool(tool: dict):
    return {
        "type": "function",
        "function": {
            "name": f"query-{tool['collection_name']}",
            "description": f"Runs a vector similarity search over a database named {tool['collection_name']} and returns the top {tool['k']} chunks",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the user query"},
                },
                "required": ["query"],
            },
        },
    }

def vdb_openai_query_mapping(tool: dict):
    return {"func": query, "args": {"collection_name": tool["collection_name"], "k": tool["k"]}}

def vdb_toolset_creator(tools: list[dict]):
    toolset = []
    for t in tools:
        if "qa" in t["name"]:
            toolset.append(vdb_qa_tool(t))
        elif "query" in t["name"]:
            toolset.append(vdb_query_tool(t))
    return toolset

def vdb_openai_toolset_creator(tools: list[dict]):
    definitions, mappings = [], {}
    for t in tools:
        if "query" in t["name"]:
            definitions.append(vdb_openai_query_tool(t))
            mappings[f"{t['name']}-{t['collection_name']}"] = vdb_openai_query_mapping(t)
    return definitions, mappings