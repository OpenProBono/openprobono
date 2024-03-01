from langchain.agents import Tool

from milvusdb import SESSION_PDF, qa, query


def vdb_qa_tool(tool: dict):
    def vdb_tool(tool: dict, question: str):
        return qa(tool["database_name"], question, tool["k"])

    async def async_vdb_tool(tool: dict, question: str):
        return vdb_tool(tool, question)
    
    tool_func = lambda q: vdb_tool(tool, q)
    co_func = lambda q: async_vdb_tool(tool, q)
    if 'description' in tool.keys:
        description = tool["description"]
    else:
        description = f"""Tool used to answer questions using the {tool["k"]} most relevant text chunks from a vector database named {tool["database_name"]}."""
    return Tool(
        name = tool["name"],
        func = tool_func,
        coroutine = co_func,
        description = description
    )

def vdb_query_tool(tool: dict):
    def vdb_tool(tool: dict, q: str):
        return query(tool["database_name"], q, tool["k"])

    async def async_vdb_tool(tool: dict, q: str):
        return vdb_tool(tool, q)
    
    tool_func = lambda q: vdb_tool(tool, q)
    co_func = lambda q: async_vdb_tool(tool, q)
    if 'description' in tool.keys:
        description = tool["description"]
    else:
        description = f"""Tool used to query a vector database named {tool["database_name"]} and return the {tool["k"]} most relevant text chunks."""
    return Tool(
        name = tool["name"],
        func = tool_func,
        coroutine = co_func,
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

def vdb_toolset_creator(tools: list[dict]):
    toolset = []
    for t in tools:
        if "qa" in t["name"]:
            toolset.append(vdb_qa_tool(t))
        elif "query" in t["name"]:
            toolset.append(vdb_query_tool(t))
    return toolset