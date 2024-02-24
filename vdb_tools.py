from milvusdb import qa, query, SESSION_PDF
from langchain.agents import Tool

def vdb_qa_tool(tool: dict):
    def vdb_tool(tool: dict, question: str):
        return qa(tool["database_name"], question, tool["k"])

    async def async_vdb_tool(tool: dict, question: str):
        return vdb_tool(tool, question)
    
    tool_func = lambda q: vdb_tool(tool, q)
    co_func = lambda q: async_vdb_tool(tool, q)

    return Tool(
        name = tool["name"],
        func = tool_func,
        coroutine = co_func,
        description = f"""Tool used to answer questions using the {tool["k"]} most relevant text chunks from a vector database named {tool["database_name"]}."""
    )

def vdb_query_tool(tool: dict):
    def vdb_tool(tool: dict, q: str):
        return query(tool["database_name"], q, tool["k"])

    async def async_vdb_tool(tool: dict, q: str):
        return vdb_tool(tool, q)
    
    tool_func = lambda q: vdb_tool(tool, q)
    co_func = lambda q: async_vdb_tool(tool, q)
    return Tool(
        name = tool["name"],
        func = tool_func,
        coroutine = co_func,
        description = f"""Tool used to query a vector database named {tool["database_name"]} and return the {tool["k"]} most relevant text chunks."""
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
            description = f"Tool used to query a vector database containing the following sources: {source_summaries.keys()}"
        )

def session_qa_tool(session_id: str, source_summaries: dict):
    def qa_tool(q: str):
        return qa(SESSION_PDF, q, session_id=session_id)
    
    async def async_qa_tool(q: str):
        return qa_tool(q)
    
    tool_func = lambda q: qa_tool(q)
    co_func = lambda q: async_qa_tool(q)
    return Tool(
            name = "session_qa_tool",
            func = tool_func,
            coroutine = co_func,
            description = f"Tool used to answer questions using the 4 most relevant text chunks from a vector database containing the following sources: {source_summaries.keys()}"
        )

def vdb_toolset_creator(tools: list[dict]):
    toolset = []
    for t in tools:
        if t["name"] == "qa":
            toolset.append(vdb_qa_tool(t))
        elif t["name"] == "query":
            toolset.append(vdb_query_tool(t))
    return toolset