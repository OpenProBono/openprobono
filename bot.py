from queue import Queue
from typing import Any

import langchain
from anyio.from_thread import start_blocking_portal
from bs4 import ResultSet
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from matplotlib.backend_tools import ToolSetCursor

from milvusdb import session_source_summaries
from models import BotRequest, ChatRequest
from search_tools import search_toolset_creator, serpapi_toolset_creator, search_openai_tool
from vdb_tools import session_query_tool, vdb_toolset_creator, vdb_openai_toolset_creator, vdb_openai_tool

from openai import OpenAI
import json

langchain.debug = True

tools_template = """GENERAL INSTRUCTIONS
    You are a legal expert. Your task is to decide which tools to use to answer a user's question. You can use up to X tools, and you can use tools multiple times with different inputs as well.

    These are the tools which are at your disposale
    {tools}

    When choosing tools, use this template:
    {{"tool": "name of the tool", "input": "input given to the tool"}}

    USER QUESTION
    {input}
    
    ANSWER FORMAT
    {{"tools":["<FILL>"]}}"""


multiple_tools_template = """You are a legal expert, tasked with answering any questions about law. ALWAYS use tools to answer questions.

Combine tool results into a coherent answer. ALWAYS return a "SOURCES" part in your answer.

If you used a web source, include the URL in the citation. If you used a database source, include relevant metadata in the citation.
"""

answer_template = """GENERAL INSTRUCTIONS
    You are a legal expert. Your task is to compose a response to the user's question using the information in the given context. 

    CONTEXT:
    {context}

    USER QUESTION
    {input}"""

# 1 subquestion

# decide x tools to use and their inputs
# out of these tools and their descriptions, choose up to x ToolS

# -> run all the tools and returns ResultSet


# -> here are the results from the tools, create your answer

# OPB bot main function
def opb_bot(r: ChatRequest, bot: BotRequest):
    class MyCallbackHandler(BaseCallbackHandler):
        def __init__(self, q):
            self.q = q
        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            self.q.put(token)

    if(r.history[-1][0].strip() == ""):
        return "Hi, how can I assist you today?"
    else:
        q = Queue()
        job_done = object()

        bot_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-1106', request_timeout=60*5, streaming=True, callbacks=[MyCallbackHandler(q)])
        chat_history = []
        for tup in r.history[1:len(r.history) - 1]:
            if tup[0]:
                chat_history.append(HumanMessage(content=tup[0]))
            if tup[1]:
                chat_history.append(AIMessage(content=tup[1]))
        #------- agent definition -------#
        toolset = []
        if(bot.search_tool_method == "google_search"):
            toolset += search_toolset_creator(bot)
        else:
            toolset += serpapi_toolset_creator(bot)
        toolset += vdb_toolset_creator(bot.vdb_tools)
        # sesssion tool
        source_summaries = session_source_summaries(r.session_id)
        if source_summaries:
            toolset.append(session_query_tool(r.session_id, source_summaries))
            # system_message += f'The session_query_tool sources have these summaries: {source_summaries}.' #this temporary change for testings

        prompt: ChatPromptTemplate = hub.pull("hwchase17/openai-tools-agent")
        prompt.messages[0].prompt.template = multiple_tools_template
        agent = create_openai_tools_agent(bot_llm, toolset, prompt)

        async def task(prompt):
            #definition of llm used for bot
            prompt = bot.message_prompt + prompt
            # Create an agent executor by passing in the agent and tools
            agent_executor = AgentExecutor(agent=agent, tools=toolset, verbose=False, return_intermediate_steps=False)
            ret = await agent_executor.ainvoke({"input" :prompt, "chat_history": chat_history})
            q.put(job_done)
            return ret

        with start_blocking_portal() as portal:
            portal.start_task_soon(task, r.history[-1][0])
            content = ""
            while True:
                next_token = q.get(True)
                if next_token is job_done:
                    return content
                content += next_token

def openai_bot(r: ChatRequest, bot: BotRequest):
    if(r.history[-1][0].strip() == ""):
        return "Hi, how can I assist you today?"
    client = OpenAI()
    # Step 1: send the conversation and available functions to the model
    messages = []
    for tup in r.history:
        if tup[0]:
            messages.append({"role": "user", "content": tup[0]})
        if tup[1]:
            messages.append({"role": "assistant", "content": tup[1]})
    # [openai tool definitions, tool name -> actual function to call on response mappings]
    toolset = vdb_openai_toolset_creator(bot.vdb_tools)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        tools=toolset,
        tool_choice="auto",  # auto is default, but we'll be explicit
        temperature=0
    )
    response_message = response.choices[0].message
    messages.append(response_message)  # extend conversation with assistant's reply
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            print(f"function_name = {function_name}")
            function_args = json.loads(tool_call.function.arguments)
            vdb_tool = next((t for t in bot.vdb_tools if function_name == f"{t['name']}_{t['collection_name']}"), None)
            print(f"vdb_tool = {vdb_tool}")
            search_tool = None # TODO: openai search tools
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            if vdb_tool:
                tool_response = vdb_openai_tool(vdb_tool, function_args)
            elif search_tool:
                tool_response = search_openai_tool(search_tool, function_args)
            else:
                tool_response = "error: unable to run tool"
            print(f"tool_response = {tool_response}")
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": tool_response,
                }
            )  # extend conversation with function response
        # Step 4: send the info for each function call and function response to the model
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            temperature=0
        )  # get a new response from the model where it can see the function response
        return second_response.choices[0].message.content
    else:
        return response_message.content