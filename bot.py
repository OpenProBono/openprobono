from queue import Queue
from typing import Any

import langchain
from anyio.from_thread import start_blocking_portal
from bs4 import ResultSet
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from matplotlib.backend_tools import ToolSetCursor

from milvusdb import session_source_summaries
from models import BotRequest, ChatRequest
from search_tools import search_toolset_creator, serpapi_toolset_creator
from vdb_tools import session_query_tool, vdb_toolset_creator

langchain.debug = True

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
        memory_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-1106')

        memory = ConversationSummaryBufferMemory(llm=memory_llm, max_token_limit=2000, memory_key="memory", return_messages=True)
        for i in range(1, len(r.history)-1):
            memory.save_context({'input': r.history[i][0]}, {'output': r.history[i][1]})

        #------- agent definition -------#
        system_message = 'You are a helpful AI assistant. ALWAYS use tools to answer questions.'
        system_message += bot.user_prompt
        system_message += '. If you used a tool, ALWAYS return a "SOURCES" part in your answer.'
        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        }
        toolset = []
        if(bot.search_tool_method == "google_search"):
            toolset += search_toolset_creator(bot)
        else:
            toolset += serpapi_toolset_creator(bot)
        toolset += vdb_toolset_creator(bot.vdb_tools)
        source_summaries = session_source_summaries(r.session_id)
        if source_summaries:
            toolset.append(session_query_tool(r.session_id, source_summaries))
            # system_message += f'The session_query_tool sources have these summaries: {source_summaries}.' #this temporary change for testings

        async def task(prompt):
            #definition of llm used for bot
            prompt = bot.message_prompt + prompt
            agent = initialize_agent(
                tools=toolset,
                llm=bot_llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=False,
                agent_kwargs=agent_kwargs,
                memory=memory,
                #return_intermediate_steps=True
            )
            agent.agent.prompt.messages[0].content = system_message
            ret = await agent.ainvoke(prompt)
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




                