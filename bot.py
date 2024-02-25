from queue import Queue
from typing import Any

import langchain
from anyio.from_thread import start_blocking_portal
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders.youtube import YoutubeLoader
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from search_tools import search_toolset_creator, serpapi_toolset_creator
from vdb_tools import vdb_toolset_creator, session_query_tool
from milvusdb import session_source_summaries
from models import BotRequest, ChatRequest

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
            system_message += f' When citing the session_query_tool, include the filename and page. The sources have these summaries: {source_summaries}.'

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
                
        


#TODO: cache vector db with bot_id
#TODO: do actual chat memory
#TODO: try cutting off intro and outro part of videos
def youtube_bot(r: BotRequest):
    if(r.user_prompt is None or r.user_prompt == ""):
        r.user_prompt = "Respond in the same style as the youtuber in the context below."

    prompt_template = r.user_prompt + """
    \n\nContext: {context}
    \n\n\n\n
    Question: {question}
    Response:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}

    embeddings = OpenAIEmbeddings()
    bot_path = "./youtube_bots/" + r.bot_id
    try:
        vectordb = FAISS.load_local(bot_path, embeddings)
    except:
        text = ""
        for url in r.youtube_urls:
            try:
                # Load the audio
                loader = YoutubeLoader.from_youtube_url(
                    url, add_video_info=False
                )
                docs = loader.load()
                # Combine doc
                combined_docs = [doc.page_content for doc in docs]
                text += " ".join(combined_docs)
            except:
                print("Error occured while loading transcript from video with url: " + url)

        # Split them
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        splits = text_splitter.split_text(text)

        # Build an index
        vectordb = FAISS.from_texts(splits, embeddings)
        vectordb.save_local(bot_path)

    # Build a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
    )

    query = r.history[-1][0]
    #check for empty query
    if(query.strip() == ""):
        return ""
    else:
        return qa_chain.run(r.message_prompt + query)