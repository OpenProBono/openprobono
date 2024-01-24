import uuid

import firebase_admin
import langchain
from typing import Annotated, List, Union, Tuple
from fastapi import Body, FastAPI
from firebase_admin import credentials, firestore
import gradio as gr
from langchain import PromptTemplate
from langchain.agents import (AgentExecutor, AgentOutputParser, AgentType,
                              LLMSingleActionAgent, Tool, ZeroShotAgent,
                              initialize_agent)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import \
    FinalStreamingStdOutCallbackHandler
from langchain.chains import LLMChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import (TextLoader, UnstructuredURLLoader,
                                        YoutubeLoader)
from langchain.document_loaders.blob_loaders.youtube_audio import \
    YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory, ChatMessageHistory
from langchain.prompts import (BaseChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain.schema import AgentAction, AgentFinish, AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union
from anyio.from_thread import start_blocking_portal
from queue import Queue
import re
from serpapi.google_search import GoogleSearch

langchain.debug = True

# OPB bot main function
def opb_bot(
    history,
    bot_id,
    tools,
    user_prompt = "", 
    session = ""):

    class MyCallbackHandler(BaseCallbackHandler):
        def __init__(self, q):
            self.q = q
        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            self.q.put(token)

    if(history[-1][0].strip() == ""):
        return "Hi, how can I assist you today?"
    else:
        q = Queue()
        job_done = object()

        bot_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-1106', request_timeout=60*5, streaming=True, callbacks=[MyCallbackHandler(q)])
        memory_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-1106')

        memory = ConversationSummaryBufferMemory(llm=memory_llm, max_token_limit=2000, memory_key="memory", return_messages=True)
        for i in range(1, len(history)-1):
            memory.save_context({'input': history[i][0]}, {'output': history[i][1]})

        ##----------------------- tools -----------------------##

        #Filter search results retured by serpapi to only include relavant results
        def filtered_search(results):
            new_dict = {}
            if('sports_results' in results):
                new_dict['sports_results'] = results['sports_results']
            if('organic_results' in results):
                new_dict['organic_results'] = results['organic_results']
            return new_dict

        toolset = []
        tool_names = []
        for t in tools:
            def search_tool(qr):
                data = {"search": t['txt'] + " " + qr, 'prompt': t['prompt'], 'timestamp': firestore.SERVER_TIMESTAMP}
                db.collection(root_path + "search").document(session).collection('searches').document("search" + get_uuid_id()).set(data)
                return filtered_search(GoogleSearch({
                    'q': t['txt'] + " " + qr,
                    'num': 5
                    }).get_dict())

            async def async_search_tool(qr):
                return search_tool(qr)

            toolset.append(Tool(
                name = t["name"],
                func = search_tool,
                coroutine = async_search_tool,
                description = t["prompt"]
            )) 
            tool_names.append(t["name"])

        ##----------------------- end of tools -----------------------##


        #------- agent definition -------#
        system_message = 'You are a helpful AI assistant. ALWAYS use tools to answer questions.'
        system_message += user_prompt
        system_message += '. If you used a tool, ALWAYS return a "SOURCES" part in your answer.'
        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        }

        async def task(prompt):
            #definition of llm used for bot
            prompt = "Using the tools at your disposal, answer the following question: " + prompt
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
            ret = await agent.arun(prompt)
            q.put(job_done)
            return ret

        with start_blocking_portal() as portal:
            portal.start_task_soon(task, history[-1][0])
            content = ""
            while True:
                next_token = q.get(True)
                if next_token is job_done:
                    return content
                content += next_token
                
        


#TODO: cache vector db with bot_id
#TODO: do actual chat memory
#TODO: try cutting off intro and outro part of videos
def youtube_bot(
    history,
    bot_id,
    youtube_urls = [],
    user_prompt = "",
    session = ""):

    if(user_prompt is None or user_prompt == ""):
        user_prompt = "Respond in the same style as the youtuber in the context below."

    prompt_template = user_prompt + """
    \n\nContext: {context}
    \n\n\n\n
    Question: {question}
    Response:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}

    embeddings = OpenAIEmbeddings()
    bot_path = "./youtube_bots/" + bot_id
    try:
        vectordb = FAISS.load_local(bot_path, embeddings)
    except:
        text = ""
        for url in youtube_urls:
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

    query = history[-1][0]
    #check for empty query
    if(query.strip() == ""):
        return ""
    else:
        return qa_chain.run(query)