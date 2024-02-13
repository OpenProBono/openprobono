import sys
from queue import Queue
from re import A, U
from typing import Any
from urllib import response

import langchain
from anyio.from_thread import start_blocking_portal
from langchain import PromptTemplate
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.agents.format_scratchpad.openai_tools import \
    format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import \
    OpenAIToolsAgentOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from tools import search_toolset_creator, serpapi_toolset_creator

langchain.debug = True

class BotRequest(BaseModel):
    history: list
    user_prompt: str = ""
    message_prompt: str = ""
    tools: list = []
    youtube_urls: list = []
    beta: bool = False
    search_tool_method: str = "serpapi"
    session: str = None
    bot_id: str = None
    api_key: str = None

# OPB bot main function
def opb_bot(r: BotRequest):
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
        system_message += r.user_prompt
        system_message += '. If you used a tool, ALWAYS return a "SOURCES" part in your answer.'
        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        }

        if(r.search_tool_method == "google_search"):
            toolset = search_toolset_creator(r)
        else:
            toolset = serpapi_toolset_creator(r)

        async def task(prompt):
            #definition of llm used for bot
            prompt = r.message_prompt + prompt
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
            portal.start_task_soon(task, r.history[-1][0])
            content = ""
            while True:
                next_token = q.get(True)
                if next_token is job_done:
                    return content
                content += next_token

def beta_bot(r: BotRequest):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    memory_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-1106')

    memory = ConversationSummaryBufferMemory(llm=memory_llm, max_token_limit=2000, memory_key="memory", return_messages=True)
    for i in range(1, len(r.history)-1):
        memory.save_context({'input': r.history[i][0]}, {'output': r.history[i][1]})

    if(r.search_tool_method == "google_search"):
        toolset = search_toolset_creator(r)
    else:
        toolset = serpapi_toolset_creator(r)
    print("toolset")
    print(toolset)
    llm_with_tools = llm.bind_tools(toolset)

    MEMORY_KEY = "chat_history"
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are very powerful assistant, but bad at calculating lengths of words.",
            ),
            #MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    print("agent")
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            #"chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=toolset, verbose=True)
    print("invoke")
    print(agent_executor.invoke({"input": r.history[-1][0],})) #"chat_history": memory}))

# need to make this whole chain work with message history and documents/vdb
def relevance_check(r: BotRequest):
    query = r.history[-1][0].strip()
    messages = [
        SystemMessage(
            content="You are a bot which checks to see if the user's query is a valid legal question. If it is, you should return a YES response. If it is not, you should return a NO response."
        ),
        HumanMessage(
            content=query
        ),
    ]
    model = ChatOpenAI(model="gpt-4")
    return model(messages)

def user_info_check(r: BotRequest):
    query = r.history[-1][0].strip()
    messages = [
        SystemMessage(
            content="You are a bot which checks to see if the user's query needs futher information from the user to answer. If you need further information from the user, return a YES response. If it does not, you should return a NO response."
        ),
        HumanMessage(
            content=query
        ),
    ]
    model = ChatOpenAI(model="gpt-4")
    return model(messages)

def further_research(r: BotRequest):
    query = r.history[-1][0].strip()
    response = r.history[-1][0].strip()
    messages = [
        SystemMessage(
            content="You are a bot which checks to see if the response needs further research to answer given query to far. If it does, you should return a response describing what topic to research further. If it does not, you should return a NO response."
        ),
        HumanMessage(
            content=query
        ),
        AIMessage(
            content=response
        ),
    ]
    model = ChatOpenAI(model="gpt-4")
    return model(messages)

def anticipate_user_actions(r: BotRequest):
    query = r.history[-1][0].strip()
    response = r.history[-1][0].strip()
    messages = [
        SystemMessage(
            content="You are a bot which amends our response to fully satisfy the user's query. Anticipate the user's needs and desires, helping them with additional actions they may want to take."
        ),
        HumanMessage(
            content=query
        ),
        AIMessage(
            content=response
        ),
    ]
    model = ChatOpenAI(model="gpt-4")
    return model(messages)

def final_check(r: BotRequest):
    query = r.history[-1][0].strip()
    response = r.history[-1][0].strip()
    messages = [
        SystemMessage(
            content="You are a bot which summarizes all of the research and information for the end user. You should return a response that is a summary of all the information gathered so far, including the sources of the information. Be very thorough and detailed. Define any legalese that the end user may not know. Be sure to include a SOURCES part in your response."
        ),
        HumanMessage(
            content=query
        ),
        AIMessage(
            content=response
        ),
    ]
    model = ChatOpenAI(model="gpt-4")
    return model(messages)

def call_chain(r: BotRequest):
    rc = relevance_check(r)
    if(relevance_check(r).content == "NO"):
        return "I cannot help you with that."
    
    info_check_response = user_info_check(r)
    #if i need more infomation from the user, ask it for it
    print(info_check_response.content)
    if(info_check_response.content != "NO"):
        print("yes")
    else:
        print("no")
    
    opb_bot_response = opb_bot(r)
    r.history[-1].append(opb_bot_response)

    further_research_response = further_research(r)
    if(further_research_response.content != "NO"):
        while(further_research_response.content != "NO"):
            r.history.append(["Enhance your answer by doing further research about " + further_research_response, ""])
            opb_bot_response = opb_bot(r)
            r.history[-1].append(opb_bot_response)
            further_research_response = further_research(r)
            
    return r.history[-1][1]


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

# print(relevance_check(BotRequest(history=[["What are the health effects of alcohol?", ""]])))
# print(user_info_check(BotRequest(history=[["What is the country with the highest alcohol age limit?", ""]])))
# print(user_info_check(BotRequest(history=[["Is it legal to make a right turn on red in NYC?", ""]])))