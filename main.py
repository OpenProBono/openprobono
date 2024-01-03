#fastapi implementation of openprobono bot
# from serpapi import GoogleSearch
import uuid

import firebase_admin
import langchain
from fastapi import FastAPI
from firebase_admin import credentials, firestore
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
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import (BaseChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain.schema import AgentAction, AgentFinish, AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from pydantic import BaseModel

# GoogleSearch.SERP_API_KEY = "5567e356a3e19133465bc68755a124268543a7dd0b2809d75b038797b43626ab"

cred = credentials.Certificate("../../creds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

root_path = "API"

def get_uuid_id():
    return str(uuid.uuid4())

def store_conversation(conversation, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, youtube_urls, session, api_key):
    (human, ai) = conversation[-1]
    data = {"human": human, "ai": ai, "t1name": t1name, 't1txt': t1txt, "t1prompt":t1prompt, "t2name": t2name, "t2txt":t2txt, "t2prompt":t2prompt, 'user_prompt': user_prompt, 'youtube_urls': youtube_urls, 'timestamp':  firestore.SERVER_TIMESTAMP, 'api_key': api_key}
    db.collection("API" + "conversations").document(session).collection('conversations').document("msg" + str(len(conversation))).set(data)

def create_bot(bot_id, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, youtube_urls):
    data = {"t1name": t1name, 't1txt': t1txt, "t1prompt":t1prompt, "t2name": t2name, "t2txt":t2txt, "t2prompt":t2prompt, 'user_prompt': user_prompt, 'youtube_urls': youtube_urls, 'timestamp':  firestore.SERVER_TIMESTAMP}
    db.collection("bots").document(bot_id).set(data)

def load_bot(bot_id):
    bot = db.collection("bots").document(bot_id).get()
    if(bot.exists):
        return bot.to_dict()
    else:
        return None
    
prompt_template = """Respond in the same style as the context below.
{context}
Question: {question}
Response:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT} 

#TODO: cache vector db with bot_id
def process(
    history, 
    t1name = "government-search", 
    t1txt = "site:*.gov | site:*.edu | site:*scholar.google.com", 
    t1prompt = "Useful for when you need to answer questions or find resources about government and laws. Always cite your sources.", 
    t2name = "case-search", 
    t2txt = "site:*case.law | site:*.gov | site:*.edu | site:*courtlistener.com | site:*scholar.google.com", 
    t2prompt = "Use for finding case law. Always cite your sources.", 
    user_prompt = "",
    youtube_urls = [],
    session = ""):

    text = ""
    for url in urls:
        loader = YoutubeLoader.from_youtube_url(
            url, add_video_info=False
        )

        docs = loader.load()

        # Combine doc
        combined_docs = [doc.page_content for doc in docs]
        text += " ".join(combined_docs)

    # Split them
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_text(text)

    # Build an index
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_texts(splits, embeddings)

    
    # Build a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
    )

    
    query = history[-1][0]
    if(query.strip() == ""):
        history[-1][1] = ""
    else:
        history[-1][1] = qa_chain.run(query)
    return history

# def openprobono_bot(history, 
#     t1name = "government-search", 
#     t1txt = "site:*.gov | site:*.edu | site:*scholar.google.com", 
#     t1prompt = "Useful for when you need to answer questions or find resources about government and laws. Always cite your sources.", 
#     t2name = "case-search", 
#     t2txt = "site:*case.law | site:*.gov | site:*.edu | site:*courtlistener.com | site:*scholar.google.com", 
#     t2prompt = "Use for finding case law. Always cite your sources.", 
#     user_prompt = "", 
#     session = ""):
#     if(history[-1][0].strip() == ""):
#         history[-1][1] = "Hi, how can I assist you today?"
#         return history 
#     else:
#         prompt = history[-1][0]
#         history_langchain_format = ChatMessageHistory()
#         for i in range(1, len(history)-1):
#             (human, ai) = history[i]
#             history_langchain_format.add_user_message(human)
#             history_langchain_format.add_ai_message(ai)
#         memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format, memory_key="memory")
#         ##----------------------- tools -----------------------##

#         # def gov_search(q):
#         #     data = {"search": t1txt + " " + q, 'prompt':t1prompt,'timestamp': firestore.SERVER_TIMESTAMP}
#         #     db.collection(root_path + "search").document(session).collection('searches').document("search" + get_uuid_id()).set(data)
#         #     return filtered_search(GoogleSearch({
#         #         'q': t1txt + " " + q,
#         #         'num': 5
#         #         }).get_dict())

#         # def case_search(q):
#         #     data = {"search": t2txt + " " + q, 'prompt': t2prompt, 'timestamp': firestore.SERVER_TIMESTAMP}
#         #     db.collection(root_path + "search").document(session).collection('searches').document("search" + get_uuid_id()).set(data)
#         #     return filtered_search(GoogleSearch({
#         #         'q': t2txt + " " + q,
#         #         'num': 5
#         #         }).get_dict())

#         # async def async_gov_search(q):
#         #     return gov_search(q)

#         # async def async_case_search(q):
#         #     return case_search(q)

#         # #Filter search results retured by serpapi to only include relavant results
#         # def filtered_search(results):
#         #     new_dict = {}
#         #     if('sports_results' in results):
#         #         new_dict['sports_results'] = results['sports_results']
#         #     if('organic_results' in results):
#         #         new_dict['organic_results'] = results['organic_results']
#         #     return new_dict

#         #Definition and descriptions of tools aviailable to the bot
#         tools = []
#         #     Tool(
#         #         name=t1name,
#         #         func=gov_search,
#         #         coroutine=async_gov_search,
#         #         description=t1prompt,
#         #     ),
#         #     Tool(
#         #         name=t1name,
#         #         func=case_search,
#         #         coroutine=async_case_search,
#         #         description=t2prompt,
#         #     )
#         # ]
#         ##----------------------- end of tools -----------------------##

#         system_message = 'You are a helpful AI assistant. ALWAYS use tools to answer questions.'
#         system_message += user_prompt
#         system_message += '. If you used a tool, ALWAYS return a "SOURCES" part in your answer.'
#         agent_kwargs = {
#             "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
#         }
       
#         #definition of llm used for bot
#         prompt = "Using the tools at your disposal, answer the following question: " + prompt
#         bot_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613', request_timeout=60*5)
#         agent = initialize_agent(
#             tools=tools,
#             llm=bot_llm,
#             agent=AgentType.OPENAI_FUNCTIONS,
#             verbose=False,
#             agent_kwargs=agent_kwargs,
#             memory=memory,
#             #return_intermediate_steps=True
#         )
#         agent.agent.prompt.messages[0].content = system_message
#         history[-1][1] = agent.run(prompt)
#         return history


class BotRequest(BaseModel):
    history: list
    t1name: str = None
    t1txt: str = None
    t1prompt: str = None
    t2name: str = None
    t2txt: str = None
    t2prompt: str = None
    user_prompt: str = ""
    youtube_urls: list = []
    session: str = None
    bot_id: str = None
    api_key: str = None

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "It's OpenProBono !!"}

@app.post("/youtube")
def bot(request: BotRequest):
    request_dict = request.dict()
    history = request_dict['history']
    t1name = request_dict['t1name']
    t1txt = request_dict['t1txt']
    t1prompt = request_dict['t1prompt']
    t2name = request_dict['t2name']
    t2txt = request_dict['t2txt']
    t2prompt = request_dict['t2prompt']
    user_prompt = request_dict['user_prompt']
    youtube_urls = request_dict['youtube_urls']
    session = request_dict['session']
    bot_id = request_dict['bot_id']
    api_key = request_dict['api_key']

    #if api key is valid (TODO: change this to a real api key)
    if(api_key == 'xyz'):
        #if bot_id is not provided, create a new bot id
        if bot_id is None or bot_id == "":
            bot_id = get_uuid_id()
            create_bot(bot_id, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, youtube_urls)
        #if bot_id is provided, load the bot
        else:
            bot = load_bot(bot_id)
            #if bot is not found, create a new bot
            if(bot is None):
                create_bot(bot_id, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, youtube_urls)
            #else load bot settings
            else:
                t1name = bot['t1name']
                t1txt = bot['t1txt']
                t1prompt = bot['t1prompt']
                t2name = bot['t2name']
                t2txt = bot['t2txt']
                t2prompt = bot['t2prompt']
                user_prompt = bot['user_prompt']
                youtube_urls = bot['youtube_urls']
        #get new response from ai
        chat = process(history, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, youtube_urls, session)
        #store conversation (log the api_key)
        store_conversation(chat, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, youtube_urls, session, api_key)
        #return the chat and the bot_id
        return {"message": "Success", "chat": chat, "bot_id": bot_id}
    else:
        return {"message": "Invalid API Key"}


request_OPB = """
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"history":[["hi",""]],"api_key":"xyz"}' \
  http://35.232.62.221/bot
"""

request_youtube = """
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"history":[["hi",""]],"youtube_urls":["https://www.youtube.com/watch?v=frIvwrdHUrg"], "api_key":"xyz"}' \
  http://35.232.62.221/youtube
"""