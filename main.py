#fastapi implementation of openprobono bot
# from serpapi import GoogleSearch
import uuid

import firebase_admin
import langchain
from typing import Annotated
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

def get_uuid_id():
    return str(uuid.uuid4())

def store_conversation(conversation, user_prompt, youtube_urls, session, api_key):
    (human, ai) = conversation[-1]
    if(session is None or session == ""):
        session = get_uuid_id()
    data = {"human": human, "ai": ai, 'user_prompt': user_prompt, 'youtube_urls': youtube_urls, 'timestamp':  firestore.SERVER_TIMESTAMP, 'api_key': api_key}
    db.collection("API_youtube_" + "conversations").document(session).collection('conversations').document("msg" + str(len(conversation))).set(data)

def create_bot(bot_id, user_prompt, youtube_urls):
    data = {'user_prompt': user_prompt, 'youtube_urls': youtube_urls, 'timestamp':  firestore.SERVER_TIMESTAMP}
    db.collection("youtube_bots").document(bot_id).set(data)

def load_bot(bot_id):
    bot = db.collection("youtube_bots").document(bot_id).get()
    if(bot.exists):
        return bot.to_dict()
    else:
        return None

#TODO: cache vector db with bot_id
#TODO: do actual chat memory
#TODO: try cutting off intro and outro part of videos
def call_agent(
    history, 
    user_prompt = "",
    youtube_urls = [],
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
    #check for empty query
    if(query.strip() == ""):
        history[-1][1] = ""
    else:
        history[-1][1] = qa_chain.run(query)

    return history

def process(history, user_prompt, youtube_urls, session, bot_id, api_key):
    #if api key is valid (TODO: change this to a real api key)
    if(api_key == 'xyz' or api_key == 'gradio'):
        try:
            #if bot_id is not provided, create a new bot id
            if bot_id is None or bot_id == "":
                bot_id = get_uuid_id()
                create_bot(bot_id, user_prompt, youtube_urls)
            #if bot_id is provided, load the bot
            else:
                bot = load_bot(bot_id)
                #if bot is not found, create a new bot
                if(bot is None):
                    create_bot(bot_id, user_prompt, youtube_urls)
                #else load bot settings
                else:
                    user_prompt = bot['user_prompt']
                    youtube_urls = bot['youtube_urls']
            #get new response from ai
            chat = call_agent(history, user_prompt, youtube_urls, session)
            #store conversation (log the api_key)
            store_conversation(chat, user_prompt, youtube_urls, session, api_key)
            #return the chat and the bot_id
            return {"message": "Success", "chat": chat, "bot_id": bot_id}
        except:
            return {"message": "Failure: Internal Error"}
    else:
        return {"message": "Invalid API Key"}

class YoutubeRequest(BaseModel):
    history: list
    user_prompt: str = ""
    youtube_urls: list = []
    session: str = None
    bot_id: str = None
    api_key: str = None

api = FastAPI()

@api.get("/")
def read_root():
    return {"message": "It's OpenProBono !!"}

tips = """Keys to good response:
- Make sure videos includes only the youtuber talking, because we are grabbing the youtube generated captions, there is no way to differenciate between voices or backgroudn game audio which got captioned
- There maybe mispellinngs / mistakes in the captions which cannot be avoided, espeically with foreign names/words
- Include many / longer videos to get better results
"""

@api.post("/youtube")
def bot(request: Annotated[
        YoutubeRequest,
        Body(
            openapi_examples={
                "create new youtube bot": {
                    "summary": "create new youtube bot",
                    "description": "Returns: {message: 'Success', chat: [[user message, ai reply]], bot_id: the new bot_id which was created}",
                    "value": {
                        "history": [["hi", ""]],
                        "youtube_urls":["https://www.youtube.com/watch?v=wnRTpHKTJgM", "https://www.youtube.com/watch?v=QHjuFAbUkg0"],
                        "api_key":"xyz",
                    },
                },
                "call a previously created bot": {
                    "summary": "call a previously created bot",
                    "description": "Use a bot_id to call a bot that has already been created. \n\n  Returns: {message: 'Success', chat: [[user message, ai reply]], bot_id: the bot id}",
                    "value": {
                        "history": [["hello there", ""]],
                        "bot_id": "e71f312c-f943-4d03-bb4f-c7c14f617625",
                        "api_key":"xyz",
                    },
                },
                "full descriptions of every parameter": {
                    "summary": "full descriptions of every parameter",
                    "description": "This is an description of all the parameters that can be used. \n\n history: a list of messages in the conversation. (currently chat history is not working, ignores everything but last user message) \n\n user_prompt: prompt to use for the bot, will use default if empty. \n\n session: session id, used for analytics/logging conversations, not necessary \n\n youtube_urls: a list of youtube urls used to create a new bot (only used if no bot_id is passed). \n\n bot_id: a bot id used to call previously created bots \n\n api_key: api key necessary for auth",
                    "value": {
                        "history": [["user message 1", "ai replay 1"], ["user message 2", "ai replay 2"], ["user message 3", "ai replay 3"]],
                        "user_prompt": "prompt to use for the bot, will use default if empty",
                        "session": "session id, used for analytics/logging conversations, not necessary",
                        "youtube_urls":["url of youtube video", "url of youtube video", "urls are ignored if bot_id is passed"],
                        "bot_id": "id of bot previously created",
                        "api_key": "api key necessary for auth",
                    },
                },
            },
        )]):
    request_dict = request.dict()
    history = request_dict['history']
    user_prompt = request_dict['user_prompt']
    youtube_urls = request_dict['youtube_urls']
    session = request_dict['session']
    bot_id = request_dict['bot_id']
    api_key = request_dict['api_key']
    return process(history, user_prompt, youtube_urls, session, bot_id, api_key)

def gradio_process(prompt, youtube_urls, bot_id):
    print("gradio_process")
    request = YoutubeRequest(
        history=[[prompt, ""]], 
        youtube_urls=[url.strip() for url in youtube_urls.split(",")],
        bot_id=bot_id,
        api_key="gradio",
    )
    print("request")
    response = bot(request)
    print("response")
    return response['chat'][-1][1], response['bot_id']

with gr.Blocks() as app:
    prompt = gr.Textbox(label="Prompt")
    youtube_urls = gr.Textbox(label="Youtube URLs")
    bot_id = gr.Textbox(label="Bot ID (optional, if included will ignore youtube urls)") #better explanation
    submit = gr.Button("Submit")
    reply = gr.Textbox(label="Output", interactive=False)
    submit.click(gradio_process, inputs=[prompt, youtube_urls, bot_id], outputs=[reply, bot_id])

gr.mount_gradio_app(api, app, path="/test")

request_OPB = """
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"history":[["hi",""]],"api_key":"xyz"}' \
  http://35.232.62.221/bot
"""

request_youtube = """
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"history":[["whats up lawerence?",""]],"youtube_urls":["https://www.youtube.com/watch?v=wnRTpHKTJgM", "https://www.youtube.com/watch?v=QHjuFAbUkg0"], "api_key":"xyz"}' \
  http://35.232.62.221/youtube
"""

request_youtube = """
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"history":[["whats up?",""]],"bot_id":"8e35157b-9717-4f7d-bc34-e3365ea98673", "api_key":"xyz"}' \
  http://35.232.62.221/youtube
"""


#8e35157b-9717-4f7d-bc34-e3365ea98673