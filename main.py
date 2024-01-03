#fastapi implementation of openprobono bot
# from serpapi import GoogleSearch
import uuid

import firebase_admin
import langchain
from fastapi import FastAPI
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
    
prompt_template = """Respond in the same style as the youtuber in the context below.
{context}
Question: {question}
Response:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT} 

#TODO: cache vector db with bot_id
#TODO: do actual chat memory
def process(
    history, 
    user_prompt = "",
    youtube_urls = [],
    session = ""):

    text = ""
    for url in youtube_urls:
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

class YoutubeRequest(BaseModel):
    history: list
    user_prompt: str = ""
    youtube_urls: list
    session: str = None
    bot_id: str = None
    api_key: str = None

api = FastAPI()

@api.get("/")
def read_root():
    return {"message": "It's OpenProBono !!"}

@api.post("/youtube")
def bot(request: YoutubeRequest):
    request_dict = request.dict()
    history = request_dict['history']
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
        chat = process(history, user_prompt, youtube_urls, session)
        #store conversation (log the api_key)
        store_conversation(chat, user_prompt, youtube_urls, session, api_key)
        #return the chat and the bot_id
        return {"message": "Success", "chat": chat, "bot_id": bot_id}
    else:
        return {"message": "Invalid API Key"}

def gradio_test_process(prompt, youtube_urls):
    youtube_urls = [url.strip() for url in youtube_urls.split(",")]
    history = [[prompt, ""]]
    chat = process(history, "", youtube_urls, "")
    return chat[-1][1]

prompt = gr.Textbox(value="hi", label="Prompt")
youtube_urls = gr.Textbox(value="https://www.youtube.com/watch?v=wnRTpHKTJgM, https://www.youtube.com/watch?v=QHjuFAbUkg0", label="Youtube URLs")
output = gr.Textbox(label="Output")
app = gr.Interface(
    fn = gradio_test_process,
    inputs = [
        prompt,
        youtube_urls,
    ],
    outputs = [output],
)

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