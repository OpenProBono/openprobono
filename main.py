#fastapi implementation
import os
import re
import uuid
from json import loads
from typing import Annotated

import firebase_admin
from fastapi import Body, FastAPI, UploadFile
from firebase_admin import auth, credentials, firestore
from requests import session

from bot import BotRequest, ChatRequest, opb_bot
from milvusdb import session_upload_pdf, session_source_summaries, delete_expr, crawl_and_scrape, session_upload_ocr, US, COLLECTIONS, SESSION_PDF
from models import ChatBySession, FetchSession, InitializeSession
from new_bot import flow

#sdvlp session
#1076cca8-a1fa-415a-b5f8-c11da178d224

#which version of db we are using
version= "vf23_db"
bot_collection = "bots"
conversation_collection = "conversations"

firebase_config = loads(os.environ["Firebase"])
cred = credentials.Certificate(firebase_config)
firebase_admin.initialize_app(cred)
db = firestore.client()

def get_uuid_id():
    return str(uuid.uuid4())
 
#make this a real api key check
def admin_check(api_key):
    return api_key in ["xyz"]

def store_conversation(r: ChatRequest, output):
    if(r.session_id is None or r.session_id == ""):
        r.session_id = get_uuid_id()

    data = r.model_dump()
    data["history"] = len(r.history)
    data["human"] = r.history[-1][0]
    data["bot"] = output

    t = firestore.SERVER_TIMESTAMP
    data["timestamp"] = t

    db.collection(conversation_collection + version).document(r.session_id).collection('conversations').document("msg" + str(len(r.history))).set(data)
    db.collection(conversation_collection + version).document(r.session_id).set({"last_message_timestamp": t}, merge=True)

def set_session_to_bot(session_id: str, bot_id: str):
    db.collection(conversation_collection + version).document(session_id).set({"bot_id": bot_id}, merge=True)

def load_session(r: ChatBySession):
    msgs = db.collection(conversation_collection + version).document(r.session_id).collection('conversations').order_by("timestamp", direction=firestore.Query.ASCENDING).get()
    history = []
    for msg in msgs:
        conversation = msg.to_dict()
        msg_pair = [conversation["human"], conversation["bot"]]
        history.append(msg_pair)
    history.append([r.message, ""])
    metadata =  db.collection(conversation_collection + version).document(r.session_id).get()
    return ChatRequest(history=history, bot_id=metadata.to_dict()["bot_id"], session_id=r.session_id, api_key=r.api_key)

def fetch_session(r: FetchSession):
    msgs = db.collection(conversation_collection + version).document(r.session_id).collection('conversations').order_by("timestamp", direction=firestore.Query.ASCENDING).get()
    history = []
    for msg in msgs:
        conversation = msg.to_dict()
        msg_pair = [conversation["human"], conversation["bot"]]
        history.append(msg_pair)
    return ChatRequest(history=history, bot_id=msgs[0].to_dict()["bot_id"], session_id=r.session_id, api_key=r.api_key)

#TODO: add a way to load_conversation via session id, and use it to pass along session id instead of history for invoke_bot
    
def store_bot(r: BotRequest, bot_id: str):
    data = r.model_dump()
    data['timestamp'] = firestore.SERVER_TIMESTAMP
    db.collection(bot_collection + version).document(bot_id).set(data)

def load_bot(bot_id):
    bot = db.collection(bot_collection + version).document(bot_id).get()
    if(bot.exists):
        return BotRequest(**bot.to_dict())
    else:
        return None
    
#Checks if api key is valid (TODO: change this to a real api key check)
def api_key_check(api_key):
    return api_key == 'xyz' or api_key == 'gradio' or api_key == 'deniz_key'

def process_chat(r: ChatRequest):
    if(api_key_check(r.api_key) == False):
        return {"message": "Invalid API Key"}

    try:
        bot = load_bot(r.bot_id)
        if(bot is None):
            return {"message": "Failure: No bot found with bot id: " + r.bot_id}

        output = opb_bot(r, bot)

        #store conversation (and also log the api_key)
        store_conversation(r, output)

        #return the chat and the bot_id
        return {"message": "Success", "output": output, "bot_id": r.bot_id}
    except Exception as error:
        return {"message": "Failure: Internal Error: " + str(error)}
        
def process_flow(r: ChatRequest):
    if(api_key_check(r.api_key) == False):
        return {"message": "Invalid API Key"}
    try:
        bot = load_bot(r.bot_id)
        if(bot is None):
            return {"message": "Failure: No bot found with bot id: " + r.bot_id}

        output = flow(r, bot)

        #store conversation (and also log the api_key)
        store_conversation(r, output)

        #return the chat and the bot_id
        return {"message": "Success", "output": output, "bot_id": r.bot_id}
    except Exception as error:
        return {"message": "Failure: Internal Error: " + str(error)}
        

# FastAPI 

api = FastAPI()

@api.get("/", tags=["General"])
def read_root():
    return {"message": "API is alive"}

@api.post("/flow", tags=["History Chat (new agent flow)"])
def new_flow(request: ChatRequest):
    return process_flow(request)

@api.post("/invoke_bot", tags=["History Chat"])
def chat(request: Annotated[
        ChatRequest,
        Body(
            openapi_examples={
                "call a bot using history": {
                    "summary": "call a bot using history",
                    "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was used}",
                    "value": {
                        "history": [["hi!", ""]],
                        "bot_id": "ae885648-4fc7-4de6-ba81-67cc58c57d4c",
                        "api_key":"xyz",
                    },
                },
                "call a bot using history 2": {
                    "summary": "call a bot using history 2",
                    "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was used}",
                    "value": {
                        "history": [["hi!", "Hi, how can I assist you today?"], ["I need help with something", ""]],
                        "bot_id": "ae885648-4fc7-4de6-ba81-67cc58c57d4c",
                        "api_key":"xyz",
                    },
                },
                "call opb bot": {
                    "summary": "call opb bot",
                    "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was used}",
                    "value": {
                        "history": [["hi!", ""]],
                        "bot_id": "39e6d5c3-4e3c-4281-93d7-4f7c8db8833b",
                        "api_key":"xyz",
                    },
                },
            },
        )]):
    
    return process_chat(request)

@api.post("/initialize_session_chat", tags=["Init Session"])
def init_session(request: Annotated[
        InitializeSession,
        Body(
            openapi_examples={
                "init session": {
                    "summary": "initialize a session",
                    "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was used, session_id: the session_id which was created",
                    "value": {
                        "message": "hi, I need help",
                        "bot_id": "83f74a4e-0f8f-4142-b4e7-92a20f688a0b",
                        "api_key":"xyz",
                    },
                },
            },
        )]):
    session_id = get_uuid_id()
    set_session_to_bot(session_id, request.bot_id)
    cr = ChatRequest(history=[[request.message, ""]], bot_id=request.bot_id, session_id=session_id, api_key=request.api_key)
    response =  process_chat(cr)
    try:
        return {"message": "Success", "output": response["output"], "bot_id": request.bot_id, "session_id": session_id}
    except:
        return response
    
# @api.post("/initialize_session_site", tags=["Init Session"])
# def init_session_site(request: Annotated[
#         InitializeSessionScrapeSite,
#         Body(
#             openapi_examples={
#                 "init session": {
#                     "summary": "initialize a session",
#                     "description": "Returns: {message: 'Success', urls: every url which was scraped, bot_id: the bot_id which was used, session_id: the session_id which was created",
#                     "value": {
#                         "site": "https://sdvlp.org/",
#                         "bot_id": "83f74a4e-0f8f-4142-b4e7-92a20f688a0b",
#                         "api_key":"xyz",
#                     },
#                 },
#             },
#         )]):
#     session_id = get_uuid_id()
#     set_session_to_bot(session_id, request.bot_id)
#     response = crawl_and_scrape(request.site, session_id)
#     try:
#         return {"message": "Success", "urls": response, "bot_id": request.bot_id, "session_id": session_id}
#     except:
#         return response

@api.post("/chat_session", tags=["Session Chat"])
def chat_session(request: Annotated[
        ChatBySession,
        Body(
            openapi_examples={
                "call a bot using session": {
                    "summary": "call a bot using session",
                    "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was used, session_id: the session_id which was used}",
                    "value": {
                        "message": "hi, I need help",
                        "session_id": "some session id",
                        "api_key":"xyz",
                    },
                },
            },
        )]):
    cr = load_session(request)
    response = process_chat(cr)
    try:
        return {"message": "Success", "output": response["output"], "bot_id": response["bot_id"], "session_id": cr.session_id}
    except:
        return response
    

@api.post("/fetch_session", tags=["Session Chat"])
def get_session(request: Annotated[
        FetchSession,
        Body(
            openapi_examples={
                "fetch chat history via session": {
                    "summary": "fetch chat history via session",
                    "description": "Returns: {message: 'Success', history: list of messages, bot_id: the bot_id which was used, session_id: the session_id which was used}",
                    "value": {
                        "session_id": "some session id",
                        "api_key":"xyz",
                    },
                },
            },
        )]):
    cr = fetch_session(request)
    return {"message": "Success", "history": cr.history, "bot_id": cr.bot_id, "session_id": cr.session_id}

@api.post("/create_bot", tags=["Bot"])
def create_bot(request: Annotated[
        BotRequest,
        Body(
            openapi_examples={
                "create new bot": {
                    "summary": "create new bot",
                    "description": "Returns: {message: 'Success', bot_id: the new bot_id which was created}",
                    "value": {
                        "search_tools": [{
                            "name": "google_search",
                            "txt": "",
                            "prompt": "Tool used to search the web, useful for current events or facts"
                        }, {
                            "name": "wikipedia",
                            "txt": "site:*wikipedia.com",
                            "prompt": "Tool used to search the wikipedia, useful for facts and biographies"
                        }],
                        "vdb_tools": [{
                            "name": "query",
                            "collection_name": US,
                            "k": 4
                        }],
                        "api_key":"xyz",
                    },
                },
                "create opb bot": {
                    "summary": "create opb bot",
                    "description": "Returns: {message: 'Success', bot_id: the new bot_id which was created}",
                    "value": {
                        "search_tools": [{
                            "name": "government-search",
                            "txt": "site:*.gov | site:*.edu | site:*scholar.google.com",
                            "prompt": "Useful for when you need to answer questions or find resources about government and laws. Always cite your sources."
                        }, {
                            "name": "case-search",
                            "txt": "site:*case.law | site:*.gov | site:*.edu | site:*courtlistener.com | site:*scholar.google.com",
                            "prompt": "Use for finding case law. Always cite your sources."
                        }],
                        "vdb_tools": [{
                            "name": "query",
                            "collection_name": US,
                            "k": 4
                        }],
                        "api_key":"xyz"
                    },
                },
                "full descriptions of every parameter": {
                    "summary": "Description and Tips",
                    "description": "full descriptions",
                    "value": {
                        "user_prompt": "prompt to use for the bot, this is appended to the regular prompt",
                        "message_prompt": "prompt to use for the bot, this is appended each message",
                        "search_tools": [{
                            "name": "name for tool",
                            "txt": "where to put google search syntax to filter or whitelist results",
                            "prompt": "description for agent to know when to use the tool"
                        }],
                        "vdb_tools": [{
                            "name": "name for tool, must be one of: qa, query",
                            "collection_name": f"name of database to query, must be one of: {', '.join(list(COLLECTIONS)[:-1])} {list(COLLECTIONS)[-1]}",
                            "k": "the number of text chunks to return when querying the database"
                        }],
                        "beta": "whether to use beta features or not, if they are available",
                        "search_tool_method": "which search tool to use, between google_search and serpapi: default is serpapi",
                        "api_key": "api key necessary for auth",
                    },
                },
            },
        )]):
    
    if(api_key_check(request.api_key) == False):
        return {"message": "Invalid API Key"}
    
    bot_id = get_uuid_id()
    store_bot(request, bot_id)

    return {"message": "Success", "bot_id": bot_id}

@api.post("/upload_file", tags=["User Upload"])
def vectordb_upload(file: UploadFile, session_id: str, summary: str = None):
        return session_upload_pdf(file, session_id, summary if summary else file.filename)

@api.post("/upload_files", tags=["User Upload"])
def vectordb_upload(files: list[UploadFile], session_id: str, summaries: list[str] = None):
    if not summaries:
        summaries = [file.filename for file in files]
    elif len(files) != len(summaries):
        return {"message": f"Failure: did not find equal numbers of files and summaries, instead found {len(files)} files and {len(summaries)} summaries."}
    failures = []
    for i, file in enumerate(files):
        result = session_upload_pdf(file, session_id, summaries[i])
        if result["message"].startswith("Failure"):
            failures.append(f"Upload #{i + 1} of {len(files)} failed. Internal message: {result['message']}")
            
    if len(failures) == 0:
        return {"message": f"Success: {len(files)} files uploaded"}
    return {"message": f"Warning: {len(failures)} failures occurred: {failures}"}

@api.post("/upload_file_ocr", tags=["User Upload"])
def vectordb_upload_ocr(file: UploadFile, session_id: str, summary: str = None):
    return session_upload_ocr(file, session_id, summary if summary else file.filename)

@api.post("/delete_file", tags=["Vector Database"])
def delete_file(filename: str, session_id: str):
    return delete_expr(SESSION_PDF, f"source=='{filename}' and session_id=='{session_id}'")

@api.post("/delete_files", tags=["Vector Database"])
def delete_files(filenames: list[str], session_id: str):
    for filename in filenames:
        delete_expr(SESSION_PDF, f"source=='{filename}' and session_id=='{session_id}'")
    return {"message": f"Success: deleted {len(filenames)} files"}

@api.post("/get_session_files", tags=["Vector Database"])
def get_session_files(session_id: str):
    source_summaries = session_source_summaries(session_id)
    files = list(source_summaries.keys())
    return {"message": f"Success: found {len(files)} files", "result": files}

@api.post("/delete_session_files", tags=["Vector Database"])
def delete_session_files(session_id: str):
    return delete_expr(SESSION_PDF, f"session_id=='{session_id}'")

@api.post("/upload_site", tags=["Admin Upload"])
def vectordb_upload_site(site: str, collection_name: str, description: str, api_key: str):
    if(not admin_check(api_key)):
        return {"message": "Failure: API key invalid"}
    return crawl_and_scrape(site, collection_name, description)

# TODO: implement this (reuse code from user/session uploads)
# @api.post("/upload_files", tags=["Admin Upload"]) 
# def vectordb_upload(files: list[UploadFile], collection_name: str, api_key: str, summaries: list[str] = None):
#     if(not admin_check(api_key)):
#         return {"message": "Failure: API key invalid"}
#     if not summaries:
#         summaries = [file.filename for file in files]
#     elif len(files) != len(summaries):
#         return {"message": f"Failure: did not find equal numbers of files and summaries, instead found {len(files)} files and {len(summaries)} summaries."}
#     failures = []
#     for i, file in enumerate(files):
#         result = collection_upload_pdf(file, collection_name, summaries[i])
#         if result["message"].startswith("Failure"):
#             failures.append(f"Upload #{i + 1} of {len(files)} failed. Internal message: {result['message']}")
            
#     if len(failures) == 0:
#         return {"message": f"Success: {len(files)} file{'s' if len(files) > 1 else ''} uploaded"}
#     return {"message": f"Warning: {len(failures)} failure{'s' if len(failures) > 1 else ''} occurred: {failures}"}