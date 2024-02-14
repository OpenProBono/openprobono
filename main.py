#fastapi implementation
import os
import uuid
from json import loads
from typing import Annotated

import firebase_admin
from fastapi import Body, FastAPI
from firebase_admin import credentials, firestore

from bot import BotRequest, ChatRequest, opb_bot

#which version of db we are using
version= "vf13"

bot_collection = "bots"
conversation_collection = "conversations"

firebase_config = loads(os.environ["Firebase"])
cred = credentials.Certificate(firebase_config)
firebase_admin.initialize_app(cred)
db = firestore.client()

def get_uuid_id():
    return str(uuid.uuid4())

def store_conversation(r: ChatRequest, output):
    if(r.session is None or r.session == ""):
        r.session = get_uuid_id()

    data = r.model_dump()
    data["history"] = len(r.history)
    data["human"] = r.history[-1][0]
    data["bot"] = output

    t = firestore.SERVER_TIMESTAMP
    data["timestamp"] = t

    db.collection(conversation_collection + version).document(r.session).collection('conversations').document("msg" + str(len(r.history))).set(data)
    db.collection(conversation_collection + version).document(r.session).set({"last_message_timestamp": t}, merge=True)

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
        

# FastAPI 

api = FastAPI()

@api.get("/", tags=["General"])
def read_root():
    return {"message": "API is alive"}

@api.post("/invoke_bot", tags=["Chat"])
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

@api.post("/create_bot", tags=["Bot"])
def create_bot(request: Annotated[
        BotRequest,
        Body(
            openapi_examples={
                "create new bot": {
                    "summary": "create new bot",
                    "description": "Returns: {message: 'Success', bot_id: the new bot_id which was created}",
                    "value": {
                        "tools": [{
                            "name": "google_search",
                            "txt": "",
                            "prompt": "Tool used to search the web, useful for current events or facts"
                        }, {
                            "name": "wikipedia",
                            "txt": "site:*wikipedia.com",
                            "prompt": "Tool used to search the wikipedia, useful for facts and biographies"
                        }
                        ],
                        "api_key":"xyz",
                    },
                },
                "create opb bot": {
                    "summary": "create opb bot",
                    "description": "Returns: {message: 'Success', bot_id: the new bot_id which was created}",
                    "value": {
                        "tools": [{
                            "name": "government-search",
                            "txt": "site:*.gov | site:*.edu | site:*scholar.google.com",
                            "prompt": "Useful for when you need to answer questions or find resources about government and laws. Always cite your sources."
                        }, {
                            "name": "case-search",
                            "txt": "site:*case.law | site:*.gov | site:*.edu | site:*courtlistener.com | site:*scholar.google.com",
                            "prompt": "Use for finding case law. Always cite your sources."
                        }
                        ],
                        "api_key":"xyz",
                    },
                },
                "full descriptions of every parameter": {
                    "summary": "Description and Tips",
                    "description": "full descriptions",
                    "value": {
                        "user_prompt": "prompt to use for the bot, this is appended to the regular prompt",
                        "message_prompt": "prompt to use for the bot, this is appended each message",
                        "tools": [{
                            "name": "name for tool, doesn't matter really i think, currently all tools are google_search_tools",
                            "txt": "where to put google search syntax to filter or whitelist results",
                            "prompt": "description for agent to know when to use the tool"
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