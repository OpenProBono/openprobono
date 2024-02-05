#fastapi implementation
import os
import uuid
from typing import Annotated

import firebase_admin
from fastapi import Body, FastAPI
from firebase_admin import credentials, firestore

from bot import BotRequest, opb_bot, youtube_bot

#Reread Supervisor's configuration file and restart the service by running these commands:
#  sudo supervisorctl reread
#  sudo supervisorctl update

#  sudo supervisorctl status
#  sudo supervisorctl restart fastapi-app

#where to change nginx config
#  /etc/nginx/sites-available/fastapi-app

#Test that the configuration file is OK and restart NGINX:
#  sudo nginx -t
#  sudo systemctl restart nginx


cred = credentials.Certificate("../../creds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
# opb bot db root path has api prefix
root_path = 'api_'

# manually set api key for now
OPENAI_API_KEY = db.collection("third_party_api_keys").document("openai").get().to_dict()['key']
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def get_uuid_id():
    return str(uuid.uuid4())

def store_conversation(r: BotRequest, output):
    human = r.history[-1][0]
    ai = output
    t = firestore.SERVER_TIMESTAMP
    if(r.session is None or r.session == ""):
        r.session = get_uuid_id()
    data = {"human": human, "ai": ai, 'user_prompt': r.user_prompt, 'message_prompt': r.message_prompt, 'tools': r.tools, 'youtube_urls': r.youtube_urls, 'timestamp': t, 'api_key': r.api_key, "bot_id":r.bot_id}
    db.collection(root_path + "conversations").document(r.session).collection('conversations').document("msg" + str(len(r.history))).set(data)
    db.collection(root_path + "conversations").document(r.session).set({"last_message_timestamp": t}, merge=True)

def create_bot(r: BotRequest):
    data = {'user_prompt': r.user_prompt, 'message_prompt': r.message_prompt, 'youtube_urls': r.youtube_urls, 'tools': r.tools, 'timestamp':  firestore.SERVER_TIMESTAMP}
    db.collection("all_bots").document(r.bot_id).set(data)

def load_bot(bot_id):
    bot = db.collection("all_bots").document(bot_id).get()
    if(bot.exists):
        return bot.to_dict()
    else:
        return None

#checks api key, determines which to call (youtube or opb, eventually will be all together)
def process(r: BotRequest):
    #if api key is valid (TODO: change this to a real api key check)
    if(r.api_key == 'xyz' or r.api_key == 'gradio' or r.api_key == 'deniz_key'):
        try:
            warn = ""
            #if bot_id is not provided, create a new bot id
            if r.bot_id is None or r.bot_id == "":
                r.bot_id = get_uuid_id()
                create_bot(r)
            #if bot_id is provided, load the bot
            else:
                bot = load_bot(r.bot_id)
                #if bot is not found, create a new bot
                if(bot is None):
                    return {"message": "Failure: No bot found with bot id: " + r.bot_id}
                #else load bot settings
                else:
                    #if user_prompt or youtube_urls are provided, warn user that they are being ignored
                    if(r.user_prompt is not None and r.user_prompt != ""):
                        warn +=  " Warning: user_prompt is ignored because bot_id is provided\n"
                    if(r.youtube_urls is not None and r.youtube_urls != []):
                        warn +=  " Warning: youtube_urls is ignored because bot_id is provided\n"
                    if(r.tools is not None and r.tools != []):
                        warn +=  " Warning: tools is ignored because bot_id is provided\n"
                    r.user_prompt = bot['user_prompt'] if "user_prompt" in bot.keys() else ""
                    r.message_prompt = bot['message_prompt'] if "message_prompt" in bot.keys() else ""
                    r.youtube_urls = bot['youtube_urls'] or []
                    r.tools = bot['tools'] or []

            #ONLY use youtube bot if youtube_urls is not empty
            if(r.youtube_urls is not None and r.youtube_urls != []):
                output = youtube_bot(r)
            else:
                output = opb_bot(r)
                
            #store conversation (log the api_key)
            store_conversation(r, output)

            #return the chat and the bot_id
            return {"message": "Success" + warn, "output": output, "bot_id": r.bot_id}
        except Exception as error:
            return {"message": "Failure: Internal Error: " + str(error)}
    else:
        return {"message": "Invalid API Key"}

# FastAPI 

api = FastAPI()

@api.get("/", tags=["General"])
def read_root():
    return {"message": "API is alive"}

helper = """
This is an description of all the parameters that can be used. \n\n history: a list of messages in the conversation. (currently chat history is not working, ignores everything but last user message)
\n\n user_prompt: prompt to use for the bot, will use default if empty. \n\n session: session id, used for analytics/logging conversations, not necessary
\n\n youtube_urls: a list of youtube urls used to create a new bot. \n\n bot_id: a bot id used to call previously created bots \n\n api_key: api key necessary for auth
\n\n
Keys to good response:
- Can use this tool to grab videos from playlist https://www.thetubelab.com/get-all-urls-of-youtube-playlist-channel/
- Make sure videos includes only the youtuber talking, because we are grabbing the youtube generated captions, there is no way to differenciate between voices or background game audio which got captioned
- There maybe mispellings / mistakes in the captions which cannot be avoided, espeically with foreign names/words
- Include many / longer videos to get better results
- BotID saves the parameters for the bot, so you can use the same bot multiple times
    - the two parameters saved are user_prompt and youtube_urls
    - if you pass in a bot_id, it will ignore the both of these parameters
"""

@api.post("/youtube", tags=["Youtube API"])
def youtube_bot_request(request: Annotated[
        BotRequest,
        Body(
            openapi_examples={
                "create new youtube bot": {
                    "summary": "create new youtube bot",
                    "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the new bot_id which was created}",
                    "value": {
                        "history": [["hi", ""]],
                        "youtube_urls":["https://youtu.be/6XEOVaL5a1Q", "https://youtu.be/5Qu-TCVCO3Q"],
                        "api_key":"xyz",
                    },
                },
                "create new youtube bot with custom prompt": {
                    "summary": "create new youtube bot with a custom prompt",
                    "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the new bot_id which was created}",
                    "value": {
                        "history": [["hi", ""]],
                        "user_prompt": "Respond like the youtuber in the context below.",
                        "youtube_urls":["https://youtu.be/6XEOVaL5a1Q", "https://youtu.be/5Qu-TCVCO3Q"],
                        "api_key":"xyz",
                    },
                },
                "full descriptions of every parameter": {
                    "summary": "Description and Tips",
                    "description": helper,
                    "value": {
                        "history": [["user message 1", "ai replay 1"], ["user message 2", "ai replay 2"], ["user message 3", "ai replay 3"]],
                        "user_prompt": "prompt to use for the bot, will use the default of \"Respond in the same style as the youtuber in the context below.\" if empty",
                        "session": "session id, used for analytics/logging conversations, not necessary",
                        "tools": "tools to be used my the agent, not used in current version",
                        "youtube_urls": ["url of youtube video", "url of youtube video", "url of youtube video"],
                        "bot_id": "id of bot previously created, if bot_id is passed then youtube_urls and user_prompt are ignored",
                        "api_key": "api key necessary for auth",
                    },
                },
            },
        )]):
    return process(request)

@api.post("/bot", tags=["General"])
def chat(request: Annotated[
        BotRequest,
        Body(
            openapi_examples={
                "call a bot": {
                    "summary": "call a bot using a bot_id",
                    "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the new bot_id which was used}",
                    "value": {
                        "bot_id": "6e39115b-c771-49af-bb12-4cef3d072b45",
                        "api_key":"xyz",
                    },
                },
            },
        )]):
    
    return process(request)

@api.post("/create_bot", tags=["General"])
def new_bot(request: Annotated[
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
                "full descriptions of every parameter": {
                    "summary": "Description and Tips",
                    "description": "full descriptions",
                    "value": {
                        "history": [["user message 1", "ai replay 1"], ["user message 2", "ai replay 2"], ["user message 3", "ai replay 3"]],
                        "user_prompt": "prompt to use for the bot, this is appended to the regular prompt",
                        "message_prompt": "prompt to use for the bot, this is appended each message",
                        "session": "session id, used for analytics/logging conversations, not necessary",
                        "tools": [{
                            "name": "name for tool, doesn't matter really i think, currently all tools are google_search_tools",
                            "txt": "where to put google search syntax to filter or whitelist results",
                            "prompt": "description for agent to know when to use the tool"
                        }],
                        "api_key": "api key necessary for auth",
                    },
                },
            },
        )]):
    request.bot_id = get_uuid_id()
    create_bot(request)
    return {"message": "Success", "bot_id": request.bot_id}