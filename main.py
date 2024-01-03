#fastapi implementation of openprobono bot
from fastapi import FastAPI
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from pydantic import BaseModel
import uuid

cred = credentials.Certificate("../../creds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def get_uuid_id():
    return str(uuid.uuid4())

def store_conversation(conversation, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session, api_key):
    (human, ai) = conversation[-1]
    data = {"human": human, "ai": ai, "t1name": t1name, 't1txt': t1txt, "t1prompt":t1prompt, "t2name": t2name, "t2txt":t2txt, "t2prompt":t2prompt, 'user_prompt': user_prompt, 'timestamp':  firestore.SERVER_TIMESTAMP, 'api_key': api_key}
    db.collection("API" + "conversations").document(session).collection('conversations').document("msg" + str(len(conversation))).set(data)

def create_bot(bot_id, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt):
    data = {"t1name": t1name, 't1txt': t1txt, "t1prompt":t1prompt, "t2name": t2name, "t2txt":t2txt, "t2prompt":t2prompt, 'user_prompt': user_prompt, 'timestamp':  firestore.SERVER_TIMESTAMP}
    db.collection("bots").document(bot_id).set(data)

def load_bot(bot_id):
    bot = db.collection("bots").document(bot_id).get()
    if(bot.exists):
        return bot.to_dict()
    else:
        return None
    

def openai_bot(history, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session):
    if(history[-1][0].strip() == ""):
        history[-1][1] = "Hi, how can I assist you today?"
        return history 
    else:
        history[-1][1] = "Hi, how can I assist you today?"
        return history 


class BotRequest(BaseModel):
    history: list
    t1name: str = None
    t1txt: str = None
    t1prompt: str = None
    t2name: str = None
    t2txt: str = None
    t2prompt: str = None
    user_prompt: str = None
    session: str = None
    bot_id: str = None
    api_key: str = None

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "It's OpenProBono !!"}

@app.post("/bot")
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
    session = request_dict['session']
    bot_id = request_dict['bot_id']
    api_key = request_dict['api_key']

    #if api key is valid (TODO: change this to a real api key)
    if(api_key == 'xyz'):
        #if bot_id is not provided, create a new bot id
        if bot_id is None or bot_id == "":
            bot_id = get_uuid_id()
            create_bot(bot_id, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt)
        #if bot_id is provided, load the bot
        else:
            bot = load_bot(bot_id)
            #if bot is not found, create a new bot
            if(bot is None):
                create_bot(bot_id, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt)
            else:
                t1name = bot['t1name']
                t1txt = bot['t1txt']
                t1prompt = bot['t1prompt']
                t2name = bot['t2name']
                t2txt = bot['t2txt']
                t2prompt = bot['t2prompt']
                user_prompt = bot['user_prompt']
        #get new response from ai
        chat = openai_bot(history, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session)
        #store conversation (log the api_key)
        store_conversation(chat, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session, api_key)
        #return the chat and the bot_id
        return {"message": "Success", "chat": chat, "bot_id": bot_id}
    else:
        return {"message": "Invalid API Key"}


request = """
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"history":[["hi",""]],"api_key":"xyz"}' \
  http://35.232.62.221/bot
"""