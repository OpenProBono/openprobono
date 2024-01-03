#fastapi implementation of openprobono bot
from fastapi import FastAPI
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from pydantic import BaseModel

cred = credentials.Certificate("../../creds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def store_conversation(conversation, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session, api_key):
    (human, ai) = conversation[-1]
    data = {"human": human, "ai": ai, "t1name": t1name, 't1txt': t1txt, "t1prompt":t1prompt, "t2name": t2name, "t2txt":t2txt, "t2prompt":t2prompt, 'user_prompt': user_prompt, 'timestamp':  firestore.SERVER_TIMESTAMP, 'api_key': api_key}
    db.collection("API" + "conversations").document(session).collection('conversations').document("msg" + str(len(conversation))).set(data)


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
    if(api_key == 'xyz'):
        chat = openai_bot(history, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session)
        store_conversation(chat, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session, api_key)
        return {"message": "Success", "chat": chat}
    else:
        return {"message": "Invalid API Key"}


request = """
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"history":[["hi",""]],"api_key":"xyz"}' \
  http://35.232.62.221/bot
"""