# Create hello world FastAPI app
from fastapi import FastAPI
from pydantic import BaseModel

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
    return request_dict


request = """
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"history":[],"":"xyz"}' \
  http://35.232.62.221/bot
"""