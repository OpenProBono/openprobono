# Create hello world FastAPI app
from fastapi import FastAPI
from pydantic import BaseModel

class BotRequest(BaseModel):
    history: list
    t1name: str | None = None
    t1txt: str | None = None
    t1prompt: str | None = None
    t2name: str | None = None
    t2txt: str | None = None
    t2prompt: str | None = None
    user_prompt: str | None = None
    session: str | None = None
    bot_id: str | None = None
    api_key: str | None = None

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
  --data '{"history":[],"api_key":"xyz"}' \
  http://35.232.62.221/bot
"""