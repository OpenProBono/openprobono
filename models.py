from pydantic import BaseModel
import uuid

def get_uuid_id():
    return str(uuid.uuid4())

class ChatRequest(BaseModel):
    history: list
    bot_id: str
    session_id: str = None
    api_key: str

class ChatBySession(BaseModel):
    message: str
    session_id: str
    api_key: str

class InitializeSession(BaseModel):
    message: str
    bot_id: str
    api_key: str

class InitializeSessionScrapeSite(BaseModel):
    site: str
    bot_id: str
    api_key: str

class FetchSession(BaseModel):
    session_id: str
    api_key: str

class BotRequest(BaseModel):
    user_prompt: str = ""
    message_prompt: str = ""
    search_tools: list = []
    vdb_tools: list = []
    youtube_urls: list = []
    beta: bool = False
    search_tool_method: str = "serpapi"
    engine: str = "langchain"
    api_key: str