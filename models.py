from pydantic import BaseModel
from enum import Enum
from typing import List
import uuid

def get_uuid_id():
    return str(uuid.uuid4())

class SearchMethodEnum(str, Enum):
    serpapi = 'serpapi'
    dynamic_serpapi = 'dynamic_serpapi'
    google = 'google'
    courtlistener = 'courtlistener'

class SearchTool(BaseModel):
    method: SearchMethodEnum = SearchMethodEnum.serpapi
    name: str
    prompt: str
    prefix: str = ""

class VDBMethodEnum(str, Enum):
    qa = 'qa'
    query = 'query'

class VDBTool(BaseModel):
    method: VDBMethodEnum = VDBMethodEnum.qa
    name: str
    collection_name: str
    k: int
    prompt: str = ""
    prefix: str = ""

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

class EngineEnum(str, Enum):
    langchain = 'langchain'
    openai = 'openai'  

class BotRequest(BaseModel):
    user_prompt: str = ""
    message_prompt: str = ""
    model: str = "gpt-3.5-turbo-0125"
    search_tools: List[SearchTool] = []
    vdb_tools: List[VDBTool] = []
    engine: EngineEnum = EngineEnum.langchain
    api_key: str