from pydantic import BaseModel


class ChatRequest(BaseModel):
    history: list
    bot_id: str
    session: str = None
    api_key: str

class BotRequest(BaseModel):
    user_prompt: str = ""
    message_prompt: str = ""
    tools: list = []
    youtube_urls: list = []
    beta: bool = False
    search_tool_method: str = "serpapi"
    api_key: str