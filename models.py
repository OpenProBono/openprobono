"""Written by Arman Aydemir. This file contains the main models/classes."""
import uuid
from enum import Enum, unique
from typing import List

from pydantic import BaseModel


def get_uuid_id() -> str:
    """Get a string UUID4 ID."""
    return str(uuid.uuid4())


@unique
class SearchMethodEnum(str, Enum):
    """Enumeration class representing different search methods."""

    serpapi = "serpapi"
    dynamic_serpapi = "dynamic_serpapi"
    google = "google"
    courtlistener = "courtlistener"


class SearchTool(BaseModel):
    """Model class representing a search tool.

    Attributes
    ----------
        method (SearchMethodEnum): The search method to be used.
        name (str): The name of the search tool.
        prompt (str): The prompt for the search tool.
        prefix (str): The prefix for the search tool.

    """

    method: SearchMethodEnum = SearchMethodEnum.serpapi
    name: str
    prompt: str
    prefix: str = ""

@unique
class VDBMethodEnum(str, Enum):
    """Enumeration class representing different VDB methods."""

    qa = "qa"
    query = "query"


class VDBTool(BaseModel):
    """Model class representing a VDB tool.

    Attributes
    ----------
        method (VDBMethodEnum): The VDB method to be used.
        name (str): The name of the VDB tool.
        collection_name (str): The collection name for the VDB tool.
        k (int): K is the number of chunks to return for the VDB tool.
        prompt (str): The prompt for the VDB tool.
        prefix (str): The prefix for the VDB tool.

    """

    method: VDBMethodEnum = VDBMethodEnum.qa
    name: str
    collection_name: str
    k: int
    prompt: str = ""
    prefix: str = ""


class ChatRequest(BaseModel):
    """Model class representing a chat request.

    Attributes
    ----------
        history (list): The chat history.
        bot_id (str): The ID of the bot.
        session_id (str): The session ID.
        api_key (str): The API key.

    """

    history: list
    bot_id: str
    session_id: str = None
    api_key: str


class ChatBySession(BaseModel):
    """Model class representing a chat request by session.

    Attributes
    ----------
        message (str): The chat message.
        session_id (str): The session ID.
        api_key (str): The API key.

    """

    message: str
    session_id: str
    api_key: str


class InitializeSession(BaseModel):
    """Model class representing an initialize session request.

    Attributes
    ----------
        message (str): The initialization message.
        bot_id (str): The ID of the bot.
        api_key (str): The API key.

    """

    message: str
    bot_id: str
    api_key: str


class InitializeSessionScrapeSite(BaseModel):
    """Model class representing an initialize session request for scraping a site.

    Attributes
    ----------
        site (str): The site to be scraped.
        bot_id (str): The ID of the bot.
        api_key (str): The API key.

    """

    site: str
    bot_id: str
    api_key: str


class FetchSession(BaseModel):
    """Model class representing a fetch session request.

    Attributes
    ----------
        session_id (str): The session ID.
        api_key (str): The API key.

    """

    session_id: str
    api_key: str


@unique
class EngineEnum(str, Enum):
    """Enumeration class representing different engine options."""

    langchain = "langchain"
    openai = "openai"
    hive = "hive"
    anthropic = "anthropic"

@unique
class AnthropicChatModel(str, Enum):
    """Enumeration class representing different Anthropic chat models."""

    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

@unique
class HiveChatModel(str, Enum):
    """Enumeration class representing different Hive chat models."""

    HIVE_7B = "hive-7b"
    HIVE_70B = "hive-70b"

@unique
class OpenAIChatModel(str, Enum):
    """Enumeration class representing different OpenAI chat models."""

    GPT_3_5 = "gpt-3.5-turbo-0125"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo-preview"

class BotRequest(BaseModel):
    """Model class representing a bot request.

    Attributes
    ----------
        user_prompt (str): The user prompt.
        message_prompt (str): The message prompt.
        model (str): The model to be used.
        search_tools (List[SearchTool]): The list of search tools.
        vdb_tools (List[VDBTool]): The list of VDB tools.
        engine (EngineEnum): The engine to be used.
        api_key (str): The API key.

    """

    user_prompt: str = ""
    message_prompt: str = ""
    model: str = OpenAIChatModel.GPT_3_5
    search_tools: List[SearchTool] = []
    vdb_tools: List[VDBTool] = []
    engine: EngineEnum = EngineEnum.langchain
    api_key: str
