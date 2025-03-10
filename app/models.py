"""Written by Arman Aydemir. This file contains the main models/classes."""
from __future__ import annotations

import uuid
from enum import Enum, unique
from typing import List

from pydantic import BaseModel

from app.prompts import COMBINE_TOOL_OUTPUTS_TEMPLATE


def get_uuid_id() -> str:
    """Get a string UUID4 ID."""
    return str(uuid.uuid4())

def get_int64() -> int:
    """Get a 64-bit integer ID."""
    max_int64 = 2 ** 63 - 1
    int64 = uuid.uuid1().int>>64
    return int64 % max_int64

@unique
class SearchMethodEnum(str, Enum):
    """Enumeration class representing different search methods."""

    serpapi = "serpapi"
    dynamic_serpapi = "dynamic_serpapi"
    google = "google"
    courtlistener = "courtlistener"
    courtroom5 = "courtroom5"
    dynamic_courtroom5 = "dynamic_courtroom5"
    bailii = "bailii"

@unique
class SummaryMethodEnum(str, Enum):
    """Enumeration class representing different ways of summarizing text."""

    stuffing = "stuffing"
    stuff_reduce = "stuff_reduce"
    map_reduce = "map_reduce"
    refine = "refine"
    gemini_full = "gemini_full" #is special case because it uses gemini instead of same llm as chat itself.


@unique
class EngineEnum(str, Enum):
    """Enumeration class representing different engine options."""

    openai = "openai"
    hive = "hive"
    anthropic = "anthropic"
    google = "google"


@unique
class AnthropicModelEnum(str, Enum):
    """Enumeration class representing different Anthropic chat models."""

    claude_3_5_sonnet = "claude-3-5-sonnet-20240620"
    claude_3_opus = "claude-3-opus-20240229"
    claude_3_sonnet = "claude-3-sonnet-20240229"
    claude_3_haiku = "claude-3-haiku-20240307"


@unique
class GoogleModelEnum(str, Enum):
    """Enumeration class representing different Google models."""

    gemini_1_5_flash = "gemini-1.5-flash"
    gemini_1_5_pro = "gemini-1.5-pro"


@unique
class HiveModelEnum(str, Enum):
    """Enumeration class representing different Hive chat models."""

    hive_7b = "hive-7b"
    hive_70b = "hive-70b"


@unique
class OpenAIModelEnum(str, Enum):
    """Enumeration class representing different OpenAI models."""

    gpt_3_5 = "gpt-3.5-turbo-0125"
    gpt_3_5_1106 = "gpt-3.5-turbo-1106"
    gpt_3_5_instruct = "gpt-3.5-turbo-instruct"
    gpt_4 = "gpt-4"
    gpt_4o = "gpt-4o"
    gpt_4o_240806 = "gpt-4o-2024-08-06"
    gpt_4o_mini = "gpt-4o-mini"
    gpt_4_turbo = "gpt-4-turbo-preview"
    gpt_4_1106 = "gpt-4-turbo-1106-preview"
    mod_stable = "text-moderation-stable"
    mod_latest = "text-moderation-latest"
    embed_large = "text-embedding-3-large" # 3072 dimensions, can project down
    embed_small = "text-embedding-3-small" # 1536 dimensions, can project down
    embed_ada_2 = "text-embedding-ada-002" # 1536 dimensions, can't project down


@unique
class VoyageModelEnum(str, Enum):
    """Enumeration class representing different Voyage embedding models."""

    large_2_instruct = "voyage-large-2-instruct" # 16000 context length, 1024 dim
    law = "voyage-law-2" # 16000 context length, 1024 dim


@unique
class MilvusMetadataEnum(str, Enum):
    """Enumeration class representing different ways of storing metadata in Milvus.

    json = a single `metadata` field containing json
    field = explicitly defined metadata fields
    no_field = no metadata field
    """

    json = "json"
    field = "field"
    no_field = "none"


class ChatModelParams(BaseModel):
    """Define a chat model for RAG."""

    engine: EngineEnum = EngineEnum.openai
    model: str = OpenAIModelEnum.gpt_4o_mini.value
    seed: int = 0
    temperature: float = 0.0


class EncoderParams(BaseModel):
    """Define the embedding model for a Collection."""

    name: str = OpenAIModelEnum.embed_small.value
    dim: int = 768


class BotRequest(BaseModel):
    """Model class representing a bot request.

    Attributes
    ----------
        system_prompt (str): The system prompt.
        message_prompt (str): The message prompt.
        model (str): The model to be used.
        search_tools (List[SearchTool]): The list of search tools.
        vdb_tools (List[VDBTool]): The list of VDB tools.
        engine (EngineEnum): The engine to be used.
        api_key (str): The API key.

    """

    system_prompt: str = COMBINE_TOOL_OUTPUTS_TEMPLATE
    message_prompt: str = ""
    search_tools: List[SearchTool] = []
    vdb_tools: List[VDBTool] = []
    chat_model: ChatModelParams = ChatModelParams()
    api_key: str = ""


class OpinionSearchRequest(BaseModel):
    """Model class representing an opinion search request.

    Attributes
    ----------
    query : str
        The query
    k : int, optional
        The number of results to return, by default 10
    jurisdictions : list[str] | None, optional
        The two-letter abbreviations of a state or territory, e.g. 'NJ' or 'TX',
        to filter query results by state. Use 'us-app' for federal appellate,
        'us-dis' for federal district, 'us-sup' for supreme court, 'us-misc'
        for federal special. By default None.
    keyword_query: str | None, optional
        The users keyword query, by default None
    after_date : str | None, optional
        The after date for the query date range in YYYY-MM-DD format, by default None
    before_date : str | None, optional
        The before date for the query date range in YYYY-MM-DD format, by default None

    """

    query: str
    k: int = 10
    jurisdictions: list[str] | None = None
    keyword_query: str | None = None
    after_date: str | None = None
    before_date: str | None = None

class OpinionFeedback(BaseModel):
    """Model class representing an opinion feedback request.

    Attributes
    ----------
        feedback_text (str): The feedback text
        opinion_id (int): The opinion ID.
        api_key (str): The API key.

    """

    feedback_text: str
    opinion_id: int
    api_key: str = ""

class LISTTerm(BaseModel):
    """Model class representing a term in the LIST taxonomy.

    Attributes
    ----------
        code (str): The term code
        title (str): The term title
        definition (str): The term definition
        parent_codes (list[str]): The parent term codes
        taxonomies (list[str]): The taxonomies the term belongs to
        children (list[LISTTerm]): The terms children

    """

    code: str
    title: str
    definition: str
    parent_codes: list[str] = []
    taxonomies: list[str] = []
    children: list[LISTTerm] = []

class LISTTermProb(BaseModel):
    """Model class representing a prediction of a LIST term from an issue classifier.

    Attributes
    ----------
        title (str): The term title
        probability (float): The term probability

    """

    title: str
    probability: float

class SearchTool(BaseModel):
    """Model class representing a search tool.

    Attributes
    ----------
        method (SearchMethodEnum): The search method to be used.
        summary_method (SummaryMethodEnum): The summary method to be used (not always used depending on the type of serach method, currently only dynamic serpapi)
        name (str): The name of the search tool.
        prompt (str): The prompt for the search tool.
        prefix (str): The prefix for the search tool.
        jurisdictions (list[str]): The jurisdictions for the search tool.

    """

    method: SearchMethodEnum = SearchMethodEnum.dynamic_serpapi
    chat_model: ChatModelParams = ChatModelParams(model=OpenAIModelEnum.gpt_4o_mini)
    summary_method: SummaryMethodEnum = SummaryMethodEnum.stuff_reduce
    name: str
    prompt: str
    prefix: str = ""
    bot_id: str = ""
    jurisdictions: list[str] = []

class VDBTool(BaseModel):
    """Model class representing a VDB tool.

    Attributes
    ----------
        name (str): The name of the VDB tool.
        collection_name (str): The collection name for the VDB tool.
        k (int): K is the number of chunks to return for the VDB tool.
        prompt (str): The prompt for the VDB tool.
        session_id (str | None): The session id if querying session data, else None.

    """

    name: str
    collection_name: str
    k: int
    prompt: str = ""
    session_id: str | None = None


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
    api_key: str = ""


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
    api_key: str = ""

class InitializeSession(BaseModel):
    """Model class representing an initialize session request.

    Attributes
    ----------
        bot_id (str): The ID of the bot.
        api_key (str): The API key.

    """

    bot_id: str
    api_key: str = ""

class InitializeSessionChat(BaseModel):
    """Model class representing an initialize session request with a message.

    Attributes
    ----------
        message (str): The initialization message.
        bot_id (str): The ID of the bot.
        api_key (str): The API key.

    """

    message: str
    bot_id: str
    api_key: str = ""


class FetchSession(BaseModel):
    """Model class representing a fetch session request.

    Attributes
    ----------
        session_id (str): The session ID.
        api_key (str): The API key.

    """

    session_id: str
    api_key: str = ""

class SessionFeedback(BaseModel):
    """Model class representing a session feedback request.

    Attributes
    ----------
        feedback_text (str): The feedback text from the consumer
        session_id (str): The session ID.
        api_key (str): The API key.

    """

    feedback_text: str
    session_id: str
    api_key: str = ""