"""Written by Arman Aydemir. Used to access and store data in the Firestore database."""
from __future__ import annotations

import os
from json import loads
from typing import List, Optional

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_document import DocumentSnapshot
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.query_results import QueryResultsList
from langfuse.decorators import observe

from app.logger import setup_logger
from app.models import (
    BotRequest,
    ChatRequest,
    EncoderParams,
    FetchSession,
    MilvusMetadataEnum,
    OpinionFeedback,
    SessionFeedback,
    User,
    get_uuid_id,
)

# Set up logger
logger = setup_logger()

# which version of db we are using
DB_VERSION = "_vf16"
BOT_COLLECTION = "bots"
MILVUS_COLLECTION = "milvus"
MILVUS_SOURCES = "sources"
MILVUS_CHUNKS = "chunks"
CONVERSATION_COLLECTION = "conversations"

firebase_config = loads(os.environ["Firebase"])
cred = credentials.Certificate(firebase_config)
firebase_admin.initialize_app(cred)
db = firestore.client()

def api_key_check(api_key: str) -> bool:
    """Check if api key is valid.

    Parameters
    ----------
    api_key : str
        the api key

    Returns
    -------
    bool
        true if valid

    """
    return db.collection("api_keys").document(api_key).get().exists


def admin_check(api_key: str) -> bool:
    """Check if api key is valid admin key.

    Parameters
    ----------
    api_key : str
        the api key

    Returns
    -------
    bool
        true if valid admin key

    """
    result = db.collection("api_keys").document(api_key).get()
    if result.exists:
        return result.to_dict()["admin"]
    return False

@observe(capture_input=False, capture_output=False)
def store_conversation_history(r: ChatRequest) -> bool:
    """Store the conversation history in the database.

    Args:
    ----
        r (ChatRequest): Chat request object containing history

    Returns:
    -------
        bool: True if successful, False otherwise

    """
    if r.session_id is None or r.session_id == "":
        r.session_id = get_uuid_id()

    data = r.model_dump()
    data["num_msg"] = len(r.history)
    data["timestamp"] = firestore.SERVER_TIMESTAMP

    session = db.collection(CONVERSATION_COLLECTION + DB_VERSION).document(r.session_id)
    session.set(data, merge=True)

    return True

@observe()
def store_session_feedback(r: SessionFeedback) -> bool:
    """Store session feedback from consumers. Overwrites the current feedback if there is.

    Parameters
    ----------
    r
        SessionFeedback obj containing session_id and feedback_text

    Returns
    -------
    bool
       True if successful, False otherwise

    """
    data = r.model_dump()
    data.pop("api_key", None)
    data.pop("session_id", None)
    if not data["categories"]:
        del data["categories"]
    session = db.collection(CONVERSATION_COLLECTION + DB_VERSION).document(r.session_id)
    if not session.get().exists:
        return False
    session.set(
        {"feedback": firestore.ArrayUnion([data])},
        merge=True,
    )
    return True

@observe()
def store_opinion_feedback(r: OpinionFeedback) -> bool:
    """Store opinion feedback. Adds to list of feedback.

    Parameters
    ----------
    r
        OpinionFeedback obj containing opinion_id and feedback_text

    Returns
    -------
    bool
       True if successful, False otherwise

    """
    # TODO: generalize to store_vdb_source_feedback()
    collection_name = "test_firebase"
    milvus = db.collection(MILVUS_COLLECTION)
    milvus_coll = milvus.document(collection_name)
    coll_sources = milvus_coll.collection(MILVUS_SOURCES)
    source = coll_sources.document(str(r.opinion_id))
    if not source.get().exists:
        return False
    source.set(
        {"feedback_list": firestore.ArrayUnion([r.feedback_text])},
        merge=True,
    )
    return True


def set_session_to_bot(session_id: str, bot_id: str) -> bool:
    """Set the session to use the bot.

    Args:
    ----
        session_id (str): the session uuid
        bot_id (str): the bot uuid

    Returns:
    -------
        bool: True if successful, False otherwise

    """
    db.collection(CONVERSATION_COLLECTION + DB_VERSION).document(session_id).set(
        {"bot_id": bot_id}, merge=True)
    return True


def fetch_session(r: FetchSession) -> ChatRequest:
    """Load the associated session data from database.

    Args:
    ----
        r (FetchSession): Obj containing the session data

    Returns:
    -------
        ChatRequest: The chat request object with data from firestore

    """
    session_data = (
        db.collection(CONVERSATION_COLLECTION + DB_VERSION)
        .document(r.session_id).get()
    ) #loading session data from db

    session_data = session_data.to_dict()

    return ChatRequest( #building the actual ChatRequest object
        history=session_data.get("history", []),
        bot_id=session_data["bot_id"],
        session_id=r.session_id,
        user=r.user,
        timestamp=str(session_data.get("timestamp", "")),
        title=session_data.get("title", ""),
        file_count=session_data.get("file_count", 0),
    )

def fetch_sessions_by(bot_id: Optional[str], firebase_uid: Optional[str], user: User) -> List[dict]:
    """
    Fetch sessions from Firebase that match the given criteria.
    
    Parameters
    ----------
    bot_id : Optional[str]
        The bot ID to filter sessions by. If None, returns sessions for the user regardless of bot.
    firebase_uid : Optional[str]
        The Firebase UID of the user whose sessions are being fetched.
    user : User
        The authenticated user making the request.
    
    Returns
    -------
    List[dict]
        A list of session dicts that match the criteria.
    """
    if(bot_id is None and firebase_uid is None):
        return []
    
    sessions_ref = db.collection(CONVERSATION_COLLECTION + DB_VERSION)
    
    # Start with base query
    query = sessions_ref
    
    if bot_id:
        # Get the bot to check ownership
        
        bot = load_bot(bot_id)
        if not bot or bot.user.firebase_uid != user.firebase_uid:
            # If not bot owner, only return sessions for this user and bot
            query = query.where("bot_id", "==", bot_id).where(filter=FieldFilter("user.firebase_uid", "==", user.firebase_uid))

        else:
            # Bot owner can see all sessions for their bot
            query = query.where("bot_id", "==", bot_id)
    else:
        # No bot specified, only return user's sessions
        query = query.where(filter=FieldFilter("user.firebase_uid", "==", user.firebase_uid))

    docs = query.get()
    sessions = [doc.to_dict() for doc in docs]
    return sessions

def store_bot(r: BotRequest, bot_id: str) -> bool:
    """Store the bot in the database.

    Parameters
    ----------
    r : BotRequest
        The bot object to store.
    bot_id : str
        The bot id to use.

    Returns
    -------
    bool
        True if successful, False otherwise.

    """
    data = r.model_dump()
    data["timestamp"] = firestore.SERVER_TIMESTAMP
    db.collection(BOT_COLLECTION + DB_VERSION).document(bot_id).set(data)


def load_bot(bot_id: str) -> BotRequest:
    """Load the bot from the database using the bot_id.

    Parameters
    ----------
    bot_id : str
        The bot id to load.

    Returns
    -------
    BotRequest
        The bot object.

    """
    bot = db.collection(BOT_COLLECTION + DB_VERSION).document(bot_id).get()
    if bot.exists:
        return BotRequest(**bot.to_dict())

    return None

def browse_bots(user: User) -> dict:
    """Browse Bots by api key.

    Parameters
    ----------
    user : User
        The user obj.

    Returns
    -------
    dict
        the bots, indexed by bot id

    """
    bot_ref = db.collection(BOT_COLLECTION + DB_VERSION)
    
    # Filter bots by the user's firebase_uid
    logger.debug("Filtering bots for firebase_uid: %s", user.firebase_uid)
    query = bot_ref.where(filter=FieldFilter("user.firebase_uid", "==", user.firebase_uid))
    
    data: QueryResultsList[DocumentSnapshot] = query.get()
    logger.debug("Found %d bots for user %s", len(data), user.firebase_uid)
    data_dict = {}
    for datum in data:
        data_dict[datum.id] = datum.to_dict()
    return data_dict

def load_vdb(collection_name: str) -> dict:
    """Load the parameters for a collection from the database.

    Parameters
    ----------
    collection_name : str
        The name of the collection that uses the parameters.

    Returns
    -------
    dict
        The collection parameters: encoder, metadata_format, fields.

    """
    data = db.collection(MILVUS_COLLECTION).document(collection_name).get()
    if data.exists:
        return data.to_dict()

    return None

def store_vdb(
    collection_name: str,
    encoder: EncoderParams,
    metadata_format: MilvusMetadataEnum,
    fields: list | None = None,
) -> bool:
    """Store the configuration of a Milvus collection in the database.

    Parameters
    ----------
    collection_name : str
        The collection that uses the configuration.
    encoder : EncoderParams
        The EncoderParams object to store.
    metadata_format : MilvusMetadataEnum
        The MilvusMetadataEnum object to store.
    fields : list
        The list of field names to store if metadata_format is field.

    Returns
    -------
    bool
        True if successful, False otherwise.

    """
    data = {
        "encoder": encoder.model_dump(),
        "metadata_format": metadata_format,
        "timestamp": firestore.SERVER_TIMESTAMP,
    }
    if fields is not None:
        data["fields"] = fields
    db.collection(MILVUS_COLLECTION).document(collection_name).set(data)
    return True

def load_vdb_source(
    collection_name: str,
    source_id: int,
) -> firestore.firestore.DocumentReference:
    """Load source data for entities in a Milvus collection from Firebase.

    This can be used to load an existing source or create a new one.

    Parameters
    ----------
    collection_name : str
        The name of the Milvus collection containing the source.
    source_id : int
        The id of the source.

    Returns
    -------
    DocumentReference
        The source as a document reference object.

    """
    milvus = db.collection(MILVUS_COLLECTION)
    milvus_coll = milvus.document(collection_name)
    coll_sources = milvus_coll.collection(MILVUS_SOURCES)
    return coll_sources.document(str(source_id))

def load_vdb_chunk(
    collection_name: str,
    source_id: int,
    chunk_id: int,
) -> firestore.firestore.DocumentReference:
    """Load chunk data for an entity in a Milvus collection in Firebase.

    This can be used to load an existing chunk or create a new one.

    Parameters
    ----------
    collection_name : str
        The name of the Milvus collection containing the chunk.
    source_id : int
        The id of the source from which the chunk originated.
    chunk_id : int
        The id of the chunk.

    Returns
    -------
    DocumentReference
        The chunk as a document reference object.

    """
    milvus = db.collection(MILVUS_COLLECTION)
    milvus_coll = milvus.document(collection_name)
    coll_sources = milvus_coll.collection(MILVUS_SOURCES)
    source = coll_sources.document(str(source_id))
    source_chunks = source.collection(MILVUS_CHUNKS)
    return source_chunks.document(str(chunk_id))

def get_batch() -> firestore.firestore.WriteBatch:
    """Get a batch object for use with Firestore.

    Used to batch write operations. See:
    https://firebase.google.com/docs/firestore/manage-data/transactions

    Returns
    -------
    WriteBatch
        A batch to use with Firestore.

    """
    return db.batch()


@observe()
def get_cached_response(bot_id: str, firebase_uid: str, message: str) -> str | None:
    """If a conversation is already in the database, return the LLM response.

    Parameters
    ----------
    bot_id : str
        The bot id for the cached response
    firebase_uid : str
        The user key for the cached response
    message : str
        The message for the cached response

    Returns
    -------
    str | None
        A cached LLM response, if it exists

    """
    sessions = db.collection(CONVERSATION_COLLECTION + DB_VERSION)
    matched_sessions = (
        sessions
        .where(filter=FieldFilter("firebase_uid", "==", firebase_uid))
        .where(filter=FieldFilter("bot_id", "==", bot_id))
        .get()
    )
    for session in matched_sessions:
        d = session.to_dict()
        if "history" not in d or not d["history"]:
            continue
        user_messages = [
            msg for msg in d["history"]
            if "role" in msg and msg["role"] == "user"
        ]
        if len(user_messages) == 1 and user_messages[0]["content"] == message:
            return d["history"][-1]["content"]
    return None

def delete_bot(bot_id: str, user: User) -> bool:
    """Delete a bot from the database.
    
    Only the user who created the bot can delete it.

    Parameters
    ----------
    bot_id : str
        The ID of the bot to delete
    user : User
        The user requesting the deletion

    Returns
    -------
    bool
        True if deletion was successful, False otherwise
    """
    # First, load the bot to check ownership
    bot = load_bot(bot_id)
    
    # If bot doesn't exist, return False
    if not bot:
        logger.warning("Attempted to delete non-existent bot %s by user %s", 
                      bot_id, user.firebase_uid)
        return False
    
    # Check if the requesting user is the bot creator
    if bot.user.firebase_uid != user.firebase_uid:
        logger.warning("Unauthorized deletion attempt of bot %s by user %s", 
                      bot_id, user.firebase_uid)
        return False
    
    # Delete the bot
    try:
        db.collection(BOT_COLLECTION + DB_VERSION).document(bot_id).delete()
        logger.info("Bot %s successfully deleted by user %s", 
                   bot_id, user.firebase_uid)
        return True
    except Exception as e:
        logger.error("Error deleting bot %s: %s", bot_id, str(e))
        return False
