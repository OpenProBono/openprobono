"""Written by Arman Aydemir. Used to access and store data in the Firestore database."""
import os
from json import loads

import firebase_admin
from firebase_admin import credentials, firestore
from langfuse import Langfuse

from bot import BotRequest, ChatRequest, opb_bot, openai_bot
from models import ChatBySession, FetchSession, get_uuid_id

langfuse = Langfuse()

# sdvlp session
# 1076cca8-a1fa-415a-b5f8-c11da178d224

# which version of db we are using
VERSION = "vm12_lang"
BOT_COLLECTION = "bots"
conversation_collection = "conversations"

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
    doc = db.collection("api_keys").document(api_key).get()
    return doc.exists #just check if key exists in DB


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
    doc = db.collection("api_keys").document(api_key).get()
    if doc.exists: #if api key exists, check if it is an admin key
        return doc.to_dict()["admin"]

    return False

def store_conversation(r: ChatRequest, output: str) -> bool:
    """Store the conversation in the database.

    Args:
    ----
        r (ChatRequest): Chat request object
        output (str): The output from the bot

    Returns:
    -------
        bool: True if successful, False otherwise

    """
    if r.session_id is None or r.session_id == "":
        r.session_id = get_uuid_id()

    data = r.model_dump()
    data["history"] = len(r.history)
    data["human"] = r.history[-1][0]
    data["bot"] = output

    t = firestore.SERVER_TIMESTAMP
    data["timestamp"] = t

    db.collection(conversation_collection + VERSION).document(r.session_id).collection(
        "conversations").document("msg" + str(len(r.history))).set(data)

    db.collection(conversation_collection + VERSION).document(r.session_id).set(
        {"last_message_timestamp": t}, merge=True)
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
    db.collection(conversation_collection + VERSION).document(session_id).set(
        {"bot_id": bot_id}, merge=True)
    return True


def load_session(r: ChatBySession) -> ChatRequest:
    """Load the associated session to continue the conversation.

    Args:
    ----
        r (ChatBySession): Obj containing the session id and message from user

    Returns:
    -------
        ChatRequest: The chat request object with appended user message

    """
    msgs = (
        db.collection(conversation_collection + VERSION)
        .document(r.session_id)
        .collection("conversations")
        .order_by("timestamp", direction=firestore.Query.ASCENDING)
        .get()
    )
    history = []
    for msg in msgs:
        conversation = msg.to_dict()
        msg_pair = [conversation["human"], conversation["bot"]]
        history.append(msg_pair)
    history.append([r.message, ""])
    metadata = (
        db.collection(conversation_collection + VERSION).document(r.session_id).get()
    )
    return ChatRequest(
        history=history,
        bot_id=metadata.to_dict()["bot_id"],
        session_id=r.session_id,
        api_key=r.api_key,
    )


def fetch_session(r: FetchSession) -> ChatRequest:
    """Load the associated session data from database.

    Args:
    ----
        r (FetchSession): Obj containing the session data

    Returns:
    -------
        ChatRequest: The chat request object with data from firestore

    """
    msgs = (
        db.collection(conversation_collection + VERSION)
        .document(r.session_id)
        .collection("conversations")
        .order_by("timestamp", direction=firestore.Query.ASCENDING)
        .get()
    ) #loading all the messages from database
    history = []
    for msg in msgs: #turning messages from db into a history list
        conversation = msg.to_dict()
        msg_pair = [conversation["human"], conversation["bot"]]
        history.append(msg_pair)
    return ChatRequest( #building the actual ChatRequest object
        history=history,
        bot_id=msgs[0].to_dict()["bot_id"],
        session_id=r.session_id,
        api_key=r.api_key,
    )

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
    db.collection(BOT_COLLECTION + VERSION).document(bot_id).set(data)


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
    bot = db.collection(BOT_COLLECTION + VERSION).document(bot_id).get()
    if bot.exists:
        return BotRequest(**bot.to_dict())

    return None


def process_chat(r: ChatRequest) -> dict:
    try:
        bot = load_bot(r.bot_id)
        if bot is None:
            return {"message": "Failure: No bot found with bot id: " + r.bot_id}

        if bot.engine == "langchain":
            output = opb_bot(r, bot)
        elif bot.engine == "openai":
            output = openai_bot(r, bot)
        else:
            return {"message": f"Failure: invalid bot engine {bot.engine}"}

        # store conversation (and also log the api_key)
        store_conversation(r, output)

        # return the chat and the bot_id
        return {"message": "Success", "output": output, "bot_id": r.bot_id}
    except Exception as error:
        return {"message": "Failure: Internal Error: " + str(error)}
