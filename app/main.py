"""Written by Arman Aydemir. This file contains the main API code for the backend."""
from __future__ import annotations

from typing import Annotated

from fastapi import Body, FastAPI

from app.bot import anthropic_bot, openai_bot
from app.db import (
    api_key_check,
    fetch_session,
    load_bot,
    load_session,
    set_session_to_bot,
    store_bot,
    store_conversation,
)
from app.models import (
    BotRequest,
    ChatBySession,
    ChatRequest,
    EngineEnum,
    FetchSession,
    InitializeSession,
    get_uuid_id,
)


# this is to ensure tracing with langfuse
# @asynccontextmanager
# async def lifespan(app: FastAPI):  # noqa: ARG001, ANN201, D103
#     # Operation on startup

#     yield  # wait until shutdown

#     # Flush all events to be sent to Langfuse on shutdown and
#     # terminate all Threads gracefully. This operation is blocking.
#     langfuse.flush()


def process_chat(r: ChatRequest) -> dict:
    # try:
    bot = load_bot(r.bot_id)
    if bot is None:
        return {"message": "Failure: No bot found with bot id: " + r.bot_id}

    match bot.chat_model.engine:
        case EngineEnum.openai:
            output = openai_bot(r, bot)
        case EngineEnum.anthropic:
            output = anthropic_bot(r, bot)
        case _:
            return {"message": f"Failure: invalid bot engine {bot.chat_model.engine}"}

    # store conversation (and also log the api_key)
    store_conversation(r, output)

    # except Exception as error:
    #     error.
    #     return {"message": "Failure: Internal Error: " + str(error)}
    # return the chat and the bot_id
    return {"message": "Success", "output": output, "bot_id": r.bot_id}


api = FastAPI()


@api.get("/", tags=["General"])
def read_root() -> dict:
    """Just a simple message to check if the API is alive."""
    return {"message": "API is alive"}


@api.post("/invoke_bot", tags=["History Chat"])
def chat(
        request: Annotated[
            ChatRequest,
            Body(
                openapi_examples={
                    "call a bot using history": {
                        "summary": "call a bot using history",
                        "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was "  # noqa: E501
                                       "used}",
                        "value": {
                            "history": [{"role": "user", "content": "hi"}],
                            "bot_id": "some bot id",
                            "api_key": "xyz",
                        },
                    },
                },
            ),
        ]) -> dict:
    """Call a bot with history (only for backwards compat, could be deprecated)."""
    if not api_key_check(request.api_key):
        return {"message": "Invalid API Key"}
    return process_chat(request)


@api.post("/initialize_session_chat", tags=["Init Session"])
def init_session(
        request: Annotated[
            InitializeSession,
            Body(
                openapi_examples={
                    "init session": {
                        "summary": "initialize a session",
                        "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was "  # noqa: E501
                                       "used, session_id: the session_id which was created",  # noqa: E501
                        "value": {
                            "message": "hi, I need help",
                            "bot_id": "some bot id",
                            "api_key": "xyz",
                        },
                    },
                },
            ),
        ]) -> dict:
    """Initialize a new session with a message."""
    if not api_key_check(request.api_key):
        return {"message": "Invalid API Key"}
    
    session_id = get_uuid_id()
    set_session_to_bot(session_id, request.bot_id)
    cr = ChatRequest(
        history=[{"role": "user", "content": request.message}],
        bot_id=request.bot_id,
        session_id=session_id,
        api_key=request.api_key,
    )
    response = process_chat(cr)
    try:
        return {
            "message": "Success",
            "output": response["output"],
            "bot_id": request.bot_id,
            "session_id": session_id,
        }
    except:
        return response


@api.post("/chat_session", tags=["Session Chat"])
def chat_session(
        request: Annotated[
            ChatBySession,
            Body(
                openapi_examples={
                    "call a bot using session": {
                        "summary": "call a bot using session",
                        "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was "  # noqa: E501
                                       "used, session_id: the session_id which was used}",  # noqa: E501
                        "value": {
                            "message": "hi, I need help",
                            "session_id": "some session id",
                            "api_key": "xyz",
                        },
                    },
                },
            ),
        ]) -> dict:
    """Continue a chat session with a message."""
    if not api_key_check(request.api_key):
        return {"message": "Invalid API Key"}
    
    cr = load_session(request)
    response = process_chat(cr)
    try:
        return {
            "message": "Success",
            "output": response["output"],
            "bot_id": response["bot_id"],
            "session_id": cr.session_id,
        }
    except:
        return response


@api.post("/fetch_session", tags=["Session Chat"])
def get_session(
        request: Annotated[
            FetchSession,
            Body(
                openapi_examples={
                    "fetch chat history via session": {
                        "summary": "fetch chat history via session",
                        "description": "Returns: {message: 'Success', history: list of messages, bot_id: the bot_id "  # noqa: E501
                                       "which was used, session_id: the session_id which was used}",  # noqa: E501
                        "value": {
                            "session_id": "some session id",
                            "api_key": "xyz",
                        },
                    },
                },
            ),
        ]) -> dict:
    """Fetch the chat history and details of a session."""
    if not api_key_check(request.api_key):
        return {"message": "Invalid API Key"}
    
    cr = fetch_session(request)
    return {
        "message": "Success",
        "history": cr.history,
        "bot_id": cr.bot_id,
        "session_id": cr.session_id,
    }


@api.post("/create_bot", tags=["Bot"])
def create_bot(
        request: Annotated[
            BotRequest,
            Body(
                openapi_examples={
                    "create bot": {
                        "summary": "create opb bot",
                        "description": "Returns: {message: 'Success', bot_id: the new bot_id which was created}",  # noqa: E501
                        "value": {
                            "search_tools": [
                                {
                                    "name": "government-search",
                                    "method": "serpapi",
                                    "prefix": "site:*.gov | site:*.edu | site:*scholar.google.com",  # noqa: E501
                                    "prompt": "Use to answer questions or find resources about "  # noqa: E501
                                              "government and laws.",
                                },
                                {
                                    "name": "case-search",
                                    "method": "courtlistener",
                                    "prompt": "Use to find case law.",
                                },
                            ],
                            "vdb_tools": [
                                {
                                    "collection_name": "USCode",
                                    "k": 4,
                                    "prompt": "Use to find information about federal laws and regulations.",  # noqa: E501
                                },
                            ],
                            "chat_model": {
                                "engine": "langchain",
                                "model": "gpt-3.5-turbo-0125",
                            },
                            "api_key": "xyz",
                        },
                    },
                    "full descriptions of every parameter": {
                        "summary": "Description and Tips",
                        "description": "full descriptions",
                        "value": {
                            "system_prompt": "prompt to use for the bot, replaces the default prompt",  # noqa: E501
                            "message_prompt": "prompt to use for the bot, this is appended for each message, default is none",  # noqa: E501
                            "search_tools": [
                                {
                                    "name": "name for tool",
                                    "method": "which search method to use, must be one of: serpapi, dynamic_serpapi, "  # noqa: E501
                                              "google, courtlistener",
                                    "prefix": "where to put google search syntax to filter or whitelist results, "  # noqa: E501
                                              "but is also just generally a prefix to add to query passed to tool by "  # noqa: E501
                                              "llm",
                                    "prompt": "description for agent to know when to use the tool",  # noqa: E501
                                },
                            ],
                            "vdb_tools": [
                                {
                                    "collection_name": "name of database to query, must be one of: USCode, NCGeneralStatutes, CAP, courtlistener",  # noqa: E501
                                    "k": "the number of text chunks to return when querying the database",  # noqa: E501
                                    "prompt": "description for agent to know when to use the tool",  # noqa: E501
                                },
                            ],
                            "chat_model": {
                                "engine": "which library to use for model calls, must be one of: langchain, openai, hive, anthropic, huggingface. "  # noqa: E501
                                      "Default is langchain.",
                                "model": "model to be used, openai models work on langchain and openai engines, default is gpt-3.5-turbo-0125",  # noqa: E501
                            },
                            "api_key": "api key necessary for auth",
                        },
                    },
                },
            ),
        ]) -> dict:
    """Create a new bot."""
    if not api_key_check(request.api_key):
        return {"message": "Invalid API Key"}

    bot_id = get_uuid_id()
    store_bot(request, bot_id)

    return {"message": "Success", "bot_id": bot_id}


# @api.post("/upload_file", tags=["User Upload"])
# def upload_file(file: UploadFile, session_id: str, summary: str | None = None) -> dict:
#     """File upload by user.

#     Parameters
#     ----------
#     file : UploadFile
#         file to upload.
#     session_id : str
#         the session to associate the file with.
#     summary: str, optional
#         A summary of the file written by the user, by default None.

#     Returns
#     -------
#     dict
#         Success or failure message.

#     """
#     try:
#         return file_upload(file, session_id, summary)
#     except Exception as error:
#         return {"message": "Failure: Internal Error: " + str(error)}


# @api.post("/upload_files", tags=["User Upload"])
# def upload_files(files: list[UploadFile],
#     session_id: str, summaries: list[str] | None = None) -> dict:
#     """Upload multiple files by user.

#     Parameters
#     ----------
#     files : list[UploadFile]
#         files to upload.
#     session_id : str
#         the session to associate the file with.
#     summaries : list[str] | None, optional
#         summaries given by the user, by default None

#     Returns
#     -------
#     dict
#         Success or failure message.

#     """
#     if not summaries:
#         summaries = [None] * len(files)
#     elif len(files) != len(summaries):
#         return {
#             "message": f"Failure: did not find equal numbers of files and summaries, "
#                 f"instead found {len(files)} files and {len(summaries)} summaries.",
#         }

#     failures = []
#     for i, file in enumerate(files):
#         result = file_upload(file, session_id, summaries[i])
#         if result["message"].startswith("Failure"):
#             failures.append(
#                 f"Upload #{i + 1} of {len(files)} failed. "
#                 f"Internal message: {result['message']}",
#             )

#     if len(failures) == 0:
#         return {"message": f"Success: {len(files)} files uploaded"}
#     return {"message": f"Warning: {len(failures)} failures occurred: {failures}"}


# @api.post("/upload_file_ocr", tags=["User Upload"])
# def vectordb_upload_ocr(file: UploadFile,
#         session_id: str, summary: str | None = None) -> dict:
#     """Upload a file by user and use OCR to extract info."""
#     return session_upload_ocr(file, session_id, summary if summary else None)


# @api.post("/delete_file", tags=["Vector Database"])
# def delete_file(filename: str, session_id: str):
#     """Delete a file from the sessions database.

#     Parameters
#     ----------
#     filename : str
#         filename to delete.
#     session_id : str
#         session to delete the file from.

#     """
#     return delete_expr(
#         SESSION_DATA,
#         f"metadata['source']=='{filename}' and session_id=='{session_id}'",
#     )


# @api.post("/delete_files", tags=["Vector Database"])
# def delete_files(filenames: list[str], session_id: str) -> dict:
#     """Delete multiple files from the database.

#     Parameters
#     ----------
#     filenames : list[str]
#         filenames to delete.
#     session_id : str
#         session to delete the file from

#     Returns
#     -------
#     dict
#         Success message with number of files deleted.

#     """
#     for filename in filenames:
#         delete_file(filename, session_id)
#     return {"message": f"Success: deleted {len(filenames)} files"}


# @api.post("/get_session_files", tags=["Vector Database"])
# def get_session_files(session_id: str) -> dict:
#     """Get names of all files associated with a session.

#     Parameters
#     ----------
#     session_id : str
#         session to get files from.

#     Returns
#     -------
#     dict
#         Success message with list of filenames.

#     """
#     source_summaries = session_source_summaries(session_id)
#     files = list(source_summaries.keys())
#     return {"message": f"Success: found {len(files)} files", "result": files}


# @api.post("/delete_session_files", tags=["Vector Database"])
# def delete_session_files(session_id: str):
#     """Delete all files associated with a session.

#     Parameters
#     ----------
#     session_id : str
#         _description_

#     Returns
#     -------
#     _type_
#         _description_
#     """
#     return delete_expr(SESSION_DATA, f"session_id=='{session_id}'")


# @api.post("/upload_site", tags=["Admin Upload"])
# def vectordb_upload_site(site: str, collection_name: str,
#         description: str, api_key: str):
#     if not admin_check(api_key):
#         return {"message": "Failure: API key invalid"}
#     return crawl_upload_site(collection_name, description, site)
