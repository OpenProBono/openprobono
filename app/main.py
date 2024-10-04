"""Written by Arman Aydemir. This file contains the main API code for the backend."""
from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Security,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from langfuse.decorators import langfuse_context, observe

from app.bot import anthropic_bot, anthropic_bot_stream, openai_bot, openai_bot_stream
from app.db import (
    admin_check,
    api_key_check,
    browse_bots,
    fetch_session,
    get_cached_response,
    load_bot,
    load_session,
    set_session_to_bot,
    store_bot,
    store_conversation,
    store_opinion_feedback,
    store_session_feedback,
)
from app.logger import get_git_hash, setup_logger
from app.milvusdb import (
    SESSION_DATA,
    crawl_upload_site,
    delete_expr,
    fetch_session_data_files,
    file_upload,
    session_upload_ocr,
)
from app.models import (
    BotRequest,
    ChatBySession,
    ChatRequest,
    EngineEnum,
    FetchSession,
    InitializeSession,
    InitializeSessionChat,
    OpinionFeedback,
    OpinionSearchRequest,
    SessionFeedback,
    get_uuid_id,
)
from app.opinion_search import add_opinion_summary, count_opinions, opinion_search

langfuse_context.configure(release=get_git_hash())
logger = setup_logger()

X_API_KEY = APIKeyHeader(name="X-API-Key")


def api_key_auth(x_api_key: str = Depends(X_API_KEY)) -> str:
    """Authenticate API key. Source: https://stackoverflow.com/questions/67942766/fastapi-api-key-as-parameter-secure-enough.

    Parameters
    ----------
    x_api_key : str, optional
        api key string, by default Depends(X_API_KEY)

    Raises
    ------
    HTTPException
        if the API key is invalid

    """
    if not api_key_check(x_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key. Check that you are passing a 'X-API-Key' on your header.",
        )
    return x_api_key



async def process_chat_stream(r: ChatRequest, message: str):
    bot = load_bot(r.bot_id)
    if bot is None:
        yield "An error occurred."
        return
    else:
        if not r.history:
            r.history = [{"role": "system", "content": bot.system_prompt}]
        if message:
            r.history.append({"role": "user", "content": message})

        match bot.chat_model.engine:
            case EngineEnum.openai:
                for chunk in openai_bot_stream(r, bot):
                    yield chunk
                    # Add a small delay to avoid blocking the event loop
                    await asyncio.sleep(0)
            case EngineEnum.anthropic:
                for chunk in anthropic_bot_stream(r, bot):
                    yield chunk
                    # Add a small delay to avoid blocking the event loop
                    await asyncio.sleep(0)
            case _:
                yield Exception("Invalid bot engine for streaming")

@observe(capture_input=False, capture_output=False)
def process_chat(r: ChatRequest, message: str) -> dict:
    # trace bot id
    langfuse_context.update_current_trace(metadata={"bot_id": r.bot_id})
    # check if bot exists
    bot = load_bot(r.bot_id)
    if bot is None:
        error = "Failure: No bot found with bot id: " + r.bot_id
        langfuse_context.update_current_observation(level="ERROR", status_message=error)
        return {"message": error}

    # set conversation history
    system_prompt_msg = {"role": "system", "content": bot.system_prompt}
    if not r.history or system_prompt_msg not in r.history:
        r.history.insert(0, system_prompt_msg)
    if message:
        r.history.append({"role": "user", "content": message})
    else:
        # invoke bot does not pass a new message, so get it from history
        user_messages = [
            msg for msg in r.history
            if "role" in msg and msg["role"] == "user"
        ]
        message = user_messages[-1]["content"] if len(user_messages) > 0 else ""

    # trace input
    langfuse_context.update_current_trace(input=message)

    # see if the response is cached
    # requirements:
    #  - the same bot id
    #  - the same API key
    #  - only 1 user message with the same content as the message here
    cached_response = get_cached_response(r.bot_id, r.api_key, message)
    if cached_response is not None:
        output = cached_response
    else:
        match bot.chat_model.engine:
            case EngineEnum.openai:
                output = openai_bot(r, bot)
            case EngineEnum.anthropic:
                output = anthropic_bot(r, bot)
            case _:
                error = f"Failure: invalid bot engine {bot.chat_model.engine}"
                langfuse_context.update_current_observation(
                    level="ERROR",
                    status_message=error,
                )
                return {"message": error}

    # store conversation (and also log the api_key)
    store_conversation(r, output)
    # trace session id and output
    langfuse_context.update_current_trace(session_id=r.session_id, output=output)
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
                            "bot_id": "custom_4o_dynamic",
                            "api_key": "xyz",
                        },
                    },
                },
            ),
        ],
        api_key: str = Security(api_key_auth)) -> dict:
    """Call a bot with history (only for backwards compat, could be deprecated)."""
    request.api_key = api_key
    return process_chat(request, "")

@api.post("/initialize_session", tags=["Init Session"])
def init_session(
        request: Annotated[
            InitializeSession,
            Body(
                openapi_examples={
                    "init session": {
                        "summary": "initialize a session",
                        "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was "  # noqa: E501
                                       "used, session_id: the session_id which was created",
                        "value": {
                            "bot_id": "some bot id",
                        },
                    },
                },
            ),
        ],
        api_key: str = Security(api_key_auth)) -> dict:
    """Initialize a new session with a message."""
    request.api_key = api_key

    session_id = get_uuid_id()
    set_session_to_bot(session_id, request.bot_id)
    return {
        "message": "Success",
        "bot_id": request.bot_id,
        "session_id": session_id,
    }


@api.post("/initialize_session_chat", tags=["Init Session"])
def init_session_chat(
        request: Annotated[
            InitializeSessionChat,
            Body(
                openapi_examples={
                    "init session": {
                        "summary": "initialize a session",
                        "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was "  # noqa: E501
                                       "used, session_id: the session_id which was created",
                        "value": {
                            "message": "hi, I need help",
                            "bot_id": "some bot id",
                        },
                    },
                },
            ),
        ],
        api_key: str = Security(api_key_auth)) -> dict:
    """Initialize a new session with a message."""
    request.api_key = api_key

    session_id = get_uuid_id()
    set_session_to_bot(session_id, request.bot_id)
    cr = ChatRequest(
        history=[],
        bot_id=request.bot_id,
        session_id=session_id,
        api_key=request.api_key,
    )
    response = process_chat(cr, request.message)
    try:
        return {
            "message": "Success",
            "output": response["output"],
            "bot_id": request.bot_id,
            "session_id": session_id,
        }
    except:
        return response

@api.post("/initialize_session_chat_stream", tags=["Init Session"], response_model=str)
def init_session_chat_stream(
        request: Annotated[
            InitializeSessionChat,
            Body(
                openapi_examples={
                    "init session": {
                        "summary": "initialize a session",
                        "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was "  # noqa: E501
                                       "used, session_id: the session_id which was created",
                        "value": {
                            "message": "hi",
                            "bot_id": "custom_4o_dynamic",
                            "api_key": "xyz",
                        },
                    },
                },
            ),
        ],
        background_tasks: BackgroundTasks,
        api_key: str = Security(api_key_auth)) -> dict:
    """Initialize a new session with a message."""
    request.api_key = api_key

    session_id = get_uuid_id()
    set_session_to_bot(session_id, request.bot_id)
    cr = ChatRequest(
        history=[],
        bot_id=request.bot_id,
        session_id=session_id,
        api_key=request.api_key,
    )

    async def stream_and_store():
        full_response = ""
        yield cr.session_id #return the session id first (only in init)
        async for chunk in process_chat_stream(cr, request.message):
            full_response += chunk
            yield chunk
        background_tasks.add_task(store_conversation, cr, full_response)

    return StreamingResponse(stream_and_store(), media_type="text/event-stream")



@api.post("/chat_session", tags=["Session Chat"])
def chat_session(
        request: Annotated[
            ChatBySession,
            Body(
                openapi_examples={
                    "call a bot using session": {
                        "summary": "call a bot using session",
                        "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was "  # noqa: E501
                                       "used, session_id: the session_id which was used}",
                        "value": {
                            "message": "hi, I need help",
                            "session_id": "some session id",
                        },
                    },
                },
            ),
        ],
        api_key: str = Security(api_key_auth))  -> dict:
    """Continue a chat session with a message."""
    request.api_key = api_key

    cr = load_session(request)
    response = process_chat(cr, request.message)
    try:
        return {
            "message": "Success",
            "output": response["output"],
            "bot_id": response["bot_id"],
            "session_id": cr.session_id,
        }
    except:
        return response

@api.post("/chat_session_stream", tags=["Session Chat"])
def chat_session_stream(
        request: Annotated[
            ChatBySession,
            Body(
                openapi_examples={
                    "call a bot using session": {
                        "summary": "call a bot using session",
                        "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was "  # noqa: E501
                                       "used, session_id: the session_id which was used}",
                        "value": {
                            "message": "hi, I need help",
                            "session_id": "some session id",
                        },
                    },
                },
            ),
        ],
        background_tasks: BackgroundTasks,
        api_key: str = Security(api_key_auth))  -> StreamingResponse:
    """Continue a chat session with a message."""
    request.api_key = api_key

    cr = load_session(request)

    async def stream_and_store():
        full_response = ""
        async for chunk in process_chat_stream(cr, request.message):
            full_response += chunk
            yield chunk
        background_tasks.add_task(store_conversation, cr, full_response)


    return StreamingResponse(stream_and_store(), media_type="text/event-stream")



@api.post("/fetch_session", tags=["Session Chat"])
def get_session(
        request: Annotated[
            FetchSession,
            Body(
                openapi_examples={
                    "fetch chat history via session": {
                        "summary": "fetch chat history via session",
                        "description": "Returns: {message: 'Success', history: list of messages, bot_id: the bot_id "  # noqa: E501
                                       "which was used, session_id: the session_id which was used}",
                        "value": {
                            "session_id": "some session id",
                        },
                    },
                },
            ),
        ],
        api_key: str = Security(api_key_auth))  -> dict:
    """Fetch the chat history and details of a session."""
    request.api_key = api_key

    cr = fetch_session(request)
    return {
        "message": "Success",
        "history": cr.history,
        "bot_id": cr.bot_id,
        "session_id": cr.session_id,
    }

@api.post(path="/session_feedback", tags=["Session Chat"])
def session_feedback(
        request: Annotated[
            SessionFeedback,
            Body(
                openapi_examples={
                    "submit session feedback": {
                        "summary": "submit session feedback",
                        "description": "Returns: {message: 'Success'} or {message: 'Failure'}",
                        "value": {
                            "feedback_text": "some feedback text",
                            "session_id": "some session id",
                        },
                    },
                },
            ),
        ],
        api_key: str = Security(api_key_auth))  -> dict:
    """Submit feedback to a specific session."""
    request.api_key = api_key

    return {"message": "Success" if store_session_feedback(request) else "Failure"}


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
                                    "prefix": "site:*.gov | site:*.edu | site:*scholar.google.com",
                                    "prompt": "Use to answer questions or find resources about "
                                              "government and laws.",
                                },
                            ],
                            "vdb_tools": [
                                {
                                    "name": "session-query",
                                    "collection_name": "SessionData",
                                    "k": 4,
                                    "prompt": "Used to search user uploaded data. Only available if a user has uploaded a file.",
                                },
                            ],
                            "chat_model": {
                                "engine": "openai",
                                "model": "gpt-3.5-turbo-0125",
                            },
                        },
                    },
                    "full descriptions of every parameter": {
                        "summary": "Description and Tips",
                        "description": "full descriptions",
                        "value": {
                            "system_prompt": "prompt to use for the bot, replaces the default prompt",
                            "message_prompt": "prompt to use for the bot, this is appended for each message, default is none",  # noqa: E501
                            "search_tools": [
                                {
                                    "name": "name for tool",
                                    "method": "which search method to use, must be one of: serpapi, dynamic_serpapi, "  # noqa: E501
                                              "google, courtlistener",
                                    "prefix": "where to put google search syntax to filter or whitelist results, "  # noqa: E501
                                              "but is also just generally a prefix to add to query passed to tool by "  # noqa: E501
                                              "llm",
                                    "prompt": "description for agent to know when to use the tool",
                                },
                            ],
                            "vdb_tools": [
                                {
                                    "name": "name for tool",
                                    "collection_name": "name of database to query, must be one of: courtlistener",  # noqa: E501
                                    "k": "the number of text chunks to return when querying the database",
                                    "prompt": "description for agent to know when to use the tool",
                                },
                            ],
                            "chat_model": {
                                "engine": "which library to use for model calls, must be one of: openai, hive, anthropic. "  # noqa: E501
                                      "Default is openai.",
                                "model": "model to be used, default is gpt-3.5-turbo-0125",  # noqa: E501
                            },
                        },
                    },
                },
            ),
        ],
        api_key: str = Security(api_key_auth)) -> dict:
    """Create a new bot."""
    request.api_key = api_key

    bot_id = get_uuid_id()
    store_bot(request, bot_id)

    return {"message": "Success", "bot_id": bot_id}

@api.post("/view_bots", tags=["Bot"])
def view_bots(api_key: str = Security(api_key_auth)) -> dict:
    return {"message": "Success", "data": browse_bots(api_key)}

@api.post("/upload_file", tags=["User Upload"])
def upload_file(file: UploadFile, session_id: str, summary: str | None = None,
                api_key: str = Security(api_key_auth)) -> dict:
    """File upload by user.

    Parameters
    ----------
    file : UploadFile
        file to upload.
    session_id : str
        the session to associate the file with.
    summary: str, optional
        A summary of the file written by the user, by default None.
    api_key: str
        The api key

    Returns
    -------
    dict
        Success or failure message.

    """
    logger.info(f"api_key {api_key} uploading file")
    try:
        return file_upload(file, session_id, summary)
    except Exception as error:
        return {"message": "Failure: Internal Error: " + str(error)}


@api.post("/upload_files", tags=["User Upload"])
def upload_files(files: list[UploadFile],
    session_id: str, summaries: list[str] | None = None,
    api_key: str = Security(api_key_auth)) -> dict:
    """Upload multiple files by user.

    Parameters
    ----------
    files : list[UploadFile]
        files to upload.
    session_id : str
        the session to associate the file with.
    summaries : list[str] | None, optional
        summaries given by the user, by default None

    Returns
    -------
    dict
        Success or failure message.

    """
    logger.info(f"api_key {api_key} uploading files")
    if not summaries:
        summaries = [None] * len(files)
    elif len(files) != len(summaries):
        return {
            "message": f"Failure: did not find equal numbers of files and summaries, "
                f"instead found {len(files)} files and {len(summaries)} summaries.",
        }

    failures = []
    for i, file in enumerate(files):
        result = file_upload(file, session_id, summaries[i])
        if result["message"].startswith("Failure"):
            failures.append(
                f"Upload #{i + 1} of {len(files)} failed. "
                f"Internal message: {result['message']}",
            )

    if len(failures) == 0:
        return {"message": f"Success: {len(files)} files uploaded"}
    return {"message": f"Warning: {len(failures)} failures occurred: {failures}"}


@api.post("/upload_file_ocr", tags=["User Upload"])
def vectordb_upload_ocr(file: UploadFile,
        session_id: str, summary: str | None = None,
        api_key: str = Security(api_key_auth)) -> dict:
    """Upload a file by user and use OCR to extract info."""
    logger.info(f"api_key {api_key} uploading file with OCR")
    return session_upload_ocr(file, session_id, summary if summary else None)


@api.post("/delete_file", tags=["Vector Database"])
def delete_file(filename: str, session_id: str, api_key: str = Security(api_key_auth)) -> dict[str, str]:
    """Delete a file from the sessions database.

    Parameters
    ----------
    filename : str
        filename to delete.
    session_id : str
        session to delete the file from.
    api_key : str
        api key

    """
    logger.info(f"api_key {api_key} deleting file {filename}")
    return delete_expr(
        SESSION_DATA,
        f"metadata['source']=='{filename}' and session_id=='{session_id}'",
    )


@api.post("/delete_files", tags=["Vector Database"])
def delete_files(filenames: list[str], session_id: str, api_key: str = Security(api_key_auth)) -> dict:
    """Delete multiple files from the database.

    Parameters
    ----------
    filenames : list[str]
        filenames to delete.
    session_id : str
        session to delete the file from
    api_key : str
        api key

    Returns
    -------
    dict
        Success message with number of files deleted.

    """
    logger.info(f"api_key {api_key} deleting files")
    for filename in filenames:
        delete_file(filename, session_id)
    return {"message": f"Success: deleted {len(filenames)} files"}


@api.post("/get_session_files", tags=["Vector Database"])
def get_session_files(session_id: str, api_key: str = Security(api_key_auth)) -> dict:
    """Get names of all files associated with a session.

    Parameters
    ----------
    session_id : str
        session to get files from.
    api_key : str
        api key

    Returns
    -------
    dict
        Success message with list of filenames.

    """
    logger.info(f"api_key {api_key} getting session files for session {session_id}")

    files = fetch_session_data_files(session_id=session_id)
    return {"message": f"Success: found {len(files)} files", "result": files}


@api.post("/delete_session_files", tags=["Vector Database"])
def delete_session_files(session_id: str, api_key: str = Security(api_key_auth)):
    """Delete all files associated with a session.

    Parameters
    ----------
    session_id : str
        _description_

    Returns
    -------
    _type_
        _description_
    """
    logger.info(f"api_key {api_key} deleting session files for session {session_id}")
    return delete_expr(SESSION_DATA, f'metadata["session_id"] in ["{session_id}"]')


@api.post("/upload_site", tags=["Admin Upload"])
def vectordb_upload_site(site: str, collection_name: str,
        description: str, api_key: str = Security(api_key_auth)):
    if not admin_check(api_key):
        return {"message": "Failure: API key invalid"}
    return crawl_upload_site(collection_name, description, site)


@api.post("/search_opinions", tags=["Opinion Search"])
def search_opinions(
    req: OpinionSearchRequest,
    api_key: str = Security(api_key_auth),
) -> dict:
    try:
        results = opinion_search(req)
    except Exception as error:
        return {"message": "Failure: Internal Error: " + str(error)}
    else:
        return {"message": "Success", "results": results}


@api.get("/get_opinion_summary", tags=["Opinion Search"])
def get_opinion_summary(
    opinion_id: int,
    api_key: str = Security(api_key_auth),
) -> dict:
    try:
        summary = add_opinion_summary(opinion_id)
    except Exception as error:
        return {"message": "Failure: Internal Error: " + str(error)}
    else:
        return {"message": "Success", "result": summary}

@api.get("/get_opinion_count", tags=["Opinion Search"])
def get_opinion_count(api_key: str = Security(api_key_auth)) -> dict:
    return {"message": "Success", "opinion_count": count_opinions()}


@api.post(path="/opinion_feedback", tags=["Opinion Search"])
def opinion_feedback(
        request: Annotated[
            OpinionFeedback,
            Body(
                openapi_examples={
                    "submit opinion feedback": {
                        "summary": "submit opinion feedback",
                        "description": "Returns: {message: 'Success'} or {message: 'Failure'}",
                        "value": {
                            "feedback_text": "some feedback text",
                            "opinion": "some opinion id",
                        },
                    },
                },
            ),
        ],
        api_key: str = Security(api_key_auth))  -> dict:
    """Submit feedback to a specific session."""
    request.api_key = api_key

    return {"message": "Success" if store_opinion_feedback(request) else "Failure"}
