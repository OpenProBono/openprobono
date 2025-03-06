"""Written by Arman Aydemir. This file contains the main API code for the backend."""
from __future__ import annotations

import asyncio
import json
from typing import Annotated

from fastapi import (
    Body,
    Depends,
    FastAPI,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from langfuse.decorators import langfuse_context, observe

from app.bot import (
    anthropic_bot,
    anthropic_bot_stream,
    openai_bot,
    openai_bot_stream,
)
from app.bot_helper import format_session_history, title_chat
from app.db import (
    browse_bots,
    fetch_session,
    fetch_sessions_by,
    get_cached_response,
    load_bot,
    set_session_to_bot,
    store_bot,
    store_conversation_history,
    store_opinion_feedback,
    store_session_feedback,
    delete_bot,
)
from app.logger import get_git_hash, setup_logger
from app.milvusdb import (
    SESSION_DATA,
    count_resources,
    delete_expr,
    file_upload,
    get_expr,
    metadata_fields,
    query_iterator,
    session_upload_ocr,
)
from app.models import (
    BotRequest,
    ChatBySession,
    ChatRequest,
    CollectionManageRequest,
    CollectionSearchRequest,
    EngineEnum,
    FetchSession,
    FetchSessions,
    InitializeSession,
    InitializeSessionChat,
    OpinionFeedback,
    OpinionSearchRequest,
    SessionFeedback,
    User,
    VDBTool,
    get_uuid_id,
)
from app.opinion_search import add_opinion_summary, opinion_search
from app.vdb_tools import format_vdb_tool_results, run_vdb_tool
from app.user_auth import get_current_user

langfuse_context.configure(release=get_git_hash())
logger = setup_logger()

@observe(capture_input=False, capture_output=False)
async def process_chat_stream(r: ChatRequest, message: str):
    # tracing
    langfuse_context.update_current_trace(
        session_id=r.session_id,
        metadata={"bot_id": r.bot_id},
    )
    bot = load_bot(r.bot_id)
    if bot is None:
        error = "Failure: No bot found with bot id: " + r.bot_id
        langfuse_context.update_current_observation(level="ERROR", status_message=error)
        logger.error(error)
        yield {
            "type": "response",
            "content": error,
        }
        return

    if not message:
        # invoke bot does not pass a new message, so get it from history
        user_messages = [
            msg for msg in r.history
            if "role" in msg and msg["role"] == "user"
        ]
        message = user_messages[-1]["content"] if len(user_messages) > 0 else ""

    r.history.append({"role": "user", "content": message})
    # trace input
    langfuse_context.update_current_trace(input=message)

    if not r.title:
        r.title = title_chat(bot, message)

    full_response = ""
    match bot.chat_model.engine:
        case EngineEnum.openai:
            # set conversation history
            system_prompt_msg = {"role": "system", "content": bot.system_prompt}
            if not r.history or system_prompt_msg not in r.history:
                r.history.insert(0, system_prompt_msg)
            for chunk in openai_bot_stream(r, bot):
                if isinstance(chunk, dict) and chunk["type"] == "response":
                    full_response += chunk["content"]
                yield chunk
                # Add a small delay to avoid blocking the event loop
                await asyncio.sleep(0)
        case EngineEnum.anthropic:
            for chunk in anthropic_bot_stream(r, bot):
                if chunk["type"] == "response":
                    full_response += chunk["content"]
                elif chunk["type"] == "tool_result":
                    # any intermediate response is preamble to a tool call, clear it
                    full_response = ""
                yield chunk
                # Add a small delay to avoid blocking the event loop
                await asyncio.sleep(0)
        case _:
            error = "Failure: Invalid bot engine for streaming"
            logger.error(error)
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message=error,
            )
            yield {
                "type": "response",
                "content": error,
            }
    # trace and store
    if full_response:
        langfuse_context.update_current_trace(output=full_response)
        r.history.append({"role": "assistant", "content": full_response})
        store_conversation_history(r)


@observe(capture_input=False, capture_output=False)
def process_chat(r: ChatRequest, message: str) -> dict:
    # tracing
    langfuse_context.update_current_trace(
        session_id=r.session_id,
        metadata={"bot_id": r.bot_id},
    )
    # check if bot exists
    bot = load_bot(r.bot_id)
    if bot is None:
        error = "Failure: No bot found with bot id: " + r.bot_id
        langfuse_context.update_current_observation(level="ERROR", status_message=error)
        return {"message": error}

    if not message:
        # invoke bot does not pass a new message, so get it from history
        user_messages = [
            msg for msg in r.history
            if "role" in msg and msg["role"] == "user"
        ]
        message = user_messages[-1]["content"] if len(user_messages) > 0 else ""
        if not message:
            error = "Failure: message not found in history."
            langfuse_context.update_current_observation(level="ERROR", status_message=error)
            return {"message": error}
    else:
        r.history.append({"role": "user", "content": message})

    # trace input
    langfuse_context.update_current_trace(input=message)

    # see if the response is cached
    # requirements:
    #  - the same bot id
    #  - the same API key
    #  - only 1 user message with the same content as the message here
    cached_response = get_cached_response(r.bot_id, r.user.firebase_uid, message)
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

    # store conversation
    r.history.append({"role": "assistant", "content": output})
    store_conversation_history(r)
    # trace session id and output
    langfuse_context.update_current_trace(session_id=r.session_id, output=output)
    # return the chat and the bot_id
    return {"message": "Success", "output": output, "bot_id": r.bot_id}


api = FastAPI(
    dependencies=[Depends(get_current_user)]
)


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
                        },
                    },
                },
            ),
        ],
        user: User = Depends(get_current_user)) -> dict:
    """Call a bot with history (only for backwards compat, could be deprecated)."""
    request.user = user
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
        ]) -> dict:
    """Initialize a new session with a message."""
    print(request.user)

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
        user: User = Depends(get_current_user)) -> dict:
    """Initialize a new session with a message."""
    request.user = user

    session_id = get_uuid_id()
    set_session_to_bot(session_id, request.bot_id)
    cr = ChatRequest(
        history=[],
        bot_id=request.bot_id,
        session_id=session_id,
        user=request.user,
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
                        },
                    },
                },
            ),
        ],
        user: User = Depends(get_current_user)) -> dict:
    """Initialize a new session with a message."""
    request.user = user

    session_id = get_uuid_id()
    set_session_to_bot(session_id, request.bot_id)
    cr = ChatRequest(
        history=[],
        bot_id=request.bot_id,
        session_id=session_id,
        user=request.user
    )

    async def stream_response():
        yield cr.session_id + "\n" #return the session id first (only in init)
        async for chunk in process_chat_stream(cr, request.message):
            yield json.dumps(chunk) + "\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


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
        user: User = Depends(get_current_user))  -> dict:
    """Continue a chat session with a message."""
    request.user = user

    session_obj = FetchSession(session_id=request.session_id, user=request.user)
    cr = fetch_session(session_obj)
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
        user: User = Depends(get_current_user))  -> StreamingResponse:
    """Continue a chat session with a message."""
    request.user = user

    session_obj = FetchSession(session_id=request.session_id, user=request.user)
    cr = fetch_session(session_obj)

    async def stream_response():
        async for chunk in process_chat_stream(cr, request.message):
            yield json.dumps(chunk) + "\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


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
        user: User = Depends(get_current_user))  -> dict:
    """Fetch the chat history and details of a session."""
    request.user = user

    cr = fetch_session(request)
    return {"message": "Success"} | cr.model_dump()

@api.post("/fetch_sessions", tags=["Session Chat"])
def fetch_sessions(
    request: FetchSessions = Body(
        ...,
        openapi_examples={
            "fetch sessions": {
                "summary": "Fetch sessions by criteria",
                "description": "Returns all sessions associated with a bot, a user, or both. If bot_id is provided, sessions "
                               "will be filtered by that bot; otherwise, all sessions for the authenticated user are returned.",
                "value": {
                    "bot_id": "default_bot"  # optional field
                },
            },
        },
    ),
    user: User = Depends(get_current_user)
) -> dict:
    """
    Fetch all sessions associated with a bot, a user, or both at once.
    
    The endpoint uses the Firebase UID from the authenticated user and optionally filters by bot ID.
    If a bot_id is provided, only the bot creator can see all sessions - other users will only see their own sessions.
    
    Returns
    -------
    dict
        A dictionary with a "message" and the list of matching "sessions".
    """
    sessions = fetch_sessions_by(bot_id=request.bot_id, firebase_uid=request.firebase_uid, user=user)
    return {"message": "Success", "sessions": sessions}


@api.post("/fetch_session_formatted_history", tags=["Session Chat"])
def get_formatted_session_history(
    request: Annotated[
        FetchSession,
        Body(
            openapi_examples={
                "fetch formatted chat history via session": {
                    "summary": "fetch chat history via session",
                    "description": "Returns: {message: 'Success', history: list of formatted messages",
                    "value": {
                        "session_id": "some session id",
                    },
                },
            },
        ),
    ],
    user: User = Depends(get_current_user),
)  -> dict:
    """Fetch the formatted history of a session for front end display."""
    request.user = user
    logger.info("user firebase uid %s getting session %s history", user.firebase_uid, request.session_id)
    cr = fetch_session(request)
    bot = load_bot(cr.bot_id)
    history = format_session_history(cr, bot)
    return {"message": "Success", "history": history}


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
        user: User = Depends(get_current_user))  -> dict:
    """Submit feedback to a specific session."""
    request.user = user

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
                            "name": "Legal Research Assistant",
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
                            "name": "My Custom Bot",
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
        ]) -> dict:
    """Create a new bot."""

    bot_id = get_uuid_id()
    store_bot(request, bot_id)

    return {"message": "Success", "bot_id": bot_id, "name": request.name}


@api.post("/view_bot", tags=["Bot"])
def view_bot(bot_id: str, user: User = Depends(get_current_user)) -> dict:
    logger.info("User %s viewing bot %s", user.firebase_uid, bot_id)
    return {"message": "Success", "data": load_bot(bot_id)}


@api.post("/view_bots", tags=["Bot"])
def view_bots(user: User = Depends(get_current_user)) -> dict:
    logger.info("User %s viewing bots", user.firebase_uid)
    bots = browse_bots(user)
    return {"message": "Success", "data": bots}


@api.post("/upload_file", tags=["User Upload"])
def upload_file(file: UploadFile, session_id: str, summary: str | None = None,
                user: User = Depends(get_current_user)) -> dict:
    """File upload by user.

    Parameters
    ----------
    file : UploadFile
        file to upload.
    session_id : str
        the session to associate the file with.
    summary: str, optional
        A summary of the file written by the user, by default None.
    user: User
        The user obj.

    Returns
    -------
    dict
        Success or failure message.

    """
    logger.info("User %s uploading file", user.firebase_uid)
    cr = fetch_session(FetchSession(session_id=session_id, user=user))
    result = file_upload(file, session_id, summary)
    if result["message"] == "Success":
        cr.file_count += 1
        cr.history.append({"role": "user", "content": f"file:{file.filename}"})
        store_conversation_history(cr)
    return result


@api.post("/upload_files", tags=["User Upload"])
def upload_files(
    files: list[UploadFile],
    session_id: str,
    summaries: list[str] | None = None,
    user: User = Depends(get_current_user),
) -> dict:
    """Upload multiple files by user.

    Parameters
    ----------
    files : list[UploadFile]
        files to upload.
    session_id : str
        the session to associate the file with.
    summaries : list[str] | None, optional
        summaries given by the user, by default None
    user: User
        The user obj.

    Returns
    -------
    dict
        Success or failure message.

    """
    logger.info("User %s uploading files", user.firebase_uid)
    if not summaries:
        summaries = [None] * len(files)
    elif len(files) != len(summaries):
        return {
            "message": f"Failure: did not find equal numbers of files and summaries, "
                f"instead found {len(files)} files and {len(summaries)} summaries.",
        }

    cr = fetch_session(FetchSession(session_id=session_id, user=user))
    results = []
    fail_occurred = False
    success_occurred = False
    for i, file in enumerate(files):
        result = file_upload(file, session_id, summaries[i])
        if result["message"].startswith("Failure"):
            fail_occurred = True
            results.append({
                "id": file.filename,
                "message": result["message"],
            })
        else:
            cr.file_count += 1
            results.append({
                "message": "Success",
                "id": file.filename,
                "insert_count": result["insert_count"],
            })
            cr.history.append({"role": "user", "content": f"file:{file.filename}"})
            success_occurred = True
    if success_occurred:
        store_conversation_history(cr)
    if fail_occurred:
        return {"message": "Failure: not all files were uploaded", "results": results}
    return {"message": "Success", "results": results}


@api.post("/upload_file_ocr", tags=["User Upload"])
def vectordb_upload_ocr(file: UploadFile,
        session_id: str, summary: str | None = None,
        user: User = Depends(get_current_user)) -> dict:
    """Upload a file by user and use OCR to extract info."""
    logger.info("User %s uploading file with OCR", user.firebase_uid)
    cr = fetch_session(FetchSession(session_id=session_id, user=user))
    result = session_upload_ocr(file, session_id, summary if summary else None)
    if result["message"] == "Success":
        cr.file_count += 1
        store_conversation_history(cr)
    return result


@api.post("/delete_file", tags=["Vector Database"])
def delete_file(filename: str, session_id: str, user: User = Depends(get_current_user)) -> dict:
    """Delete a file from the sessions database.

    Parameters
    ----------
    filename : str
        filename to delete.
    session_id : str
        session to delete the file from.
    user: User
        The user obj.

    """
    logger.info("User %s deleting file %s", user.firebase_uid, filename)
    cr = fetch_session(FetchSession(session_id=session_id, user=user))
    expr = (
        f"metadata['filename']=='{filename}' and "
        f"metadata['session_id']=='{session_id}'"
    )
    result = delete_expr(
        SESSION_DATA,
        expr,
        session_id,
    )
    if result["delete_count"] == 0:
        logger.warning("session %s file %s not found", session_id, filename)
    elif result["message"] == "Success":
        cr.file_count -= 1
        store_conversation_history(cr)
    return result


@api.post("/delete_files", tags=["Vector Database"])
def delete_files(filenames: list[str], session_id: str, user: User = Depends(get_current_user)) -> dict:
    """Delete multiple files from the database.

    Parameters
    ----------
    filenames : list[str]
        filenames to delete.
    session_id : str
        session to delete the file from.
    user: User
        The user obj.

    Returns
    -------
    dict
        Success message with number of files deleted.

    """
    logger.info("User %s deleting files", user.firebase_uid)
    results = []
    fail_occurred = False
    for filename in filenames:
        result = delete_file(filename, session_id, user)
        results.append(result)
        if result["message"] != "Success":
            fail_occurred = True
    message = "Success"
    if fail_occurred:
        message = "Failure: not all files were deleted successfully"
    return {"message": message, "results": results}


@api.post("/get_session_files", tags=["Vector Database"])
def get_session_files(session_id: str, user: User = Depends(get_current_user)) -> dict:
    """Get names of all files associated with a session.

    Parameters
    ----------
    session_id : str
        session to get files from.
    user: User
        The user obj.

    Returns
    -------
    dict
        Success message with list of filenames.

    """
    logger.info("User %s getting session files for session %s", user.firebase_uid, session_id)
    cr = fetch_session(FetchSession(session_id=session_id, user=user))
    result = get_expr(
        collection_name=SESSION_DATA,
        expr = f"metadata['session_id']=='{session_id}'",
    )
    if result["message"] != "Success":
        return {"message": "Failure: unable to get session files from Milvus"}
    files = list({data["metadata"]["filename"] for data in result["result"]})
    num_files = len(files)
    message = "Success"
    if cr.file_count != num_files:
        message = (
            f"Warning: file_count is {cr.file_count} "
            f"but {num_files} are in Milvus"
        )
        logger.error(
            "Session %s file_count is %d but %d are in Milvus",
            session_id,
            cr.file_count,
            num_files,
        )
    return {"message": message, "file_count": num_files, "results": files}


@api.post("/delete_session_files", tags=["Vector Database"])
def delete_session_files(session_id: str, user: User = Depends(get_current_user)) -> dict:
    """Delete all files associated with a session.

    Parameters
    ----------
    session_id : str
        session to delete files from.
    user : User
        user obj

    Returns
    -------
    dict
        Success message with delete count

    """
    logger.info("user %s deleting session files for session %s", user.firebase_uid, session_id)
    cr = fetch_session(FetchSession(session_id=session_id, user=user))
    result = delete_expr(
        SESSION_DATA,
        f"metadata['session_id']=='{session_id}'",
        session_id,
    )
    if result["message"] == "Success":
        cr.file_count = 0
        store_conversation_history(cr)
    else:
        logger.error("Unable to delete files in Milvus for session %s", session_id)
    return result


@api.post("/search_opinions", tags=["Opinion Search"])
def search_opinions(
    req: OpinionSearchRequest,
    user: User = Depends(get_current_user),
) -> dict:
    try:
        results = opinion_search(req)
    except Exception as error:
        return {"message": "Failure: Internal Error: " + str(error)}
    else:
        return {"message": "Success", "results": results}


@api.post("/search_resources", tags=["Resource Search"])
def search_resources(
    req: CollectionSearchRequest,
    user: User = Depends(get_current_user),
) -> dict:
    vdb_tool = VDBTool(name="test-tool", collection_name=req.resource_group, k=req.k)
    tool_response = run_vdb_tool(vdb_tool, req.model_dump())
    formatted_results = format_vdb_tool_results(tool_response, vdb_tool)
    return {"message": "Success", "results": formatted_results}


@api.get("/get_opinion_summary", tags=["Opinion Search"])
def get_opinion_summary(
    opinion_id: int,
    user: User = Depends(get_current_user),
) -> dict:
    try:
        summary = add_opinion_summary(opinion_id)
    except Exception as error:
        return {"message": "Failure: Internal Error: " + str(error)}
    else:
        return {"message": "Success", "result": summary}


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
        user: User = Depends(get_current_user))  -> dict:
    """Submit feedback to a specific session."""
    request.user = user
    return {"message": "Success" if store_opinion_feedback(request) else "Failure"}


@api.post("/search_collection", tags=["Resource Search"])
def search_collection(
    req: CollectionSearchRequest,
) -> dict:
    vdb_tool = VDBTool(name="test-tool", collection_name=req.collection, k=req.k)
    tool_response = run_vdb_tool(vdb_tool, req.model_dump(exclude_unset=True))
    formatted_results = format_vdb_tool_results(tool_response, vdb_tool)
    return {"message": "Success", "results": formatted_results}


@api.get("/resource_count/{collection_name}", tags=["Resource Search"])
def get_resource_count(
    collection_name: str,
) -> dict:
    return {"message": "Success", "resource_count": count_resources(collection_name)}


@api.post("/browse_collection", tags=["Resource Search"])
def browse_collection(
        req: CollectionManageRequest,
        page: int = 1,
        per_page: int = 200,
        user: User = Depends(get_current_user)
):
    """Browse a collection."""
    from datetime import UTC, datetime

    from app.courtlistener import courtlistener_collection, jurisdiction_codes
    from app.milvusdb import fuzzy_keyword_query
    from app.models import VDBMethodEnum

    expr = ""
    output_fields = ["text", *metadata_fields(req.collection)]
    if req.collection in {"search_collection_vj1", "search_collection_gemini"}:
        entity_id_key = "url"
    elif req.collection == SESSION_DATA:
        entity_id_key = "filename"
    else:
        entity_id_key = "id"
    if req.collection == courtlistener_collection:
        if req.source:
            expr = f"metadata['case_name'] like '%{req.source}%'"
        if req.keyword_query:
            expr += (" and " if expr else "")
            expr += " and ".join([
                f"TEXT_MATCH(text, '{word}')"
                for word in req.keyword_query.split()
            ])
        if req.jurisdictions:
            valid_jurisdics = []
            # look up each str in dictionary, append matches as lists
            for juris in req.jurisdictions:
                if juris.lower() in jurisdiction_codes:
                    valid_jurisdics += jurisdiction_codes[juris.lower()].split(" ")
            # clear duplicate federal district jurisdictions if they exist
            valid_jurisdics = list(set(valid_jurisdics))
            expr += (" and " if expr else "")
            expr += f"metadata['court_id'] in {valid_jurisdics}"
        if req.after_date:
            expr += (" and " if expr else "")
            expr += f"metadata['date_filed']>'{req.after_date}'"
        if req.before_date:
            expr += (" and " if expr else "")
            expr += f"metadata['date_filed']<'{req.before_date}'"
    else:
        if req.source:
            expr = f"metadata['{entity_id_key}'] like '%{req.source}%'"
        if req.keyword_query:
            tool_keyword_query = req.keyword_query
            keyword_query = fuzzy_keyword_query(tool_keyword_query)
            expr += (" and " if expr else "")
            expr += f"text like '% {keyword_query} %'"
        if req.jurisdictions:
            valid_jurisdics = [j.upper() for j in req.jurisdictions]
            # look up each str in dictionary, append matches as lists
            for juris in req.jurisdictions:
                if juris.lower() in jurisdiction_codes:
                    valid_jurisdics += jurisdiction_codes[juris.lower()].split(" ")
            # clear duplicate federal district jurisdictions if they exist
            valid_jurisdics = list(set(valid_jurisdics))
            expr += (" and " if expr else "")
            expr += f"ARRAY_CONTAINS_ANY(metadata['jurisdictions'], {valid_jurisdics})"
        if req.after_date:
            # convert YYYY-MM-DD to epoch time
            after_date = datetime.strptime(
                req.after_date,
                "%Y-%m-%d",
            ).replace(tzinfo=UTC)
            expr += (" and " if expr else "")
            expr += f"metadata['timestamp']>{after_date.timestamp()}"
        if req.before_date:
            # convert YYYY-MM-DD to epoch time
            before_date = datetime.strptime(
                req.before_date,
                "%Y-%m-%d",
            ).replace(tzinfo=UTC)
            expr += (" and " if expr else "")
            expr += f"metadata['timestamp']<{before_date.timestamp()}"
    q_iter = query_iterator(req.collection, expr, output_fields, 1000)
    source_ids = set()
    res = []
    has_next = True
    # skip the first (page - 1) * per_page sources
    while len(source_ids) < (page - 1) * per_page:
        res = q_iter.next()
        if not res:
            has_next = False
            break
        for hit in res:
            source_id = hit["metadata"][entity_id_key]
            if source_id not in source_ids:
                source_ids.add(source_id)
                if len(source_ids) == (page - 1) * per_page:
                    break
    if not has_next:
        return {"message": "Success", "has_next": False, "results": []}
    last_id = None
    res = [hit for hit in res if hit["metadata"][entity_id_key] not in source_ids]
    page_results = []
    while len(source_ids) < page * per_page:
        if not res:
            res = q_iter.next()
            if not res:
                has_next = False
                break
        for hit in res:
            source_id = hit["metadata"][entity_id_key]
            if source_id not in source_ids:
                last_id = source_id
                source_ids.add(source_id)
                if len(source_ids) == page * per_page:
                    break
            page_results.append(hit)
        res = []
    q_iter.close()
    last_id_expr = last_id if isinstance(last_id, int) else f"'{last_id}'"
    expr = f"metadata['{entity_id_key}']=={last_id_expr}"
    q_iter = query_iterator(req.collection, expr, output_fields, 1000)
    last_id_chunks = []
    res = q_iter.next()
    while len(res) > 0:
        last_id_chunks += res
        res = q_iter.next()
    q_iter.close()
    page_results = [
        hit for hit in page_results
        if hit["metadata"][entity_id_key] != last_id
    ]
    tool_output = {"message": "Success", "result": page_results + last_id_chunks}
    vdb_tool = VDBTool(
        name="test-tool",
        collection_name=req.collection,
        method=VDBMethodEnum.get_source,
    )
    formatted_results = format_vdb_tool_results(tool_output, vdb_tool)
    return {"message": "Success", "has_next": has_next, "results": formatted_results}


@api.delete("/delete_bot/{bot_id}", tags=["Bot"])
def delete_bot_endpoint(
    bot_id: str,
    user: User = Depends(get_current_user)
) -> dict:
    """
    Delete a bot.
    
    Only the creator of the bot can delete it.
    
    Parameters
    ----------
    bot_id : str
        The ID of the bot to delete
    user : User
        The authenticated user making the request
        
    Returns
    -------
    dict
        Success or failure message
    """
    logger.info("User %s attempting to delete bot %s", user.firebase_uid, bot_id)
    
    # Call the delete_bot function from db.py
    success = delete_bot(bot_id, user)
    
    if success:
        return {"message": "Success", "bot_id": bot_id}
    else:
        return {"message": "Failure: Bot not found or you don't have permission to delete it"}
