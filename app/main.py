"""Written by Arman Aydemir. This file contains the main API code for the backend."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Annotated, List, Optional

from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from langfuse.decorators import langfuse_context, observe

from app.bot import (
    anthropic_bot,
    anthropic_bot_stream,
    google_bot,
    google_bot_stream,
    openai_bot,
    openai_bot_stream,
)
from app.bot_helper import format_session_history, title_chat
from app.chat_models import chat_str_openai
from app.db import (
    browse_bots,
    browse_public_bots,
    browse_public_vdbs,
    browse_vdbs,
    delete_bot,
    fetch_session,
    fetch_sessions_by,
    get_cached_response,
    get_dataset,
    get_labeled_dataset,
    get_user_datasets,
    get_user_labeled_datasets,
    load_bot,
    load_vdb,
    set_session_to_bot,
    store_bot,
    store_conversation_history,
    store_eval_dataset,
    store_labeled_eval_dataset,
    store_opinion_feedback,
    store_session_feedback,
    update_labeled_session,
)
from app.logger import get_git_hash, setup_logger
from app.milvusdb import (
    SESSION_DATA,
    count_resources,
    create_collection,
    delete_collection,
    delete_expr,
    get_expr,
    query_iterator,
    session_upload_ocr,
    upload_resource,
)
from app.models import (
    AnthropicModelEnum,
    BotRequest,
    ChatBySession,
    ChatRequest,
    EngineEnum,
    EvalDataset,
    EvalSession,
    FetchSession,
    FetchSessions,
    GoogleModelEnum,
    HiveModelEnum,
    InitializeSession,
    InitializeSessionChat,
    InputGeneratorRequest,
    LabeledEvalDataset,
    LabeledEvalSession,
    LabelingAspect,
    OpenAIModelEnum,
    OpinionFeedback,
    OpinionSearchRequest,
    SessionFeedback,
    User,
    VDBManageRequest,
    VDBRequest,
    VDBSearchRequest,
    VDBTool,
    get_uuid_id,
)
from app.opinion_search import add_opinion_summary, opinion_search
from app.user_auth import get_current_user
from app.vdb_tools import format_vdb_tool_results, get_browse_expr, run_vdb_tool

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
        case EngineEnum.google:
            for chunk in google_bot_stream(r, bot):
                if isinstance(chunk, dict) and chunk["type"] == "response":
                    full_response += chunk["content"]
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
            case EngineEnum.google:
                output = google_bot(r, bot)
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
    logger.info(f"Succesffuly fetched {len(sessions)} sessions for user {user.firebase_uid}")
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
                                    "vdb_id": "SessionData",
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
                                    "vdb_id": "name of database to query, must be one of: courtlistener, bailii",  # noqa: E501
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


@api.post("/view_public_bots", tags=["Bot"])
def view_public_bots(user: User = Depends(get_current_user)) -> dict:
    """
    Get all public bots available in the system.
    
    Parameters
    ----------
    user : User
        The authenticated user making the request
        
    Returns
    -------
    dict
        Dictionary containing public bots
    """
    logger.info("User %s viewing public bots", user.firebase_uid)
    public_bots = browse_public_bots()
    return {"message": "Success", "data": public_bots}


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
    result = upload_resource(
        collection_name=SESSION_DATA,
        resource_type="file",
        resource=file,
        session_id=session_id,
        user_summary=summary,
    )
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
        result = upload_resource(
            collection_name=SESSION_DATA,
            resource_type="file",
            resource=file,
            session_id=session_id,
            user_summary=summaries[i],
        )
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
    result = delete_expr(SESSION_DATA, expr, session_id)
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
    expr = f"metadata['session_id']=='{session_id}'"
    result = get_expr(SESSION_DATA, expr)
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
    expr = f"metadata['session_id']=='{session_id}'"
    result = get_expr(SESSION_DATA, expr, session_id)
    if result["message"] == "Success":
        cr.file_count = 0
        store_conversation_history(cr)
    else:
        logger.error("Unable to delete files in Milvus for session %s", session_id)
    return result


@api.post("/search_opinions", tags=["Opinion Search"])
def search_opinions(
    req: OpinionSearchRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    logger.info("User %s searching opinions with request %s", user.firebase_uid, req)
    try:
        results = opinion_search(req)
    except Exception as error:
        logger.exception("Error searching opinions")
        return {"message": "Failure: Internal Error: " + str(error)}
    else:
        return {"message": "Success", "results": results}


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
    req: VDBSearchRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    logger.info("User %s searching collection with request %s", user.firebase_uid, req)
    vdb_tool = VDBTool(name="test-tool", vdb_id=req.vdb_id, k=req.k)
    tool_response = run_vdb_tool(vdb_tool, req.model_dump(exclude_unset=True))
    formatted_results = format_vdb_tool_results(tool_response, vdb_tool)
    return {"message": "Success", "results": formatted_results}


@api.get("/resource_count/{collection_name}", tags=["Resource Search"])
def resource_count(
    collection_name: str,
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    msg = "User %s counting resources in collection %s"
    logger.info(msg, user.firebase_uid, collection_name)
    return {"message": "Success", "resource_count": count_resources(collection_name)}


@api.post("/browse_collection", tags=["Resource Search"])
def browse_collection(
    req: VDBManageRequest,
    user: Annotated[User, Depends(get_current_user)],
    page: int = 1,
    per_page: int = 200,
) -> dict:
    """Browse a collection.

    Parameters
    ----------
    req : VDBManageRequest
        VDBManageRequest object containing the collection name and other parameters.
    user : Annotated[User, Depends(get_current_user)]
        The current authenticated user.
    page : int, optional
        The page number to retrieve, by default 1
    per_page : int, optional
        The number of items to return per page, by default 200

    Returns
    -------
    dict
        A dictionary containing the message and results of the browse operation.

    """
    from app.models import MilvusMetadataEnum, VDBMethodEnum

    logger.info("User %s browsing collection with request %s", user.firebase_uid, req)
    vdb = load_vdb(req.vdb_id)
    if vdb.metadata_format == MilvusMetadataEnum.json:
        fields = ["metadata"]
    elif vdb.metadata_format == MilvusMetadataEnum.field:
        fields = [f.name for f in vdb.extra_fields]
    else:
        fields = []
    output_fields = ["text", *fields]
    expr = get_browse_expr(req)
    try:
        q_iter = query_iterator(req.vdb_id, expr, output_fields, 1000)
    except:
        logger.exception("Error getting query iterator for collection %s", req.vdb_id)
        return {
            "message": "Success",
            "collection_name": vdb.name,
            "has_next": False,
            "results": [],
        }
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
            if "url" in hit["metadata"]:
                source_id = hit["metadata"]["url"]
            else:
                source_id = hit["metadata"]["id"]
            if source_id not in source_ids:
                source_ids.add(source_id)
                if len(source_ids) == (page - 1) * per_page:
                    break
    if not has_next:
        return {
            "message": "Success",
            "collection_name": vdb.name,
            "has_next": False,
            "results": [],
        }
    last_id = None
    res = [
        hit for hit in res
        if ("id" in hit["metadata"] and hit["metadata"]["id"] not in source_ids)
        or ("url" in hit["metadata"] and hit["metadata"]["url"] not in source_ids)
    ]
    page_results = []
    while len(source_ids) < page * per_page:
        if not res:
            res = q_iter.next()
            if not res:
                has_next = False
                break
        for hit in res:
            if "id" in hit["metadata"]:
                source_id = hit["metadata"]["id"]
            else:
                source_id = hit["metadata"]["url"]
            if source_id not in source_ids:
                last_id = source_id
                source_ids.add(source_id)
                if len(source_ids) == page * per_page:
                    break
            page_results.append(hit)
        res = []
    q_iter.close()
    last_id_expr = last_id if isinstance(last_id, int) else f"'{last_id}'"
    expr = (
        f"metadata['id']=={last_id_expr} or "
        f"metadata['url']=={last_id_expr}"
    )
    q_iter = query_iterator(req.vdb_id, expr, output_fields, 1000)
    last_id_chunks = []
    res = q_iter.next()
    while len(res) > 0:
        last_id_chunks += res
        res = q_iter.next()
    q_iter.close()
    page_results = [
        hit for hit in page_results
        if ("id" in hit["metadata"] and hit["metadata"]["id"] != last_id)
        or ("url" in hit["metadata"] and hit["metadata"]["url"] != last_id)
    ]
    tool_output = {"message": "Success", "result": page_results + last_id_chunks}
    vdb_tool = VDBTool(
        name="test-tool",
        vdb_id=req.vdb_id,
        method=VDBMethodEnum.get_source,
    )
    formatted_results = format_vdb_tool_results(tool_output, vdb_tool)
    return {
        "message": "Success",
        "collection_name": vdb.name,
        "has_next": has_next,
        "results": formatted_results,
    }

@api.get("/view_collection/{vdb_id}", tags=["Collection"])
def view_collection(
    user: Annotated[User, Depends(get_current_user)],
    vdb_id: str,
) -> dict:
    """Get collection info by vdb_id.

    Parameters
    ----------
    user : Annotated[User, Depends
        The authenticated user
    vdb_id : str
        The VDB ID

    Returns
    -------
    dict
        Containing `message` key indicating success,
        and `data` containing  VDBRequest object on success

    """
    logger.info("User %s viewing collection %s", user.firebase_uid, vdb_id)
    vdb = load_vdb(vdb_id)
    if vdb is None:
        return {"message": "Failure: collection not found"}
    return {"message": "Success", "data": vdb}

@api.get("/view_user_collections", tags=["Collection"])
def view_user_collections(user: Annotated[User, Depends(get_current_user)]) -> dict:
    """View all collections created by this user.

    Parameters
    ----------
    user : User
        The authenticated user

    Returns
    -------
    dict
        Success message with collections dictionary

    """
    logger.info("User %s requesting their collections", user.firebase_uid)
    vdbs = browse_vdbs(user)
    for vdb_id in vdbs:
        vdbs[vdb_id]["resource_count"] = count_resources(vdb_id)
    return {"message": "Success", "data": vdbs}


@api.get("/view_public_collections", tags=["Collection"])
def view_public_collections(
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """View all public collections.

    Parameters
    ----------
    user : User
        The authenticated user

    Returns
    -------
    dict
        Success message with collections dictionary

    """
    logger.info("User %s requesting public collections", user.firebase_uid)
    public_vdbs = browse_public_vdbs()
    for vdb_id in public_vdbs:
        public_vdbs[vdb_id]["resource_count"] = count_resources(vdb_id)
    return {"message": "Success", "data": public_vdbs}


@api.post("/upload_resources", tags=["Upload"])
def upload_resources(
    user: Annotated[User, Depends(get_current_user)],
    resource_type: str,
    target_id: str,
    files: Annotated[list[UploadFile], File()] = ...,
    urls: Annotated[list[str], Form()] = ...,
    summaries: list[str] | None = None,
) -> dict:
    """Upload resources (files or URLs) to a collection or session.

    Parameters
    ----------
    user : User
        The authenticated user
    resource_type : str
        Type of resources being added, either "file" or "url"
    target_id : str
        Collection ID or session ID
    files : list[UploadFile], optional
        Files to upload, by default None
    urls : list[str], optional
        URLs to scrape and add, by default None
    summaries : list[str], optional
        Summaries provided by the user for each resource, by default None

    """
    logger.info("User %s uploading resources to %s", user.firebase_uid, target_id)

    error_msg = None
    if resource_type not in ["file", "url"]:
        error_msg = "Failure: resource_type must be 'file' or 'url'"
    if resource_type == "file":
        if not files:
            error_msg = "Failure: no files provided"
        if summaries and len(summaries) != len(files):
            error_msg = (
                f"Failure: did not find equal numbers of files and summaries, "
                f"instead found {len(files)} files and {len(summaries)} summaries."
            )
    else: # url
        if not urls:
            error_msg = "Failure: no URLs provided"
        if summaries and len(summaries) != len(urls):
            error_msg = (
                f"Failure: did not find equal numbers of URLs and summaries, "
                f"instead found {len(urls)} URLs and {len(summaries)} summaries."
            )
    if error_msg:
        logger.error(error_msg)
        return {"message": error_msg}

    is_collection = target_id.startswith("col_")

    if is_collection:
        vdb = load_vdb(target_id)
        if vdb is None:
            return {"message": f"Failure: collection with ID {target_id} not found"}
    else:
        session = fetch_session(FetchSession(session_id=target_id, user=None))

    results = []
    fail_occurred = False

    if resource_type == "file":
        for i, file in enumerate(files):
            result = upload_resource(
                collection_name=target_id,
                resource_type="file",
                resource=file,
                session_id=None if is_collection else target_id,
                user_summary=summaries[i] if summaries else None,
            )
            if result["message"] != "Success":
                fail_occurred = True
            elif not is_collection:
                session.file_count += 1
                session.history.append({"role": "user", "content": f"file:{file.filename}"})
            results.append({"id": file.filename, "message": result["message"], "insert_count": result.get("insert_count")})

        if not is_collection:
            store_conversation_history(session)

    elif resource_type == "url":
        for i, url in enumerate(urls):
            try:
                # For URLs, we need to create a search result like object
                search_result = {
                    "link": url,
                    "title": url,  # Use URL as title if we don't have one
                    "source": "user_upload",
                }

                # Create a dummy search tool with basic parameters
                # This is needed for the upload_resource function
                from app.models import SearchTool
                dummy_search_tool = SearchTool(name="url_upload", prompt="")

                result = upload_resource(
                    collection_name=target_id,
                    resource_type="url",
                    resource=search_result,
                    session_id=None if is_collection else target_id,
                    search_tool=dummy_search_tool,
                    user_summary=summaries[i] if summaries else None,
                )

                if result["message"] != "Success":
                    fail_occurred = True
                results.append({"id": url, "message": result["message"], "insert_count": result.get("insert_count")})
            except Exception as e:
                logger.exception("Failed to upload resource: %s", url)
                fail_occurred = True
                results.append({"id": url, "message": f"Failure: {e!s}"})

    return {"message": "Failure: not all resources were added" if fail_occurred else "Success", "results": results}


@api.post("/remove_resources", tags=["Collection"])
def remove_resources(
    resource_ids: list[str],
    resource_type: str,
    vdb_id: str,
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Remove resources (files or URLs) from a collection.

    Parameters
    ----------
    resource_ids : List[str]
        IDs of resources to delete (filenames or URLs)
    resource_type : str
        Type of resources being removed, either "file" or "url"
    vdb_id : str
        The ID of the collection to remove resources from
    user : User
        The authenticated user

    Returns
    -------
    dict
        Success message with number of resources deleted
    """
    logger.info("User %s removing resources from collection %s", user.firebase_uid, vdb_id)
    
    if resource_type not in ["file", "url"]:
        return {"message": "Failure: resource_type must be 'file' or 'url'"}
    
    # Check if collection exists and user has access
    vdb = load_vdb(vdb_id)
    if vdb is None:
        return {"message": f"Failure: collection with ID {vdb_id} not found"}
    
    results = []
    fail_occurred = False
    
    for resource_id in resource_ids:
        try:
            # Create appropriate expression based on resource type
            if resource_type == "file":
                expr = f"metadata['filename']=='{resource_id}'"
            else:  # url
                expr = f"metadata['url']=='{resource_id}'"
            
            # Delete matching resources
            result = delete_expr(vdb_id, expr)
            
            if result["delete_count"] == 0:
                logger.warning("collection %s resource %s not found", vdb_id, resource_id)
                results.append({
                    "id": resource_id,
                    "message": "Warning: resource not found",
                    "delete_count": 0
                })
            else:
                results.append({
                    "id": resource_id,
                    "message": "Success",
                    "delete_count": result["delete_count"]
                })
        except Exception as e:
            fail_occurred = True
            results.append({
                "id": resource_id,
                "message": f"Failure: {str(e)}",
                "delete_count": 0
            })
    
    message = "Success"
    if fail_occurred:
        message = "Failure: not all resources were deleted successfully"
    
    return {"message": message, "results": results}


@api.delete("/delete_collection/{collection_id}", tags=["Collection"])
def delete_vdb(
    collection_id: str,
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Delete a collection.

    Only the creator of the collection can delete it.

    Parameters
    ----------
    collection_id : str
        The ID of the collection to delete
    data : dict
        Dict containing the user information
    user : User
        The authenticated user making the request

    Returns
    -------
    dict
        Success or failure message

    """
    msg = "User %s attempting to delete collection %s"
    logger.info(msg,user.firebase_uid, collection_id)
    res = delete_collection(collection_id, user)
    if res:
        return {"message": "Success", "collection_id": collection_id}
    return {
        "message": "Failure: Collection not found or you don't have permission to delete it."}


@api.post("/create_collection", tags=["Collection"])
def create_vdb(
    req: VDBRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """Create a new collection.

    Parameters
    ----------
    req : VDBRequest
        Containing the collection information (name, description, etc.)
    user : User
        The authenticated user making the request

    Returns
    -------
    dict
        Success or failure message with the new collection ID

    """
    logger.info("User %s creating a new collection: %s", user.firebase_uid, req.name)

    # get a unique ID for the collection
    vdb_id = "col_" + get_uuid_id().replace("-", "_")

    coll = create_collection(req, vdb_id)

    if coll is None:
        return {"message": "Failure: The schema is invalid or the collection already exists."}
    return {"message": "Success", "vdb_id": vdb_id}

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


@api.post("/run_eval_dataset", tags=["Evaluation"])
def run_eval_dataset(
    background_tasks: BackgroundTasks,
    dataset: Annotated[
        EvalDataset,
        Body(
            openapi_examples={
                "create dataset": {
                    "summary": "Create an evaluation dataset",
                    "description": "Creates a dataset with inputs and bots for evaluation",
                    "value": {
                        "name": "Test Dataset",
                        "description": "A dataset for testing bot performance",
                        "inputs": ["What is the capital of France?", "Explain quantum computing"],
                        "bot_ids": ["bot_id_1", "bot_id_2"]
                    },
                },
            },
        ),
    ],
    user: User = Depends(get_current_user)
) -> dict:
    """Create a new evaluation dataset with inputs and bots.
    
    This endpoint creates a dataset that can be used to evaluate multiple bots
    against the same set of inputs. It initializes sessions for each input-bot pair,
    runs the bots on the inputs, and stores the results asynchronously.
    
    Parameters
    ----------
    background_tasks : BackgroundTasks
        FastAPI background tasks handler
    dataset : EvalDataset
        The dataset to create, containing inputs and bot IDs
    user : User
        The authenticated user creating the dataset
        
    Returns
    -------
    dict
        Success message with the dataset ID
    """
    # Set the user
    dataset.user = user
    
    # Generate dataset ID
    dataset_id = get_uuid_id()
    
    # Initialize sessions list
    dataset.sessions = []
    
    # Store the initial dataset
    store_eval_dataset(dataset, dataset_id)
    
    # Define the background task function
    def process_eval_dataset(dataset, dataset_id, user):
        sessions = []
        
        # Create sessions for each input-bot pair
        for input_idx, input_text in enumerate(dataset.inputs):
            for bot_idx, bot_id in enumerate(dataset.bot_ids):
                # Check if bot exists and user has access
                bot = load_bot(bot_id)
                if not bot:
                    logger.error(f"Bot {bot_id} not found for dataset {dataset_id}")
                    continue
                
                # Create a new session for this input-bot pair
                session_id = get_uuid_id()
                set_session_to_bot(session_id, bot_id)
                
                # Initialize the session with the input
                cr = ChatRequest(
                    history=[{"role": "user", "content": input_text}],
                    bot_id=bot_id,
                    session_id=session_id,
                    user=user,
                )
                
                # Call the bot to get the output
                response = process_chat(cr, input_text)
                output_text = response.get("output", "Error: No output generated")
                
                # Create and store the session
                eval_session = EvalSession(
                    input_idx=input_idx,
                    bot_idx=bot_idx,
                    input_text=input_text,
                    output_text=output_text,
                    bot_id=bot_id,
                    session_id=session_id
                )
                sessions.append(eval_session)
                
                # Store the conversation history with the bot's response
                store_conversation_history(cr)
                
                # Update the dataset with the current sessions
                dataset.sessions = sessions
                store_eval_dataset(dataset, dataset_id)
        
        # Final update to the dataset
        dataset.sessions = sessions
        store_eval_dataset(dataset, dataset_id)
        logger.info(f"Completed evaluation dataset {dataset_id} with {len(sessions)} sessions")
    
    # Add the task to background tasks
    background_tasks.add_task(process_eval_dataset, dataset, dataset_id, user)
    
    # Return immediately with the dataset ID
    return {
        "message": "Success",
        "dataset_id": dataset_id,
        "status": "Processing evaluation dataset in the background"
    }

@api.get("/get_user_datasets", tags=["Evaluation"])
def get_datasets(user: User = Depends(get_current_user)) -> dict:
    """Get all evaluation datasets for the authenticated user.
    
    Parameters
    ----------
    user : User
        The authenticated user
        
    Returns
    -------
    dict
        Success message with the datasets
    """
    datasets = get_user_datasets(user)
    return {"message": "Success", "datasets": datasets}

@api.get("/get_dataset_sessions/{dataset_id}", tags=["Evaluation"])
def get_dataset_sessions(dataset_id: str, user: User = Depends(get_current_user)) -> dict:
    """Get all sessions for a specific dataset.
    
    Parameters
    ----------
    dataset_id : str
        The ID of the dataset
    user : User
        The authenticated user
        
    Returns
    -------
    dict
        Success message with the sessions
    """
    dataset = get_dataset(dataset_id)
    if not dataset:
        return {"message": "Failure: Dataset not found"}
    
    if dataset.user.firebase_uid != user.firebase_uid:
        return {"message": "Failure: You don't have permission to access this dataset"}
    
    # Create a more structured view of the sessions
    structured_sessions = {}
    for session in dataset.sessions:
        structured_sessions[session.session_id] = session

    return {
        "message": "Success", 
        "dataset": {
            "name": dataset.name,
            "description": dataset.description,
            "inputs": dataset.inputs,
            "bot_ids": dataset.bot_ids,
            "sessions": structured_sessions
        }
    }

@api.post("/create_labeled_dataset", tags=["Evaluation"])
def create_labeled_dataset(
    dataset_name: str,
    dataset_id: str,
    labeling_aspects: List[LabelingAspect],
    user: User = Depends(get_current_user)
) -> dict:
    """Create a new labeled evaluation dataset from an existing dataset.
    
    This endpoint creates a labeled dataset based on an existing evaluation dataset.
    It copies the inputs, bot IDs, and sessions from the original dataset and prepares
    them for labeling with multiple aspects.
    
    Parameters
    ----------
    dataset_name : str
        given name for labeled dataset
    dataset_id : str
        The ID of the existing dataset to label
    labeling_aspects : List[LabelingAspect]
        The list of aspects to evaluate for each response
    user : User
        The authenticated user creating the labeled dataset
        
    Returns
    -------
    dict
        Success message with the labeled dataset ID
    """
    # Get the original dataset
    original_dataset = get_dataset(dataset_id)
    if not original_dataset:
        return {"message": "Failure: Dataset not found"}
    
    # Check user permission
    if original_dataset.user.firebase_uid != user.firebase_uid:
        return {"message": "Failure: You don't have permission to access this dataset"}
    
    # Create a new labeled dataset
    labeled_dataset_id = get_uuid_id()
    
    # Initialize labeled sessions
    labeled_sessions = []
    for session in original_dataset.sessions:
        labeled_session = LabeledEvalSession(
            session_id=session.session_id,
            input_idx=session.input_idx,
            bot_idx=session.bot_idx,
            input_text=session.input_text,
            output_text=session.output_text,
            bot_id=session.bot_id,
            aspects=labeling_aspects,
            labeled=False
        )
        labeled_sessions.append(labeled_session)
    
    # Create the labeled dataset
    labeled_dataset = LabeledEvalDataset(
        name=dataset_name,
        description=original_dataset.description,
        labeling_aspects=labeling_aspects,
        original_dataset_id=dataset_id,
        inputs=original_dataset.inputs,
        bot_ids=original_dataset.bot_ids,
        sessions=labeled_sessions,
        progress=0.0,
        completed=False,
        user=user
    )
    
    # Store the labeled dataset
    store_labeled_eval_dataset(labeled_dataset, labeled_dataset_id)
    
    return {
        "message": "Success",
        "labeled_dataset_id": labeled_dataset_id
    }

@api.get("/get_user_labeled_datasets", tags=["Evaluation"])
def get_labeled_datasets(user: User = Depends(get_current_user)) -> dict:
    """Get all labeled evaluation datasets for the authenticated user.
    
    Parameters
    ----------
    user : User
        The authenticated user
        
    Returns
    -------
    dict
        Success message with the labeled datasets
    """
    datasets = get_user_labeled_datasets(user)
    return {"message": "Success", "datasets": datasets}

@api.get("/get_labeled_dataset/{dataset_id}", tags=["Evaluation"])
def get_labeled_dataset_endpoint(dataset_id: str, user: User = Depends(get_current_user)) -> dict:
    """Get a specific labeled evaluation dataset.
    
    Parameters
    ----------
    dataset_id : str
        The ID of the labeled dataset
    user : User
        The authenticated user
        
    Returns
    -------
    dict
        Success message with the labeled dataset
    """
    dataset = get_labeled_dataset(dataset_id)
    if not dataset:
        return {"message": "Failure: Labeled dataset not found"}
    
    if dataset.user.firebase_uid != user.firebase_uid:
        return {"message": "Failure: You don't have permission to access this dataset"}
    
    return {
        "message": "Success",
        "dataset": dataset.model_dump()
    }

@api.post("/update_labeled_session", tags=["Evaluation"])
def update_labeled_session_endpoint(
    dataset_id: str,
    session_id: str,
    aspect_ratings: dict,  # Dictionary mapping aspect_id to rating value
    notes: Optional[str] = None,
    user: User = Depends(get_current_user)
) -> dict:
    """Update a labeled session within a labeled evaluation dataset.
    
    Parameters
    ----------
    dataset_id : str
        The ID of the labeled dataset
    session_id : str
        The ID of the session to update
    aspect_ratings : dict
        Dictionary mapping aspect_id to rating value (can be int for ranking, bool for thumbs, float for score)
    notes : Optional[str], optional
        Evaluation notes, by default None
    user : User
        The authenticated user
        
    Returns
    -------
    dict
        Success message
    """
    success = update_labeled_session(
        dataset_id=dataset_id,
        session_id=session_id,
        aspect_ratings=aspect_ratings,
        notes=notes,
        user=user
    )
    
    if not success:
        return {"message": "Failure: Could not update labeled session"}
    
    # Get the updated dataset to return progress
    dataset = get_labeled_dataset(dataset_id)
    
    return {
        "message": "Success",
        "progress": dataset.progress,
        "completed": dataset.completed
    }

@api.post("/input_generator", tags=["Evaluation"])
def input_generator_endpoint(
    request: Annotated[
        InputGeneratorRequest,
        Body(
            openapi_examples={
                "generate inputs": {
                    "summary": "Generate a list of inputs based on a prompt",
                    "description": "Returns a list of generated inputs based on the provided prompt",
                    "value": {
                        "prompt": "Generate legal questions about contract law"
                    },
                },
            },
        ),
    ],
    user: User = Depends(get_current_user)
) -> dict:
    """Generate a list of inputs based on a prompt using GPT-4o.
    
    Parameters
    ----------
    request : InputGeneratorRequest
        The request containing the prompt
    user : User
        The authenticated user
        
    Returns
    -------
    dict
        A dictionary containing the generated inputs
    """
    # Set up the request with the user
    request.user = user
    
    # Create the messages for the GPT-4o call
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates a list of inputs based on prompts."},
        {"role": "user", "content": f"Generate a list of inputs based on the following prompt:\n\n{request.prompt}"}
    ]
    
    # Call GPT-4o using the chat_str_openai function
    try:
        response = chat_str_openai(
            messages=messages,
            model=OpenAIModelEnum.gpt_4o.value,
            temperature=0.7  # Use a slightly higher temperature for creativity
        )
        
        # Parse the response into a list of strings
        # First, try to parse as a list if it looks like one
        if response.startswith("1.") or response.startswith("-") or response.startswith("*"):
            # Split by newlines and clean up
            inputs = [line.strip() for line in response.split("\n") 
                     if line.strip() and not line.strip().isdigit()]
            
            # Remove numbering or bullet points
            inputs = [re.sub(r"^\d+\.\s*|\*\s*|-\s*", "", line) for line in inputs]
        else:
            # If not in list format, split by newlines
            inputs = [line.strip() for line in response.split("\n") if line.strip()]
        
        # Filter out any empty strings
        inputs = [input_str for input_str in inputs if input_str]
        
        return {
            "message": "Success",
            "inputs": inputs
        }
    except Exception as e:
        return {
            "message": f"Error: {str(e)}",
            "inputs": []
        }

@api.get("/available_models", tags=["Bot"])
def get_available_models() -> dict:
    """Return available models for each engine type based on the enums in models.py.
    
    Returns:
        dict: A dictionary with engine types as keys and lists of available models as values
    """
    models = {
        "openai": [model.value for model in OpenAIModelEnum],
        "anthropic": [model.value for model in AnthropicModelEnum],
        "google": [model.value for model in GoogleModelEnum],
        "hive": [model.value for model in HiveModelEnum]
    }
    
    return {
        "message": "Success",
        "data": models
    }
