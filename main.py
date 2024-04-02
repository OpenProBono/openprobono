"""Written by Arman Aydemir. This file contains the main API code for the backend."""
from contextlib import asynccontextmanager
from typing import Annotated, Optional

import langfuse
from fastapi import Body, FastAPI, UploadFile

from db import (
    admin_check,
    api_key_check,
    fetch_session,
    load_session,
    process_chat,
    set_session_to_bot,
    store_bot,
)
from milvusdb import (
    COLLECTIONS,
    SESSION_PDF,
    US,
    crawl_and_scrape,
    delete_expr,
    session_source_summaries,
    session_upload_ocr,
    upload_documents,
)
from models import (
    BotRequest,
    ChatBySession,
    ChatRequest,
    FetchSession,
    InitializeSession,
    get_uuid_id,
)
from pdfs import summarized_chunks_pdf


# this is to ensure tracing with langfuse
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001, ANN201, D103
    # Operation on startup

    yield  # wait until shutdown

    # Flush all events to be sent to Langfuse on shutdown and
    # terminate all Threads gracefully. This operation is blocking.
    langfuse.flush()


api = FastAPI(lifespan=lifespan)


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
                            "history": [["hi!", ""]],
                            "bot_id": "ae885648-4fc7-4de6-ba81-67cc58c57d4c",
                            "api_key": "xyz",
                        },
                    },
                    "call a bot using history 2": {
                        "summary": "call a bot using history 2",
                        "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was "  # noqa: E501
                                       "used}",
                        "value": {
                            "history": [
                                ["hi!", "Hi, how can I assist you today?"],
                                ["I need help with something", ""],
                            ],
                            "bot_id": "ae885648-4fc7-4de6-ba81-67cc58c57d4c",
                            "api_key": "xyz",
                        },
                    },
                    "call opb bot": {
                        "summary": "call opb bot",
                        "description": "Returns: {message: 'Success', output: ai_reply, bot_id: the bot_id which was "  # noqa: E501
                                       "used}",
                        "value": {
                            "history": [["hi!", ""]],
                            "bot_id": "39e6d5c3-4e3c-4281-93d7-4f7c8db8833b",
                            "api_key": "xyz",
                        },
                    },
                },
            ),
        ]) -> dict:
    """Call a bot with history (only for backwards compat, could be deprecated)."""
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
                            "bot_id": "83f74a4e-0f8f-4142-b4e7-92a20f688a0b",
                            "api_key": "xyz",
                        },
                    },
                },
            ),
        ]) -> dict:
    """Initialize a new session with a message."""
    session_id = get_uuid_id()
    set_session_to_bot(session_id, request.bot_id)
    cr = ChatRequest(
        history=[[request.message, ""]],
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
                                    "prompt": "Useful for when you need to answer questions or find resources about "  # noqa: E501
                                              "government and laws.",
                                },
                                {
                                    "name": "case-search",
                                    "method": "courtlistener",
                                    "prompt": "Use for finding case law.",
                                },
                            ],
                            "vdb_tools": [
                                {
                                    "name": "USCode_query",
                                    "method": "query",
                                    "collection_name": US,
                                    "k": 4,
                                    "prompt": "Useful for finding information about US Code",  # noqa: E501
                                },
                            ],
                            "engine": "langchain",
                            "api_key": "xyz",
                        },
                    },
                    "full descriptions of every parameter": {
                        "summary": "Description and Tips",
                        "description": "full descriptions",
                        "value": {
                            "user_prompt": "prompt to use for the bot, this is appended to the regular prompt",  # noqa: E501
                            "message_prompt": "prompt to use for the bot, this is appended each message",  # noqa: E501
                            "model": "model to be used, currently only openai models, default is gpt-3.5-turbo-0125",  # noqa: E501
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
                                    "name": "name for tool",
                                    "method": "which search method to use, must be one of: qa, query",  # noqa: E501
                                    "collection_name": f"name of database to query, must be one of: {', '.join(list(COLLECTIONS))}",  # noqa: E501
                                    "k": "the number of text chunks to return when querying the database",  # noqa: E501
                                    "prompt": "description for agent to know when to use the tool",  # noqa: E501
                                    "prefix": "a prefix to add to query passed to tool by llm",  # noqa: E501
                                },
                            ],
                            "engine": "which library to use for model calls, must be one of: langchain, openai. "  # noqa: E501
                                      "Default is langchain.",
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


@api.post("/upload_file", tags=["User Upload"])
def upload_file(file: UploadFile,
        session_id: str, summary: Optional[str] = None) -> dict:
    """File upload by user.

    Parameters
    ----------
    file : UploadFile
        file to upload.
    session_id : str
        the session to associate the file with.
    summary : Optional[str], optional
        summary given by the user, by default None

    Returns
    -------
    dict
        Success or failure message.

    """
    docs = summarized_chunks_pdf(
        file, session_id, summary if summary else file.filename,
    )
    return upload_documents(SESSION_PDF, docs)


@api.post("/upload_files", tags=["User Upload"])
def upload_files(files: list[UploadFile],
    session_id: str, summaries: Optional[list[str]] = None) -> dict:
    """Upload multiple files by user.

    Parameters
    ----------
    files : list[UploadFile]
        files to upload.
    session_id : str
        the session to associate the file with.
    summaries : Optional[list[str]], optional
        summaries given by the user, by default None

    Returns
    -------
    dict
        Success or failure message.

    """
    if not summaries:
        summaries = [file.filename for file in files]
    elif len(files) != len(summaries):
        return {
            "message": f"Failure: did not find equal numbers of files and summaries, "
                f"instead found {len(files)} files and {len(summaries)} summaries.",
        }

    failures = []
    for i, file in enumerate(files):
        docs = summarized_chunks_pdf(file, session_id, summaries[i])
        result = upload_documents(SESSION_PDF, docs)
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
        session_id: str, summary: Optional[str] = None) -> dict:
    """Upload a file by user and use OCR to extract info."""
    return session_upload_ocr(file, session_id, summary if summary else file.filename)


@api.post("/delete_file", tags=["Vector Database"])
def delete_file(filename: str, session_id: str) -> dict:
    """Delete a file from the database.

    Parameters
    ----------
    filename : str
        filename to delete.
    session_id : str
        session to delete the file from.

    """
    return delete_expr(SESSION_PDF,
            f"source=='{filename}' and session_id=='{session_id}'")


@api.post("/delete_files", tags=["Vector Database"])
def delete_files(filenames: list[str], session_id: str) -> dict:
    """Delete multiple files from the database.

    Parameters
    ----------
    filenames : list[str]
        filenames to delete.
    session_id : str
        session to delete the file from

    Returns
    -------
    dict
        Success message with number of files deleted.

    """
    for filename in filenames:
        delete_expr(SESSION_PDF, f"source=='{filename}' and session_id=='{session_id}'")
    return {"message": f"Success: deleted {len(filenames)} files"}


@api.post("/get_session_files", tags=["Vector Database"])
def get_session_files(session_id: str) -> dict:
    """Get names of all files associated with a session.

    Parameters
    ----------
    session_id : str
        session to get files from.

    Returns
    -------
    dict
        Success message with list of filenames.

    """
    source_summaries = session_source_summaries(session_id)
    files = list(source_summaries.keys())
    return {"message": f"Success: found {len(files)} files", "result": files}


@api.post("/delete_session_files", tags=["Vector Database"])
def delete_session_files(session_id: str):
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
    return delete_expr(SESSION_PDF, f"session_id=='{session_id}'")


@api.post("/upload_site", tags=["Admin Upload"])
def vectordb_upload_site(site: str, collection_name: str,
        description: str, api_key: str):
    if not admin_check(api_key):
        return {"message": "Failure: API key invalid"}
    return crawl_and_scrape(site, collection_name, description)
