"""Tests for API endpoints."""
from __future__ import annotations

import os
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from app import main
from app.logger import setup_logger

logger = setup_logger()

test_api_key = os.environ["OPB_TEST_API_KEY"]
client = TestClient(main.api, headers={"X-API-KEY": test_api_key})
test_bot_id = "default_bot"
test_session_id = "test_session"
test_session_files_id = "test_session_with_files"
test_message = "hi"
test_history = [{"role":"user", "content":"hi"}]
test_opinion_id = 1
test_site = "https://www.openprobono.com/"
test_query = "foo"
http_ok = 200


def response_test(status_code: int, response_json: dict) -> None:
    """Test for API responses.

    Parameters
    ----------
    status_code : int
        The response status code
    response_json : dict
        The response object

    """
    assert status_code == http_ok, f"Unexpected status code: {status_code}"
    assert "message" in response_json, f"'message' not found in response: {response_json}"
    assert isinstance(response_json["message"], str), f"'message' unexpected data type: expected str, got {type(response_json['message'])}"
    assert response_json["message"] == "Success", f"'message' unexpected value: expected 'Success', got {response_json['message']}"


def test_read_root() -> None:
    """Tests that the API is alive."""
    response = client.get("/")
    assert response.status_code == http_ok, f"Unexpected status code: {response.status_code}"
    response_json = response.json()
    assert response_json == {"message": "API is alive"}, f"Unexpected response: {response_json}"


@pytest.mark.parametrize(
    ("bot_id", "session_id", "history"),
    [(test_bot_id, test_session_id, test_history)],
)
def test_chat(bot_id: str, session_id: str, history: list) -> None:
    """Tests calling a bot using history (no new message).

    Parameters
    ----------
    bot_id : str
        The bot for this session
    session_id : str
        The id of this session
    history : list
        The history for this session

    """
    from app.models import ChatRequest
    test_chat_request = ChatRequest(
        bot_id=bot_id,
        session_id=session_id,
        history=history,
    )
    response = client.post("/invoke_bot", json=test_chat_request.model_dump())
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "bot_id" in response_json, "'bot_id' not found in response"
    assert isinstance(response_json["bot_id"], str), f"'bot_id' unexpected data type: expected str, got {type(response_json['bot_id'])}"
    assert "output" in response_json, "'output' not found in response"
    assert isinstance(response_json["output"], str), f"'output' unexpected data type: expected str, got {type(response_json['output'])}"


@pytest.mark.parametrize("bot_id", [test_bot_id])
def test_init_session(bot_id: str) -> None:
    """Tests initializing a session.

    Parameters
    ----------
    bot_id : str
        The bot for this session.

    """
    from app.models import InitializeSession
    init_session = InitializeSession(bot_id=bot_id)
    response = client.post("/initialize_session", json=init_session.model_dump())
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "bot_id" in response_json, "'bot_id' not found in response"
    assert isinstance(response_json["bot_id"], str), f"'bot_id' unexpected data type: expected str, got {type(response_json['bot_id'])}"
    assert "session_id" in response_json, "'session_id' not found in response"
    assert isinstance(response_json["session_id"], str), f"'session_id' unexpected data type: expected str, got {type(response_json['session_id'])}"


@pytest.mark.parametrize(("bot_id", "message"), [(test_bot_id, test_message)])
def test_init_session_chat(bot_id: str, message: str) -> None:
    """Tests initializing a session with a message.

    Parameters
    ----------
    bot_id : str
        The bot for this session
    message : str
        The initial message for this session

    """
    from app.models import InitializeSessionChat
    test_initialize_session = InitializeSessionChat(bot_id=bot_id, message=message)
    response = client.post(
        "/initialize_session_chat",
        json=test_initialize_session.model_dump(),
    )
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "bot_id" in response_json, "'bot_id' not found in response"
    assert isinstance(response_json["bot_id"], str), f"'bot_id' unexpected data type: expected str, got {type(response_json['bot_id'])}"
    assert "output" in response_json, "'output' not found in response"
    assert isinstance(response_json["output"], str), f"'output' unexpected data type: expected str, got {type(response_json['output'])}"
    assert "session_id" in response_json, "'session_id' not found in response"
    assert isinstance(response_json["session_id"], str), f"'session_id' unexpected data type: expected str, got {type(response_json['session_id'])}"


@pytest.mark.parametrize(("session_id", "message"), [(test_session_id, test_message)])
def test_chat_session(session_id: str, message: str) -> None:
    """Tests continuing a session with a message.

    Parameters
    ----------
    session_id : str
        The session to test
    message : str
        The message for this session

    """
    from app.models import ChatBySession
    chat_session = ChatBySession(session_id=session_id, message=message)
    response = client.post(
        "/chat_session",
        json=chat_session.model_dump(),
    )
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "bot_id" in response_json, "'bot_id' not found in response"
    assert isinstance(response_json["bot_id"], str), f"'bot_id' unexpected data type: expected str, got {type(response_json['bot_id'])}"
    assert "output" in response_json, "'output' not found in response"
    assert isinstance(response_json["output"], str), f"'output' unexpected data type: expected str, got {type(response_json['output'])}"
    assert "session_id" in response_json, "'session_id' not found in response"
    assert isinstance(response_json["session_id"], str), f"'session_id' unexpected data type: expected str, got {type(response_json['session_id'])}"
    assert response_json["session_id"] == session_id, f"'session_id's do not match: expected {session_id}, got {response_json['session_id']}"


@pytest.mark.parametrize("session_id", [test_session_id])
def test_fetch_session(session_id: str) -> None:
    """Tests initializing a session.

    Parameters
    ----------
    session_id : str
        The session to test.

    """
    from app.models import ChatRequest, FetchSession
    fetch_session = FetchSession(session_id=session_id)
    response = client.post("/fetch_session", json=fetch_session.model_dump())
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    del response_json["message"]
    try:
        logger.info("Validating object is a ChatRequest")
        _ = ChatRequest(**response_json)
    except Exception as e:
        pytest.fail(f"ChatRequest validation failed: {e}")


@pytest.mark.parametrize("session_id", [test_session_id])
def test_format_session_history(session_id: str) -> None:
    """Tests fetching formatted session history.

    Parameters
    ----------
    session_id : str
        The session to test

    """
    from app.models import FetchSession
    fetch_session = FetchSession(session_id=session_id)
    response = client.post(
        "/fetch_session_formatted_history",
        json=fetch_session.model_dump(),
    )
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "history" in response_json, "'history' not found in response"
    assert isinstance(response_json["history"], list), f"'history' unexpected data type: expected list, got {type(response_json['history'])}"


@pytest.mark.parametrize("session_id", [test_session_id])
def test_session_feedback(session_id: str) -> None:
    """Tests leaving feedback for a session.

    Parameters
    ----------
    session_id : str
        The session receiving feedback

    """
    from app.models import SessionFeedback
    test_submit_feedback = SessionFeedback(
        feedback_text="test feedback",
        session_id=session_id,
    )
    response = client.post("/session_feedback", json=test_submit_feedback.model_dump())
    response_json = response.json()
    response_test(response.status_code, response_json)


def test_create_bot() -> None:
    """Tests creating a bot with default parameters."""
    from app.models import BotRequest
    bot_request = BotRequest()
    response = client.post("/create_bot", json=bot_request.model_dump())
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "bot_id" in response_json, "'bot_id' not found in response"
    assert isinstance(response_json["bot_id"], str), f"'bot_id' unexpected data type: expected str, got {type(response_json['bot_id'])}"


@pytest.mark.parametrize("bot_id", [test_bot_id])
def test_view_bot(bot_id: str) -> None:
    """Tests viewing bot parameters.

    Parameters
    ----------
    bot_id : str
        The bot to test.

    """
    from app.models import BotRequest
    data = {"bot_id": bot_id}
    response = client.get("/view_bot", params=data)
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "data" in response_json, "'data' not found in response"
    assert isinstance(response_json["data"], dict), f"'data' unexpected data type: expected dict, got {type(response_json['data'])}"
    try:
        _ = BotRequest(**response_json["data"])
    except Exception as e:
        pytest.fail(f"BotRequest initialization failed: {e}")


def test_view_bots() -> None:
    """Tests viewing public bots."""
    from app.models import BotRequest
    response = client.post("/view_bots")
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "data" in response_json, "'data' not found in response"
    assert isinstance(response_json["data"], dict), f"'data' unexpected data type: expected dict, got {type(response_json['data'])}"
    try:
        for bot_id in response_json["data"]:
            logger.info("Validating bot ID %s", bot_id)
            _ = BotRequest(**response_json["data"][bot_id])
    except Exception as e:
        pytest.fail(f"BotRequest validation failed: {e}")


def _create_test_file(
    filename: str,
    content: str | None = None,
) -> tuple[str, tuple[str, BytesIO, str]]:
    """Create a test file tuple ready for HTTP request. Helper function.

    Parameters
    ----------
    filename : str
        The filename
    content : str, optional
        The file content, by default None

    Returns
    -------
    tuple[str, tuple[str, BytesIO, str]]
        Tuple of (field_name, (filename, file_object, content_type))
        Ready to use in the files parameter of a request

    """
    if content is None:
        content = f"test content for {filename}\n"

    file_obj = BytesIO(content.encode())
    content_type = "text/plain"  # You could make this a parameter if needed

    return ("file", (filename, file_obj, content_type))


def _create_test_files(
    filenames: list[str],
    contents: list[str] | None = None,
) -> list[tuple[str, tuple[str, BytesIO, str]]]:
    """Create multiple test files. Helper function.

    Parameters
    ----------
    filenames : list[str]
        The names of the files
    contents : list[str], optional
        The contents of the files, by default None

    Returns
    -------
    list[tuple[str, tuple[str, BytesIO, str]]]
        Tuple of (field_name, (filename, file_object, content_type))
        Ready to use in the files parameter of a request

    """
    if contents is not None:
        assert len(filenames) == len(contents), "Filenames and contents are not the same length."
    content_type = "text/plain"  # You could make this a parameter if needed
    return [
        ("files", (
            filenames[i],
            BytesIO((f"test text {i+1}\n" if contents is None else contents[i]).encode()),
            content_type,
        ))
        for i in range(len(filenames))
    ]

@pytest.mark.parametrize("session_id", [test_session_id])
def test_upload_file(session_id: str) -> None:
    """Tests uploading a file to a session.

    Parameters
    ----------
    session_id : str
        The session to test

    """
    files = [_create_test_file("test_text.txt", "test text\n")]
    params = {"session_id": session_id}
    response = client.post("/upload_file", params=params, files=files)
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "insert_count" in response_json, "'insert_count' not found in response"
    assert isinstance(response_json["insert_count"], int), f"'insert_count' unexpected data type: expected int, got {type(response_json['insert_count'])}"
    assert response_json["insert_count"] > 0, "'insert_count' is not a positive number"


@pytest.mark.parametrize("session_id", [test_session_id])
def test_upload_files(session_id: str) -> None:
    """Tests uploading files to a session.

    Parameters
    ----------
    session_id : str
        The session to test

    """
    files = _create_test_files(["test_text.txt", "test_text2.txt"])
    params = {"session_id": session_id}
    response = client.post("/upload_files", params=params, files=files)
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "results" in response_json, "'results' not found in response"
    assert isinstance(response_json["results"], list), f"'results' unexpected data type: expected list, got {type(response_json['results'])}"
    assert len(response_json["results"]) == len(files), f"'results' unexpected length: expected {len(files)}, got {len(response_json['results'])}"
    for result in response_json["results"]:
        logger.info("Validating file upload result %s", result)
        assert "insert_count" in result, "'insert_count' not found in result"
        assert isinstance(result["insert_count"], int), f"'insert_count' unexpected data type: expected int, got {type(result['insert_count'])}"
        assert result["insert_count"] > 0, "'insert_count' is not a positive number"


@pytest.mark.parametrize("session_id", [test_session_id])
def test_vectordb_upload_ocr(session_id: str) -> None:
    """Tests uploading a file using OCR to a session.

    Parameters
    ----------
    session_id : str
        The session to test

    """
    files = [_create_test_file("test_text.txt", "test text\n")]
    params = {"session_id": session_id}
    response = client.post("/upload_file_ocr", params=params, files=files)
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "insert_count" in response_json, "'insert_count' not found in response"
    assert isinstance(response_json["insert_count"], int), f"'insert_count' unexpected data type: expected int, got {type(response_json['insert_count'])}"
    assert response_json["insert_count"] > 0, "'insert_count' is not a positive number"


@pytest.mark.parametrize("session_id", [test_session_id])
def test_delete_file(session_id: str) -> None:
    """Tests deleting a file from a session.

    Parameters
    ----------
    session_id : str
        The session to test

    """
    filename = "test_text.txt"
    # First upload a test file
    files = [_create_test_file(filename, "test text\n")]
    params = {"session_id": session_id}
    logger.info("Uploading test file")
    upload_response = client.post("/upload_file", params=params, files=files)
    assert upload_response.status_code == http_ok, "Upload failed while testing deletion."

    # Then test deletion
    params = {
        "filename": filename,
        "session_id": session_id,
    }
    response = client.post("/delete_file", params=params)
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "delete_count" in response_json, "'delete_count' not found in response"
    assert isinstance(response_json["delete_count"], int), f"'delete_count' unexpected data type: expected int, got {type(response_json['delete_count'])}"
    assert response_json["delete_count"] > 0, "'delete_count' is not a positive number"


@pytest.mark.parametrize("session_id", [test_session_id])
def test_delete_files(session_id: str) -> None:
    """Tests deleting files from a session.

    Parameters
    ----------
    session_id : str
        The session to test

    """
    filename1 = "test_text.txt"
    filename2 = "test_text2.txt"
    filenames = [filename1, filename2]
    # First upload test files
    files = _create_test_files(filenames)
    params = {"session_id": session_id}
    logger.info("Uploading test files")
    upload_response = client.post("/upload_files", params=params, files=files)
    assert upload_response.status_code == http_ok, "Upload failed while testing deletion."

    # Then test deletion
    params = {"session_id": session_id}
    response = client.post("/delete_files", params=params, json=filenames)
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "results" in response_json, "'results' not found in response"
    assert isinstance(response_json["results"], list), f"'results' unexpected data type: expected list, got {type(response_json['results'])}"
    assert len(response_json["results"]) == len(filenames), f"'results' unexpected length: expected {len(filenames)}, got {len(response_json['results'])}"
    for result in response_json["results"]:
        logger.info("Validating file delete result %s", result)
        assert "delete_count" in result, "'delete_count' not found in result"
        assert isinstance(result["delete_count"], int), f"'delete_count' unexpected data type: expected int, got {type(result['delete_count'])}"
        assert result["delete_count"] > 0, "'delete_count' is not a positive number"


@pytest.mark.parametrize("session_id", [test_session_files_id])
def test_get_session_files(session_id: str) -> None:
    """Tests getting all files from a session.

    Parameters
    ----------
    session_id : str
        The session to test

    """
    params = {"session_id": session_id}
    response = client.post("/get_session_files", params=params)
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "results" in response_json, "'results' not found in response"
    assert isinstance(response_json["results"], list), f"'results' unexpected data type: expected list, got {type(response_json['results'])}"
    assert "file_count" in response_json, "'file_count' not found in response"
    assert isinstance(response_json["file_count"], int), f"'file_count' unexpected data type: expected int, got {type(response_json['file_count'])}"


@pytest.mark.parametrize("session_id", [test_session_id])
def test_delete_session_files(session_id: str) -> None:
    """Tests deleting all files from a session.

    Parameters
    ----------
    session_id : str
        The session to test

    """
    params = {"session_id": session_id}
    response = client.post("/delete_session_files", params=params)
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)


@pytest.mark.parametrize("site", [test_site])
def test_upload_site(site: str) -> None:
    """Tests scraping and uploading a website.

    Parameters
    ----------
    site : str
        The site to scrape and upload

    """
    test_description = "OpenProBono website uploaded for testing."
    test_collection_name = "DevTest"
    params = {
        "site": site,
        "collection_name": test_collection_name,
        "description": test_description,
    }
    response = client.post("/upload_site", params=params)
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "results" in response_json, "'results' not found in response"
    assert isinstance(response_json["results"], list), f"'results' unexpected data type: expected list, got {type(response_json['results'])}"


@pytest.mark.parametrize("query", [test_query])
def test_search_opinions(query: str) -> None:
    """Tests opinion search.

    Parameters
    ----------
    query : str
        The search query

    """
    data = {"query": query}
    response = client.post("/search_opinions", json=data)
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "results" in response_json, "'results' not found in response"
    assert isinstance(response_json["results"], list), f"'results' unexpected data type: expected list, got {type(response_json['results'])}"


@pytest.mark.parametrize("opinion_id", [test_opinion_id])
def test_get_opinion_summary(opinion_id: int) -> None:
    """Tests getting an opinion summary.

    Parameters
    ----------
    opinion_id : int
        The opinion to test

    """
    data = {"opinion_id": opinion_id}
    response = client.get("/get_opinion_summary", params=data)
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "result" in response_json, "'result' not found in response"
    assert isinstance(response_json["result"], str), f"'result' unexpected data type: expected str, got {type(response_json['result'])}"


def test_get_opinion_count() -> None:
    """Tests getting the current opinion count."""
    response = client.get("/get_opinion_count")
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
    assert "opinion_count" in response_json, "'opinion_count' not found in response"
    assert isinstance(response_json["opinion_count"], int), f"'opinion_count' unexpected data type: expected int, got {type(response_json['opinion_count'])}"


@pytest.mark.parametrize("opinion_id", [test_opinion_id])
def test_opinion_feedback(opinion_id: int) -> None:
    """Tests leaving feedback for an opinion.

    Parameters
    ----------
    opinion_id : str
        The opinion receiving feedback

    """
    from app.models import OpinionFeedback
    opinion_feedback = OpinionFeedback(
        feedback_text="test feedback",
        opinion_id=opinion_id,
    )
    response = client.post("/opinion_feedback", json=opinion_feedback.model_dump())
    response_json = response.json()
    logger.info(response_json)
    response_test(response.status_code, response_json)
