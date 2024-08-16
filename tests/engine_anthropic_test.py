import os

from fastapi.testclient import TestClient

from app import main
from app.models import (
    AnthropicModelEnum,
    BotRequest,
    ChatModelParams,
    EngineEnum,
    InitializeSessionChat,
)
from app.prompts import FILTERED_CASELAW_PROMPT

client = TestClient(main.api, headers={"X-API-KEY": os.environ["OPB_TEST_API_KEY"]})

def test_courtroom5_anthropic_bot() -> None:
    search_tool = {
        "name": "courtroom5",
        "method": "courtroom5",
        "prefix": "",
        "prompt": "Tool used to search government and legal resources",
    }
    test_bot_request = BotRequest(
        chat_model=ChatModelParams(
            engine=EngineEnum.anthropic,
            model=AnthropicModelEnum.claude_3_5_sonnet,
        ),
        search_tools=[search_tool],
    )
    response = client.post("/create_bot", json=test_bot_request.model_dump())
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["message"]== "Success"
    assert "bot_id" in response_json
    assert isinstance(response_json["bot_id"], str)
    assert len(response_json["bot_id"]) == 36

    bot_id = response_json["bot_id"]
    test_initialize_session = InitializeSessionChat(
        bot_id=bot_id,
        message="What is the rule in Florida related to designating an "
                "email address for service in litigation?",
    )
    response = client.post(
        "/initialize_session_chat", json=test_initialize_session.model_dump(),
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["message"] == "Success"
    assert "bot_id" in response_json
    assert isinstance(response_json["bot_id"], str)
    assert len(response_json["bot_id"]) == 36
    assert "output" in response_json
    assert isinstance(response_json["output"], str)
    assert "session_id" in response_json
    print(response_json["session_id"])
    assert isinstance(response_json["session_id"], str)
    assert len(response_json["session_id"]) == 36

def test_courtlistener_anthropic_bot() -> None:
        search_tool = {
            "name": "filtered-case-search",
            "method": "courtlistener",
            "prompt": FILTERED_CASELAW_PROMPT,
        }
        chat_model = ChatModelParams(
            engine=EngineEnum.anthropic,
            model=AnthropicModelEnum.claude_3_5_sonnet,
        )
        test_bot_request = BotRequest(
            chat_model=chat_model,
            search_tools=[search_tool],
        )
        response = client.post("/create_bot", json=test_bot_request.model_dump())
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["message"] == "Success"
        assert "bot_id" in response_json
        assert isinstance(response_json["bot_id"], str)
        assert len(response_json["bot_id"]) == 36
        bot_id = response_json["bot_id"]
        test_initialize_session = InitializeSessionChat(
            bot_id=bot_id,
            message="Tell me about cases related to copyright that were adjudicated in the state of California since 2000.",
        )
        response = client.post(
            "/initialize_session_chat", json=test_initialize_session.model_dump(),
        )
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["message"] == "Success"
        assert "bot_id" in response_json
        assert isinstance(response_json["bot_id"], str)
        assert len(response_json["bot_id"]) == 36
        assert "output" in response_json
        assert isinstance(response_json["output"], str)
        assert "session_id" in response_json
        print(response_json["session_id"])
        assert isinstance(response_json["session_id"], str)
        assert len(response_json["session_id"]) == 36
