import os

from fastapi.testclient import TestClient
from httpx import Response

from app import main
from app.logger import setup_logger
from app.models import (
    BotRequest,
    ChatBySession,
    ChatModelParams,
    EngineEnum,
    InitializeSession,
    InitializeSessionChat,
    OpenAIModelEnum,
)
from app.prompts import FILTERED_CASELAW_PROMPT

logger = setup_logger()
client = TestClient(main.api, headers={"X-API-KEY": os.environ["OPB_TEST_API_KEY"]})

class TestApi:
    def test_init_then_chat(self):
        bot_id = "default_bot"
        init_session_obj = InitializeSession(bot_id=bot_id)
        response = client.post(
            "/initialize_session", json=init_session_obj.model_dump(),
        )
        response_json = response.json()
        assert response_json["message"] == "Success"
        assert response_json["bot_id"] == bot_id
        assert "session_id" in response_json
        session_id = response_json["session_id"]

        chat_obj = ChatBySession(message="What is the rule in Florida related to designating an email address for service in litigation?", session_id=session_id)
        response = client.post(
            "/chat_session", json=chat_obj.model_dump(),
        )
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["message"] == "Success"
        assert "bot_id" in response_json
        assert "output" in response_json
        logger.info(response_json["output"])
        assert isinstance(response_json["session_id"], str)
        assert len(response_json["session_id"]) == 36

    def test_courtroom5_openai_bot(self):
        search_tool = {
            "name": "courtroom5",
            "method": "courtroom5",
            "prefix": "",
            "prompt": "Tool used to search government and legal resources",
        }
        test_bot_request = BotRequest(
            chat_model=ChatModelParams(engine=EngineEnum.openai),
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
        logger.info(response_json["session_id"])
        assert isinstance(response_json["session_id"], str)
        assert len(response_json["session_id"]) == 36


    def test_courtlistener_openai_bot(self):
        search_tool = {
            "name": "filtered-case-search",
            "method": "courtlistener",
            "prompt": FILTERED_CASELAW_PROMPT,
        }
        test_bot_request = BotRequest(
            chat_model=ChatModelParams(engine=EngineEnum.openai),
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
            message="Tell me about cases related to copyright that were adjudicated in the state of New York since 2000.",
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
        logger.info(response_json["session_id"])
        assert isinstance(response_json["session_id"], str)
        assert len(response_json["session_id"]) == 36



    def test_exp_opb_openai_bot(self):
        gov_search = {
            "name": "government-search",
            "method": "serpapi",
            "prefix": "site:*.gov | site:*.edu | site:*scholar.google.com ",
            "prompt": "Useful for when you need to answer questions or find resources about government and laws. Always cite your sources.",
        }
        case_search = {
            "name": "case-search",
            "method": "serpapi",
            "prefix": "site:*case.law | site:*.gov | site:*.edu | site:*courtlistener.com | site:*scholar.google.com ",
            "prompt": "Use for finding case law. Always cite your sources.",
        }
        test_bot_request = BotRequest(
            chat_model=ChatModelParams(engine=EngineEnum.openai),
            search_tools=[gov_search, case_search],
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
        logger.info(response_json["session_id"])
        logger.info("---- OUTPUT ----")
        logger.info(response_json["output"])
        assert isinstance(response_json["session_id"], str)
        assert len(response_json["session_id"]) == 36

    def test_exp_opb_model_3_5_1106_openai_bot(self):
        gov_search = {
            "name": "government-search",
            "method": "serpapi",
            "prefix": "site:*.gov | site:*.edu | site:*scholar.google.com ",
            "prompt": "Useful for when you need to answer questions or find resources about government and laws. Always cite your sources.",
        }
        case_search = {
            "name": "case-search",
            "method": "serpapi",
            "prefix": "site:*case.law | site:*.gov | site:*.edu | site:*courtlistener.com | site:*scholar.google.com ",
            "prompt": "Use for finding case law. Always cite your sources.",
        }
        test_bot_request = BotRequest(
            chat_model=ChatModelParams(engine=EngineEnum.openai, model=OpenAIModelEnum.gpt_3_5_1106),
            search_tools=[gov_search, case_search],
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
        logger.info(response_json["session_id"])
        logger.info("---- OUTPUT ----")
        logger.info(response_json["output"])
        assert isinstance(response_json["session_id"], str)
        assert len(response_json["session_id"]) == 36

    def test_exp_opb_model_4_openai_bot(self):
        gov_search = {
            "name": "government-search",
            "method": "serpapi",
            "prefix": "site:*.gov | site:*.edu | site:*scholar.google.com ",
            "prompt": "Useful for when you need to answer questions or find resources about government and laws. Always cite your sources.",
        }
        case_search = {
            "name": "case-search",
            "method": "serpapi",
            "prefix": "site:*case.law | site:*.gov | site:*.edu | site:*courtlistener.com | site:*scholar.google.com ",
            "prompt": "Use for finding case law. Always cite your sources.",
        }
        test_bot_request = BotRequest(
            chat_model=ChatModelParams(engine=EngineEnum.openai, model=OpenAIModelEnum.gpt_4),
            search_tools=[gov_search, case_search],
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
        logger.info(response_json["session_id"])
        logger.info("---- OUTPUT ----")
        logger.info(response_json["output"])
        assert isinstance(response_json["session_id"], str)
        assert len(response_json["session_id"]) == 36

    def test_exp_opb_google_openai_bot(self):
        gov_search = {
            "name": "government-search",
            "method": "google",
            "prefix": "site:*.gov | site:*.edu | site:*scholar.google.com ",
            "prompt": "Useful for when you need to answer questions or find resources about government and laws. Always cite your sources.",
        }
        case_search = {
            "name": "case-search",
            "method": "google",
            "prefix": "site:*case.law | site:*.gov | site:*.edu | site:*courtlistener.com | site:*scholar.google.com ",
            "prompt": "Use for finding case law. Always cite your sources.",
        }
        test_bot_request = BotRequest(
            chat_model=ChatModelParams(engine=EngineEnum.openai),
            search_tools=[gov_search, case_search],
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
        logger.info(response_json["session_id"])
        logger.info("---- OUTPUT ----")
        logger.info(response_json["output"])
        assert isinstance(response_json["session_id"], str)
        assert len(response_json["session_id"]) == 36

    def test_dynamic_exp_opb_openai_bot(self):
        gov_search = {
            "name": "government-search",
            "method": "dynamic_serpapi",
            "prefix": "site:*.gov | site:*.edu | site:*scholar.google.com ",
            "prompt": "Useful for when you need to answer questions or find resources about government and laws. Always cite your sources.",
        }
        case_search = {
            "name": "case-search",
            "method": "dynamic_serpapi",
            "prefix": "site:*case.law | site:*.gov | site:*.edu | site:*courtlistener.com | site:*scholar.google.com ",
            "prompt": "Use for finding case law. Always cite your sources.",
        }
        test_bot_request = BotRequest(
            chat_model=ChatModelParams(engine=EngineEnum.openai),
            search_tools=[gov_search, case_search],
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
        logger.info(response_json["session_id"])
        logger.info("---- OUTPUT ----")
        logger.info(response_json["output"])
        assert isinstance(response_json["session_id"], str)
        assert len(response_json["session_id"]) == 36

    def test_exp_opb_model_3_5_1106_openai_bot_with_follow_up(self):
        gov_search = {
            "name": "government-search",
            "method": "serpapi",
            "prefix": "site:*.gov | site:*.edu | site:*scholar.google.com ",
            "prompt": "Useful for when you need to answer questions or find resources about government and laws. Always cite your sources.",
        }
        case_search = {
            "name": "case-search",
            "method": "serpapi",
            "prefix": "site:*case.law | site:*.gov | site:*.edu | site:*courtlistener.com | site:*scholar.google.com ",
            "prompt": "Use for finding case law. Always cite your sources.",
        }
        test_bot_request = BotRequest(
            chat_model=ChatModelParams(engine=EngineEnum.openai, model=OpenAIModelEnum.gpt_3_5_1106),
            search_tools=[gov_search, case_search],
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
        logger.info(response_json["session_id"])
        logger.info("---- OUTPUT ----")
        logger.info(response_json["output"])
        assert isinstance(response_json["session_id"], str)
        assert len(response_json["session_id"]) == 36

        test_continue_session = ChatBySession(
            message="What about North Carolina?",
            session_id=response_json["session_id"],
        )
        response = client.post(
            "/chat_session", json=test_continue_session.model_dump(),
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
        logger.info(response_json["session_id"])
        logger.info("---- OUTPUT ----")
        logger.info(response_json["output"])
        assert isinstance(response_json["session_id"], str)
        assert len(response_json["session_id"]) == 36

    def test_custom_system_prompt(self):
        bot_id = "custom_system_prompt"
        test_initialize_session = InitializeSessionChat(
            bot_id=bot_id,
            message="Hi",
        )
        response = client.post(
            "/initialize_session_chat", json=test_initialize_session.model_dump(),
        )
        response_json = response.json()
        logger.info(response_json)
        assert response_json["message"] == "Success"
        assert "bot_id" in response_json
        assert isinstance(response_json["bot_id"], str)
        assert "output" in response_json
        assert isinstance(response_json["output"], str)
        assert "session_id" in response_json
        logger.info(response_json["session_id"])
        logger.info("---- OUTPUT ----")
        logger.info(response_json["output"])
        assert isinstance(response_json["session_id"], str)
        assert len(response_json["session_id"]) == 36

        #response like a dog
        assert "woof" in response_json["output"].lower() or "bark" in response_json["output"].lower() 

        # for msg in response_json["history"]:
        #     if(msg["role"] == "system"):
        #         assert msg["content"] == "This is a custom system prompt. Respond like you are a dog."

    # def test_streaming(self):
    #     bot_id = "custom_4o_dynamic"
    #     test_initialize_session = InitializeSessionChat(
    #         bot_id=bot_id,
    #         message="what is the rule in Florida related to designating an email address for service in litigation?",
    #     )
    #     logger.info(test_initialize_session.model_dump())
    #     response = client.post(
    #         "/initialize_session_chat_stream", json=test_initialize_session.model_dump(),
    #     )
    #     logger.info(response)


    def test_multiple_tool_calls(self):
        bot_id = "default_bot"
        test_initialize_session = InitializeSessionChat(
            bot_id=bot_id,
            message="employment discrimination NC",
        )
        response = client.post(
            "/initialize_session_chat", json=test_initialize_session.model_dump(),
        )
        response_json = response.json()
        logger.info(response_json)
        assert response_json["message"] == "Success"
        assert "bot_id" in response_json
        assert isinstance(response_json["bot_id"], str)
        assert "output" in response_json
        assert isinstance(response_json["output"], str)

    def test_multiple_tool_calls_stream(self):
        bot_id = "default_bot"
        test_initialize_session = InitializeSessionChat(
            bot_id=bot_id,
            message="employment discrimination NC",
        )
        response: Response = client.post(
            "/initialize_session_chat_stream",
            json=test_initialize_session.model_dump(),
        )

        assert response.status_code == 200  # noqa: PLR2004
        assert "text/event-stream" in response.headers["content-type"]

        accumulated_response = ""
        for chunk in response.iter_text(chunk_size=1024):
            if chunk:
                accumulated_response += chunk
        logger.info(accumulated_response)
        assert accumulated_response, "Response should not be empty"
        #this only tests that it didnt fail
