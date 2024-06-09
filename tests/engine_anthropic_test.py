import unittest
import os
from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.api)

API_KEY = os.environ["OPB_TEST_API_KEY"]

class ApiTests(unittest.TestCase):
    # #_courtroom5_v1
    # test_bot_vdb_id = "37394099-4c05-474f-8a35-28bcc4dc68ca"
    # test_session_vdb_id = "0c393d97-a70e-4b2f-b3e8-5c4326e6e10c"
    # in botsvm12_lang
    test_bot_vdb_id = "2c580482-046e-4118-87dd-4f3abeb391b2"
    test_bot_search_id = "7ab97742-ec5e-4663-862e-019f57ced68a"
    # in conversationsvm12_lang
    # 23c66d05-a5c1-4b30-af3a-9c22536d0e49
    test_session_vdb_id = "c703b319-41be-4e7d-b2a0-e546f3bfc49e"

    def test_courtroom5_anthropic_bot(self):
        from app.models import BotRequest, InitializeSession, ChatModelParams, EngineEnum, AnthropicModelEnum

        search_tool = {
            "name": "courtroom5",
            "method": "courtroom5",
            "prefix": "",
            "prompt": "Tool used to search government and legal resources",
        }
        test_bot_request = BotRequest(
            api_key=API_KEY,
            chat_model=ChatModelParams(engine=EngineEnum.anthropic, model=AnthropicModelEnum.claude_3_opus),
            search_tools=[search_tool],
        )
        response = client.post("/create_bot", json=test_bot_request.model_dump())
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(response_json["message"], "Success")
        self.assertTrue("bot_id" in response_json)
        self.assertTrue(isinstance(response_json["bot_id"], str))
        self.assertEqual(len(response_json["bot_id"]), 36)
        bot_id = response_json["bot_id"]
        test_initialize_session = InitializeSession(
            api_key=API_KEY,
            bot_id=bot_id,
            message="What is the rule in Florida related to designating an "
                    "email address for service in litigation?",
        )
        response = client.post(
            "/initialize_session_chat", json=test_initialize_session.model_dump(),
        )
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(response_json["message"], "Success")
        self.assertTrue("bot_id" in response_json)
        self.assertTrue(isinstance(response_json["bot_id"], str))
        self.assertEqual(len(response_json["bot_id"]), 36)
        self.assertTrue("output" in response_json)
        self.assertTrue(isinstance(response_json["output"], str))
        self.assertTrue("session_id" in response_json)
        print(response_json["session_id"])
        self.assertTrue(isinstance(response_json["session_id"], str))
        self.assertEqual(len(response_json["session_id"]), 36)
