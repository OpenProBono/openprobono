import unittest

from fastapi.testclient import TestClient

import main


class UUIDTests(unittest.TestCase):
    uuid = main.get_uuid_id()

    def test_uuid_type(self):
        self.assertTrue(isinstance(self.uuid, str))

    def test_uuid_length(self):
        self.assertEqual(len(self.uuid), 36)


class ApiKeyTests(unittest.TestCase):
    def test_validAdminKey(self):
        key = 'xyz'
        self.assertTrue(main.admin_check(key))
        self.assertTrue(main.api_key_check(key))

    def test_validKey(self):
        key = 'gradio'
        self.assertFalse(main.admin_check(key))
        self.assertTrue(main.api_key_check(key))

    def test_invalidKey(self):
        key = 'abc'
        self.assertFalse(main.admin_check(key))
        self.assertFalse(main.api_key_check(key))


# api methods


client = TestClient(main.api)


class ApiTests(unittest.TestCase):
    # in botsvm12_lang
    test_bot_vdb_id = "29396e1e-d722-4f81-b2a7-c5ee6cdba1ed"
    test_bot_search_id = "1a41e424-0434-457e-9a9e-2b3c7ed0dccd"
    # in conversationsvm12_lang
    test_session_vdb_id = "be45df8f-0f4e-466a-9c7b-7ab56b5837c8"

    def test_read_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "API is alive"})

    def test_create_bot_vdb(self):
        from models import BotRequest, VDBTool
        vdb_tool = VDBTool(
            collection_name="USCode",
            k=4,
        )
        test_bot_request = BotRequest(api_key='xyz', vdb_tools=[vdb_tool])
        response = client.post("/create_bot", json=test_bot_request.model_dump())
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        print(response_json)
        self.assertEqual(response_json["message"], "Success")
        self.assertTrue("bot_id" in response_json)
        self.assertTrue(isinstance(response_json["bot_id"], str))
        self.assertEqual(len(response_json["bot_id"]), 36)

    def test_create_bot_search(self):
        from models import BotRequest, SearchTool
        search_tool = SearchTool(
            name="government-search",
            prompt="Useful for when you need to answer questions or find resources about government and laws.",
            prefix="site:*.gov | site:*.edu | site:*scholar.google.com",
        )
        test_bot_request = BotRequest(api_key='xyz', search_tools=[search_tool])
        response = client.post("/create_bot", json=test_bot_request.model_dump())
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        print(response_json)
        self.assertEqual(response_json["message"], "Success")
        self.assertTrue("bot_id" in response_json)
        self.assertTrue(isinstance(response_json["bot_id"], str))
        self.assertEqual(len(response_json["bot_id"]), 36)

    # TODO: determine if session tests should be replicated using a bot with search tool only, or both
    def test_init_session(self):
        from models import InitializeSession
        test_initialize_session = InitializeSession(api_key='xyz', bot_id=self.test_bot_vdb_id, message='test')
        response = client.post("/initialize_session_chat", json=test_initialize_session.model_dump())
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        print(response_json)
        self.assertEqual(response_json["message"], "Success")
        self.assertTrue("bot_id" in response_json)
        self.assertTrue(isinstance(response_json["bot_id"], str))
        self.assertEqual(len(response_json["bot_id"]), 36)
        self.assertTrue("output" in response_json)
        self.assertTrue(isinstance(response_json["output"], str))
        self.assertTrue("session_id" in response_json)
        self.assertTrue(isinstance(response_json["session_id"], str))
        self.assertEqual(len(response_json["session_id"]), 36)

    def test_chat_session(self):
        from models import ChatBySession
        test_chat_session = ChatBySession(api_key='xyz', session_id=self.test_session_vdb_id, message='test')
        response = client.post("/chat_session", json=test_chat_session.model_dump())
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(response_json["message"], "Success")
        self.assertTrue("bot_id" in response_json)
        self.assertTrue(isinstance(response_json["bot_id"], str))
        self.assertEqual(len(response_json["bot_id"]), 36)
        self.assertTrue("output" in response_json)
        self.assertTrue(isinstance(response_json["output"], str))
        self.assertTrue("session_id" in response_json)
        self.assertTrue(isinstance(response_json["session_id"], str))
        self.assertEqual(len(response_json["session_id"]), 36)

    def test_fetch_session(self):
        from models import FetchSession
        test_fetch_session = FetchSession(api_key='xyz', session_id=self.test_session_vdb_id)
        response = client.post("/fetch_session", json=test_fetch_session.model_dump())
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(response_json["message"], "Success")
        self.assertTrue("bot_id" in response_json)
        self.assertTrue(isinstance(response_json["bot_id"], str))
        self.assertEqual(len(response_json["bot_id"]), 36)
        self.assertTrue("session_id" in response_json)
        self.assertTrue(isinstance(response_json["session_id"], str))
        self.assertEqual(len(response_json["session_id"]), 36)
        self.assertTrue("history" in response_json)
        self.assertTrue(isinstance(response_json["history"], list))
