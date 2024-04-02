import unittest

from fastapi.testclient import TestClient

import main


class UUIDTests(unittest.TestCase):
    import models

    uuid = models.get_uuid_id()

    def test_uuid_type(self):
        self.assertTrue(isinstance(self.uuid, str))

    def test_uuid_length(self):
        self.assertEqual(len(self.uuid), 36)


client = TestClient(main.api)


class ApiTests(unittest.TestCase):
    test_bot_vdb_id = "16322f79-abb0-40da-82d9-54a506514a74"
    #test_bot_search_id = "15060a2d-fd2b-4f4a-9bc9-d354b6bcb9b4" botsvm12_lang/aa644d04-9292-4a1b-8026-98c65008c7ef
    # in conversationsvf23_db
    test_session_vdb_id = "d8e03899-a586-4365-b860-6b406070a781"

    def test_read_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "API is alive"})

    def test_create_bot_vdb_query_US(self):
        from milvusdb import US
        from models import BotRequest, VDBTool

        vdb_tool = VDBTool(name="query", collection_name=US, k=4)
        test_bot_request = BotRequest(api_key="xyz", vdb_tools=[vdb_tool])
        response = client.post("/create_bot", json=test_bot_request.model_dump())
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(response_json["message"], "Success")
        self.assertTrue("bot_id" in response_json)
        self.assertTrue(isinstance(response_json["bot_id"], str))
        self.assertEqual(len(response_json["bot_id"]), 36)

    def test_create_bot_search(self):
        from models import BotRequest

        search_tool = {
            "name": "google_search",
            "txt": "",
            "prompt": "Tool used to search the web, useful for current events or facts",
        }
        test_bot_request = BotRequest(api_key="xyz", search_tools=[search_tool])
        response = client.post("/create_bot", json=test_bot_request.model_dump())
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(response_json["message"], "Success")
        self.assertTrue("bot_id" in response_json)
        self.assertTrue(isinstance(response_json["bot_id"], str))
        self.assertEqual(len(response_json["bot_id"]), 36)

    # TODO: determine if session tests should be replicated using 
    # a bot with search tool only, or both
    def test_init_session(self):
        from models import InitializeSession

        test_initialize_session = InitializeSession(
            api_key="xyz", bot_id=self.test_bot_vdb_id, message="test"
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

    def test_chat_session(self):
        from models import ChatBySession

        test_chat_session = ChatBySession(
            api_key="xyz", session_id=self.test_session_vdb_id, message="test"
        )
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

        test_fetch_session = FetchSession(
            api_key="xyz", session_id=self.test_session_vdb_id,
        )
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


# TODO: more unit tests
if __name__ == "__main__":
    unittest.main()
