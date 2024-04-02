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
    #m12_db
    test_bot_vdb_id = "f273054b-2831-45b1-a2ba-af683fe9a102"
    test_session_vdb_id = "42e31a0c-450d-4210-8af1-4ac19ccc6180"

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

    def test_courtlistener_bot(self):
        from models import BotRequest, InitializeSession

        search_tool = {
            "name": "courtlistener",
            "method": "courtlistener",
            "txt": "",
            "prompt": "Tool used to search courtlistener for legal cases and opinions",
        }
        test_bot_request = BotRequest(api_key="xyz", search_tools=[search_tool])
        response = client.post("/create_bot", json=test_bot_request.model_dump())
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(response_json["message"], "Success")
        self.assertTrue("bot_id" in response_json)
        self.assertTrue(isinstance(response_json["bot_id"], str))
        self.assertEqual(len(response_json["bot_id"]), 36)
        bot_id = response_json["bot_id"]
        test_initialize_session = InitializeSession(
            api_key="xyz", bot_id=bot_id, message="find me cases related to covid",
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


    # TODO: determine if session tests should be replicated using 
    # a bot with search tool only, or both
    def test_init_session(self):
        from models import InitializeSession

        test_initialize_session = InitializeSession(
            api_key="xyz", bot_id=self.test_bot_vdb_id, message="test",
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

    def test_empty_init(self):
        from models import InitializeSession

        test_initialize_session = InitializeSession(
            api_key="xyz", bot_id=self.test_bot_vdb_id, message="",
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
        self.assertTrue(response_json["output"] == "Hi, how can I assist you today?")
        self.assertTrue("session_id" in response_json)
        print(response_json["session_id"])
        self.assertTrue(isinstance(response_json["session_id"], str))
        self.assertEqual(len(response_json["session_id"]), 36)

    def test_init_and_continue_session(self):
        from models import ChatBySession, InitializeSession

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

        test_cont_session = ChatBySession(
            api_key="xyz", session_id=response_json["session_id"], message="test2",
        )
        response = client.post("/chat_session", json=test_cont_session.model_dump())
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
