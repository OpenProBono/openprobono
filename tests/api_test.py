import unittest

from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.api)

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
    
    def test_read_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "API is alive"})

    def test_fetch_session(self):
        from app.models import FetchSession
        test_fetch_session = FetchSession(api_key='axyz', session_id=self.test_session_vdb_id)
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