import os
import unittest

from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.api, headers={"X-API-KEY": os.environ["OPB_TEST_API_KEY"]})

API_KEY = os.environ["OPB_TEST_API_KEY"]

class ApiTests(unittest.TestCase):
    test_session_id = "test_session"

    def test_read_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "API is alive"})

    def test_fetch_session(self):
        from app.models import FetchSession
        test_fetch_session = FetchSession(session_id=self.test_session_id)
        response = client.post("/fetch_session", json=test_fetch_session.model_dump())
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(response_json["message"], "Success")
        self.assertTrue("bot_id" in response_json)
        self.assertTrue(isinstance(response_json["bot_id"], str))

        self.assertTrue("session_id" in response_json)
        self.assertTrue(isinstance(response_json["session_id"], str))
        self.assertEqual(response_json["session_id"], self.test_session_id)
        self.assertTrue("history" in response_json)
        self.assertTrue(isinstance(response_json["history"], list))