import os
import unittest

from fastapi.testclient import TestClient

from app.logger import setup_logger
import app.main as main

logger = setup_logger()

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

    def test_session_feedback(self):
        from app.models import InitializeSessionChat, SessionFeedback
        bot_id = "default_bot"
        test_initialize_session = InitializeSessionChat(
            bot_id=bot_id,
            message="hi",
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

        test_submit_feedback = SessionFeedback(feedback_text="test feedback",
                                session_id=response_json["session_id"])
        response = client.post(
            "/session_feedback", json=test_submit_feedback.model_dump(),
        )
        response = response.json()
        assert response_json["message"] == "Success"

    def test_opinion_feedback(self):
        from app.models import OpinionFeedback

        test_opinion_feedback = OpinionFeedback(
            feedback_text="test feedback2",
            opinion_id=522541,
        )
        response = client.post(
            "/opinion_feedback", json=test_opinion_feedback.model_dump(),
        )
        response_json = response.json()
        logger.info(response_json)
        assert response_json["message"] == "Success"


