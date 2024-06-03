# import unittest
# import os

# from fastapi.testclient import TestClient

# import main

# client = TestClient(main.api)

# API_KEY = os.environ["OPB_TEST_API_KEY"]

# class ApiTests(unittest.TestCase):
#     # #_courtroom5_v1
#     # test_bot_vdb_id = "37394099-4c05-474f-8a35-28bcc4dc68ca"
#     # test_session_vdb_id = "0c393d97-a70e-4b2f-b3e8-5c4326e6e10c"
#     # in botsvm12_lang
#     test_bot_vdb_id = "00bf175d-e360-42bb-a58d-4896086ef215"
#     test_bot_search_id = "7ab97742-ec5e-4663-862e-019f57ced68a"
#     # in conversationsvm12_lang
#     # 23c66d05-a5c1-4b30-af3a-9c22536d0e49
#     test_session_vdb_id = "c703b319-41be-4e7d-b2a0-e546f3bfc49e"

#     def test_create_bot_vdb_query_US(self):
#         from models import BotRequest, VDBTool

#         vdb_tool = VDBTool(collection_name="USCode", k=4)
#         test_bot_request = BotRequest(api_key=API_KEY, vdb_tools=[vdb_tool])
#         response = client.post("/create_bot", json=test_bot_request.model_dump())
#         self.assertEqual(response.status_code, 200)
#         response_json = response.json()
#         self.assertEqual(response_json["message"], "Success")
#         self.assertTrue("bot_id" in response_json)
#         self.assertTrue(isinstance(response_json["bot_id"], str))
#         self.assertEqual(len(response_json["bot_id"]), 36)

#     def test_create_bot_serpapi_search(self):
#         from models import BotRequest

#         search_tool = {
#             "method": "serpapi",
#             "name": "google_search",
#             "prefix": "",
#             "prompt": "Tool used to search the web, useful for current events or facts",
#         }
#         test_bot_request = BotRequest(api_key=API_KEY, search_tools=[search_tool])
#         response = client.post("/create_bot", json=test_bot_request.model_dump())
#         self.assertEqual(response.status_code, 200)
#         response_json = response.json()
#         print(response_json)
#         self.assertEqual(response_json["message"], "Success")
#         self.assertTrue("bot_id" in response_json)
#         self.assertTrue(isinstance(response_json["bot_id"], str))
#         self.assertEqual(len(response_json["bot_id"]), 36)

#     def test_create_bot_google_search(self):
#         from models import BotRequest

#         search_tool = {
#             "method": "google",
#             "name": "google_search",
#             "prefix": "",
#             "prompt": "Tool used to search the web, useful for current events or facts",
#         }
#         test_bot_request = BotRequest(api_key=API_KEY, search_tools=[search_tool])
#         response = client.post("/create_bot", json=test_bot_request.model_dump())
#         self.assertEqual(response.status_code, 200)
#         response_json = response.json()
#         self.assertEqual(response_json["message"], "Success")
#         self.assertTrue("bot_id" in response_json)
#         self.assertTrue(isinstance(response_json["bot_id"], str))
#         self.assertEqual(len(response_json["bot_id"]), 36)

#     def test_courtroom5_bot(self):
#         from models import BotRequest, InitializeSession

#         search_tool = {
#             "name": "courtroom5",
#             "method": "courtroom5",
#             "prefix": "",
#             "prompt": "Tool used to search government and legal resources",
#         }
#         test_bot_request = BotRequest(api_key=API_KEY, search_tools=[search_tool])
#         response = client.post("/create_bot", json=test_bot_request.model_dump())
#         self.assertEqual(response.status_code, 200)
#         response_json = response.json()
#         self.assertEqual(response_json["message"], "Success")
#         self.assertTrue("bot_id" in response_json)
#         self.assertTrue(isinstance(response_json["bot_id"], str))
#         self.assertEqual(len(response_json["bot_id"]), 36)
#         bot_id = response_json["bot_id"]
#         test_initialize_session = InitializeSession(
#             api_key=API_KEY,
#             bot_id=bot_id,
#             message="What is the rule in Florida related to designating an "
#                     "email address for service in litigation?",
#         )
#         response = client.post(
#             "/initialize_session_chat", json=test_initialize_session.model_dump(),
#         )
#         self.assertEqual(response.status_code, 200)
#         response_json = response.json()
#         self.assertEqual(response_json["message"], "Success")
#         self.assertTrue("bot_id" in response_json)
#         self.assertTrue(isinstance(response_json["bot_id"], str))
#         self.assertEqual(len(response_json["bot_id"]), 36)
#         self.assertTrue("output" in response_json)
#         self.assertTrue(isinstance(response_json["output"], str))
#         self.assertTrue("session_id" in response_json)
#         print(response_json["session_id"])
#         self.assertTrue(isinstance(response_json["session_id"], str))
#         self.assertEqual(len(response_json["session_id"]), 36)

#     # def test_dynamic_courtroom5_bot(self):
#     #     from models import BotRequest, InitializeSession

#     #     search_tool = {
#     #         "name": "dynamic_courtroom5",
#     #         "method": "dynamic_courtroom5",
#     #         "prefix": "",
#     #         "prompt": "Tool used to search government and legal resources",
#     #     }
#     #     test_bot_request = BotRequest(api_key=API_KEY, search_tools=[search_tool])
#     #     response = client.post("/create_bot", json=test_bot_request.model_dump())
#     #     self.assertEqual(response.status_code, 200)
#     #     response_json = response.json()
#     #     self.assertEqual(response_json["message"], "Success")
#     #     self.assertTrue("bot_id" in response_json)
#     #     self.assertTrue(isinstance(response_json["bot_id"], str))
#     #     self.assertEqual(len(response_json["bot_id"]), 36)
#     #     bot_id = response_json["bot_id"]
#     #     test_initialize_session = InitializeSession(
#     #         api_key=API_KEY, bot_id=bot_id, 
#     #         message="What is the rule in Florida related to designating an "
#     #                 "email address for service in litigation?",
#     #     )
#     #     response = client.post(
#     #         "/initialize_session_chat", json=test_initialize_session.model_dump(),
#     #     )
#     #     self.assertEqual(response.status_code, 200)
#     #     response_json = response.json()
#     #     self.assertEqual(response_json["message"], "Success")
#     #     self.assertTrue("bot_id" in response_json)
#     #     self.assertTrue(isinstance(response_json["bot_id"], str))
#     #     self.assertEqual(len(response_json["bot_id"]), 36)
#     #     self.assertTrue("output" in response_json)
#     #     self.assertTrue(isinstance(response_json["output"], str))
#     #     self.assertTrue("session_id" in response_json)
#     #     print(response_json["session_id"])
#     #     self.assertTrue(isinstance(response_json["session_id"], str))
#     #     self.assertEqual(len(response_json["session_id"]), 36)

#     # def test_dynamic_serpapi_bot(self):
#     #     from models import BotRequest, InitializeSession

#     #     search_tool = {
#     #         "name": "dynamic_serpapi",
#     #         "method": "dynamic_serpapi",
#     #         "prefix": "",
#     #         "prompt": "Tool used to search government and legal resources",
#     #     }
#     #     test_bot_request = BotRequest(api_key=API_KEY, search_tools=[search_tool])
#     #     response = client.post("/create_bot", json=test_bot_request.model_dump())
#     #     self.assertEqual(response.status_code, 200)
#     #     response_json = response.json()
#     #     self.assertEqual(response_json["message"], "Success")
#     #     self.assertTrue("bot_id" in response_json)
#     #     self.assertTrue(isinstance(response_json["bot_id"], str))
#     #     self.assertEqual(len(response_json["bot_id"]), 36)
#     #     bot_id = response_json["bot_id"]
#     #     test_initialize_session = InitializeSession(
#     #         api_key=API_KEY, bot_id=bot_id,
#     #         message="What is the rule in Florida related to designating an "
#     #                 "email address for service in litigation?",
#     #     )
#     #     response = client.post(
#     #         "/initialize_session_chat", json=test_initialize_session.model_dump(),
#     #     )
#     #     self.assertEqual(response.status_code, 200)
#     #     response_json = response.json()
#     #     self.assertEqual(response_json["message"], "Success")
#     #     self.assertTrue("bot_id" in response_json)
#     #     self.assertTrue(isinstance(response_json["bot_id"], str))
#     #     self.assertEqual(len(response_json["bot_id"]), 36)
#     #     self.assertTrue("output" in response_json)
#     #     self.assertTrue(isinstance(response_json["output"], str))
#     #     self.assertTrue("session_id" in response_json)
#     #     print(response_json["session_id"])
#     #     self.assertTrue(isinstance(response_json["session_id"], str))
#     #     self.assertEqual(len(response_json["session_id"]), 36)

#     def test_courtlistener_bot(self):
#         from models import BotRequest, InitializeSession

#         search_tool = {
#             "name": "courtlistener",
#             "method": "courtlistener",
#             "prefix": "",
#             "prompt": "Tool used to search courtlistener for legal cases and opinions",
#         }
#         test_bot_request = BotRequest(api_key=API_KEY, search_tools=[search_tool])
#         response = client.post("/create_bot", json=test_bot_request.model_dump())
#         self.assertEqual(response.status_code, 200)
#         response_json = response.json()
#         self.assertEqual(response_json["message"], "Success")
#         self.assertTrue("bot_id" in response_json)
#         self.assertTrue(isinstance(response_json["bot_id"], str))
#         self.assertEqual(len(response_json["bot_id"]), 36)
#         bot_id = response_json["bot_id"]
#         test_initialize_session = InitializeSession(
#             api_key=API_KEY,
#             bot_id=bot_id,
#             message="find me cases related to covid",
#         )
#         response = client.post(
#             "/initialize_session_chat", json=test_initialize_session.model_dump(),
#         )
#         self.assertEqual(response.status_code, 200)
#         response_json = response.json()
#         print(response_json)
#         self.assertEqual(response_json["message"], "Success")
#         self.assertTrue("bot_id" in response_json)
#         self.assertTrue(isinstance(response_json["bot_id"], str))
#         self.assertEqual(len(response_json["bot_id"]), 36)
#         self.assertTrue("output" in response_json)
#         self.assertTrue(isinstance(response_json["output"], str))
#         self.assertTrue("session_id" in response_json)
#         print(response_json["session_id"])
#         self.assertTrue(isinstance(response_json["session_id"], str))
#         self.assertEqual(len(response_json["session_id"]), 36)


#     # TODO: determine if session tests should be replicated using 
#     # a bot with search tool only, or both
#     def test_init_session(self):
#         from models import InitializeSession

#         test_initialize_session = InitializeSession(
#             api_key=API_KEY, bot_id=self.test_bot_vdb_id, message="test",
#         )
#         response = client.post(
#             "/initialize_session_chat", json=test_initialize_session.model_dump(),
#         )
#         self.assertEqual(response.status_code, 200)
#         response_json = response.json()
#         self.assertEqual(response_json["message"], "Success")
#         self.assertTrue("bot_id" in response_json)
#         self.assertTrue(isinstance(response_json["bot_id"], str))
#         self.assertEqual(len(response_json["bot_id"]), 36)
#         self.assertTrue("output" in response_json)
#         self.assertTrue(isinstance(response_json["output"], str))
#         self.assertTrue("session_id" in response_json)
#         print(response_json["session_id"])
#         self.assertTrue(isinstance(response_json["session_id"], str))
#         self.assertEqual(len(response_json["session_id"]), 36)

#     def test_empty_init(self):
#         from models import InitializeSession

#         test_initialize_session = InitializeSession(
#             api_key=API_KEY, bot_id=self.test_bot_vdb_id, message="",
#         )
#         response = client.post(
#             "/initialize_session_chat", json=test_initialize_session.model_dump(),
#         )
#         self.assertEqual(response.status_code, 200)
#         response_json = response.json()
#         self.assertEqual(response_json["message"], "Success")
#         self.assertTrue("bot_id" in response_json)
#         self.assertTrue(isinstance(response_json["bot_id"], str))
#         self.assertEqual(len(response_json["bot_id"]), 36)
#         self.assertTrue("output" in response_json)
#         self.assertTrue(isinstance(response_json["output"], str))
#         self.assertTrue(response_json["output"] == "Hi, how can I assist you today?")
#         self.assertTrue("session_id" in response_json)
#         print(response_json["session_id"])
#         self.assertTrue(isinstance(response_json["session_id"], str))
#         self.assertEqual(len(response_json["session_id"]), 36)

#     def test_init_and_continue_session(self):
#         from models import ChatBySession, InitializeSession

#         test_initialize_session = InitializeSession(
#             api_key=API_KEY, bot_id=self.test_bot_vdb_id, message="test"
#         )
#         response = client.post(
#             "/initialize_session_chat", json=test_initialize_session.model_dump(),
#         )
#         self.assertEqual(response.status_code, 200)
#         response_json = response.json()
#         self.assertEqual(response_json["message"], "Success")
#         self.assertTrue("bot_id" in response_json)
#         self.assertTrue(isinstance(response_json["bot_id"], str))
#         self.assertEqual(len(response_json["bot_id"]), 36)
#         self.assertTrue("output" in response_json)
#         self.assertTrue(isinstance(response_json["output"], str))
#         self.assertTrue("session_id" in response_json)
#         print(response_json["session_id"])
#         self.assertTrue(isinstance(response_json["session_id"], str))
#         self.assertEqual(len(response_json["session_id"]), 36)

#         test_cont_session = ChatBySession(
#             api_key=API_KEY, session_id=response_json["session_id"], message="test2",
#         )
#         response = client.post("/chat_session", json=test_cont_session.model_dump())
#         self.assertEqual(response.status_code, 200)
#         response_json = response.json()
#         print(response_json)
#         self.assertEqual(response_json["message"], "Success")
#         self.assertTrue("bot_id" in response_json)
#         self.assertTrue(isinstance(response_json["bot_id"], str))
#         self.assertEqual(len(response_json["bot_id"]), 36)
#         self.assertTrue("output" in response_json)
#         self.assertTrue(isinstance(response_json["output"], str))
#         self.assertTrue("session_id" in response_json)
#         self.assertTrue(isinstance(response_json["session_id"], str))
#         self.assertEqual(len(response_json["session_id"]), 36)

#     def test_chat_session(self):
#         from models import ChatBySession
#         test_chat_session = ChatBySession(api_key=API_KEY, session_id=self.test_session_vdb_id, message='test')
#         response = client.post("/chat_session", json=test_chat_session.model_dump())
#         self.assertEqual(response.status_code, 200)
#         response_json = response.json()
#         self.assertEqual(response_json["message"], "Success")
#         self.assertTrue("bot_id" in response_json)
#         self.assertTrue(isinstance(response_json["bot_id"], str))
#         self.assertEqual(len(response_json["bot_id"]), 36)
#         self.assertTrue("output" in response_json)
#         self.assertTrue(isinstance(response_json["output"], str))
#         self.assertTrue("session_id" in response_json)
#         self.assertTrue(isinstance(response_json["session_id"], str))
#         self.assertEqual(len(response_json["session_id"]), 36)

#     def test_fetch_session(self):
#         from models import FetchSession
#         test_fetch_session = FetchSession(api_key=API_KEY, session_id=self.test_session_vdb_id)
#         response = client.post("/fetch_session", json=test_fetch_session.model_dump())
#         self.assertEqual(response.status_code, 200)
#         response_json = response.json()
#         self.assertEqual(response_json["message"], "Success")
#         self.assertTrue("bot_id" in response_json)
#         self.assertTrue(isinstance(response_json["bot_id"], str))
#         self.assertEqual(len(response_json["bot_id"]), 36)
#         self.assertTrue("session_id" in response_json)
#         self.assertTrue(isinstance(response_json["session_id"], str))
#         self.assertEqual(len(response_json["session_id"]), 36)
#         self.assertTrue("history" in response_json)
#         self.assertTrue(isinstance(response_json["history"], list))

