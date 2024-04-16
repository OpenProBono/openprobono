import time
from pathlib import Path
import json
from main import chat
from models import ChatRequest

# bot_id = "2f294354-2101-42ac-a9a2-2e16da92e240" #dynamic courtroom5 search
bot_id = "18443765-4f72-4def-a161-68528223d3a3" #regular courtroom5 search

api_key = "xyz"
question = (
    "What is the rule in {state} "
    "related to designating an email address for service in litigation?"
)
states = []
if Path("states").is_file():
    with Path("states").open() as f:
        states = f.readlines()

responses = []
for state in states[]:
    request = ChatRequest(
        history=[(question.format(state=state), None)],
        bot_id=bot_id, api_key=api_key)

    response = chat(request)
    responses.append(response["output"])
    time.sleep(1)


with Path("states-responses.txt").open("w") as f:
    f.write(json.dumps(responses))