import json
import os
import time
from pathlib import Path

from app.main import chat
from app.models import ChatRequest

# bot_id = "2f294354-2101-42ac-a9a2-2e16da92e240" #dynamic courtroom5 search
# bot_id = "18443765-4f72-4def-a161-68528223d3a3" #regular courtroom5 search
# bot_id = "401d8762-b0c7-450b-8f41-13d41fae37c8" #gpt4 openai engine
#bot_id = "401d8762-b0c7-450b-8f41-13d41fae37d1" #gpt4 dyanmic


#bot_id = "401d8762-b0c7-450b-8f41-13d41fae37c9" #gpt4o clasic

# firebase_config = loads(os.environ["Firebase"])
# cred = credentials.Certificate(firebase_config)
# firebase_admin.initialize_app(cred)
# db = firestore.client()


def main():
    bot_id = "401d8762-b0c7-450b-8f41-13d41fae37c2" #gpt4o dynamic

    api_key = os.environ.get("OPB_TEST_API_KEY")

    question = (
        "What is the rule in {state} "
        "related to designating an email address for service in litigation?"
    )
    states = []
    if Path("app/eval/states").is_file():
        with Path("app/eval/states").open() as f:
            states = f.readlines()

    responses = []
    for state in states:
        state = state.strip()
        print(state)
        request = ChatRequest(
            history=[{"role": "user", "content": question.format(state=state)}],
            bot_id=bot_id,
            api_key=api_key)

        response = chat(request)
        print(response)
        responses.append(response["output"])
        print(state + " done")
        time.sleep(1)



    with Path("states-responses_j8_q_k3_dyserp_nr10" + str(bot_id) + ".txt").open("w") as f:
        f.write(json.dumps(responses))

main()