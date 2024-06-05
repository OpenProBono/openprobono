from chat_models import chat_openai
from models import OpenAIModelEnum
import json
from pathlib import Path


files_to_compare = [
    "states-responsesdaaa_401d8762-b0c7-450b-8f41-13d41fae37c9.txt",
    "states-responsesdynaj4a_401d8762-b0c7-450b-8f41-13d41fae37c2.txt"
]

question = (
    "What is the rule in {state} "
    "related to designating an email address for service in litigation?"
)
states = []
if Path("states").is_file():
    with Path("states").open() as f:
        states = f.readlines()

responses = []
for file in files_to_compare:
    if Path(file).is_file():
        with Path(file).open() as f:
            responses.append(json.loads(f.read()))

for state, response1, response2 in zip(states, responses[0], responses[1]):
    state = state.strip()
    # print(state)
    # print(response1)
    # print(response2)

    print(state + " done")
    prompt = "Compare these two responses to the question. Respond with which option is better and one sentence on why. If you think they are equal, respond with 'equal'."

    messages = [
    {
        "content": "The question: " + question.format(state=state) + "\n\nResponse 1: " + response1 + "\n\nResponse 2: " + response2,
        "role": "user"
    },
    {
        "content": prompt,
        "role": "system"
    }]

    

    response = chat_openai(messages, OpenAIModelEnum.gpt_4_turbo)
    print(response)
    print("THIS IS RESPONSE")