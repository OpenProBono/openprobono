from chat_models import chat_openai
from models import OpenAIModelEnum
import json
from pathlib import Path

EVALUATION_SYSTEM_MSG = "You are a fair evaluator language model."
EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""

eval_temperature = 0


def evaluate_answers(
    answers: list,
    questions: list,
) -> None:
    """Evaluate generated answers. Modifies the given answer file in place for better checkpointing."""
    results = []
    for answer, question in zip(answers, questions):
        eval_prompt = EVALUATION_PROMPT.format(
            instruction = question,
            response = answer,
        )
        eval_sys_msg = {"role": "system", "content": EVALUATION_SYSTEM_MSG}
        eval_msg = {"role": "user", "content": eval_prompt}
        eval_response = chat_openai([eval_sys_msg, eval_msg], OpenAIModelEnum.gpt_4_turbo, temperature=eval_temperature)
        eval_result = eval_response.choices[0].message.content
        feedback, score = (item.strip() for item in eval_result.split("[RESULT]"))
        print(feedback)
        print("^^ feedback")
        print(score)
        print("^^ score")
        print("================")
        results.append((answer, question, feedback, score))
    return results

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

questions = [question.format(state=state) for state in states]

responses = []
for file in files_to_compare:
    if Path(file).is_file():
        with Path(file).open() as f:
            responses.append(json.loads(f.read()))

eval1 = evaluate_answers(responses[0], questions)
eval2 = evaluate_answers(responses[1], questions)

with Path("eval-of-" + files_to_compare[0]).open("w") as f:
    f.write(json.dumps(eval1))

with Path("eval-of-" + files_to_compare[1]).open("w") as f:
    f.write(json.dumps(eval2))