"""Run evaluation methods on agents and chains."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from main import chat
from models import ChatRequest
from prompts import EVALUATION_PROMPT

API_KEY = os.environ["OPB_TEST_API_KEY"]

def evaluate_agent(
    bot_id: str,
    questions: list[str],
    true_answers: list[str],
) -> tuple[list[str], list[str]]:
    """Evaluate an agent's generated answers to questions against their true answers.

    Parameters
    ----------
    bot_id : str
        The bot_id of the agent to evaluate.
    questions : list[str]
        The questions to ask the agent.
    true_answers : list[str]
        The true answers to the questions.

    Returns
    -------
    tuple[list[str], list[str]]
        feedbacks, scores (between 1 and 5)

    """
    generated_answers = [answer_question(q, bot_id) for q in questions]
    feedbacks, scores = [], []
    for i in range(len(questions)):
        feedback, score = evaluate_answer(
            questions[i],
            generated_answers[i],
            true_answers[i],
            bot_id,
        )
        feedbacks.append(feedback)
        scores.append(score)
    return feedbacks, scores


def legalbench_ruleqa() -> tuple[list[str], list[str]]:
    """Load questions and answers from legalbench ruleQA.

    Returns
    -------
    tuple[list[str], list[str]]
        questions, answers

    """
    # load ruleqa
    ruleqa = pd.read_csv("data/legalbench-ruleqa.csv")
    questions, answers = zip(*ruleqa[["text", "answer"]].to_numpy().tolist())
    return list(questions), list(answers)


def litigation_state_emails() -> list[str]:
    """Generate questions for registering an email for litigation in every state.

    Returns
    -------
    list[str]
        questions

    """
    question = (
        "What is the rule in {state} "
        "related to designating an email address for service in litigation?"
    )
    states = []
    if Path("states").is_file():
        with Path("states").open() as f:
            states = f.readlines()
    return [question.format(state=state) for state in states]


def answer_question(question: str, bot_id: str) -> str:
    """Answer a question using an agent.

    Parameters
    ----------
    question : str
        The question to answer.
    bot_id : str
        The bot_id of the agent to use.

    Returns
    -------
    str
        The generated answer.

    """
    request = ChatRequest(
        history=[(question, None)],
        bot_id=bot_id, api_key=API_KEY,
    )
    return chat(request)["output"]


def evaluate_answer(
    question: str,
    generated_answer: str,
    true_answer: str,
    bot_id: str,
) -> tuple[str, str]:
    """Evaluate an answer against a reference answer.

    Parameters
    ----------
    question : str
        The question to evaluate the answer against.
    generated_answer : str
        The generated answer to evaluate.
    true_answer : str
        The true answer to evaluate against.
    bot_id : str
        The bot_id of the evaluator.

    Returns
    -------
    tuple[str, str]
        The evaluation feedback and the evaluation score.

    """
    eval_prompt = EVALUATION_PROMPT.format(
        instruction=question,
        response=generated_answer,
        reference_answer=true_answer,
    )
    request = ChatRequest(
        history=[(eval_prompt, None)],
        bot_id=bot_id, api_key=API_KEY,
    )
    eval_result = chat(request)["output"]
    feedback, score = (item.strip() for item in eval_result.split("[RESULT]"))
    return feedback, score
