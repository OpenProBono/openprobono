"""Run evaluation methods on agents and chains."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from unstructured.documents.elements import ElementMetadata, Text
from unstructured.partition.auto import partition

from main import chat
from milvusdb import get_expr
from models import ChatRequest
from prompts import EVALUATION_PROMPT

API_KEY = os.environ["OPB_TEST_API_KEY"]

# for loading eval data by statute
root_dir = "data/chapter_urls/"
chapter_names = sorted(os.listdir(root_dir))
# for loading by chapter
evalset_urls = "data/NC-court/court-urls"
chapter_pdf_urls = ""
if Path(evalset_urls).exists():
    with Path(evalset_urls).open() as f:
        chapter_pdf_urls = [line.strip() for line in f.readlines()]


def load_statute_urls(chapter_name: str):
    with Path(root_dir + chapter_name).open() as f:
        return [line.strip() for line in f.readlines()]


def generate_statute_elements():
    for chapter in chapter_names:
        statute_urls = load_statute_urls(chapter)
        for statute_url in statute_urls:
            elements = partition(statute_url)
            yield statute_url, elements


def resume_statute_elements(chapter_name: str, statute_url: str):
    resume_chapter = next(iter([chapter for chapter in chapter_names if chapter == chapter_name]), None)
    resume_chapter_idx = chapter_names.index(resume_chapter) if resume_chapter else 0
    for i in range(resume_chapter_idx, len(chapter_names)):
        statute_urls = load_statute_urls(chapter_names[i])
        resume_statute = next(iter([statute for statute in statute_urls if statute == statute_url]), None)
        resume_statute_idx = statute_urls.index(resume_statute) if resume_statute else 0
        for j in range(resume_statute_idx, len(statute_urls)):
            elements = partition(url=statute_urls[j], content_type="application/pdf")
            yield statute_url, elements


def generate_chapter_elements():
    for chapter_pdf_url in chapter_pdf_urls:
        elements = partition(url=chapter_pdf_url, content_type="application/pdf")
        yield chapter_pdf_url, elements


# for loading eval data from a milvus Collection
def vdb_source_documents(collection_name: str, source: str) -> list[Text]:
    expr = f"metadata['url']=='{source}'"
    hits = get_expr(collection_name, expr)["result"]
    return [
        Text(
            text=hit["text"],
            metadata=ElementMetadata(
                url=source,
                page_number=hit["metadata"]["page_number"],
            ),
        ) for hit in hits
    ]


def load_statute_elements(collection_name: str):
    for chapter in chapter_names:
        statute_urls = load_statute_urls(chapter)
        for statute_url in statute_urls:
            yield vdb_source_documents(collection_name, statute_url)


def load_chapter_elements(collection_name: str):
    for chapter_pdf_url in chapter_pdf_urls:
        yield vdb_source_documents(collection_name, chapter_pdf_url)


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
