"""Run evaluation methods on agents and chains."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from unstructured.documents.elements import ElementMetadata, Text
from unstructured.partition.auto import partition

from chat_models import chat as llm_chat
from main import chat
from milvusdb import get_expr
from models import ChatModelParams, ChatRequest
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


def legalbench(subset: str) -> Dataset:
    """Load a subset of data from legalbench.

    Parameters
    ----------
    subset : str
        The subset of legalbench to load.

    Returns
    -------
    Dataset
        The subset of data from legalbench.

    """
    return load_dataset(
        "nguha/legalbench",
        subset,
        split="test",
        trust_remote_code=True,
    )


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


def compare_evaluation(
    models: list[str],
    model_evals: list[pd.DataFrame],
    compare_path: str,
) -> None:
    """Compare generated answers with true answers between every pair of models.

    Parameters
    ----------
    models : list[str]
        The models to compare.
    model_evals : list[pd.DataFrame]
        The results of the evaluation for each model.
    compare_path : str
        The path to load/store the comparison results.

    """
    from tqdm.auto import tqdm

    from prompts import COMPARISON_PROMPT

    evaluators = [
        ChatModelParams(engine="openai"),
        ChatModelParams(engine="openai", model="gpt-4o"),
    ]
    questions = model_evals[0]["question"]

    # Load the comparison results if they exist.
    if Path(compare_path).exists():
        with Path(compare_path).open() as f:
            results = json.load(f)
    else:
        results = {}

    # Evaluate every pair of models.
    for evaluator in evaluators:
        print("Evaluating with ", evaluator.model)
        if evaluator.model not in results:
            results[evaluator.model] = {}
        for i in range(len(models)):
            model_a = models[i]
            true_answer = model_evals[i]["true_answer"]
            answer_a = model_evals[i]["generated_answer"]
            print(" Model A: ", model_a)
            for j in range(i + 1, len(models)):
                model_b = models[j]
                answer_b = model_evals[j]["generated_answer"]
                matchup = f"{model_a} vs. {model_b}"
                print("  Model B: ", model_b)
                if matchup not in results[evaluator.model]:
                    results[evaluator.model][matchup] = []
                for k, question in tqdm(enumerate(questions)):
                    # skip completed comparisons
                    if question in [
                        output["question"]
                        for output in results[evaluator.model][matchup]
                    ]:
                        continue

                    eval_prompt = COMPARISON_PROMPT.format(
                        question=question,
                        true_answer=true_answer[k],
                        answer_a=answer_a[k],
                        answer_b=answer_b[k],
                    )
                    eval_response = llm_chat(
                        [{"role":"system", "content":eval_prompt}],
                        evaluator,
                    )
                    eval_result = eval_response.choices[0].message.content
                    feedback, result = (
                        item.strip() for item in eval_result.split("[RESULT]")
                    )
                    feedback = feedback.split("Feedback:")[-1].strip()
                    results[evaluator.model][matchup].append(
                        {"question": question, "feedback": feedback, "result": result},
                    )
                    # save the results
                    with Path(compare_path).open("w") as f:
                        json.dump(results, f)

    # Plot the comparison results.
    for evaluator in results:
        print("Plotting ", evaluator, " evaluations")
        compare_win_matrix_plot(results[evaluator], evaluator)


def compare_win_matrix(
    matchups: dict[str, list[dict[str, str]]],
    models: list[str],
) -> list[list[float]]:
    """Calculate win matrix from matchups.

    Parameters
    ----------
    matchups : dict[str, list[dict[str, str]]]
        The matchups to compare. E.g.
        {"A vs. B": [{"question": "...", "feedback": "...", "result": "..."}]}.
    models : list[str]
        The models to compare.

    Returns
    -------
    list[list[float]]
        The win rate matrix

    """
    matrix = [[0] * len(models) for _ in range(len(models))]
    for matchup in matchups:
        matchup_models = matchup.split(" vs. ")
        model_a = matchup_models[0]
        model_b = matchup_models[1]
        results = matchups[matchup]
        num_ties = 0
        for result in results:
            if result["result"] == "A" or result["result"] == "Answer A" or \
                "answer a" in result["result"].lower() or \
                    "better answer is a" in result["result"].lower():

                matrix[models.index(model_a)][models.index(model_b)] += 1
            elif "tie" in result["result"].lower() or \
                "equal" in result["result"].lower() or \
                    "neither" in result["result"].lower():

                num_ties += 1
            elif result["result"] != "B" and result["result"] != "Answer B" \
                and "answer b" not in result["result"].lower():

                print(f"unknown result for matchup {matchup}: {result}")

        # divide number of wins by number of results to get a's win rate
        matrix[models.index(model_a)][models.index(model_b)] /= len(results)
        # model b win rate = 1 - model a's - tie rate
        model_a_win_rate = matrix[models.index(model_a)][models.index(model_b)]
        tie_rate = num_ties / len(results)
        model_b_win_rate = 1 - model_a_win_rate - tie_rate
        matrix[models.index(model_b)][models.index(model_a)] = model_b_win_rate
    return matrix


def compare_win_matrix_plot(
    matchups: dict[str, list[dict[str, str]]],
    evaluator: str,
) -> None:
    """Plot the win matrix from matchups.

    Parameters
    ----------
    matchups : dict[str, list[dict[str, str]]]
        The matchups to compare.
    evaluator : str
        The evaluator to plot.

    """
    # get the list of models in the matchups
    models = []
    for matchup in matchups:
        matchup_models = matchup.split(" vs. ")
        if matchup_models[0] not in models:
            models.append(matchup_models[0])
        if matchup_models[1] not in models:
            models.append(matchup_models[1])

    # create a sample matrix (2D array)
    matrix = compare_win_matrix(matchups, models)

    # assume 'matrix' is your 2D array (matrix)
    _, ax = plt.subplots()
    _ = ax.imshow(matrix, cmap="coolwarm", interpolation="nearest")

    # annotate each element in the matrix with its value
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            ax.annotate(f"{matrix[i][j]:.2f}", (j, i), ha="center", va="center")

    # add labels on each row and column
    ax.set_xticks(np.arange(len(matrix[0])))
    ax.set_yticks(np.arange(len(matrix)))

    # set the tick labels for rows and columns
    ax.set_xticklabels(models)
    ax.set_yticklabels(models)
    ax.set_title(f"Win Rate Matrix - {evaluator} evaluator")

    plt.show()
