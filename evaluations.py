"""Run evaluation methods on agents and chains."""
from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestDataset, TestsetGenerator

import milvusdb
from encoder import OPENAI_3_SMALL
from pdfs import get_documents_pdf

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable


def synthetic_testset_uscode(source: str) -> TestDataset:
    """Generate a set of synthetic data from a hardcoded path.

    Parameters
    ----------
    source : str
        the US Code source file to look up

    Returns
    -------
    TestDataset
        columns are: question, contexts, ground_truth, evolution_type, episode_done

    """
    documents = get_documents_pdf(milvusdb.US, source)
    # generator with openai models
    generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
    critic_llm = ChatOpenAI(model="gpt-4")
    embeddings = OpenAIEmbeddings(model=OPENAI_3_SMALL)

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings,
    )

    # generate testset
    return generator.generate_with_langchain_docs(
        documents,
        test_size=6,
        distributions={
            simple: 0.34,
            reasoning: 0.33,
            multi_context: 0.33,
        },
    )

def testset_responses(questions: list, chain: Runnable) -> tuple[list, list]:
    """Get a chains answers and used contexts for a list of questions.

    Parameters
    ----------
    questions : list
        The (possibly synthetic) questions
    chain : Runnable
        The chain used to answer the questions

    Returns
    -------
    tuple[list, list]
        answers, contexts used for each answer

    """
    answers = []
    contexts = []
    for question in questions:
        res = chain.invoke(question)
        # set() because the QA chain can cite the same chunk multiple times,
        # if chunk size is large enough
        source_indices = set(res["cited_answer"][0]["citations"])
        contexts.append([res["docs"][i - 1].page_content for i in source_indices])
        answers.append(res["cited_answer"][0]["answer"])
    return answers, contexts

testset = synthetic_testset_uscode("usc04@118-30.pdf")
test_df = testset.to_pandas()
test_df.to_json("data/evals/test/synthetic_testset.json")
test_questions = test_df["question"].to_numpy().tolist()
test_groundtruths = test_df["ground_truth"].to_numpy().tolist()
chain = milvusdb.qa_chain(milvusdb.US)
answers, contexts = testset_responses(test_questions, chain)
response_dataset = Dataset.from_dict({
    "question" : test_questions,
    "answer" : answers,
    "contexts" : contexts,
    "ground_truth" : test_groundtruths,
})
response_dataset.to_json("data/evals/test/response_dataset.json")

def evaluate_ragas(response_dataset: Dataset) -> None:
    """Evaluate a chains answers based on ground truths.

    Parameters
    ----------
    response_dataset : Dataset
        its columns should be: [question, answer, contexts, ground_truth]

    """
    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        answer_correctness,
    ]
    results = evaluate(response_dataset, metrics)
    results.to_pandas().to_json("data/evals/test/results.json")

evaluate_ragas(response_dataset)
