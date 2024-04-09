"""Run evaluation methods on agents and chains."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import bs4
import requests
from langchain.text_splitter import TokenTextSplitter
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
from ragas.testset.extractor import KeyphraseExtractor
from ragas.testset.generator import (
    InMemoryDocumentStore,
    LangchainEmbeddingsWrapper,
    LangchainLLMWrapper,
    TestDataset,
    TestsetGenerator,
)
from unstructured.partition.auto import partition

import milvusdb
from chat_models import GPT_4_TURBO
from encoders import OPENAI_3_SMALL

if TYPE_CHECKING:
    from datasets import Dataset
    from langchain_core.runnables import Runnable


def synthetic_testset(collection_name: str, expr: str) -> TestDataset | None:
    """Generate a set of synthetic data from documents in Milvus.

    Parameters
    ----------
    source : str
        the US Code source file to look up

    Returns
    -------
    TestDataset
        columns are: question, contexts, ground_truth, evolution_type, episode_done

    """
    documents = milvusdb.get_documents(collection_name, expr)
    char_count = 0
    for i in range(len(documents)):
        char_count += len(documents[i].page_content)
        del documents[i].metadata["metadata"]["filetype"]
        del documents[i].metadata["metadata"]["languages"]
    # assume ~2500 characters per page so 1 question per page
    if char_count < 500:
        return None
    question_count = min([8, max([1, char_count // 2500])])
    print(" ", question_count, " questions")
    if question_count == 1:
        distributions = {simple: 1.0}
    elif question_count == 2:
        distributions = {simple: 0.5, reasoning: 0.5}
    elif question_count == 3:
        distributions = {simple:0.34, reasoning: 0.33, multi_context: 0.33}
    else:
        distributions = {simple: 0.5, reasoning: 0.25, multi_context: 0.25}
    generator_llm = ChatOpenAI()
    critic_llm = ChatOpenAI(model=GPT_4_TURBO)
    embeddings = OpenAIEmbeddings(model=OPENAI_3_SMALL, dimensions=768)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=0)
    generator_llm_model = LangchainLLMWrapper(generator_llm)
    keyphrase_extractor = KeyphraseExtractor(llm=generator_llm_model)
    docstore = InMemoryDocumentStore(
        splitter=splitter,
        extractor=keyphrase_extractor,
        embeddings=ragas_embeddings,
    )
    docstore.add_documents(documents)
    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings,
        docstore,
    )
    testset = generator.generate(
        test_size=question_count,
        distributions=distributions,
        raise_exceptions=False,
    )
    return testset

def make_questionset(url: str):
    urlsplit = url.split("/")
    fname = urlsplit[-1].split(".")[0]
    print(fname)
    testset = synthetic_testset(milvusdb.COURTROOM5, f"metadata['url']=='{url}'")
    if testset:
        test_df = testset.to_pandas()
        test_df.to_json(f"data/evals/pdfs/{fname}.json")

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


def scrape(site: str, old_urls: list[str], get_links: bool = False):
    urls = []
    elements = []
    if get_links:
        r = requests.get(site)
        site_base = "//".join(site.split("//")[:-1])
        # converting the text 
        s = bs4.BeautifulSoup(r.content, "html.parser")

        for i in s.find_all("a"):
            if "href" in i.attrs:
                href: str = i.attrs['href']

                if "/BySection/" in href and href.endswith(".pdf") and href not in old_urls:
                    old_urls.append(href)
                    urls.append("https://www.ncleg.gov" + href)
    else:
        elements = partition(url=site, content_type="application/pdf")
    return urls, elements


def crawl_and_scrape(site: str, collection_name: str):
    urls = [site]
    new_urls, _ = scrape(site, urls, get_links=True)

    i = 0 #resume_idx
    # delete partially uploaded site
    #print(milvusdb.delete_expr(collection, f"metadata['url']=='{new_urls[i]}'"))
    while i < len(new_urls):
       cur_url = new_urls[i]
       _, elements = scrape(cur_url, urls + new_urls)
       milvusdb.upload_elements(elements, collection_name)
       i += 1
    return new_urls

#metadataField = milvusdb.FieldSchema("metadata", milvusdb.DataType.JSON, "The associated metadata")
#coll = milvusdb.create_collection(milvusdb.COURTROOM5, "NC General Statutes parsed from PDFs by section for Courtroom5", [metadataField])
import os
root_dir = "data/evals/chapter_urls/"
chapters = sorted(os.listdir(root_dir))
resume_chapter = next(iter([chapter for chapter in chapters if chapter == "Chapter146"]), None)
resume_chapter_idx = chapters.index(resume_chapter) if resume_chapter else 0
for i in range(resume_chapter_idx, len(chapters)):
    chapter = chapters[i]
    with Path(root_dir + chapter).open() as f:
        section_urls = [line.strip() for line in f.readlines()]
    resume_section = next(iter([section for section in section_urls if section == "https://www.ncleg.gov/EnactedLegislation/Statutes/PDF/BySection/Chapter_146/GS_146-32.pdf"]), None)
    resume_section_idx = section_urls.index(resume_section) if resume_section else 0
    print(f"{chapter}: {len(section_urls)} sections, {len(section_urls) - resume_section_idx} remaining")
    for j in range(resume_section_idx, len(section_urls)):
        _, elements = scrape(section_urls[j], [])
        milvusdb.upload_elements(elements, milvusdb.COURTROOM5)
        # with Path(f"data/evals/chapter_urls/{chapter}").open(mode="w") as f:
        #     f.write("\n".join(section_urls))
