"""Run evaluation methods on agents and chains."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import bs4
import requests
from langchain.text_splitter import TokenTextSplitter
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
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
from langfuse.callback import CallbackHandler

import langchain
import gc
#langchain.debug = True

import milvusdb
from encoders import OPENAI_3_SMALL

if TYPE_CHECKING:
    from datasets import Dataset
    from langchain_core.runnables import Runnable


def synthetic_testset(collection_name: str, expr: str, url: str) -> TestDataset:
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
    documents = milvusdb.get_documents(collection_name, expr)
    char_count = 0
    for i in range(len(documents)):
        char_count += len(documents[i].page_content)
        del documents[i].metadata["metadata"]["emphasized_text_contents"]
        del documents[i].metadata["metadata"]["emphasized_text_tags"]
        del documents[i].metadata["metadata"]["page_number"]
        documents[i].metadata["filename"] = url
    # assume ~2500 characters per page so 1 question per page
    question_count = 3 # min([32, max([1, char_count // 2500])])
    print(' ', question_count, ' questions')
    if question_count == 1:
        distributions = {simple: 1.0}
    elif question_count == 2:
        distributions = {simple: 0.5, reasoning: 0.5}
    elif question_count == 3:
        distributions = {simple:1}
    else:
        distributions = {
            simple: 0.5,
            reasoning: 0.25,
            multi_context: 0.25,
        }
    generator_llm = ChatOpenAI()
    critic_llm = ChatOpenAI()
    print(generator_llm.client)
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
    for node in docstore.nodes:
        if node.metadata["metadata"]["url"] != url:
            print('bad node')
    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings,
        docstore,
    )
    return generator.generate(
        test_size=question_count,
        distributions=distributions,
        raise_exceptions=False,
        with_debugging_logs=False,
    )

with Path("urls").open() as f:
    urls = [line.strip() for line in f.readlines()]
for url in [urls[0], urls[5]]:
    urlsplit = url.split("/")
    fname = urlsplit[-1].split(".")[0]
    print(fname)
    testset = synthetic_testset(milvusdb.COURTROOM5, f"metadata['url']=='{url}'", url)
    test_df = testset.to_pandas()
    test_df.to_json(f"data/evals/urls/{fname}_gpt35_bulk.json")
    gc.collect()

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
    print("site: ", site)
    r = requests.get(site)
    site_base = "//".join(site.split("//")[:-1])
    # converting the text 
    s = bs4.BeautifulSoup(r.content, "html.parser")
    urls = []

    if get_links:
        for i in s.find_all("a"):
            if "href" in i.attrs:
                href: str = i.attrs['href']

                if "/HTML/" in href and href not in old_urls:
                    old_urls.append(href)
                    urls.append("https://www.ncleg.gov" + href)

    # try:
    #     elements = partition(url=site)
    # except:
    #     elements = partition(url=site, content_type="text/html")
    print(" uploading")
    return urls


def crawl_and_scrape(site: str, collection: str, description: str):
    #metadataField = milvusdb.FieldSchema("metadata", milvusdb.DataType.JSON, "The associated metadata")
    #coll = milvusdb.create_collection(collection, description, [metadataField])
    urls = [site]
    new_urls = scrape(site, urls, get_links=True)
    print("new_urls: ", new_urls)
    #resume_url = next(iter([url for url in new_urls if url.endswith("/Chapter_151.html")]), None)
    #resume_idx = new_urls.index(resume_url)
    i = 0 #resume_idx
    # delete partially uploaded site
    #print(milvusdb.delete_expr(collection, f"metadata['url']=='{new_urls[i]}'"))
    # while i < len(new_urls):
    #     cur_url = new_urls[i]
    #     _, elements = scrape(cur_url, urls + new_urls)
    #     milvusdb.upload_elements(elements, coll)
    #     i += 1
    return new_urls


