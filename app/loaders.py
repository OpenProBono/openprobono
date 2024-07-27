"""Functions for loading text from files/urls."""
from __future__ import annotations

import io
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import requests
from bs4 import BeautifulSoup
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from langfuse.decorators import observe
from openai import OpenAI
from pymilvus import Collection
from pypandoc import ensure_pandoc_installed
from unstructured.partition.auto import partition
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.rtf import partition_rtf

if TYPE_CHECKING:
    from fastapi import UploadFile
    from unstructured.documents.elements import Element


def partition_html_str(html: str) -> list[Element]:
    """Partition an HTML string into elements.

    Parameters
    ----------
    html : str
        The HTML string.

    Returns
    -------
    list[Element]
        The extracted elements.

    """
    return partition_html(text=html)


def partition_uploadfile(file: UploadFile) -> list[Element]:
    """Partition an uploaded file into elements.

    Parameters
    ----------
    file : UploadFile
        The file to partition.

    Returns
    -------
    list[Element]
        The extracted elements.

    """
    return partition(file=file.file, metadata_filename=file.filename)


@observe(capture_output=False)
def scrape(site: str) -> list[Element]:
    """Scrape a site for text and partition it into elements.

    Parameters
    ----------
    site : str
        The URL to scrape.

    Returns
    -------
    list[Element]
        The scraped elements.

    """
    try:
        if site.endswith(".pdf"):
            r = requests.get(site, timeout=10)
            elements = partition_pdf(file=io.BytesIO(r.content))
        elif site.endswith(".rtf"):
            r = requests.get(site, timeout=10)
            ensure_pandoc_installed()
            elements = partition_rtf(file=io.BytesIO(r.content))
        else:
            elements = partition(url=site)
    except Exception as error:
        print("Error in regular partition: " + str(error))
        elements = partition(url=site, content_type="text/html")
    return elements


def scrape_with_links(
    site: str,
    old_urls: list[str],
) -> tuple[list[str], list[Element]]:
    """Scrape a site and get any links referenced on the site.

    Parameters
    ----------
    site : str
        The URL to scrape.
    old_urls : list[str]
        The list of URLs already visited.

    Returns
    -------
    tuple[list[str], list[Element]]
        URLs, elements

    """
    print("site: ", site)
    r = requests.get(site, timeout=10)
    site_base = "/".join(site.split("/")[:-1])
    # converting the text
    s = BeautifulSoup(r.content, "html.parser")
    urls = []

    # get links
    for i in s.find_all("a"):
        if "href" in i.attrs:
            href = i.attrs["href"]

            if href.startswith("/"):
                link = site_base + href
            elif href.startswith("http"):
                link = href
            else:
                link = old_urls[0]
                # skip this link

            if link not in old_urls:
                old_urls.append(link)
                urls.append(link)

    elements = scrape(site)
    return urls, elements


def quickstart_ocr(file: UploadFile) -> str:
    """Extract text from a file using OCR.

    Parameters
    ----------
    file : UploadFile
        The file to extract text from.

    Returns
    -------
    str
        The extracted text from the file.

    """
    project_id = "h2o-gpt"
    location = "us"  # Format is "us" or "eu"
    processor_id = "c99e554bb49cf45d"
    if not file.filename.endswith(".pdf"):
        process_options = documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                language_code="en",
                enable_native_pdf_parsing=True,
            ),
        )
    else:
        process_options = documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                language_code="en",
            ),
        )

    # You must set the `api_endpoint`if you use a location other than "us".
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    processor_name = client.processor_path(project_id, location, processor_id)

    # Print the processor information
    print(f"Processor Name: {processor_name}")

    # Load binary data
    raw_document = documentai.RawDocument(
        content=file.file.read(),
        mime_type=mimetypes.guess_type(file.filename)[0],
        # Refer to https://cloud.google.com/document-ai/docs/file-types
        # for supported file types
    )

    # Configure the process request
    # `processor.name` is the full resource name of the processor, e.g.:
    # `projects/{project_id}/locations/{location}/processors/{processor_id}`
    request = documentai.ProcessRequest(
        name=processor_name, raw_document=raw_document,
        process_options=process_options,
    )

    result = client.process_document(request=request)

    # For a full list of `Document` object attributes, reference this page:
    # https://cloud.google.com/document-ai/docs/reference/rest/v1/Document
    document = result.document

    # Read the text recognition output from the processor
    print("The document contains the following text:")
    print(document.text)
    return document.text


def transfer_hive(collection_name: str) -> None:
    """Transfer a collection from Milvus to Hive.

    Parameters
    ----------
    collection_name : str
        The name of the collection to transfer.

    """
    headers = {
        "Accept": "application/json",
        "Authorization": f"Token {os.environ['HIVE_ADD_PROJECT_KEY']}",
        "Content-Type": "application/json",
    }
    coll = Collection(collection_name)
    coll.load()
    q_iter = coll.query_iterator(output_fields=["text"])
    res = q_iter.next()
    num_batches = 0
    error = False
    while len(res) > 0:
        print(f"processing batch {num_batches}")
        for i, item in enumerate(res):
            if i % 100 == 0:
                print(f" i = {i}")
            data = {"text_data": item["text"]}
            attempt = 1
            num_attempts = 75
            while attempt < num_attempts:
                try:
                    response = requests.post(
                        "https://api.thehive.ai/api/v2/custom_index/add/sync",
                        headers=headers,
                        json=data,
                        timeout=75,
                    )
                    if response.status_code != 200:
                        print(response.json())
                        print(f"ERROR: status code = {response.status_code}, current pk = {item['pk']}")
                        error = True
                    break
                except:
                    time.sleep(1)
                    attempt += 1
            if error or attempt == num_attempts:
                print(f"ERROR REPORTED: attempt = {attempt}")
                error = True
                break
        num_batches += 1
        if error:
            print("ERROR REPORTED: exiting")
            break
        res = q_iter.next()
    q_iter.close()


def upload_jsonl_openai(
    jsonl_path: str,
    purpose: str,
    client: OpenAI | None = None,
) -> str:
    """Upload a .jsonl file of API requests to OpenAI.

    Parameters
    ----------
    jsonl_path : str
        Path to a .jsonl file
    purpose : str
        'assistants', 'vision', 'batch', or 'fine-tune'
    client : OpenAI | None, optional
        An OpenAI client to use, by default None.

    Returns
    -------
    str
        The uploaded file's identifier, for reference in API endpoints

    """
    client = OpenAI() if client is None else client
    res = client.files.create(file=Path.open(jsonl_path, "rb"), purpose=purpose)
    return res.id


def create_batch_openai(
    file_id: str,
    endpoint: str,
    client: OpenAI | None = None,
    description: str | None = None,
    metadata: dict | None = None,
) -> str:
    client = OpenAI() if client is None else client
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint=endpoint,
        description=description,
        metadata=metadata,
    )
    return batch.id


def download_file_openai(file_id: str, client: OpenAI | None = None) -> str:
    """Download a file from OpenAI.

    Parameters
    ----------
    file_id : str
        The file identifier
    client : OpenAI | None, optional
        An OpenAI client to use, by default None

    Returns
    -------
    str
        The file content

    """
    client = OpenAI() if client is None else client
    res = client.files.content(file_id)
    return res.text


def yield_batch_output(basedir: str) -> Generator[tuple[list, list, list]]:
    client = OpenAI()
    openai_files = client.files.list()
    batches = client.batches.list()
    for batch in batches.data:
        metadatas, texts, vectors = [], [], []
        if batch.status != "completed":
            continue

        input_file = next(
            (f for f in openai_files if batch.input_file_id == f.id),
            None,
        )
        if input_file is None:
            print("input file not found for " + batch.input_file_id)
            continue

        input_filename = input_file.filename
        print(input_filename)

        result_file_id = batch.output_file_id
        result_file_name = input_filename.split(".")[0] + "_out.jsonl"
        if not Path(basedir + result_file_name).exists():
            result = client.files.content(result_file_id).content
            with Path(basedir + result_file_name).open("wb") as f:
                f.write(result)
        with Path(basedir + result_file_name).open("r") as f:
            for i, line in enumerate(f, start=1):
                output = json.loads(line)
                with Path(basedir + input_filename).open("r") as in_f:
                    input_line = in_f.readline()
                    input_data = json.loads(input_line)
                    while input_line and input_data["custom_id"] != output["custom_id"]:
                        input_line = in_f.readline()
                        input_data = json.loads(input_line)
                custom_id_split = output["custom_id"].split("-")
                cluster_id = int(custom_id_split[0])
                opinion_id = int(custom_id_split[1])
                metadata = {}
                # metadata.update(cluster_data[cluster_id])
                # metadata.update(opinion_data[opinion_id])
                # metadata.update(docket_data[metadata["docket_id"]])
                text = input_data["body"]["input"]
                vector = output["response"]["body"]["data"][0]["embedding"]
                metadatas.append(metadata)
                texts.append(text)
                vectors.append(vector)
                if i % 5000 == 0:
                    print(f"i = {i}")
                if len(metadatas) == 1000:
                    yield metadatas, texts, vectors
                    metadatas, texts, vectors = [], [], []
        # return the last (< batch_size) lines
        if len(metadatas) > 0:
            yield metadatas, texts, vectors
