"""Functions for loading text from files/urls."""
from __future__ import annotations

import io
import mimetypes
import os
import pathlib
import time
from typing import TYPE_CHECKING

import requests
import urllib3
from bs4 import BeautifulSoup
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from langfuse.decorators import langfuse_context, observe
from openai import OpenAI
from pymilvus import Collection
from pypandoc import ensure_pandoc_installed
from PyPDF2 import PdfReader
from unstructured.documents.elements import Element, ElementMetadata
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.rtf import partition_rtf

from app.logger import setup_logger

if TYPE_CHECKING:
    from fastapi import UploadFile
    from openai.types import Batch

logger = setup_logger()


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
    elements = []
    try:
        r = requests.get(site, timeout=10)
        r.raise_for_status()
        if site.endswith(".pdf"):
            # try PyPDF first
            logger.info("scraping PDF: %s", site)
            reader = PdfReader(io.BytesIO(r.content))
            for i, page in enumerate(reader.pages, start=1):
                e = Element(metadata=ElementMetadata(url=site, page_number=i))
                e.text = page.extract_text()
                elements.append(e)
            # PyPDF can't do OCR. If the elements are all blank,
            # we probably need OCR (unstructured).
            if not any(e.text for e in elements):
                logger.info("pypdf didnt work, trying partition_pdf: %s", site)
                elements = partition_pdf(file=io.BytesIO(r.content))
        elif site.endswith(".rtf"):
            logger.info("scraping .rtf: %s", site)
            ensure_pandoc_installed()
            elements = partition_rtf(file=io.BytesIO(r.content))
        elif b"<!DOCTYPE" in r.content[:100] or b"<html" in r.content[:100]:
            logger.info("scraping html w beautifulsoup: %s", site)
            # it's HTML, use BeautifulSoup
            soup = BeautifulSoup(r.content, "html.parser")
            e = Element(metadata=ElementMetadata(url=site))
            e.text = soup.get_text()
            elements.append(e)
        else:
            logger.info("scraping fallback to unstructed: %s", site)
            # fall back to unstructured
            elements = partition(file=io.BytesIO(r.content))


    except (
        requests.exceptions.Timeout,
        urllib3.exceptions.ConnectTimeoutError,
    ) as timeout_err:
        logger.exception("Timeout error: %s", site)
        langfuse_context.update_current_observation(
            level="ERROR",
            status_message=str(timeout_err),
        )
    except (requests.exceptions.SSLError, urllib3.exceptions.SSLError) as ssl_err:
        logger.exception("SSL error: %s", site)
        langfuse_context.update_current_observation(
            level="ERROR",
            status_message=str(ssl_err),
        )
    except (
        requests.exceptions.ConnectionError,
        urllib3.exceptions.ProtocolError,
    ) as conn_err:
        logger.exception("Connection error: %s", site)
        langfuse_context.update_current_observation(
            level="ERROR",
            status_message=str(conn_err),
        )
    except Exception as error:
        logger.exception("Unexpected error: %s", site)
        langfuse_context.update_current_observation(level="ERROR", status_message=str(error))

    langfuse_context.update_current_observation(output=f"{len(elements)} elements")
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
    logger.info("site: %s", site)
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
    logger.info("Processor Name: %s", processor_name)

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
    logger.info("The document contains the following text: %s", document.text)
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
        logger.info("processing batch %s", num_batches)
        for i, item in enumerate(res):
            if i % 100 == 0:
                logger.info(" i = %s", i)
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
                        logger.error(
                            "ERROR: status code = %s, current pk = %s, JSON = %s",
                            response.status_code,
                            item["pk"],
                            response.json(),
                        )
                        error = True
                    break
                except:
                    time.sleep(1)
                    attempt += 1
            if error or attempt == num_attempts:
                logger.error("ERROR REPORTED: attempt = %s", attempt)
                error = True
                break
        num_batches += 1
        if error:
            logger.error("ERROR REPORTED: exiting")
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
    res = client.files.create(file=pathlib.Path.open(jsonl_path, "rb"), purpose=purpose)
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


def download_batch_output(
    client: OpenAI,
    batch: Batch,
    output_filename: str,
    basedir: str,
) -> None:
    """Download a batch output file from OpenAI."""
    output_file_id = batch.output_file_id
    if not pathlib.Path(basedir + output_filename).exists():
        result = client.files.content(output_file_id).content
        with pathlib.Path(basedir + output_filename).open("wb") as f:
            f.write(result)


def wait_for_batches(wait_time: int = 10 * 60) -> None:
    """After submitting batches to OpenAI, wait for them to complete or fail.

    Parameters
    ----------
    wait_time : int, optional
        Time to wait between checks, by default 10*60

    """
    completed_or_failed = False
    count = 1
    while not completed_or_failed:
        time.sleep(wait_time)
        logger.info("waited %s times, checking batches", count)
        client = OpenAI()
        completed_or_failed = True
        batches = client.batches.list()
        for page in batches.iter_pages():
            for batch in page.data:
                if batch.status not in ("completed", "failed"):
                    completed_or_failed = False
                    break
            if not completed_or_failed:
                break
        count += 1


def delete_completed_batches(basedir: str) -> None:
    client = OpenAI()
    batches = client.batches.list()
    files = client.files.list()
    for page in batches.iter_pages():
        for batch in page.data:
            # handle completed batches
            if batch.status != "completed":
                continue
            logger.info("batch %s completed", batch.id)
            in_exists, out_exists = False, False
            in_fname, out_fname = None, None
            for fpage in files.iter_pages():
                for f in fpage.data:
                    if f.id == batch.input_file_id:
                        logger.info(f.filename)
                        in_exists = True
                        in_fname = f.filename
                        split_fname = f.filename.split(".")
                        out_fname = split_fname[0] + "_out.jsonl"
                    elif f.id == batch.output_file_id:
                        logger.info(f.filename)
                        out_exists = True
                if in_exists and out_exists:
                    break
            if in_exists:
                logger.info("deleted input file in API")
                client.files.delete(batch.input_file_id)
                # move file from basedir to basedir + completed
                pathlib.Path(basedir + in_fname).rename(basedir + "completed/" + in_fname)
                logger.info("moved %s to completed/", in_fname)
                pathlib.Path(basedir + out_fname).rename(basedir + "completed/" + out_fname)
                logger.info("moved %s to completed/", out_fname)
                if batch.error_file_id is not None:
                    logger.info("batch contains an error file, downloading")
                    client.files.content(batch.error_file_id)
                    result = client.files.content(batch.error_file_id).content
                    with pathlib.Path(basedir + "completed/errors/errors_" + in_fname).open("wb") as f:
                        f.write(result)
                    logger.info("downloaded errors for %s", in_fname)
            if out_exists:
                logger.info("deleted output file")
                client.files.delete(batch.output_file_id)


def retry_failed_batches() -> None:
    client = OpenAI()
    batches = client.batches.list()
    files = client.files.list()
    restarted_infile_ids = set()
    count = 0
    for page in batches.iter_pages():
        for batch in page.data:
            # handle failed batches
            if batch.status != "failed":
                continue
            in_file_id = batch.input_file_id
            in_exists = False
            for fpage in files.iter_pages():
                for f in fpage.data:
                    if f.id == in_file_id:
                        logger.info(f.filename)
                        in_exists = True
                        break
                if in_exists:
                    break
            if not in_exists:
                logger.info("batch %s input file not found, skipping", batch.id)
                continue
            if in_file_id in restarted_infile_ids:
                logger.info("batch %s input file already restarted in another batch, skipping", batch.id)
                continue
            logger.info("batch %s failed and has input file, recreating", batch.id)
            client.batches.create(
                completion_window="24h",
                endpoint="/v1/embeddings",
                input_file_id=in_file_id,
            )
            restarted_infile_ids.add(in_file_id)
            count += 1
            if count == 25:
                return
