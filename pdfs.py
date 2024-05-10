"""Functions for chunking and loading PDFs. Written by Nick Colvin."""
from __future__ import annotations

from os import listdir
from typing import TYPE_CHECKING

from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.llms import OpenAI as LangChainOpenAI
from pymilvus import Collection
from pypdf import PdfReader
from unstructured.chunking.base import Element
from unstructured.chunking.basic import chunk_elements
from unstructured.partition.pdf import partition_pdf

from encoders import (
    EncoderParams,
    embed_strs,
    get_langchain_embedding_model,
)

if TYPE_CHECKING:
    from fastapi import UploadFile
    from pymilvus import Collection

PYPDF = "pypdf"
UNSTRUCTURED = "unstructured"

def upload_pdf(collection: Collection, data: list, session_id: str, batch_size: int):
    num_chunks = len(data[0])
    for i in range(0, num_chunks, batch_size):
        batch_vector = data[0][i: i + batch_size]
        batch_text = data[1][i: i + batch_size]
        batch_source = data[2][i: i + batch_size]
        batch_page = data[3][i: i + batch_size]
        batch = [batch_vector, batch_text, batch_source, batch_page]
        if session_id:
            batch.append([session_id] * batch_size)
        current_batch_size = len(batch[0])
        res = collection.insert(batch)
        if res.insert_count != current_batch_size:
            msg = (
                f"  error: expected {current_batch_size} insertions but got "
                f"{res.insert_count} for pages {batch_page[0]}...{batch_page[-1]}"
            )
            raise ValueError(msg)

def upload_pdfs(collection: Collection, directory: str, max_chunk_size: int,
                chunk_overlap: int, session_id: str = "", batch_size: int = 1000,
                pdf_loader: str = UNSTRUCTURED):
    files = sorted(listdir(directory))
    for fname in files:
        if not fname.endswith(".pdf"):
            continue
        if pdf_loader == UNSTRUCTURED:
            data = chunk_pdf_unstructured(
                directory,
                fname,
                COLLECTION_ENCODER[collection.name],
                max_chunk_size,
                chunk_overlap,
            )
        else: # pypdf
            data = chunk_pdf_pypdf(
                directory,
                fname,
                COLLECTION_ENCODER[collection.name],
                max_chunk_size,
                chunk_overlap,
            )
        upload_pdf(collection, data, session_id, batch_size)

def chunk_pdf_pypdf(directory, file,
                    params: EncoderParams, max_chunk_size: int, chunk_overlap: int):
    if isinstance(file, str):
        reader = PdfReader(directory + file)
        fname = file
    else:
        reader = PdfReader(file.file)
        fname = file.filename
    elements = []
    for page in reader.pages:
        element = Element()
        element.text = page.extract_text()
        element.metadata.page_number = page.page_number + 1
        elements.append(element)
    chunks = chunk_elements(
        elements,
        max_characters=max_chunk_size,
        overlap=chunk_overlap,
    )
    num_chunks = len(chunks)
    text, page_numbers = [], []
    for chunk in chunks:
        text.append(chunk.text)
        page_numbers.append(chunk.metadata.page_number)
    # vector, text, source, page
    return [
        embed_strs(text, params=params),
        text,
        [fname] * num_chunks,
        page_numbers,
    ]

def chunk_pdf_unstructured(directory, file,
                           params: EncoderParams, max_chunk_size: int,
                           chunk_overlap: int):
    if isinstance(file, str):
        fname = file
        elements = partition_pdf(filename=directory + fname)
    else:
        fname = file.filename
        elements = partition_pdf(file=file.file)
    chunks = chunk_elements(
        elements,
        max_characters=max_chunk_size,
        overlap=chunk_overlap,
    )
    num_chunks = len(chunks)
    text, page_numbers = [], []
    for chunk in chunks:
        text.append(chunk.text)
        page_numbers.append(chunk.metadata.page_number)
    # vector, text, source, page
    return [
        embed_strs(text, params=params),
        text,
        [fname] * num_chunks,
        page_numbers,
    ]

def summarized_chunks_pdf(file: UploadFile, session_id: str, summary: str,
                          max_chunk_size: int = 10000, chunk_overlap: int = 1500):
    if not file.filename.endswith(".pdf"):
        return {"message": f"Failure: {file.filename} is not a PDF file"}

    # TODO(Nick): multiple pdf loaders (maybe merge with embed_pdf?)
    # parse
    reader = PdfReader(file.file)
    documents = [
        Document(
            page_content=page.extract_text(),
            metadata={
                "source": file.filename,
                "page": page_number,
                "session_id": session_id,
                "user_summary": summary,
            },
        )
        for page_number, page in enumerate(reader.pages, start=1)
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = text_splitter.split_documents(documents)

    # summarize
    chain = load_summarize_chain(
        LangChainOpenAI(temperature=0),
        chain_type="map_reduce",
    )
    result = chain.invoke({"input_documents": documents[:200]})
    for doc in documents:
        doc.metadata["ai_summary"] = result["output_text"].strip()

    return documents

def semantic_chunks_pdf(directory: str, file: str, params: EncoderParams):
    embeddings = get_langchain_embedding_model(params)
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="standard_deviation",
    )
    elements = partition_pdf(filename=directory + file)
    return text_splitter.create_documents([element.text for element in elements])

