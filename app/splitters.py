"""Functions for chunking/splitting text data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.chunking.title import chunk_by_title

if TYPE_CHECKING:
    from unstructured.documents.elements import Element


def chunk_str(
    text: str,
    max_chunk_size: int = 2500,
    chunk_overlap: int = 250,
) -> list[str]:
    """Chunk elements using langchain `RecursiveCharacterTextSplitter`.

    Parameters
    ----------
    text : str
        The string to chunk.
    max_chunk_size : int, optional
        The maximum number of characters in a chunk, by default 2500
    chunk_overlap : int, optional
        The number of characters shared between consecutive chunks, by default 250

    Returns
    -------
    list[str]
        The chunked string.

    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = text_splitter.split_documents([Document(text)])
    return [document.page_content for document in documents]


def chunk_elements_by_title(
    elements: list[Element],
    max_characters: int = 2500,
    new_after_n_chars: int = 1000,
    overlap: int = 250,
) -> tuple[list[str], list[dict]]:
    """Chunk elements for uploading using unstructured `chunk_by_title`.

    Parameters
    ----------
    elements : list[Element]
        The elements to chunk.
    max_characters : int, optional
        See `chunk_by_title` for details, by default 2500.
    new_after_n_chars : int, optional
        See `chunk_by_title` for details, by default 1000.
    overlap : int, optional
        See `chunk_by_title` for details, by default 250.

    Returns
    -------
    tuple[list[str], list[dict]]
        texts, metadatas

    """
    chunks = chunk_by_title(
        elements,
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
        overlap=overlap,
    )
    texts, metadatas = [], []
    for i in range(len(chunks)):
        if not chunks[i].text:
            texts.append(" ")
        else:
            texts.append(chunks[i].text)
        md = chunks[i].metadata.to_dict()
        # delete fields which are empty or over 1000 characters
        maxlen = 1000
        keys_to_remove = [
            key for key in md
            if not md[key] or (isinstance(md[key], str) and len(md[key]) > maxlen) \
            or (
                isinstance(md[key], list) \
                and sum([len(str(item)) for item in md[key]]) > maxlen
            )
        ]
        for key in keys_to_remove:
            del md[key]
        metadatas.append(md)
    return texts, metadatas
