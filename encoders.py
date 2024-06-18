"""Embed text into vectors, and retrieve embedding models for chains."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import tiktoken
import voyageai
from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_openai import OpenAIEmbeddings
from openai import APITimeoutError, OpenAI

from models import EncoderParams, OpenAIModelEnum, VoyageModelEnum

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

OPENAI_MODELS = {
    OpenAIModelEnum.embed_ada_2,
    OpenAIModelEnum.embed_small,
    OpenAIModelEnum.embed_large,
}

VOYAGE_MODELS = {
    VoyageModelEnum.large_2_instruct,
    VoyageModelEnum.law,
}

def get_langchain_embedding_model(encoder: EncoderParams) -> Embeddings:
    """Get the LangChain class for a given embedding model.

    Parameters
    ----------
    encoder : EncoderParams, optional
        The parameters defining an embedding model, by default
        OPENAI_3_SMALL with 768 dimensions

    Returns
    -------
    Embeddings
        The class representing the embedding model for use in LangChain

    """
    if encoder.name in OPENAI_MODELS:
        dim = encoder.dim if encoder.name != OpenAIModelEnum.embed_ada_2 else None
        return OpenAIEmbeddings(model=encoder.name, dimensions=dim)
    return HuggingFaceHubEmbeddings(model=encoder.name)

def token_count(string: str, model: str) -> int:
    """Return the number of tokens in a text string.

    Third-generation embedding models like text-embedding-3-small
    use the cl100k_base encoding.

    Parameters
    ----------
    string : str
        The string whose tokens will be counted
    model : str
        The embedding model that will accept the string

    Returns
    -------
    int
        The number of tokens in the given string for the given model

    """
    if model in OPENAI_MODELS:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(string))
    if model in VOYAGE_MODELS:
        vo = voyageai.Client()
        return vo.count_tokens([string], model)
    raise ValueError(model)


def embed_strs(text: list[str], encoder: EncoderParams) -> list:
    """Embeds text from a list where each element is within the model max input length.

    Parameters
    ----------
    text : list[str]
        The text entries to embed
    encoder : EncoderParams
        The embedding model parameters

    Returns
    -------
    list
        A list of lists of floats representing the embedded text

    """
    if encoder.name in OPENAI_MODELS:
        return embed_strs_openai(text, encoder)
    if encoder.name in VOYAGE_MODELS:
        return embed_strs_voyage(text, encoder)
    raise ValueError(encoder.name)

def embed_strs_openai(text: list[str], encoder: EncoderParams) -> list:
    """Embed a list of strings using an OpenAI client.

    Parameters
    ----------
    text : list[str]
        The list of strings to embed
    encoder : EncoderParams
        The parameters for an OpenAI embedding model

    Returns
    -------
    list
        A list of lists of floats representing embedded strings

    """
    max_tokens = 8191
    max_array_size = 2048
    i = 0
    data = []
    client = OpenAI()
    num_strings = len(text)
    while i < num_strings:
        j = i
        while j < num_strings and j - i < max_array_size:
            tokens = token_count(text[j], encoder.name)
            if tokens > max_tokens:
                msg = f"str at index {j} is {tokens} tokens but the max is {max_tokens}"
                raise ValueError(msg)
            j += 1
        attempt = 1
        num_attempts = 75
        while attempt < num_attempts:
            try:
                args = {"input": text[i:j], "model": encoder.name}
                if encoder.name != OpenAIModelEnum.embed_ada_2:
                    args["dimensions"] = encoder.dim
                response = client.embeddings.create(**args)
                data += [data.embedding for data in response.data]
                i = j
                break
            except APITimeoutError:
                time.sleep(1)
                attempt += 1
    return data


def embed_strs_voyage(text: list[str], encoder: EncoderParams) -> list:
    """Embed a list of strings using a Voyage client.

    Parameters
    ----------
    text : list[str]
        The list of strings to embed
    encoder : EncoderParams
        The parameters for an OpenAI embedding model

    Returns
    -------
    list
        A list of lists of floats representing embedded strings

    """
    max_array_tokens = 120000
    max_tokens = 16000
    max_array_size = 128
    i = 0
    data = []
    client = voyageai.Client()
    num_strings = len(text)
    while i < num_strings:
        array_tokens = 0
        j = i
        # determine how long of an array to upload
        while j < num_strings and j - i < max_array_size:
            tokens = token_count(text[j], encoder.name)
            # determine if a string is too many tokens
            if tokens > max_tokens:
                msg = f"str at index {j} is {tokens} tokens but the max is {max_tokens}"
                raise ValueError(msg)
            # determine if the array is too many tokens
            if array_tokens + tokens <= max_array_tokens:
                array_tokens += tokens
                j += 1
        attempt = 1
        num_attempts = 75
        while attempt < num_attempts:
            try:
                response = client.embed(
                    text[i:j],
                    model="voyage-large-2-instruct",
                    input_type="document",
                ).embeddings
                data += response
                i = j
                break
            except APITimeoutError:
                time.sleep(1)
                attempt += 1
    return data
