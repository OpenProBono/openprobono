"""Embed text into vectors, and retrieve embedding models for chains."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import tiktoken
import torch
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_openai import OpenAIEmbeddings
from openai import APITimeoutError, OpenAI
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from transformers.modeling_outputs import BaseModelOutput
    from unstructured.chunking.base import Element

class EncoderParams:
    """Define the embedding model for a Collection."""

    def __init__(self: EncoderParams, name: str, dim: int) -> None:
        """Define parameters for an embedding model to pass to chains.

        Parameters
        ----------
        name : str
            The name of the embedding model
        dim : int
            The number of dimensions in the model hidden layers

        """
        self.name = name
        self.dim = dim

# load PyTorch Tensors onto GPU or Mac MPS, if possible
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# models
OPENAI_3_LARGE = "text-embedding-3-large" # 3072 dimensions, can project down
OPENAI_3_SMALL = "text-embedding-3-small" # 1536 dimensions, can project down
OPENAI_ADA_2 = "text-embedding-ada-002" # 1536 dimensions, can't project down

MPNET = "sentence-transformers/all-mpnet-base-v2"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
LEGALBERT = "nlpaueb/legal-bert-base-uncased"
BERT = "bert-base-uncased"

MAXTOKENS_OPENAI = 8191
DEFAULT_PARAMS = EncoderParams(OPENAI_3_SMALL, 768)

def get_huggingface_model(model_name: str) -> PreTrainedModel:
    """Get a HuggingFace model based on its name.

    Parameters
    ----------
    model_name : str
        The name of a model on HuggingFace. Usually in username/model_name format.

    Returns
    -------
    PreTrainedModel
        The model, loaded onto a PyTorch device.

    """
    return AutoModel.from_pretrained(model_name).to(device)

def get_huggingface_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Get a HuggingFace tokenizer based on an associated model name.

    Parameters
    ----------
    model_name : str
        The name of a model on HuggingFace. Usually in username/model_name format.

    Returns
    -------
    PreTrainedTokenizerBase
        The tokenizer, to be used with a PreTrainedModel.

    """
    return AutoTokenizer.from_pretrained(model_name)

def get_langchain_embedding_model(params: EncoderParams) -> Embeddings:
    """Get the LangChain class for a given embedding model.

    Parameters
    ----------
    params : EncoderParams
        The parameters defining an embedding model.

    Returns
    -------
    Embeddings
        The class representing the embedding model for use in LangChain

    """
    if params.name in (OPENAI_ADA_2, OPENAI_3_SMALL, OPENAI_3_LARGE):
        args = {"model": params.name}
        if params.dim:
            args["dimensions"] = params.dim
        return OpenAIEmbeddings(**args)
    if params.name in (MPNET, MINILM):
        return HuggingFaceEmbeddings(model_name=params.name,
                                     model_kwargs={"device": device})
    return HuggingFaceHubEmbeddings(model=params.name)

def embed_strs(text: list[str], params: EncoderParams) -> list:
    """Embeds text from a list where each element is within the model max input length.

    Parameters
    ----------
    text : list[str]
        The text entries to embed
    params : EncoderParams
        The embedding model parameters

    Returns
    -------
    list
        A list of lists of floats representing the embedded text

    """
    if params.name in (OPENAI_3_SMALL, OPENAI_ADA_2, OPENAI_3_LARGE):
        return embed_strs_openai(text, params)
    # TODO(Nick): do this (and tokenize_embed_chunks) in batches?
    model = get_huggingface_model(params.name)
    tokenizer = get_huggingface_tokenizer(params.name)
    embedded_tokens = []
    for s in text:
        s_tokens = tokenizer(s, return_tensors="pt",
                             padding=True, truncation=True).to(device)
        with torch.no_grad():
            model_out: BaseModelOutput = model(**s_tokens)
            embedded_tokens.append(model_out.last_hidden_state.mean(dim=1).cpu().numpy())
    return embedded_tokens

def tokenize_embed_chunks(chunks: list[Element], model: PreTrainedModel,
                          tokenizer: PreTrainedTokenizerBase) -> list:
    """Tokenize and embed a list of Elements with a given model and tokenizer.

    Parameters
    ----------
    chunks : list[Element]
        The chunks to tokenize and embed
    model : PreTrainedModel
        The HuggingFace embedding model used to embed chunks
    tokenizer : PreTrainedTokenizerBase
        The tokenizer associated with the model to tokenize chunks for embedding.

    Returns
    -------
    list
        A list of embedded chunks

    """
    chunk_embeddings = []
    for chunk in chunks:
        tokens = tokenizer(
            chunk.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            model_out: BaseModelOutput = model(**tokens)
            embeddings = model_out.last_hidden_state.mean(dim=1)
        chunk_embeddings.append(embeddings.cpu().numpy().squeeze())
    return chunk_embeddings

def token_count(string: str, model: str) -> int:
    """Return the number of tokens in a text string.

    Third-generation embedding models like text-embedding-3-small
    use the cl100k_base encoding.

    Parameters
    ----------
    string : str
        The string whose tokens will be counted
    model : str
        The OpenAI model that will accept the string

    Returns
    -------
    int
        The number of tokens in the given string for the given model

    """
    try:
      encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

def embed_strs_openai(text: list[str], params: EncoderParams) -> list:
    """Embed a list of strings using an OpenAI client.

    Parameters
    ----------
    text : list[str]
        The list of strings to embed
    params : EncoderParams
        The parameters for an OpenAI embedding model

    Returns
    -------
    list
        A list of embedded strings, could be longer than the input list
        if any single string runs out of the models context window (8191 tokens)

    """
    i = 0
    data = []
    client = OpenAI()
    while i < len(text):
        j = i
        tokens = 0
        while (
            j < len(text) and
            (tokens := tokens + token_count(text[j], params.name)) < MAXTOKENS_OPENAI
        ):
            j += 1
        attempt = 1
        num_attempts = 75
        while attempt < num_attempts:
            try:
                args = {"input": text[i:j], "model": params.name}
                if params.dim:
                    args["dimensions"] = params.dim
                response = client.embeddings.create(**args)
                data += [data.embedding for data in response.data]
                i = j
                break
            except APITimeoutError:
                time.sleep(1)
                attempt += 1
    return data
