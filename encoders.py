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
OPENAI_MODELS = {OPENAI_3_LARGE, OPENAI_3_SMALL, OPENAI_ADA_2}

MPNET = "sentence-transformers/all-mpnet-base-v2" # max input length 384
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_TRANSFORMERS = {MPNET, MINILM}

LEGALBERT = "nlpaueb/legal-bert-base-uncased"
BERT = "bert-base-uncased"
SFR_MISTRAL = "Salesforce/SFR-Embedding-Mistral" # 4096 dimensions, 32768 max tokens
UAE_LARGE = "WhereIsAI/UAE-Large-V1" # 1024 dimensions, 512 max tokens

# TODO(Nick): lookup dimensions and max tokens for huggingface models
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

def get_langchain_embedding_model(params: EncoderParams = DEFAULT_PARAMS) -> Embeddings:
    """Get the LangChain class for a given embedding model.

    Parameters
    ----------
    params : EncoderParams, optional
        The parameters defining an embedding model, by default
        OPENAI_3_SMALL with 768 dimensions

    Returns
    -------
    Embeddings
        The class representing the embedding model for use in LangChain

    """
    if params.name in OPENAI_MODELS:
        args = {"model": params.name}
        if params.dim:
            args["dimensions"] = params.dim
        return OpenAIEmbeddings(**args)
    if params.name in SENTENCE_TRANSFORMERS:
        return HuggingFaceEmbeddings(model_name=params.name,
                                     model_kwargs={"device": device})
    return HuggingFaceHubEmbeddings(model=params.name)

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

def embed_strs(text: list[str], params: EncoderParams = DEFAULT_PARAMS) -> list:
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
    if params.name in OPENAI_MODELS:
        return embed_strs_openai(text, params)
    if params.name in SENTENCE_TRANSFORMERS:
        return embed_strs_sentencetransformers(text, params)
    if params.name == SFR_MISTRAL:
        return embed_strs_mistral(text, params)
    return embed_strs_huggingface(text, params)

def embed_strs_huggingface(text: list[str], params: EncoderParams) -> list:
    """Embed a list of strings using a HuggingFace model.

    Parameters
    ----------
    text : list[str]
        The list of strings to embed
    params : EncoderParams
        The parameters for a HuggingFace embedding model

    Returns
    -------
    list
        A list of floats representing embedded strings

    """
    model = get_huggingface_model(params.name)
    tokenizer = get_huggingface_tokenizer(params.name)
    text_tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        model_out: BaseModelOutput = model(**text_tokens)
    # might want to call ndarray.squeeze() on this
    return list(model_out.last_hidden_state.mean(dim=1).cpu().numpy())

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
        A list of floats representing embedded strings

    """
    max_tokens = 8191
    max_array_size = 2048
    i = 0
    data = []
    client = OpenAI()
    num_strings = len(text)
    while i < num_strings:
        j = i
        tokens = 0
        while (
            j < num_strings and
            j - i < max_array_size and
            (tokens := tokens + token_count(text[j], params.name)) < max_tokens
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

# from https://huggingface.co/sentence-transformers/all-mpnet-base-v2
def embed_strs_sentencetransformers(text: list[str], params: EncoderParams) -> list:
    """Embed a list of strings using a sentence-transformers model.

    Parameters
    ----------
    text : list[str]
        The list of strings to embed
    params : EncoderParams
        The parameters for a sentence-transformers embedding model

    Returns
    -------
    list
        A list of floats representing embedded strings

    """
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output: BaseModelOutput,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size(),
        ).float()
        num = torch.sum(token_embeddings * input_mask_expanded, 1)
        denom = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return num / denom
    tokenizer = get_huggingface_tokenizer(params.name)
    model = get_huggingface_model(params.name)
    encoded_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt").to(device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    # Normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return list(sentence_embeddings.cpu().numpy())

# from https://huggingface.co/Salesforce/SFR-Embedding-Mistral
def embed_strs_mistral(text: list[str], params: EncoderParams) -> list:
    """Embed a list of strings using the SFR Mistral model.

    Parameters
    ----------
    text : list[str]
        The list of strings to embed
    params : EncoderParams
        The parameters for the SFR Mistral embedding model

    Returns
    -------
    list
        A list of floats representing embedded strings

    """
    def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(
                batch_size,
                device=last_hidden_states.device,
            ), sequence_lengths,
        ]

    # load model and tokenizer
    tokenizer = get_huggingface_tokenizer(params.name)
    model = get_huggingface_model(params.name)

    # get the embeddings
    max_length = 4096
    batch_dict = tokenizer(
        text,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    outputs = model(**batch_dict)
    embeddings = last_token_pool(
        outputs.last_hidden_state,
        batch_dict["attention_mask"],
    )

    # normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return list(embeddings.cpu().numpy())
