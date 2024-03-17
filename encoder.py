import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutput
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceHubEmbeddings
from langchain_openai import OpenAIEmbeddings
from unstructured.chunking.base import Element

from openai import OpenAI
import tiktoken
import time

class EncoderParams():
    """Define the embedding model for a Collection"""
    def __init__(self, name: str, dim: int) -> None:
        self.name = name
        self.dim = dim

# load PyTorch Tensors onto GPU or Mac MPS if possible
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# models
OPENAI_3_LARGE = "text-embedding-3-large"
OPENAI_3_SMALL = "text-embedding-3-small"
OPENAI_ADA_2 = "text-embedding-ada-002" # uses 1536 dimensions, cant be changed
MPNET = "sentence-transformers/all-mpnet-base-v2"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
LEGALBERT = "nlpaueb/legal-bert-base-uncased"
BERT = "bert-base-uncased"

DEFAULT_PARAMS = EncoderParams(OPENAI_3_SMALL, 768)

def get_huggingface_model(model_name: str):
    return AutoModel.from_pretrained(model_name).to(device)

def get_huggingface_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

def get_langchain_embedding_function(params: EncoderParams):
    if params.name == OPENAI_ADA_2 or params.name == OPENAI_3_SMALL:
        args = {"model": params.name}
        if params.dim:
            args["dimensions"] = params.dim
        return OpenAIEmbeddings(**args)
    if params.name == MPNET or params.name == MINILM:
        return HuggingFaceEmbeddings(model_name=params.name, model_kwargs={"device": device})
    return HuggingFaceHubEmbeddings(model=params.name)

def embed_strs(text: list[str], params: EncoderParams):
    """Embeds text from a list where each element is within the model max input length.
    
    Args
        text: the text entries to embed
        params: EncoderParams to describe an embedding model    
    """
    if params.name == OPENAI_3_SMALL or params.name == OPENAI_ADA_2 or params.name == OPENAI_3_LARGE:
        return embed_strs_openai(text, params)
    # TODO: do this (and tokenize_embed_chunks) in batches?
    model = get_huggingface_model(params.name)
    tokenizer = get_huggingface_tokenizer(params.name)
    embedded_tokens = []
    for s in text:
        s_tokens = tokenizer(s, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            model_out: BaseModelOutput = model(**s_tokens)
            embedded_tokens.append(model_out.last_hidden_state.mean(dim=1).cpu().numpy())
    return embedded_tokens

def tokenize_embed_chunks(chunks: list[Element], model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
    num_chunks = len(chunks)
    chunk_embeddings = []
    for j, chunk in enumerate(chunks, start=1):
        if j % 1000 == 0:
            print(f"  {num_chunks - j} chunks remaining")
        tokens = tokenizer(chunk.text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            model_out: BaseModelOutput = model(**tokens)
            embeddings = model_out.last_hidden_state.mean(dim=1)
        chunk_embeddings.append(embeddings.cpu().numpy().squeeze())
    return chunk_embeddings
    
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    # For third-generation embedding models like text-embedding-3-small, use the cl100k_base encoding.
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def embed_strs_openai(text: list[str], params: EncoderParams):
    i = 0
    data = []
    client = OpenAI()
    while i < len(text):
        batch_tokens = 0
        j = i
        while j < len(text) and (batch_tokens := batch_tokens + num_tokens_from_string(text[j], "cl100k_base")) < 8191:
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
            except:
                time.sleep(1)
                print(f'  retry attempt #{attempt} of {num_attempts}')
                attempt += 1
    return data