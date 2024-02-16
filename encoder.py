import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutput
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceHubEmbeddings
from langchain_openai import OpenAIEmbeddings
from unstructured.chunking.base import Element

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OPENAI = "openai"
MPNET = "sentence-transformers/all-mpnet-base-v2"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
LEGALBERT = "nlpaueb/legal-bert-base-uncased"
BERT = "bert-base-uncased"

def get_model(model_name: str):
    return AutoModel.from_pretrained(model_name).to(device)

def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

def embedding_function(model_name: str):
    if model_name == OPENAI:
        return OpenAIEmbeddings()
    elif model_name == MPNET or model_name == MINILM:
        return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
    else:
        return HuggingFaceHubEmbeddings(model=model_name)

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

def embed_query(query: str, model_name: str):
    if model_name == OPENAI:
        return [OpenAIEmbeddings().embed_query(query)]
    model = get_model(model_name)
    tokenizer = get_tokenizer(model_name)
    query_tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        model_out: BaseModelOutput = model(**query_tokens)
        return model_out.last_hidden_state.mean(dim=1).cpu().numpy()