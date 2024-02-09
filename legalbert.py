import torch
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

model_name = 'nlpaueb/legal-bert-base-uncased'
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def tokenize_embed_documents(documents: list):
    document_embeddings = []
    print(f'Embedding {len(documents)} documents')
    for i, document_chunks in enumerate(documents, start=1):
        num_chunks = len(document_chunks)
        print(f' {i}: {num_chunks} chunks')
        i += 1
        chunk_embeddings = []
        for j, chunk in enumerate(document_chunks, start=1):
            if j % 1000 == 0:
                print(f'  {num_chunks - j} chunks remaining')
            tokens = tokenizer(chunk.text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                model_out: BaseModelOutput = model(**tokens)
                embeddings = model_out.last_hidden_state.mean(dim=1)
            chunk_embeddings.append(embeddings)
        document_embeddings.append(chunk_embeddings)
    return document_embeddings

def tokenize_embed_chunks(chunks: list):
    num_chunks = len(chunks)
    chunk_embeddings = []
    for j, chunk in enumerate(chunks, start=1):
        if j % 1000 == 0:
            print(f'  {num_chunks - j} chunks remaining')
        tokens = tokenizer(chunk.text, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            model_out: BaseModelOutput = model(**tokens)
            embeddings = model_out.last_hidden_state.mean(dim=1)
        chunk_embeddings.append(embeddings.cpu().numpy().squeeze())
    return chunk_embeddings

def embed_query(query: str):
    query_tokens = tokenizer(query, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        model_out: BaseModelOutput = model(**query_tokens)
        return model_out.last_hidden_state.mean(dim=1).cpu().numpy()