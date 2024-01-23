import torch
from transformers import BertModel, BertTokenizer
from transformers.modeling_outputs import BaseModelOutput
import faiss
import pdfs

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
documents = pdfs.uscode()
# tokenize and obtain embeddings
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
        tokens = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            model_out: BaseModelOutput = model(**tokens)
            embeddings = model_out.last_hidden_state.mean(dim=1)
        chunk_embeddings.append(embeddings)
    document_embeddings.append(chunk_embeddings)

# build an index for each document
print(f'Building indices')
indices = []
for chunk_embeddings in document_embeddings:
    chunk_embeddings = torch.cat(chunk_embeddings)
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    indices.append(index)

def semantic_search(query: str, indices: list[faiss.IndexFlatL2], top_k: int = 3):
    query_tokens = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        model_out: BaseModelOutput = model(**query_tokens)
        query_embedding = model_out.last_hidden_state.mean(dim=1).numpy()
    results = []
    for i, document_index in enumerate(indices):
        print(f'Searching Document {i + 1}')
        _, similar_indices = document_index.search(query_embedding, top_k)
        similar_chunks = [(i, similar_index) for similar_index in similar_indices[0]]
        results.extend(similar_chunks)
    # sort by similarity score
    results.sort(key=lambda x: x[1])
    top_k_results = results[:top_k]
    top_k_chunks = [(documents[i][j], i) for i, j in top_k_results]
    return top_k_chunks

# example
query = 'Legal information regarding mutilating the flag'
results = semantic_search(query, indices)
print(f'Query: {query}')
print('Search Results:')
for i, (chunk, document_idx) in enumerate(results, start=1):
    print(f' {i}. Document {document_idx + 1}, Chunk: {chunk}')