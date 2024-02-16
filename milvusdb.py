import os
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import VectorStoreRetriever, Field
from langchain.docstore.document import Document
from langchain_community.vectorstores.milvus import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_openai import OpenAIEmbeddings
from pymilvus import utility, connections, Collection, CollectionSchema, FieldSchema, DataType
from json import load
from typing import List
import encoder
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.basic import chunk_elements
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from fastapi import UploadFile

with open("milvus_config.json") as f:
    connection_args = load(f)
# test connection to db, also needed to use utility functions
connections.connect(uri=connection_args["uri"], token=connection_args["token"])

# collections by jurisdiction?
US = "USCode"
NC = "NCGeneralStatutes"
USER = "UserData"
TEST = "Test"
COLLECTIONS = {US, NC, USER, TEST}
# pypdf for langchain (openai or huggingface embeddings)
PYPDF = "pypdf"
# unstructured for pytorch (huggingface embeddings)
UNSTRUCTURED = "unstructured"
PDF_LOADERS = {PYPDF, UNSTRUCTURED}
# collection -> encoder mapping
ENCODERS = {NC: encoder.OPENAI,
            US: encoder.OPENAI,
            USER: encoder.OPENAI}
SEARCH_PARAMS = {
    "anns_field": "vector",
    "param": {"metric_type": "L2", "M": 8, "efConstruction": 64},
    "output_fields": ["text", "source", "page"]
}

def load_db(collection_name: str):
    return Milvus(
        embedding_function=encoder.embedding_function(ENCODERS[collection_name]),
        collection_name=collection_name,
        connection_args=connection_args
    )

def collection_exists(collection_name: str) -> bool:
    return utility.has_collection(collection_name)

def check_params(database_name: str, query: str, k: int, user: str = None):
    if not collection_exists(database_name):
        return {"message": f"Failure: database {database_name} not found"}
    if not query or query == "":
        return {"message": "Failure: query not found"}
    if k < 0 or k > 10:
        return {"message": "Failure: k out of range"}
    if user is None and database_name == USER:
        return {"message": "Failure: missing user"}

def query(database_name: str, query: str, k: int = 4, expr: str = None, user: str = None):
    if check_params(database_name, query, k, user):
        return check_params(database_name, query, k, user)
    
    coll = Collection(database_name)
    coll.load()
    SEARCH_PARAMS["data"] = encoder.embed_query(query, ENCODERS[database_name])
    SEARCH_PARAMS["limit"] = k

    if expr:
        SEARCH_PARAMS["expr"] = expr
    if user:
        user_filter = f"user=='{user}'"
        # append to existing filter expr or create new filter
        if expr:
            SEARCH_PARAMS["expr"] += f" and {user_filter}"
        else:
            SEARCH_PARAMS["expr"] = user_filter
    res = coll.search(**SEARCH_PARAMS)
    if res:
        # on success, returns a list containing a single inner list containing result objects
        if len(res) == 1:
            hits = res[0]
            return {"message": "Success", "result": hits}
        return {"message": "Success", "result": res}
    return {"message": "Failure: unable to complete search"}

def delete_file(database_name: str, filename: str):
    # not atomic i.e. may only delete some then fail: https://milvus.io/docs/delete_data.md#Delete-Entities
    if utility.has_collection(database_name):
        coll = Collection(database_name)
        coll.load()
        return coll.delete(expr=f"source == '{filename}'")
    
def delete_user(user: str):
    # non atomic
    if utility.has_collection(USER):
        coll = Collection(USER)
        coll.load()
        return coll.delete(expr=f"user == '{user}'")

def create_collection_pdf(collection_name: str, directory: str, encoder_name: str, embedding_dim: int = None,
                          max_chunk_size: int = 1000, chunk_overlap: int = 150, max_src_length: int = 256, pdf_loader: str = PYPDF) -> bool:
    if utility.has_collection(collection_name):
        print(f"error: collection {collection_name} already exists")
        return False
    if not os.path.exists(os.path.join(directory, "description.txt")):
        print("""error: a description.txt file containing a brief description of the 
              collection must be in the same directory as the data""")
        return False
    if pdf_loader not in PDF_LOADERS:
        print(f"error: unsupported pdf loader {pdf_loader}")
        return False
    if pdf_loader == UNSTRUCTURED and encoder_name == encoder.OPENAI:
        print("error: unstructured parse only works with huggingface models, openai not yet supported")
        return False
    
    with open(os.path.join(directory, "description.txt")) as f:
        description = ' '.join(f.readlines())
    
    # TODO: if possible, support custom embedding size for huggingface models
    # TODO: support other OpenAI models
    if encoder_name != encoder.OPENAI:
        model = encoder.get_model(encoder_name)
        embedding_dim = model.config.hidden_size
    elif embedding_dim is None:
        embedding_dim = 1536
    # define schema, create collection, create index on vectors
    pk_field = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, description="The primary key", auto_id=True)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, description="The source text", max_length=2 * embedding_dim)
    embedding_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim, description="The embedded text")
    source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, description="The source file", max_length=max_src_length)
    page_field = FieldSchema(name="page", dtype=DataType.INT16, description="The page number")
    schema = CollectionSchema(fields=[pk_field, embedding_field, text_field, source_field, page_field],
                              auto_id=True, enable_dynamic_field=True, description=description)
    coll = Collection(name=collection_name, schema=schema)
    index_params = {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 8, "efConstruction": 64}}
    coll.create_index("vector", index_params=index_params, index_name="HnswL2M8eFCons64Index")

    # add collection -> encoder mapping
    if collection_name not in ENCODERS:
        ENCODERS[collection_name] = encoder_name
    upload_pdfs(collection_name, directory, pdf_loader, encoder_name, embedding_dim, max_chunk_size, chunk_overlap)
    return True

def upload_pdfs(collection_name: str, directory: str, pdf_loader: str, encoder_name: str, embedding_dim: int, max_chunk_size: int, chunk_overlap: int,
                user: str = None):
    files = sorted(os.listdir(directory))
    print(f"found {len(files)} files")
    if pdf_loader == PYPDF:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
        db = load_db(collection_name)
        embedding = encoder.embedding_function(encoder_name)
        if isinstance(embedding, OpenAIEmbeddings):
            embedding.dimensions = embedding_dim
        for i, fname in enumerate(files, start=1):
            print(f'{i}: {fname}')
            upload_pdf_pypdf(db, directory, fname, text_splitter, embedding, user)
    else: # unstructured
        coll = Collection(name=collection_name)
        model = encoder.get_model(encoder_name)
        tokenizer = encoder.get_tokenizer(encoder_name)
        for i, fname in enumerate(files, start=1):
            print(f'{i}: {fname}')
            upload_pdf_unstructured(coll, directory, fname, max_chunk_size, chunk_overlap, model, tokenizer, user=user)

def upload_pdf_pypdf(db: Milvus, directory: str, fname: str, text_splitter: TextSplitter, embedding: Embeddings, user: str = None):
    if not fname.endswith(".pdf"):
        print(' skipping')
        return
    
    loader = PyPDFLoader(directory + fname)
    print(" partitioning and chunking")
    documents = loader.load_and_split(text_splitter=text_splitter)
    num_docs = len(documents)
    for j in range(num_docs):
        # replace filepath with filename
        documents[j].metadata.update({"source": fname})
        # change pages from 0-index to 1-index
        documents[j].metadata.update({"page": documents[j].metadata["page"] + 1})
        # add user data
        if user:
            documents[j].metadata["user"] = user
    print(f" inserting {num_docs} chunks")
    ids = db.add_documents(documents=documents, embedding=embedding, connection_args=connection_args)
    if num_docs != len(ids):
        print(f" error: expected {num_docs} uploads but got {len(ids)}")

def userupload_pdf(file: UploadFile, max_chunk_size: int, chunk_overlap: int, user: str):
    elements = partition_pdf(file=file.file)
    chunks = chunk_elements(elements, max_characters=max_chunk_size, overlap=chunk_overlap)
    documents = []
    for chunk in chunks:
        doc = Document(chunk.text)
        doc.metadata["source"] = file.filename
        doc.metadata["page"] = chunk.metadata.page_number
        doc.metadata["user"] = user
        documents.append(doc)
    ids = load_db(USER).add_documents(documents=documents, embedding=OpenAIEmbeddings(), connection_args=connection_args)
    if len(documents) != len(ids):
        return {"message": f"Failure: expected {len(documents)} uploads but got {len(ids)}"}
    return {"message": f"Success: {len(documents)} chunks uploaded"}

def upload_pdf_unstructured(collection: Collection, directory: str, fname: str, max_chunk_size: int, chunk_overlap: int, model: PreTrainedModel,
                            tokenizer: PreTrainedTokenizerBase, batch_size: int = 1000, user: str = None):
    if not fname.endswith(".pdf"):
        print(' skipping')
        return
    
    print(' partitioning')
    elements = partition_pdf(filename=directory + fname)
    print(f' chunking {len(elements)} partitioned elements')
    chunks = chunk_elements(elements, max_characters=max_chunk_size, overlap=chunk_overlap)
    print(f' embedding {len(chunks)} chunks')
    chunk_embeddings = encoder.tokenize_embed_chunks(chunks, model, tokenizer)
    data = [
        chunk_embeddings, # vector
        [chunk.text for chunk in chunks], # text
        [fname] * len(chunk_embeddings), # source
        [chunk.metadata.page_number for chunk in chunks] # page number
    ]
    num_batches = len(data[0]) / batch_size if len(data[0]) % batch_size == 0 else len(data[0]) // batch_size + 1
    print(f' inserting {num_batches} batch{"es" if num_batches > 1 else ""} of embeddings')
    for i in range(0, len(data), batch_size):
        batch_vector = data[0][i: i + batch_size]
        batch_text = data[1][i: i + batch_size]
        batch_source = data[2][i: i + batch_size]
        batch_page = data[3][i: i + batch_size]
        batch = [batch_vector, batch_text, batch_source, batch_page]
        if user:
            batch.append([user] * batch_size)
        collection.insert(batch)

class FilteredRetriever(VectorStoreRetriever):
    vectorstore: VectorStoreRetriever
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)
    user_filter: str
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.vectorstore.get_relevant_documents(query=query)
        return [doc for doc in results if doc.metadata['user'] == self.user_filter]