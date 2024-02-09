import os
from langchain.embeddings.base import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import VectorStoreRetriever, Field
from langchain.docstore.document import Document
from langchain_community.vectorstores.milvus import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from pymilvus import utility, connections, Collection, CollectionSchema, FieldSchema, DataType
from json import load
from typing import List
import legalbert
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.basic import chunk_elements

with open("milvus_config.json") as f:
    connection_args = load(f)
# test connection to db, also needed to use utility functions
connections.connect(uri=connection_args["uri"], token=connection_args["token"])

# collections by jurisdiction?
US = "USCode"
NC = "NCGeneralStatutes"
USER = "UserData"

def load_db(collection_name: str, embedding_function: Embeddings = OpenAIEmbeddings()):
    return Milvus(
        embedding_function=embedding_function,
        collection_name=collection_name,
        connection_args=connection_args
    )

def collection_exists(collection_name: str) -> bool:
    return utility.has_collection(collection_name)

def create_collection_pdf(collection_name: str, directory: str, embedding_size: int = 1536,
                          chunk_size: int = 1000, chunk_overlap: int = 150, max_src_length: int = 256) -> bool:
    if utility.has_collection(collection_name):
        print(f"error: collection {collection_name} already exists")
        return False
    if not os.path.exists(os.path.join(directory, "description.txt")):
        print("""error: a description.txt file containing a brief description of the 
              collection must be in the same directory as the data""")
        return False
    
    with open(os.path.join(directory, "description.txt")) as f:
        description = ' '.join(f.readlines())
    
    # define schema, create collection, create index on vectors
    pk_field = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, description="The primary key", auto_id=True)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, description="The source text", max_length=2 * chunk_size)
    embedding_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_size, description="The embedded text")
    source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, description="The source file", max_length=max_src_length)
    page_field = FieldSchema(name="page", dtype=DataType.INT16, description="The page number")
    schema = CollectionSchema(fields=[pk_field, embedding_field, text_field, source_field, page_field],
                              auto_id=True, enable_dynamic_field=True, description=description)
    coll = Collection(name=collection_name, schema=schema)
    index_params = {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 8, "efConstruction": 64}}
    coll.create_index("vector", index_params=index_params, index_name="HnswL2M8eFCons64Index")

    upload_pdfs(collection_name, directory, chunk_size, chunk_overlap)
    return True

def upload_pdfs(collection_name: str, directory: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    db = load_db(collection_name)
    for i, fname in enumerate(sorted(os.listdir(directory)), start=1):
        print(f'{i}: {fname}')
        upload_pdf(db, directory, fname, text_splitter)

def upload_pdf(db: Milvus, directory: str, fname: str,
               text_splitter: TextSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150),
               user: str = None):
    if not fname.endswith(".pdf"):
        return
    
    loader = PyPDFLoader(directory + fname)
    documents = loader.load_and_split(text_splitter=text_splitter)
    num_docs = len(documents)
    print(f" {num_docs} documents")
    for j in range(num_docs):
        # replace filepath with filename
        documents[j].metadata.update({"source": fname})
        # change pages from 0-index to 1-index
        documents[j].metadata.update({"page": documents[j].metadata["page"] + 1})
        # add user data
        if user:
            documents[j].metadata["user"] = user
    ids = db.add_documents(documents=documents, embedding=OpenAIEmbeddings(), connection_args=connection_args)
    if num_docs != len(ids):
        print(f" error: expected {num_docs} uploads but got {len(ids)}")

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

def check_params(database_name: str, query: str, k: int):
    if not collection_exists(database_name):
        return {"message": f"Failure: database {database_name} not found"}
    if not query or query == "":
        return {"message": "Failure: query not found"}
    if k < 0 or k > 10:
        return {"message": "Failure: k out of range"}

def custom_query(database_name: str, query: str, k: int = 4):
    if check_params(database_name, query, k):
        return check_params(database_name, query, k)
    
    coll = Collection(database_name)
    coll.load()
    search_params = {
        "data": legalbert.embed_query(query),
        "anns_field": "vector",
        "param": {"metric_type": "L2", "M": 8, "efConstruction": 64},
        "limit": k
    }
    return coll.search(**search_params)

def custom_create_collection(collection_name: str, directory: str, max_chunk_size: int = 1000, chunk_overlap: int = 150, max_src_length: int = 256) -> bool:
    if utility.has_collection(collection_name):
        print(f"error: collection {collection_name} already exists")
        return False
    if not os.path.exists(os.path.join(directory, "description.txt")):
        print("""error: a description.txt file containing a brief description of the 
              collection must be in the same directory as the data""")
        return False
    
    with open(os.path.join(directory, "description.txt")) as f:
        description = ' '.join(f.readlines())
    
    # define schema, create collection, create index on vectors
    pk_field = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, description="The primary key", auto_id=True)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, description="The source text", max_length=2 * max_chunk_size)
    # NOTE: embeddings are assumed to be 768 dimensions
    embedding_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768, description="The embedded text")
    source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, description="The source file", max_length=max_src_length)
    page_field = FieldSchema(name="page", dtype=DataType.INT16, description="The page number")
    schema = CollectionSchema(fields=[pk_field, embedding_field, text_field, source_field, page_field],
                              auto_id=True, enable_dynamic_field=True, description=description)
    coll = Collection(name=collection_name, schema=schema)
    index_params = {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 8, "efConstruction": 64}}
    coll.create_index("vector", index_params=index_params, index_name="HnswL2M8eFCons64Index")

    custom_upload_pdfs(coll, directory, max_chunk_size, chunk_overlap)
    return True

def custom_upload_pdfs(collection: Collection, directory: str, max_chunk_size: int, chunk_overlap: int):
    for i, fname in enumerate(sorted(os.listdir(directory)), start=1):
        print(f'{i}: {fname}')
        custom_upload_pdf(collection, directory, fname, max_chunk_size, chunk_overlap)

def custom_upload_pdf(collection: Collection, directory: str, fname: str, max_chunk_size: int, chunk_overlap: int):
    if not fname.endswith(".pdf"):
        print(' skipping')
        return
    
    print(' partitioning')
    elements = partition_pdf(filename=directory + fname)
    chunks = chunk_elements(elements, max_characters=max_chunk_size, overlap=chunk_overlap)
    print(f' embedding {len(chunks)} chunks')
    chunk_embeddings = legalbert.tokenize_embed_chunks(chunks)
    data = [
        chunk_embeddings, # vector
        [chunk.text for chunk in chunks], # text
        [fname] * len(chunk_embeddings), # source
        [chunk.metadata.page_number for chunk in chunks] # page number
    ]
    for i in range(0, len(data), 1000):
        batch_vector = data[0][i: i + 1000]
        batch_text = data[1][i: i + 1000]
        batch_source = data[2][i: i + 1000]
        batch_page = data[3][i: i + 1000]
        batch = [batch_vector, batch_text, batch_source, batch_page]
        collection.insert(batch)

class FilteredRetriever(VectorStoreRetriever):
    vectorstore: VectorStoreRetriever
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)
    user_filter: str
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.vectorstore.get_relevant_documents(query=query)
        return [doc for doc in results if doc.metadata['user'] == self.user_filter]

# if custom_create_collection("USCodeLB", os.getcwd() + "/uscode/"):
#    print(custom_query("USCodeLB", "What is the punishment for mutilating the flag?"))