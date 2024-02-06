import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import VectorStoreRetriever, Field
from langchain.docstore.document import Document
from langchain_community.vectorstores.milvus import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from pymilvus import utility, connections, Collection, CollectionSchema, FieldSchema, DataType
from json import loads
from typing import List

connection_args = loads(os.environ["Milvus"])
# test connection to db, also needed to use utility functions
connections.connect(uri=connection_args["uri"], token=connection_args["token"])

# collections by jurisdiction?
US = "USCode"
NC = "NCGeneralStatutes"
USER = "UserData"

def load_db(collection_name: str):
    return Milvus(
        embedding_function=OpenAIEmbeddings(),
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

class FilteredRetriever(VectorStoreRetriever):
    vectorstore: VectorStoreRetriever
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)
    user_filter: str
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.vectorstore.get_relevant_documents(query=query)
        return [doc for doc in results if doc.metadata['user'] == self.user_filter]

# firebase user uploads
# from firebase_admin import credentials, firestore, initialize_app, storage
# cred = credentials.Certificate("../../creds.json")
# initialize_app(cred, {"storageBucket": 'openprobono.appspot.com'})
# db = firestore.client()
# bucket = storage.bucket()
# blob = bucket.blob('usc04@118-30.pdf')
# with open(os.getcwd() + "/data/US/usc04@118-30.pdf", 'rb') as my_file:
#     blob.upload_from_file(my_file)