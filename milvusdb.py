import os
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader
from langchain_core.vectorstores import VectorStoreRetriever, Field
from langchain.docstore.document import Document
from langchain_community.vectorstores.milvus import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_openai import OpenAIEmbeddings
from pymilvus import utility, connections, Collection, CollectionSchema, FieldSchema, DataType
from json import loads
from typing import List
from fastapi import UploadFile

connection_args = loads(os.environ["Milvus"])
# test connection to db, also needed to use utility functions
connections.connect(uri=connection_args["uri"], token=connection_args["token"])

# collections by jurisdiction?
US = "USCode"
NC = "NCGeneralStatutes"
USER = "UserData"
SEARCH_PARAMS = {
    "anns_field": "vector",
    "param": {"metric_type": "L2", "M": 8, "efConstruction": 64},
    "output_fields": ["text", "source", "page"]
}

def load_db(collection_name: str):
    return Milvus(
        embedding_function=OpenAIEmbeddings(),
        collection_name=collection_name,
        connection_args=connection_args,
        auto_id=True
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
    SEARCH_PARAMS["data"] = [OpenAIEmbeddings().embed_query(query)]
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

def create_collection_pdf(collection_name: str, directory: str, embedding_dim: int = 1536,
                          max_chunk_size: int = 1000, chunk_overlap: int = 150, max_src_length: int = 256) -> bool:
    if utility.has_collection(collection_name):
        print(f"error: collection {collection_name} already exists")
        return False
    if not os.path.exists(os.path.join(directory, "description.txt")):
        print("""error: a description.txt file containing a brief description of the 
              collection must be in the same directory as the data""")
        return False
    
    with open(os.path.join(directory, "description.txt")) as f:
        description = ' '.join(f.readlines())
    
    # TODO: if possible, support custom embedding size for huggingface models
    # TODO: support other OpenAI models
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

    upload_pdfs(collection_name, directory, embedding_dim, max_chunk_size, chunk_overlap)
    return True

def upload_pdfs(collection_name: str, directory: str, embedding_dim: int, max_chunk_size: int, chunk_overlap: int, user: str = None):
    files = sorted(os.listdir(directory))
    print(f"found {len(files)} file{'s' if len(files) > 1 else ''}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
    db = load_db(collection_name)
    embedding = OpenAIEmbeddings()
    embedding.dimensions = embedding_dim
    for i, fname in enumerate(files, start=1):
        print(f'{i}: {fname}')
        upload_pdf_pypdf(db, directory, fname, text_splitter, embedding, user)

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
    print(f" inserting {num_docs} chunk{'s' if num_docs > 1 else ''}")
    ids = db.add_documents(documents=documents, embedding=embedding, connection_args=connection_args)
    if num_docs != len(ids):
        print(f" error: expected {num_docs} upload{'s' if num_docs > 1 else ''} but got {len(ids)}")

def userupload_pdf(file: UploadFile, max_chunk_size: int, chunk_overlap: int, user: str):
    reader = PdfReader(file.file)
    documents = [
        Document(
            page_content=page.extract_text(),
            metadata={"source": file.filename, "page": page_number, "user": user},
        )
        for page_number, page in enumerate(reader.pages)
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)
    ids = load_db(USER).add_documents(documents=documents, embedding=OpenAIEmbeddings(), connection_args=connection_args)
    if len(documents) != len(ids):
        return {"message": f"Failure: expected to upload {len(documents)} chunk{'s' if len(documents) > 1 else ''} for {file.filename} but got {len(ids)}"}
    return {"message": f"Success: uploaded {file.filename} as {len(documents)} chunk{'s' if len(documents) > 1 else ''}"}

def qa(database_name: str, query: str, k: int = 4, user: str = None):
    """
    Runs query on database_name and returns an answer along with the top k source chunks

    This should be similar to db_bot, but using newer langchain LCEL

    Args
        database_name: the name of a pymilvus.Collection
        query: the user query
        k: return the top k chunks
        user: the username for filtering user data

    Returns dict with success or failure message and a result if success
    """
    if check_params(database_name, query, k, user):
        return check_params(database_name, query, k, user)
    
    db = load_db(database_name)
    retrieval_qa_chat_prompt: ChatPromptTemplate = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = OpenAI(temperature=0)
    if user:
        retriever = FilteredRetriever(vectorstore=db.as_retriever(), user_filter=user, search_kwargs={"k": k})
    else:
        retriever = db.as_retriever(search_kwargs={"k": k})
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    result = retrieval_chain.invoke({"input": query})
    cited_sources = []
    for doc in result["context"]:
        cited_sources.append({"source": doc.metadata["source"], "page": doc.metadata["page"]})
    return {"message": "Success", "result": {"answer": result["answer"].strip(), "sources": cited_sources}}

class FilteredRetriever(VectorStoreRetriever):
    vectorstore: VectorStoreRetriever
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)
    user_filter: str
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.vectorstore.get_relevant_documents(query=query)
        return [doc for doc in results if doc.metadata['user'] == self.user_filter]