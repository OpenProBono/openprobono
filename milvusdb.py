import os
from langchain import hub
from langchain.chains import create_retrieval_chain, load_summarize_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore, Field
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
SESSION_PDF = "SessionPDF"
COLLECTIONS = {US, NC}
PDF = "PDF"
HTML = "HTML"
COLLECTION_TYPES = {
    US: PDF,
    NC: PDF,
    SESSION_PDF: PDF
}
OUTPUT_FIELDS = {PDF: ["page"], HTML: []}
SEARCH_PARAMS = {
    "anns_field": "vector",
    "param": {"metric_type": "L2", "M": 8, "efConstruction": 64},
    "output_fields": ["text", "source"]
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

def check_params(collection_name: str, query: str, k: int, session_id: str = None):
    if not collection_exists(collection_name):
        return {"message": f"Failure: collection {collection_name} not found"}
    if not query or query == "":
        return {"message": "Failure: query not found"}
    if k < 1 or k > 16384:
        return {"message": f"Failure: k = {k} out of range [1, 16384]"}
    if session_id is None and collection_name == SESSION_PDF:
        return {"message": "Failure: session_id not found"}

def query(collection_name: str, query: str, k: int = 4, expr: str = None, session_id: str = None):
    if check_params(collection_name, query, k, session_id):
        return check_params(collection_name, query, k, session_id)
    
    coll = Collection(collection_name)
    coll.load()
    search_params = SEARCH_PARAMS
    search_params["data"] = [OpenAIEmbeddings().embed_query(query)]
    search_params["limit"] = k
    search_params["output_fields"] += OUTPUT_FIELDS[COLLECTION_TYPES[collection_name]]

    if expr:
        search_params["expr"] = expr
    if session_id:
        session_filter = f"session_id=='{session_id}'"
        # append to existing filter expr or create new filter
        if expr:
            search_params["expr"] += f" and {session_filter}"
        else:
            search_params["expr"] = session_filter
    res = coll.search(**search_params)
    if res:
        # on success, returns a list containing a single inner list containing result objects
        if len(res) == 1:
            hits = res[0]
            return {"message": "Success", "result": hits}
        return {"message": "Success", "result": res}
    return {"message": "Failure: unable to complete search"}

def delete_expr(collection_name: str, expr: str):
    """
    Deletes database entries according to expr.
    Not atomic, i.e. may only delete some then fail: https://milvus.io/docs/delete_data.md#Delete-Entities.
    
    Args
        collection_name: the name of a pymilvus.Collection
        expr: a boolean expression to specify conditions for ANN search
    """
    if utility.has_collection(collection_name):
        coll = Collection(collection_name)
        coll.load()
        ids = coll.delete(expr=expr)
        return {"message": f"Success: deleted {ids.delete_count} chunks"}

def create_collection(collection_name: str, directory: str, extra_fields: List = [], embedding_dim: int = 1536, max_src_length: int = 256) -> bool:
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
    # unstructured chunk lengths are sketchy
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, description="The source text", max_length=2 * embedding_dim)
    embedding_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim, description="The embedded text")
    source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, description="The source file", max_length=max_src_length)
    schema = CollectionSchema(fields=[pk_field, embedding_field, text_field, source_field] + extra_fields,
                              auto_id=True, enable_dynamic_field=True, description=description)
    coll = Collection(name=collection_name, schema=schema)
    index_params = {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 8, "efConstruction": 64}}
    coll.create_index("vector", index_params=index_params, index_name="HnswL2M8eFCons64Index")

    return True

def upload_pdfs(collection_name: str, directory: str, embedding_dim: int, max_chunk_size: int, chunk_overlap: int, session_id: str = None):
    files = sorted(os.listdir(directory))
    print(f"found {len(files)} files")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
    db = load_db(collection_name)
    embedding = OpenAIEmbeddings()
    embedding.dimensions = embedding_dim
    for i, fname in enumerate(files, start=1):
        print(f'{i}: {fname}')
        upload_pdf_pypdf(db, directory, fname, text_splitter, embedding, session_id)

def upload_pdf_pypdf(db: Milvus, directory: str, fname: str, text_splitter: TextSplitter, embedding: Embeddings, session_id: str = None):
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
        if session_id:
            documents[j].metadata["session_id"] = session_id
    print(f" inserting {num_docs} chunks")
    ids = db.add_documents(documents=documents, embedding=embedding, connection_args=connection_args)
    if num_docs != len(ids):
        print(f" error: expected {num_docs} uploads but got {len(ids)}")

def session_upload_pdf(file: UploadFile, session_id: str, summary: str, max_chunk_size: int = 1000, chunk_overlap: int = 150):
    if not file.filename.endswith(".pdf"):
        return {"message": f"Failure: {file.filename} is not a PDF file"}
    
    # parse
    reader = PdfReader(file.file)
    documents = [
        Document(
            page_content=page.extract_text(),
            metadata={"source": file.filename, "page": page_number, "session_id": session_id, "user_summary": summary},
        )
        for page_number, page in enumerate(reader.pages, start=1)
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)

    # summarize
    chain = load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce")
    result = chain.invoke({"input_documents": documents[:200]})
    for doc in documents:
        doc.metadata["ai_summary"] = result["output_text"].strip()

    # upload
    ids = load_db(SESSION_PDF).add_documents(documents=documents, embedding=OpenAIEmbeddings(), connection_args=connection_args)
    num_docs = len(documents)
    if num_docs != len(ids):
        return {"message": f"Failure: expected to upload {num_docs} chunks for {file.filename} but got {len(ids)}"}
    return {"message": f"Success: uploaded {file.filename} as {num_docs} chunks"}

def session_source_summaries(session_id: str, batch_size: int = 1000):
    coll = Collection(SESSION_PDF)
    coll.load()
    q_iter = coll.query_iterator(expr=f"session_id=='{session_id}'", output_fields= ["source", "ai_summary", "user_summary"], batch_size=batch_size)
    source_summaries = {}
    res = q_iter.next()
    while len(res) > 0:
        for item in res:
            if item["source"] not in source_summaries:
                source_summaries[item["source"]] = {"ai_summary": item["ai_summary"]}
                if item["user_summary"] != item["source"]:
                    source_summaries[item["source"]]["user_summary"] = item["user_summary"]
        res = q_iter.next()
    q_iter.close()
    return source_summaries

def qa(collection_name: str, query: str, k: int = 4, session_id: str = None):
    """
    Runs query on collection_name and returns an answer along with the top k source chunks

    Args
        collection_name: the name of a pymilvus.Collection
        query: the user query
        k: return the top k chunks
        session_id: the session id for filtering session data

    Returns dict with success message, result, and sources or else failure message
    """
    if check_params(collection_name, query, k, session_id):
        return check_params(collection_name, query, k, session_id)
    
    db = load_db(collection_name)
    retrieval_qa_chat_prompt: ChatPromptTemplate = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = OpenAI(temperature=0)
    if session_id:
        retriever = FilteredRetriever(vectorstore=db, session_filter=session_id, search_kwargs={"k": k})
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
    vectorstore: VectorStore
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)
    session_filter: str
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = []
        k = self.search_kwargs["k"]
        # TODO: determine if get_relevant_documents() kwargs param supports filtering by metadata
        # double k on each call to get_relevant_documents() until there are k filtered documents
        while len(docs) < k:
            results = self.vectorstore.as_retriever(search_kwargs={"k": k}).get_relevant_documents(query=query)
            docs += [doc for doc in results if doc.metadata['session_id'] == self.session_filter and doc not in docs]
            k = 2 * k
        return docs[:self.search_kwargs["k"]]