import mimetypes
import os
from csv import reader
from hmac import new
from json import loads
from typing import List

import requests
from bs4 import BeautifulSoup
from fastapi import UploadFile
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from langchain import hub
from langchain.chains import create_retrieval_chain, load_summarize_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import (RecursiveCharacterTextSplitter,
                                     TextSplitter)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.vectorstores import Field, VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)
from pypdf import PdfReader
from unstructured.partition.auto import partition

connection_args = loads(os.environ["Milvus"])
# test connection to db, also needed to use utility functions
connections.connect(uri=connection_args["uri"], token=connection_args["token"])

# TODO: Uncomment these variables before running the sample.
project_id = "h2o-gpt"
location = "us"  # Format is "us" or "eu"
processor_id = "c99e554bb49cf45d"
#processor_display_name = "my" # Must be unique per project, e.g.: "My Processor"

def session_upload_str(reader: str, session_id: str, summary: str, max_chunk_size: int = 1000, chunk_overlap: int = 150):
    documents = [
        Document(
            page_content=page,
            metadata={"source": summary, "page": page_number, "session_id": session_id, "user_summary": summary},
        )
        for page_number, page in enumerate([reader], start=1)
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
        return {"message": f"Failure: expected to upload {num_docs} chunk{'s' if num_docs > 1 else ''} for {summary} but got {len(ids)}"}
    return {"message": f"Success: uploaded {summary} as {num_docs} chunk{'s' if num_docs > 1 else ''}"}

def scrape(site: str, old_urls: list[str], session_id: str): 
    r = requests.get(site) 

    # converting the text 
    s = BeautifulSoup(r.content,"html.parser") 
    urls = []

    for i in s.find_all("a"): 
        href = i.attrs['href'] 
        
        if href.startswith("/"): 
            link = site+href 
        elif href.startswith("http"):
            link = href
        else:
            link = old_urls[0]
            #skip this link

        if link not in old_urls: 
            old_urls.append(link)
            urls.append(link)

    try:
        elements = partition(url=site)
    except:
        elements = partition(url=site, content_type="text/html")
    e_text = "\n\n".join([str(el) for el in elements[:-1]])
    session_upload_str(e_text, session_id, site)

    return urls

def crawl_and_scrape(site: str, session_id: str):
    urls = [site]
    new_urls = scrape(site, urls, session_id)
    while len(new_urls) > 0:
        urls += new_urls
        new_urls = scrape(site, urls, session_id)
    return urls
                
def quickstart_ocr(
    file: UploadFile,
):
    if not file.filename.endswith(".pdf"):
        process_options = documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                language_code="en",
                enable_native_pdf_parsing=True,
            )
        )
    else:
        process_options = documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                language_code="en",
            )
        )

    # You must set the `api_endpoint`if you use a location other than "us".
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com" )
    
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    processor_name = client.processor_path(project_id, location, processor_id)

    # Print the processor information
    print(f"Processor Name: {processor_name}")

    # Load binary data
    raw_document = documentai.RawDocument(
        content=file.file.read(),
        mime_type=mimetypes.guess_type(file.filename)[0], # Refer to https://cloud.google.com/document-ai/docs/file-types for supported file types
    )

    # Configure the process request
    # `processor.name` is the full resource name of the processor, e.g.:
    # `projects/{project_id}/locations/{location}/processors/{processor_id}`
    request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document, process_options=process_options)

    result = client.process_document(request=request)

    # For a full list of `Document` object attributes, reference this page:
    # https://cloud.google.com/document-ai/docs/reference/rest/v1/Document
    document = result.document

    # Read the text recognition output from the processor
    print("The document contains the following text:")
    print(document.text)
    return document.text

# collections by jurisdiction?
US = "USCode"
NC = "NCGeneralStatutes"
SESSION_PDF = "SessionPDF"
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

def check_params(database_name: str, query: str, k: int, session_id: str = None):
    if not collection_exists(database_name):
        return {"message": f"Failure: database {database_name} not found"}
    if not query or query == "":
        return {"message": "Failure: query not found"}
    if k < 0 or k > 10:
        return {"message": "Failure: k out of range"}
    if session_id is None and database_name == SESSION_PDF:
        return {"message": "Failure: missing session ID"}

def query(database_name: str, query: str, k: int = 4, expr: str = None, session_id: str = None):
    if check_params(database_name, query, k, session_id):
        return check_params(database_name, query, k, session_id)
    
    coll = Collection(database_name)
    coll.load()
    SEARCH_PARAMS["data"] = [OpenAIEmbeddings().embed_query(query)]
    SEARCH_PARAMS["limit"] = k

    if expr:
        SEARCH_PARAMS["expr"] = expr
    if session_id:
        session_filter = f"session_id=='{session_id}'"
        # append to existing filter expr or create new filter
        if expr:
            SEARCH_PARAMS["expr"] += f" and {session_filter}"
        else:
            SEARCH_PARAMS["expr"] = session_filter
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
    
def delete_session(session_id: str):
    # non atomic
    if utility.has_collection(SESSION_PDF):
        coll = Collection(SESSION_PDF)
        coll.load()
        return coll.delete(expr=f"session_id == '{session_id}'")

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
    # unstructured chunk lengths are sketchy
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

def upload_pdfs(collection_name: str, directory: str, embedding_dim: int, max_chunk_size: int, chunk_overlap: int, session_id: str = None):
    files = sorted(os.listdir(directory))
    print(f"found {len(files)} file{'s' if len(files) > 1 else ''}")
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
    print(f" inserting {num_docs} chunk{'s' if num_docs > 1 else ''}")
    ids = db.add_documents(documents=documents, embedding=embedding, connection_args=connection_args)
    if num_docs != len(ids):
        print(f" error: expected {num_docs} upload{'s' if num_docs > 1 else ''} but got {len(ids)}")

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
        return {"message": f"Failure: expected to upload {num_docs} chunk{'s' if num_docs > 1 else ''} for {file.filename} but got {len(ids)}"}
    return {"message": f"Success: uploaded {file.filename} as {num_docs} chunk{'s' if num_docs > 1 else ''}"}

def session_upload_ocr(file: UploadFile, session_id: str, summary: str, max_chunk_size: int = 1000, chunk_overlap: int = 150):
    reader = quickstart_ocr(file)
    documents = [
        Document(
            page_content=page,
            metadata={"source": file.filename, "page": page_number, "session_id": session_id, "user_summary": summary},
        )
        for page_number, page in enumerate([reader], start=1)
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
        return {"message": f"Failure: expected to upload {num_docs} chunk{'s' if num_docs > 1 else ''} for {file.filename} but got {len(ids)}"}
    return {"message": f"Success: uploaded {file.filename} as {num_docs} chunk{'s' if num_docs > 1 else ''}"}

def session_source_summaries(session_id: str):
    coll = Collection(SESSION_PDF)
    coll.load()
    q_iter = coll.query_iterator(expr=f"session_id=='{session_id}'", output_fields= ["source", "ai_summary", "user_summary"])
    source_summary = {}
    while True:
        res = q_iter.next()
        if len(res) == 0:
            q_iter.close()
            break
        for item in res:
            if item["source"] not in source_summary:
                source_summary[item["source"]] = {"ai_summary": item["ai_summary"], "user_summary": item["user_summary"]}
    return source_summary

def qa(database_name: str, query: str, k: int = 4, session_id: str = None):
    """
    Runs query on database_name and returns an answer along with the top k source chunks

    This should be similar to db_bot, but using newer langchain LCEL

    Args
        database_name: the name of a pymilvus.Collection
        query: the user query
        k: return the top k chunks
        session_id: the session id for filtering session data

    Returns dict with success or failure message and a result if success
    """
    if check_params(database_name, query, k, session_id):
        return check_params(database_name, query, k, session_id)
    
    db = load_db(database_name)
    retrieval_qa_chat_prompt: ChatPromptTemplate = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = OpenAI(temperature=0)
    if session_id:
        retriever = FilteredRetriever(vectorstore=db.as_retriever(), session_filter=session_id, search_kwargs={"k": k})
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
    session_filter: str
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.vectorstore.get_relevant_documents(query=query)
        return [doc for doc in results if doc.metadata['session_id'] == self.session_filter]
    


    