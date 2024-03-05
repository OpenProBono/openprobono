import mimetypes
import os
from csv import reader
from hmac import new
from json import loads
from signal import SIGTERM
from typing import List

import requests
from bs4 import BeautifulSoup
from fastapi import UploadFile
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from langchain import hub
from langchain.chains import create_retrieval_chain, load_summarize_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore, Field
from langchain_openai import OpenAIEmbeddings
from langchain_openai.llms import OpenAI as LangChainOpenAI
from pypdf import PdfReader
from unstructured.chunking.base import Element
from unstructured.chunking.basic import chunk_elements
from unstructured.partition.auto import partition, partition_text
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.milvus import Milvus
from networkx import circular_layout
from pymilvus import utility, connections, Collection, CollectionSchema, FieldSchema, DataType
from sqlalchemy import desc

from openai import OpenAI
import tiktoken
import time

connection_args = loads(os.environ["Milvus"])
# test connection to db, also needed to use utility functions
connections.connect(uri=connection_args["uri"], token=connection_args["token"])

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
    chain = load_summarize_chain(LangChainOpenAI(temperature=0), chain_type="map_reduce")
    result = chain.invoke({"input_documents": documents[:200]})
    for doc in documents:
        doc.metadata["ai_summary"] = result["output_text"].strip()

    # upload
    ids = load_db(SESSION_PDF).add_documents(documents=documents, embedding=OpenAIEmbeddings(), connection_args=connection_args)
    num_docs = len(documents)
    if num_docs != len(ids):
        return {"message": f"Failure: expected to upload {num_docs} chunk{'s' if num_docs > 1 else ''} for {summary} but got {len(ids)}"}
    return {"message": f"Success: uploaded {summary} as {num_docs} chunk{'s' if num_docs > 1 else ''}"}

def collection_upload_str(reader: str, collection: str, site: str, max_chunk_size: int = 1000, chunk_overlap: int = 150):
    documents = [
        Document(
            page_content=page,
            metadata={"source": site},
        )
        for page_number, page in enumerate([reader], start=1)
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)

    # summarize
    chain = load_summarize_chain(LangChainOpenAI(temperature=0), chain_type="map_reduce")
    result = chain.invoke({"input_documents": documents[:200]})
    for doc in documents:
        doc.metadata["ai_summary"] = result["output_text"].strip()

    # upload
    ids = load_db(collection).add_documents(documents=documents, embedding=OpenAIEmbeddings(), connection_args=connection_args)
    num_docs = len(documents)
    if num_docs != len(ids):
        return {"message": f"Failure: expected to upload {num_docs} chunk{'s' if num_docs > 1 else ''} for {site} but got {len(ids)}"}
    return {"message": f"Success: uploaded {site} as {num_docs} chunk{'s' if num_docs > 1 else ''}"}


def scrape(site: str, old_urls: list[str], common_elements: list[str], collection: str, get_links: bool = False): 
    print("site: ", site)
    r = requests.get(site)
    site_base = "//".join(site.split("//")[:-1])
    # converting the text 
    s = BeautifulSoup(r.content,"html.parser") 
    urls = []

    if(get_links):
        for i in s.find_all("a"): 
            if("href" in i.attrs):
                href = i.attrs['href'] 
                
                if href.startswith("/"): 
                    link = site_base+href 
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
    e_text = ""
    for el in elements:
        el = str(el)
        if(el not in common_elements):
            e_text += el + "\n\n"
    print("elements: ", e_text)
    print("site: ", site)
    collection_upload_str(e_text, collection, site)
    return [urls, elements]

def crawl_and_scrape(site: str, collection: str, description: str):
    create_collection(collection, description)
    urls = [site]
    new_urls, common_elements = scrape(site, urls, [], collection, True)
    print("new_urls: ", new_urls)
    while len(new_urls) > 0:
        cur_url = new_urls.pop()
        if site == cur_url[:len(site)]:
            urls.append(cur_url)
            add_urls, common_elements = scrape(cur_url, urls + new_urls, common_elements, collection)
            new_urls += add_urls
    print(urls)
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
CAP = "CAP"
SESSION_PDF = "SessionPDF"
COLLECTIONS = {US, NC, CAP}

PYPDF = "pypdf"
UNSTRUCTURED = "unstructured"
PDF_LOADERS = {PYPDF, UNSTRUCTURED}

PDF = "PDF"
HTML = "HTML"
COLLECTION_TYPES = {
    US: PDF,
    NC: PDF,
    SESSION_PDF: PDF,
    CAP: CAP
}

OUTPUT_FIELDS = {PDF: ["source", "page"], HTML: [], CAP: ["opinion_author", "opinion_type", "case_name_abbreviation", "decision_date", "cite", "court_name", "jurisdiction_name"]}
SEARCH_PARAMS = {
    "anns_field": "vector",
    "param": {"metric_type": "L2", "M": 8, "efConstruction": 64},
    "output_fields": ["text"]
}
INDEX_PARAMS = {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 8, "efConstruction": 64}}

def create_collection(name: str, description: str = "", extra_fields: list[FieldSchema] = [], embedding_dim: int = 768):
    if utility.has_collection(name):
        print(f"error: collection {name} already exists")
        return
    
    # TODO: if possible, support custom embedding size for huggingface models
    # TODO: support other OpenAI models
    # define schema, create collection, create index on vectors
    pk_field = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, description="The primary key", auto_id=True)
    # unstructured chunk lengths are sketchy
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, description="The source text", max_length=2 * embedding_dim)
    embedding_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim, description="The embedded text")
    schema = CollectionSchema(fields=[pk_field, embedding_field, text_field] + extra_fields,
                              auto_id=True, enable_dynamic_field=True, description=description)
    coll = Collection(name=name, schema=schema)
    coll.create_index("vector", index_params=INDEX_PARAMS, index_name="HnswL2M8eFCons64Index")

    # must call coll.load() before query/search
    return coll

# TODO: custom OpenAIEmbeddings embedding dimensions
def load_db(collection_name: str):
    return Milvus(
        embedding_function=OpenAIEmbeddings(),
        collection_name=collection_name,
        connection_args=connection_args,
        auto_id=True
    )


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    # For third-generation embedding models like text-embedding-3-small, use the cl100k_base encoding.
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def embed_strs(text: list[str], openai_engine: str = "text-embedding-3-small", dimensions: int = 768):
    if openai_engine != "text-embedding-3-small":
        print(f"error: openai_engine {openai_engine} is not yet implemented. Need to know the encoding name, max input length, and dimensions.")
        return
    i = 0
    data = []
    client = OpenAI()
    checkpoint = 1000
    while i < len(text):
        if i > checkpoint:
            print(f"  {len(text) - i} chunks remaining")
            checkpoint += 1000
        batch_tokens = 0
        j = i
        while j < len(text) and (batch_tokens := batch_tokens + num_tokens_from_string(text[j], "cl100k_base")) < 8191:
            j += 1
        attempt = 1
        while attempt < 75:
            try:
                response = client.embeddings.create(
                    input=text[i:j],
                    model=openai_engine,
                    dimensions=dimensions
                )
                data += [data.embedding for data in response.data]
                i = j
                break
            except:
                time.sleep(1)
                attempt += 1
    return data

def check_params(collection_name: str, query: str, k: int, session_id: str = None):
    if not utility.has_collection(collection_name):
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
    search_params["data"] = embed_strs([query])
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
    llm = LangChainOpenAI(temperature=0)
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

def upload_pdfs(collection: Collection, directory: str, embedding_dim: int, max_chunk_size: int, chunk_overlap: int,
                session_id: str = None, batch_size: int = 1000, pdf_loader: str = UNSTRUCTURED):
    files = sorted(os.listdir(directory))
    print(f"found {len(files)} files")
    for i, fname in enumerate(files, start=1):
        print(f'{i}: {fname}')
        if not fname.endswith(".pdf"):
            print(" skipping")
            continue
        if pdf_loader == UNSTRUCTURED:
            data = embed_pdf_unstructured(directory, fname, embedding_dim, max_chunk_size, chunk_overlap)
        else: # pypdf
            data = embed_pdf_pypdf(directory, fname, embedding_dim, max_chunk_size, chunk_overlap)
        upload_pdf_chunks(collection, data, session_id, batch_size)

def embed_pdf_pypdf(directory: str, fname: str, embedding_dim: int, max_chunk_size: int, chunk_overlap: int):
    reader = PdfReader(directory + fname)
    elements = []
    print(' extracting text')
    for page in reader.pages:
        element = Element()
        element.text = page.extract_text()
        element.metadata.page_number = page.page_number + 1
        elements.append(element)
    print(f' chunking {len(elements)} extracted elements')
    chunks = chunk_elements(elements, max_characters=max_chunk_size, overlap=chunk_overlap)
    num_chunks = len(chunks)
    text, page_numbers = [], []
    for chunk in chunks:
        text.append(chunk.text)
        page_numbers.append(chunk.metadata.page_number)
    print(f' embedding {num_chunks} chunks')
    # vector, text, source, page
    data = [
        embed_strs(text, dimensions=embedding_dim),
        text,
        [fname] * num_chunks,
        page_numbers
    ]
    return data

def embed_pdf_unstructured(directory: str, fname: str, embedding_dim: int, max_chunk_size: int, chunk_overlap: int):
    print(' partitioning')
    elements = partition(filename=directory + fname)
    print(f' chunking {len(elements)} partitioned elements')
    chunks = chunk_elements(elements, max_characters=max_chunk_size, overlap=chunk_overlap)
    num_chunks = len(chunks)
    text, page_numbers = [], []
    for chunk in chunks:
        text.append(chunk.text)
        page_numbers.append(chunk.metadata.page_number)
    print(f' embedding {num_chunks} chunks')
    # vector, text, source, page
    data = [
        embed_strs(text, dimensions=embedding_dim),
        text,
        [fname] * num_chunks,
        page_numbers
    ]
    return data

def upload_pdf_chunks(collection: Collection, data: list, session_id: str, batch_size: int):
    num_chunks = len(data[0])
    num_batches = num_chunks / batch_size if num_chunks % batch_size == 0 else num_chunks // batch_size + 1
    print(f' inserting batches')
    for i in range(0, num_chunks, batch_size):
        if i % (10 * batch_size) == 0:
            print(f'  {num_batches - (i // batch_size)} batches remaining')
        batch_vector = data[0][i: i + batch_size]
        batch_text = data[1][i: i + batch_size]
        batch_source = data[2][i: i + batch_size]
        batch_page = data[3][i: i + batch_size]
        batch = [batch_vector, batch_text, batch_source, batch_page]
        if session_id:
            batch.append([session_id] * batch_size)
        current_batch_size = len(batch[0])
        res = collection.insert(batch)
        if res.insert_count != current_batch_size:
            print(f'  error: expected {current_batch_size} insertions but got {res.insert_count} for pages {batch_page[0]}...{batch_page[-1]}')

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
    chain = load_summarize_chain(LangChainOpenAI(temperature=0), chain_type="map_reduce")
    result = chain.invoke({"input_documents": documents[:200]})
    for doc in documents:
        doc.metadata["ai_summary"] = result["output_text"].strip()

    # upload
    ids = load_db(SESSION_PDF).add_documents(documents=documents, embedding=OpenAIEmbeddings(), connection_args=connection_args)
    num_docs = len(documents)
    if num_docs != len(ids):
        return {"message": f"Failure: expected to upload {num_docs} chunks for {file.filename} but got {len(ids)}"}
    return {"message": f"Success: uploaded {file.filename} as {num_docs} chunks"}

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
    chain = load_summarize_chain(LangChainOpenAI(temperature=0), chain_type="map_reduce")
    result = chain.invoke({"input_documents": documents[:200]})
    for doc in documents:
        doc.metadata["ai_summary"] = result["output_text"].strip()

    # upload
    ids = load_db(SESSION_PDF).add_documents(documents=documents, embedding=OpenAIEmbeddings(), connection_args=connection_args)
    num_docs = len(documents)
    if num_docs != len(ids):
        return {"message": f"Failure: expected to upload {num_docs} chunk{'s' if num_docs > 1 else ''} for {file.filename} but got {len(ids)}"}
    return {"message": f"Success: uploaded {file.filename} as {num_docs} chunk{'s' if num_docs > 1 else ''}"}

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
    
def cap_data():
    embedding_dim = 768
    collection_name = "CAP"
    # description = "Case text and metadata provided by the Caselaw Access Project. Version 20210921."
    # caseid_field = FieldSchema(name="case_id", dtype=DataType.INT64, description="The id of the case on the CAP API")
    # type_field = FieldSchema(name="opinion_type", dtype=DataType.VARCHAR, description="The opinion type", max_length=128)
    # author_field = FieldSchema(name="opinion_author", dtype=DataType.VARCHAR, description="The opinion author", max_length=embedding_dim)
    # casenameabbr_field = FieldSchema(name="case_name_abbreviation", dtype=DataType.VARCHAR, description="The abbreviated name of the case", max_length=embedding_dim)
    # date_field = FieldSchema(name="decision_date", dtype=DataType.VARCHAR, description="The date of the decision", max_length=10)
    # cite_field = FieldSchema(name="cite", dtype=DataType.VARCHAR, description="The official citation", max_length=embedding_dim // 2)
    # court_field = FieldSchema(name="court_name", dtype=DataType.VARCHAR, description="The name of the court", max_length=embedding_dim // 2)
    # jurisdiction_field = FieldSchema(name="jurisdiction_name", dtype=DataType.VARCHAR, description="The name of the jurisdiction", max_length=embedding_dim // 8)
    # extra_fields = [author_field, type_field, caseid_field, casenameabbr_field, date_field, cite_field, court_field, jurisdiction_field]
    # coll = create_collection(collection_name, description, extra_fields, embedding_dim)
    coll = Collection(collection_name)
    
    root_dir = "CAP/"
    chunk_size = 1000
    chunk_overlap = 150
    batch_size = 1000
    for subdir in sorted(os.listdir(f"{os.getcwd()}/data/{root_dir}")):
        if "metadata" in subdir or "DS_Store" in subdir:
            continue
        with open(f"data/{root_dir + subdir}/{subdir}/data/data.jsonl") as f:
            lines = f.readlines()
        if "ark" in subdir:
            # the code below is for reuploading the batch that was in process if the upload crashes
            # for i, line in enumerate(lines[41499:42000]):
            #     json = loads(line)
            #     case_id = json["id"]
            #     ids = coll.delete(expr=f"case_id == {case_id}")
            #     deletions = ids.delete_count
            #     while ids.delete_count > 0:
            #         ids = coll.delete(expr=f"case_id == {case_id}")
            #         deletions += ids.delete_count
            #     print(f" deleted {deletions} chunks for case {case_id}, reinserting")
            #     opinions = json["casebody"]["data"]["opinions"]
            #     case_chunks = 0
            #     for opinion in opinions:
            #         if not opinion["text"]:
            #             continue
            #         opinion_type = opinion["type"] if opinion["type"] else "unknown"
            #         opinion_author = opinion["author"] if opinion["author"] else "unknown"
            #         elements = partition_text(text=opinion["text"])
            #         chunks = chunk_elements(elements, max_characters=chunk_size, overlap=chunk_overlap)
            #         num_opinion_chunks = len(chunks)
            #         case_chunks += num_opinion_chunks
            #         text = [chunk.text for chunk in chunks]
            #         chunk_embeddings = embed_strs(text)
            #         case_name_abbr = json["name_abbreviation"]
            #         decision_date = json["decision_date"]
            #         citations = json["citations"]
            #         official_cite = next(iter([cite for cite in citations if cite["type"] == "official"]), None)
            #         if not official_cite:
            #             print(f" error: citation not found for case id {case_id}")
            #             official_cite = "unknown"
            #         else:
            #             official_cite = official_cite["cite"]
            #         court_name = json["court"]["name"]
            #         jurisdiction_name = json["jurisdiction"]["name"]
            #         for j in range(0, num_opinion_chunks, batch_size):
            #             batch_vector = chunk_embeddings[j: j + batch_size]
            #             batch_text = text[j: j + batch_size]
            #             current_batch_size = len(batch_text)
            #             batch_author = [opinion_author] * current_batch_size
            #             batch_type = [opinion_type] * current_batch_size
            #             batch_id = [case_id] * current_batch_size
            #             batch_name_abbr = [case_name_abbr] * current_batch_size
            #             batch_decision_date = [decision_date] * current_batch_size
            #             batch_cite = [official_cite] * current_batch_size
            #             batch_court_name = [court_name] * current_batch_size
            #             batch_jurisdiction_name = [jurisdiction_name] * current_batch_size
            #             batch = [batch_vector, batch_text, batch_author, batch_type, batch_id, batch_name_abbr,
            #                     batch_decision_date, batch_cite, batch_court_name, batch_jurisdiction_name]
            #             result = coll.insert(batch)
            #             if result.insert_count != current_batch_size:
            #                 print(f" error: expected {current_batch_size} uploads but got {result.insert_count} for case id {case_id}")
            #     print(f" reinserted {case_chunks} chunks")
            lines = lines[42000:]
        
        print(f"{subdir} contains {len(lines)} cases")
        num_chunks = 0
        for i, line in enumerate(lines, start=1):
            if i % 500 == 0:
                print(f" {num_chunks} chunks processed, {len(lines) - i} cases remaining")
            json = loads(line)
            case_id = json["id"]
            opinions = json["casebody"]["data"]["opinions"]
            for opinion in opinions:
                if not opinion["text"]:
                    continue
                opinion_type = opinion["type"] if opinion["type"] else "unknown"
                opinion_author = opinion["author"] if opinion["author"] else "unknown"
                elements = partition_text(text=opinion["text"])
                chunks = chunk_elements(elements, max_characters=chunk_size, overlap=chunk_overlap)
                num_opinion_chunks = len(chunks)
                text = [chunk.text for chunk in chunks]
                chunk_embeddings = embed_strs(text)
                case_name_abbr = json["name_abbreviation"]
                decision_date = json["decision_date"]
                citations = json["citations"]
                official_cite = next(iter([cite for cite in citations if cite["type"] == "official"]), None)
                if not official_cite:
                    print(f" error: citation not found for case id {case_id}")
                    official_cite = "unknown"
                else:
                    official_cite = official_cite["cite"]
                court_name = json["court"]["name"]
                jurisdiction_name = json["jurisdiction"]["name"]
                for j in range(0, num_opinion_chunks, batch_size):
                    batch_vector = chunk_embeddings[j: j + batch_size]
                    batch_text = text[j: j + batch_size]
                    current_batch_size = len(batch_text)
                    batch_author = [opinion_author] * current_batch_size
                    batch_type = [opinion_type] * current_batch_size
                    batch_id = [case_id] * current_batch_size
                    batch_name_abbr = [case_name_abbr] * current_batch_size
                    batch_decision_date = [decision_date] * current_batch_size
                    batch_cite = [official_cite] * current_batch_size
                    batch_court_name = [court_name] * current_batch_size
                    batch_jurisdiction_name = [jurisdiction_name] * current_batch_size
                    batch = [batch_vector, batch_text, batch_author, batch_type, batch_id, batch_name_abbr,
                            batch_decision_date, batch_cite, batch_court_name, batch_jurisdiction_name]
                    result = coll.insert(batch)
                    if result.insert_count != current_batch_size:
                        print(f" error: expected {current_batch_size} uploads but got {result.insert_count} for case id {case_id}")
                num_chunks += num_opinion_chunks
        print(f"uploaded {num_chunks} chunks")
        print()