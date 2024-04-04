"""Functions for managing and searching vectors and collections in Milvus."""
from __future__ import annotations

import logging
import mimetypes
import os
from json import loads
from logging.handlers import RotatingFileHandler
from operator import itemgetter
from typing import List

import requests
from bs4 import BeautifulSoup
from fastapi import UploadFile
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from langchain.chains.summarize import load_summarize_chain
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_core.vectorstores import Field, VectorStore, VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.llms import OpenAI as LangChainOpenAI
from langfuse.callback import CallbackHandler
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from unstructured.chunking.basic import Element, chunk_elements
from unstructured.partition.auto import partition

import encoders
import prompts

langfuse_handler = CallbackHandler(public_key=os.environ["LANGFUSE_PUBLIC_KEY"], secret_key=os.environ["LANGFUSE_SECRET_KEY"])

connection_args = loads(os.environ["Milvus"])
# test connection to db, also needed to use utility functions
connections.connect(uri=connection_args["uri"], token=connection_args["token"])

# init logs
log_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s",
)
my_handler = RotatingFileHandler("vdb.log", maxBytes=5*1024*1024, backupCount=2)
my_handler.setFormatter(log_formatter)
my_handler.setLevel(logging.ERROR)

vdb_log = logging.getLogger("vdb")
vdb_log.setLevel(logging.ERROR)

vdb_log.addHandler(my_handler)


project_id = "h2o-gpt"
location = "us"  # Format is "us" or "eu"
processor_id = "c99e554bb49cf45d"


# processor_display_name = "my" # Must be unique per project, e.g.: "My Processor"

def session_upload_str(reader: str, session_id: str, summary: str, max_chunk_size: int = 1000,
                       chunk_overlap: int = 150):
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
    result = chain.invoke({"input_documents": documents[:200]}, config={"callbacks": [langfuse_handler]})
    for doc in documents:
        doc.metadata["ai_summary"] = result["output_text"].strip()

    # upload
    ids = langchain_db(SESSION_DATA).add_documents(documents=documents, embedding=OpenAIEmbeddings(),
                                             connection_args=connection_args)
    num_docs = len(documents)
    if num_docs != len(ids):
        return {
            "message": f"Failure: expected to upload {num_docs} chunk{'s' if num_docs > 1 else ''} for {summary} but "
                       f"got {len(ids)}"}
    return {"message": f"Success: uploaded {summary} as {num_docs} chunk{'s' if num_docs > 1 else ''}"}


def collection_upload_str(reader: str, collection: str, source: str, max_chunk_size: int = 10000,
                          chunk_overlap: int = 1500):
    documents = [
        Document(
            page_content=page,
            metadata={"source": source},
        )
        for page_number, page in enumerate([reader], start=1)
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)

    # summarize
    chain = load_summarize_chain(LangChainOpenAI(temperature=0), chain_type="map_reduce")
    result = chain.invoke({"input_documents": documents[:200]}, config={"callbacks": [langfuse_handler]})
    for doc in documents:
        doc.metadata["ai_summary"] = result["output_text"].strip()

    # upload
    ids = langchain_db(collection).add_documents(documents=documents, embedding=OpenAIEmbeddings(),
                                            connection_args=connection_args)
    num_docs = len(documents)
    if num_docs != len(ids):
        return {
            "message": f"Failure: expected to upload {num_docs} chunk{'s' if num_docs > 1 else ''} for {source} but got {len(ids)}"}
    return {"message": f"Success: uploaded {source} as {num_docs} chunk{'s' if num_docs > 1 else ''}"}


def scrape(site: str, old_urls: list[str], common_elements: list[str], collection: str, get_links: bool = False):
    print("site: ", site)
    r = requests.get(site)
    site_base = "//".join(site.split("//")[:-1])
    # converting the text 
    s = BeautifulSoup(r.content, "html.parser")
    urls = []

    if get_links:
        for i in s.find_all("a"):
            if "href" in i.attrs:
                href = i.attrs['href']

                if href.startswith("/"):
                    link = site_base + href
                elif href.startswith("http"):
                    link = href
                else:
                    link = old_urls[0]
                    # skip this link

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
        if el not in common_elements:
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
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    processor_name = client.processor_path(project_id, location, processor_id)

    # Print the processor information
    print(f"Processor Name: {processor_name}")

    # Load binary data
    raw_document = documentai.RawDocument(
        content=file.file.read(),
        mime_type=mimetypes.guess_type(file.filename)[0],
        # Refer to https://cloud.google.com/document-ai/docs/file-types for supported file types
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
# TODO(Nick): get collection names from Milvus
US = "USCode"
NC = "NCGeneralStatutes"
CAP = "CAP"
SESSION_DATA = "SessionData"
COURTLISTENER = "courtlistener"
COURTROOM5 = "Courtroom5_NCStatutes"
COLLECTIONS = {US, NC, CAP, COURTLISTENER, COURTROOM5}
# collection -> encoder mapping
# TODO(Nick): make this a file or use firebase?
COLLECTION_ENCODER = {
    US: encoders.DEFAULT_PARAMS,
    NC: encoders.DEFAULT_PARAMS,
    CAP: encoders.DEFAULT_PARAMS,
    COURTLISTENER: encoders.EncoderParams(encoders.OPENAI_ADA_2, None),
    SESSION_DATA: encoders.EncoderParams(encoders.OPENAI_ADA_2, None),
    COURTROOM5: encoders.DEFAULT_PARAMS,
}

PDF = "PDF"
HTML = "HTML"
JSON = "JSON"
COLLECTION_FORMAT = {
    US: PDF,
    NC: PDF,
    SESSION_DATA: JSON,
    CAP: CAP,
    COURTLISTENER: COURTLISTENER,
    COURTROOM5: JSON,
}

FORMAT_OUTPUTFIELDS = {
    PDF: ["source", "page"],
    HTML: [],
    CAP: ["opinion_author", "opinion_type", "case_name_abbreviation", "decision_date", "cite", "court_name",
          "jurisdiction_name"],
    COURTLISTENER: ["source"],
    JSON: ["metadata"],
}
# can customize index params with param field assuming you know index type
SEARCH_PARAMS = {
    "anns_field": "vector",
    "param": {},
    "output_fields": ["text"],
}
# AUTOINDEX is only supported through Zilliz, not standalone Milvus
AUTO_INDEX = {
    "index_type": "AUTOINDEX",
    "metric_type": "IP",
}

MAX_K = 16384

def create_collection(
        name: str,
        description: str = "",
        extra_fields: list[FieldSchema] | None = None,
        params: encoders.EncoderParams = encoders.DEFAULT_PARAMS,
    ) -> Collection:
    """Create a collection with a given name and other parameters.

    Parameters
    ----------
    name : str
        The name of the collection to be created
    description : str, optional
        A description for the collection, by default ""
    extra_fields : list[FieldSchema] | None, optional
        A list of fields to add to the collections schema, by default None
    params : encoders.EncoderParams, optional
        The embedding model used to create the vectors,
        by default encoders.DEFAULT_PARAMS

    Returns
    -------
    pymilvus.Collection
        The created collection. Must call load() before query/search.

    Raises
    ------
    ValueError
        If the collection name already exists

    """
    if utility.has_collection(name):
        already_exists = f"collection named {name} already exists"
        raise ValueError(already_exists)

    # TODO: if possible, support custom embedding size for huggingface models
    # TODO: support other OpenAI models
    # define schema, create collection, create index on vectors
    pk_field = FieldSchema(
        name="pk",
        dtype=DataType.INT64,
        is_primary=True,
        description="The primary key",
        auto_id=True,
    )
    # unstructured chunk lengths are sketchy
    text_field = FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        description="The source text",
        max_length=65535,
    )
    embedding_field = FieldSchema(
        name="vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=params.dim,
        description="The embedded text",
    )
    if not extra_fields:
        extra_fields = []
    schema = CollectionSchema(
        fields=[pk_field, embedding_field, text_field, *extra_fields],
        auto_id=True,
        enable_dynamic_field=True,
        description=description,
    )
    coll = Collection(name=name, schema=schema)
    coll.create_index("vector", index_params=AUTO_INDEX, index_name="auto_index")

    # save collection->encoder, collection->format
    COLLECTION_ENCODER[name] = params
    COLLECTION_FORMAT[name] = PDF
    return coll

def langchain_db(collection_name: str) -> Milvus:
    """Get an instance of the database for use in LangChain.

    Parameters
    ----------
    collection_name : str
        The name of the collection to load

    Returns
    -------
    Milvus
        A subclass of LangChain's VectorStore

    """
    return Milvus(
        embedding_function=encoders.get_langchain_embedding_model(
            COLLECTION_ENCODER[collection_name],
        ),
        collection_name=collection_name,
        connection_args=connection_args,
        auto_id=True,
    )

def query_check(collection_name: str, query: str, k: int, session_id: str = "") -> dict:
    """Check query parameters.

    Parameters
    ----------
    collection_name : str
        The name of the collection
    query : str
        The query for the collection
    k : int
        The number of vectors to return from the collection
    session_id : str, optional
        The session id if the query is to a session collection, by default ""

    Returns
    -------
    dict
        Contains a `message` if there was an error, otherwise empty

    """
    msg = {}
    if not utility.has_collection(collection_name):
        msg["message"] = f"Failure: collection {collection_name} not found"
    if not query or query == "":
        msg["message"] = "Failure: query not found"
    if k < 1 or k > MAX_K:
        msg["message"] = f"Failure: k = {k} out of range [1, {MAX_K}]"
    if not session_id and collection_name == SESSION_DATA:
        msg["message"] = "Failure: session_id not found"
    if collection_name not in COLLECTION_ENCODER:
        msg["message"] = f"Failure: encoder for collection {collection_name} not found"
    if collection_name not in COLLECTION_FORMAT:
        msg["message"] = f"Failure: format for collection {collection_name} not found"
    return msg

def query(collection_name: str, query: str,
          k: int = 4, expr: str = "", session_id: str = "") -> dict:
    """Run a query on a given collection.

    Parameters
    ----------
    collection_name : str
        the collection to query
    query : str
        the query itself
    k : int, optional
        how many chunks to return, by default 4
    expr : str, optional
        a boolean expression to specify conditions for ANN search, by default ""
    session_id : str, optional
        The session id for filtering session data, by default ""

    Returns
    -------
    dict
        With message and result on success, just message on failure

    """
    if query_check(collection_name, query, k, session_id):
        return query_check(collection_name, query, k, session_id)

    coll = Collection(collection_name)
    coll.load()
    search_params = SEARCH_PARAMS
    search_params["data"] = encoders.embed_strs(
        [query],
        COLLECTION_ENCODER[collection_name],
    )
    search_params["limit"] = k
    search_params["output_fields"] += FORMAT_OUTPUTFIELDS[COLLECTION_FORMAT[collection_name]]

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

def qa_chain(collection_name: str, k: int = 4,
             session_id: str = "") -> RunnableSerializable:
    """Create a QA chain.

    Parameters
    ----------
    collection_name : str
        The name of a Milvus Collection
    k : int, optional
        Return the top k chunks, by default 4
    session_id : str, optional
        The session id for filtering session data, by default ""

    Returns
    -------
    RunnableSerializable
        A LangChain Runnable QA chain

    """
    db = langchain_db(collection_name)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_with_tool = llm.bind_tools(
        [prompts.CitedAnswer],
        tool_choice="CitedAnswer",
    )
    output_parser = JsonOutputKeyToolsParser(key_name="CitedAnswer", return_single=True)
    output_fields = FORMAT_OUTPUTFIELDS[COLLECTION_FORMAT[collection_name]]

    def format_docs_with_id(docs: list[Document]) -> str:
        formatted = [
            f"Chunk ID: {i}\n" + "\n".join(
            [f"Chunk {f.capitalize()}: {doc.metadata[f]}" for f in output_fields]) +
            "\nChunk Text: " + doc.page_content
            # start=1 because the LLM will switch to 1-indexed IDs if chunks are large
            # TODO(Nick): format QA chain docs differently based on chunk size
            for i, doc in enumerate(docs, start=1)
        ]
        return "\n\n" + "\n\n".join(formatted)

    doc_format_chain = itemgetter("docs") | RunnableLambda(format_docs_with_id)
    output_chain = prompts.QA_PROMPT | llm_with_tool | output_parser
    if session_id:
        docs = FilteredRetriever(vectorstore=db,
                                 session_filter=session_id,
                                 search_kwargs={"k": k})
    else:
        docs = db.as_retriever(search_kwargs={"k": k})
    return (
        RunnableParallel(question=RunnablePassthrough(), docs=docs)
        .assign(context=doc_format_chain)
        .assign(cited_answer=output_chain)
        .pick(["cited_answer", "docs"])
    )

def qa(collection_name: str, query: str,
       k: int = 4, session_id: str = "") -> dict:
    """Run a QA chain to answer the query and return the top k source chunks.

    Parameters
    ----------
    collection_name : str
        The name of a Milvus Collection
    query : str
        The users query
    k : int, optional
        Return the top k chunks, by default 4
    session_id : str, optional
        The session id for filtering session data, by default ""

    Returns
    -------
    dict
        With success message, result, and sources or else failure message

    """
    if query_check(collection_name, query, k, session_id):
        return query_check(collection_name, query, k, session_id)

    chain = qa_chain(collection_name, k, session_id)
    result = chain.invoke(query)
    cited_sources = [
        {
            # i - 1 because start=1 in format_docs_with_id (see above)
            field: result["docs"][i - 1].metadata[field]
            for field in FORMAT_OUTPUTFIELDS[COLLECTION_FORMAT[collection_name]]
        }
        for i in set(result["cited_answer"][0]["citations"])
    ]
    return {
        "message": "Success",
        "result": {
            "answer": result["cited_answer"][0]["answer"],
            "sources": cited_sources,
        },
    }

def file_upload(file: UploadFile, collection_name: str, session_id: str = "") -> dict:
    """Upload a file to a collection.

    Parameters
    ----------
    file : UploadFile
        The file to upload
    collection_name : str
        The collection where the file will be uploaded
    session_id : str, optional
        The session associated with the file, by default ""

    Returns
    -------
    dict
        With a `message`, and `num_chunks` if success

    """
    elements = partition(file=file.file, metadata_filename=file.filename)
    result = upload_elements(elements, collection_name, session_id)
    if result["message"] == "Success":
        result["message"] = f"Success: uploaded {file.filename}"
    return result

def upload_elements(
        elements: list[Element],
        collection_name: str,
        session_id: str = "",
    ) -> dict:
    """Upload elements to a collection.

    Parameters
    ----------
    elements : list[Element]
        The elements to upload
    collection_name : str
        The collection where the elements will be uploaded
    session_id : str, optional
        The session associated with the elements, by default ""

    Returns
    -------
    dict
        With a `message`, and `num_chunks` if success

    """
    chunks = chunk_elements(
        elements,
        max_characters=5000,
        overlap=750,
    )
    batch_size = 1000
    texts, metadatas = [], []
    num_chunks = len(chunks)
    for i in range(num_chunks):
        texts.append(chunks[i].text)
        metadatas.append(chunks[i].metadata.to_dict())
    vectors = encoders.embed_strs(texts)
    data = [vectors, texts, metadatas]
    collection = Collection(collection_name)
    pks = []
    for i in range(0, len(chunks), batch_size):
        batch_vector = data[0][i: i + batch_size]
        batch_text = data[1][i: i + batch_size]
        batch_metadata = data[2][i: i + batch_size]
        batch = [batch_vector, batch_text, batch_metadata]
        current_batch_size = len(batch[0])
        if session_id:
            batch.append([session_id] * current_batch_size)
        res = collection.insert(batch)
        pks += res.primary_keys
        if res.insert_count != current_batch_size:
            # the upload failed, try deleting any partially uploaded data
            bad_deletes = []
            for pk in pks:
                delete_res = collection.delete(expr=f"pk=={pk}")
                if delete_res.delete_count != 1:
                    bad_deletes.append(pk)
            bad_insert = (
                f"Failure: expected {current_batch_size} insertions but got "
                f"{res.insert_count}. "
            )
            if bad_deletes:
                logging.error(
                    "dangling data",
                    extra={"session_id": session_id, "pks": bad_deletes},
                )
                bad_insert += (
                    "We were unable to delete some of your partially uploaded data. "
                    "This has been logged, and your data will eventually be deleted."
                    "If you would like more information, please email "
                    "contact@openprobono.com and mention your session_id: "
                    f"{session_id}."
                )
            else:
                bad_insert += "Any partially uploaded data has been deleted."
            return {"message": bad_insert}
    return {"message": "Success", "num_chunks": num_chunks}

def upload_documents(collection_name: str, documents: list[Document]):
    ids = langchain_db(collection_name).add_documents(
        documents=documents,
        embedding=encoders.get_langchain_embedding_model(COLLECTION_ENCODER[collection_name]),
        connection_args=connection_args,
    )
    num_docs = len(documents)
    if num_docs != len(ids):
        return {"message": f"Failure: expected to upload {num_docs} chunks but got {len(ids)}"}
    return {"message": f"Success: uploaded {num_docs} chunks"}

def get_documents(collection_name: str, expr: str) -> list[Document]:
    """Get chunks from a Milvus Collection as a list of LangChain Documents.

    Parameters
    ----------
    collection_name : str
        The name of a pymilvus.Collection
    expr : str
        The boolean expression used to filter chunks

    Returns
    -------
    list[Document]
        PDF chunks from the source PDF and collection

    """
    hits = get_expr(collection_name, expr)["result"]
    return [
        Document(
            page_content=hit["text"],
            metadata={
                field: hit[field]
                for field in FORMAT_OUTPUTFIELDS[COLLECTION_FORMAT[collection_name]]
            },
        )
        for hit in hits
    ]


def get_expr(collection_name: str, expr: str, batch_size: int = 1000) -> dict:
    """Get database entries according to a boolean expression.

    Parameters
    ----------
    collection_name : str
        The name of a pymilvus.Collection
    expr : str
        A boolean expression to filter database entries
    batch_size: int, optional
        The batch size used to fetch entries from Milvus, defaults to 1000

    Returns
    -------
    dict
        Contains `message`, `result` list if successful

    """
    if not utility.has_collection(collection_name):
        return {"message": f"Failure: collection {collection_name} does not exist"}
    coll = Collection(collection_name)
    coll.load()
    output_fields = ["text"] + FORMAT_OUTPUTFIELDS[COLLECTION_FORMAT[collection_name]]
    q_iter = coll.query_iterator(
        expr=expr,
        output_fields=output_fields,
        batch_size=batch_size,
    )
    hits = []
    res = q_iter.next()
    while len(res) > 0:
        hits += res
        res = q_iter.next()
    q_iter.close()
    return {"message": "Success", "result": hits}

def delete_expr(collection_name: str, expr: str) -> dict:
    """Delete database entries according to a boolean expression.

    Parameters
    ----------
    collection_name : str
        The name of a pymilvus.Collection
    expr : str
        A boolean expression to filter database entries

    """
    if not utility.has_collection(collection_name):
        return {"message": f"Failure: collection {collection_name} does not exist"}
    coll = Collection(collection_name)
    coll.load()
    ids = coll.delete(expr=expr)
    return {"message": f"Success: deleted {ids.delete_count} chunks"}


def session_upload_ocr(file: UploadFile, session_id: str, summary: str, max_chunk_size: int = 1000,
                       chunk_overlap: int = 150):
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
    return upload_documents(SESSION_DATA, documents)

def session_source_summaries(session_id: str, batch_size: int = 1000):
    coll = Collection(SESSION_DATA)
    coll.load()
    q_iter = coll.query_iterator(expr=f"session_id=='{session_id}'",
                                 output_fields=["source", "ai_summary", "user_summary"], batch_size=batch_size)
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
