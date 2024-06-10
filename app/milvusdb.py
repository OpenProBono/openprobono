"""Functions for managing and searching vectors and collections in Milvus."""
from __future__ import annotations

import logging
import os
import time
from json import loads
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app import encoders
from app.chat_models import summarize
from app.db import load_vdb, store_vdb
from app.loaders import partition_uploadfile, quickstart_ocr, scrape, scrape_with_links
from app.models import EncoderParams, MilvusMetadataEnum
from app.splitters import chunk_elements_by_title, chunk_str

if TYPE_CHECKING:
    from fastapi import UploadFile


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


# used to cache collection params from firebase
COLLECTION_PARAMS = {}

SEARCH_PARAMS = {
    "anns_field": "vector",
    "param": {}, # can customize index params assuming you know index type
    "output_fields": ["text"],
}

# AUTOINDEX is only supported through Zilliz, not standalone Milvus
AUTO_INDEX = {
    "index_type": "AUTOINDEX",
    "metric_type": "IP",
}
SESSION_DATA = "SessionData"
MAX_K = 16384

# core features

def load_vdb_param(
    collection_name: str,
    param_name: str,
) -> EncoderParams | MilvusMetadataEnum | list:
    """Load a vector database parameter from firebase.

    Parameters
    ----------
    collection_name : str
        The name of the collection using the parameter
    param_name : str
        The name of the desired parameter value

    Returns
    -------
    EncoderParams | MilvusMetadataEnum | list
        EncoderParams if param_name = "encoder"

        MilvusMetadataEnum if param_name = "metadata_format"

        list if param_name = "fields"

    """
    # check if the params are cached
    if collection_name in COLLECTION_PARAMS:
        return COLLECTION_PARAMS[collection_name][param_name]
    param_value = load_vdb(collection_name)[param_name]
    # create the parameter object
    match param_name:
        case "encoder":
            return EncoderParams(**param_value)
        case "metadata_format":
            return MilvusMetadataEnum(param_value)
        case "fields":
            return param_value
        case _:
            raise ValueError(param_name)


def create_collection(
    name: str,
    encoder: EncoderParams | None = None,
    description: str = "",
    extra_fields: list[FieldSchema] | None = None,
    metadata_format: MilvusMetadataEnum = MilvusMetadataEnum.json,
) -> Collection:
    """Create a collection with a given name and other parameters.

    Parameters
    ----------
    name : str
        The name of the collection to be created
    encoder : EncoderParams, optional
        The embedding model used to create the vectors,
        by default text-embedding-3-small with 768 dimensions
    description : str, optional
        A description for the collection, by default ""
    extra_fields : list[FieldSchema] | None, optional
        A list of fields to add to the collections schema, by default None
    metadata_format : MilvusMetadataEnum, optional
        The format used to store metadata other than text, by default json
        (a single field called `metadata`)

        If `json`, the `metadata` field will be made automatically

        If `no_field`, `extra_fields` should not contain fields

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
    encoder = encoder if encoder is not None else EncoderParams()
    db_fields = None

    # define schema
    pk_field = FieldSchema(
        name="pk",
        dtype=DataType.INT64,
        is_primary=True,
        description="The primary key",
        auto_id=True,
    )
    text_field = FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        description="The source text",
        max_length=65535,
    )
    embedding_field = FieldSchema(
        name="vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=encoder.dim,
        description="The embedded text",
    )

    # keep track of how the collection stores metadata
    match metadata_format:
        case MilvusMetadataEnum.json:
            extra_fields = [FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
                description="The associated metadata",
            )]
        case MilvusMetadataEnum.no_field:
            if extra_fields:
                msg = "metadata_format = no_field but extra_fields is not empty"
                raise ValueError(msg)
        case MilvusMetadataEnum.field:
            # field format allows for empty extra_fields or a single json field as long
            # as dynamic fields are enabled, otherwise should be no_field or json
            if extra_fields is None:
                extra_fields = []
            else:
                db_fields = [field.name for field in extra_fields]

    schema = CollectionSchema(
        fields=[pk_field, embedding_field, text_field, *extra_fields],
        auto_id=True,
        enable_dynamic_field=True,
        description=description,
    )

    # create collection
    coll = Collection(name=name, schema=schema)
    # create index for vector field
    coll.create_index("vector", index_params=AUTO_INDEX, index_name="auto_index")

    # save params in firebase
    store_vdb(name, encoder, metadata_format, db_fields)
    # cache params in dictionary
    COLLECTION_PARAMS[name] = {
        "encoder": encoder,
        "metadata_format": metadata_format,
    }
    if db_fields is not None:
        COLLECTION_PARAMS[name]["fields"] = db_fields
    return coll


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
    encoder = load_vdb_param(collection_name, "encoder")
    search_params["data"] = encoders.embed_strs(
        [query],
        encoder,
    )
    search_params["limit"] = k
    metadata_format = load_vdb_param(collection_name, "metadata_format")
    match metadata_format:
        case MilvusMetadataEnum.json:
            search_params["output_fields"] += ["metadata"]
        case MilvusMetadataEnum.field:
            search_params["output_fields"] += load_vdb_param(collection_name, "fields")
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
        # on success, returns a list containing a single inner list containing
        # result objects
        if len(res) == 1:
            # sort hits by ascending distance
            hits = sorted(
                [hit.to_dict() for hit in res[0]],
                key=lambda h: h["distance"],
            )
            # delete pks
            for hit in hits:
                del hit["entity"]["metadata"]["orig_elements"]
                del hit["id"]
            return {"message": "Success", "result": hits}
        return {"message": "Success", "result": res}
    return {"message": "Failure: unable to complete search"}

def source_exists(collection_name: str, source: str) -> bool:
    collection = Collection(collection_name)
    q = collection.query(expr=f"metadata['url']=='{source}'")

    return len(q) > 0

def upload_data_json(
    texts: list[str],
    metadatas: list[dict],
    collection_name: str,
    batch_size: int = 1000,
) -> dict[str, str]:
    """Upload data to a collection with json format.

    Parameters
    ----------
    texts : list[str]
        The text to be uploaded.
    metadatas : list[dict]
        The metadata of the text to be uploaded.
    collection_name : str
        The name of the Milvus collection.
    batch_size : int, optional
        The number of records to be uploaded at a time, by default 1000.

    Returns
    -------
    dict[str, str]
        With a `message` indicating success or failure

    """
    vectors = encoders.embed_strs(texts, load_vdb_param(collection_name, "encoder"))
    data = [vectors, texts, metadatas]
    collection = Collection(collection_name)
    pks = []
    for i in range(0, len(texts), batch_size):
        batch_vector = data[0][i: i + batch_size]
        batch_text = data[1][i: i + batch_size]
        batch_metadata = data[2][i: i + batch_size]
        batch = [batch_vector, batch_text, batch_metadata]
        current_batch_size = len(batch[0])
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
                    extra={"pks": bad_deletes},
                )
                bad_insert += (
                    "We were unable to delete some of your partially uploaded data. "
                    "This has been logged, and your data will eventually be deleted. "
                    "If you would like more information, please email "
                    "contact@openprobono.com."
                )
            else:
                bad_insert += "Any partially uploaded data has been deleted."
            return {"message": bad_insert}
    return {"message": "Success"}


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
    collection_format = load_vdb_param(collection_name, "metadata_format")
    match collection_format:
        case MilvusMetadataEnum.field:
            output_fields = ["text", *load_vdb_param(collection_name, "fields")]
        case MilvusMetadataEnum.json:
            output_fields = ["text", "metadata"]
        case MilvusMetadataEnum.no_field:
            output_fields = ["text"]
    q_iter = coll.query_iterator(
        expr=expr,
        output_fields=output_fields,
        batch_size=batch_size,
    )
    hits = []
    res = q_iter.next()
    while len(res) > 0:
        # delete pks
        for hit in res:
            del hit["pk"]
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


# application level features

def upload_courtlistener(collection_name: str, oo: dict) -> dict:
    if "text" not in oo or not oo["text"]:
        return {"message": "Failure: no opinion text found"}
    # chunk
    texts = chunk_str(oo["text"], 10000, 1000)
    # summarize
    summary = summarize(texts, "map_reduce")
    # metadata
    del oo["text"]
    maxlen = 1000
    keys_to_remove = [
        key for key in oo
        if not oo[key] or (isinstance(oo[key], str) and len(oo[key])) > maxlen
    ]
    for key in keys_to_remove:
        del oo[key]
    oo["ai_summary"] = summary
    metadatas = [oo] * len(texts)
    # upload
    return upload_data_json(texts, metadatas, collection_name)


def crawl_upload_site(collection_name: str, description: str, url: str) -> list[str]:
    create_collection(collection_name, description=description)
    urls = [url]
    new_urls, prev_elements = scrape_with_links(url, urls)
    strs, metadatas = chunk_elements_by_title(prev_elements, 3000, 1000, 300)
    ai_summary = summarize(strs, "map_reduce")
    for metadata in metadatas:
        metadata["ai_summary"] = ai_summary
    upload_data_json(strs, metadatas, collection_name)
    print("new_urls: ", new_urls)
    while len(new_urls) > 0:
        cur_url = new_urls.pop()
        if url == cur_url[:len(url)]:
            urls.append(cur_url)
            cur_elements = scrape(cur_url)
            new_elements = [
                element for element in cur_elements if element not in prev_elements
            ]
            strs, metadatas = chunk_elements_by_title(new_elements, 3000, 1000, 300)
            ai_summary = summarize(strs, "map_reduce")
            for metadata in metadatas:
                metadata["ai_summary"] = ai_summary
            upload_data_json(strs, metadatas, collection_name)
            prev_elements = cur_elements
    print(urls)
    return urls


def upload_site(collection_name: str, url: str, max_chars=10000, new_after_n_chars=2500, overlap=500) -> dict[str, str]:
    """Scrape, chunk, summarize, and upload a URLs contents to Milvus.

    Parameters
    ----------
    collection_name : str
        Where the chunks will be uploaded.
    url : str
        The site to scrape.

    Returns
    -------
    dict[str, str]
        With a `message` indicating success or failure

    """
    elements = scrape(url)
    if len(elements) == 0:
        return {"message": f"Failure: no elements found at {url}"}
    strs, metadatas = chunk_elements_by_title(elements, max_chars, new_after_n_chars, overlap)
    ai_summary = summarize(strs, "map_reduce")
    for metadata in metadatas:
        # metadata["timestamp"] = firestore.SERVER_TIMESTAMP
        metadata["timestamp"] = str(time.time())
        metadata["url"] = url
        metadata["ai_summary"] = ai_summary
    return upload_data_json(strs, metadatas, collection_name)


def session_upload_ocr(
    file: UploadFile,
    session_id: str,
    summary: str | None = None,
    max_chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> dict[str, str]:
    """OCR scan, chunk, summarize, and upload a file to Milvus.

    Parameters
    ----------
    file : UploadFile
        The file to scan and upload.
    session_id : str
        The session associated with the file.
    summary : str | None, optional
        A user-written summary of the file, by default None.
    max_chunk_size : int, optional
        The max length of a text chunk in chars, by default 1000.
    chunk_overlap : int, optional
        The number of chars to overlap between chunks, by default 150.

    Returns
    -------
    dict[str, str]
        With a `message` indicating success or failure

    """
    reader = quickstart_ocr(file)
    strs = chunk_str(reader, max_chunk_size, chunk_overlap)
    ai_summary = summarize(strs, "map_reduce")
    metadata = {
        "session_id": session_id,
        "source": file.filename,
        "ai_summary": ai_summary,
    }
    if summary is not None:
        metadata["user_summary"] = summary
    # upload
    return upload_data_json(strs, [metadata] * len(strs), SESSION_DATA)


def file_upload(
    file: UploadFile,
    session_id: str,
    summary: str | None = None,
) -> dict[str, str]:
    """Perform an unstructured partition, chunk_by_title, and upload a file to Milvus.

    Parameters
    ----------
    file : UploadFile
        The file to upload.
    session_id : str
        The session associated with the file.
    summary: str, optional
        A summary of the file written by the user, by default None.

    Returns
    -------
    dict[str, str]
        With a `message` indicating success or failure

    """
    # extract text
    elements = partition_uploadfile(file)
    if summary is not None:
        for i in range(len(elements)):
            elements[i].metadata["user_summary"] = summary
    # chunk text
    texts, metadatas = chunk_elements_by_title(elements)
    # add session id to metadata
    for i in range(len(metadatas)):
        metadatas[i]["session_id"] = session_id
    # upload
    return upload_data_json(texts, metadatas, SESSION_DATA)


def session_source_summaries(
    session_id: str,
    batch_size: int = 1000,
) -> dict[str, dict[str, str]]:
    """Get AI and user-written summaries of any files from a session.

    Parameters
    ----------
    session_id : str
        The session to search for file summaries.
    batch_size : int, optional
        The number of chunks to return from the query iterator at a time,
        by default 1000.

    Returns
    -------
    dict[str, dict[str, str]]
        {"source": {"ai_summary": "str", "user_summary": "str"}}

    """
    coll = Collection(SESSION_DATA)
    coll.load()
    q_iter = coll.query_iterator(
        expr=f"session_id=='{session_id}'",
        output_fields=["metadata"],
        batch_size=batch_size,
    )
    source_summaries = {}
    res = q_iter.next()
    while len(res) > 0:
        for item in res:
            metadata = item["metadata"]
            if metadata["source"] not in source_summaries:
                source_summaries[metadata["source"]] = {
                    "ai_summary": metadata["ai_summary"],
                }
                if "user_summary" in metadata:
                    source_summary = source_summaries[metadata["source"]]
                    source_summary["user_summary"] = metadata["user_summary"]
        res = q_iter.next()
    q_iter.close()
    return source_summaries
