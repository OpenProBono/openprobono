"""Functions for managing and searching vectors and collections in Milvus."""
from __future__ import annotations

import logging
import os
import time
from json import loads
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

from langfuse.decorators import langfuse_context, observe
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.db import get_batch, load_vdb, load_vdb_chunk, load_vdb_source, store_vdb
from app.encoders import embed_strs
from app.loaders import (
    partition_uploadfile,
    quickstart_ocr,
    scrape,
    scrape_with_links,
)
from app.models import EncoderParams, MilvusMetadataEnum
from app.splitters import chunk_elements_by_title, chunk_str
from app.summarization import summarize

if TYPE_CHECKING:
    from fastapi import UploadFile
    from pymilvus.orm.iterator import QueryIterator


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
# for storing session data
SESSION_DATA = "SessionData"
# limit for number of results in queries
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
    if collection_name in COLLECTION_PARAMS and \
        param_name in COLLECTION_PARAMS[collection_name]:

        return COLLECTION_PARAMS[collection_name][param_name]
    if collection_name not in COLLECTION_PARAMS:
        COLLECTION_PARAMS[collection_name] = {}
    param_value = load_vdb(collection_name)[param_name]
    # create the parameter object
    match param_name:
        case "encoder":
            COLLECTION_PARAMS[collection_name][param_name] = EncoderParams(
                **param_value,
            )
        case "metadata_format":
            COLLECTION_PARAMS[collection_name][param_name] = MilvusMetadataEnum(
                param_value,
            )
        case "fields":
            COLLECTION_PARAMS[collection_name][param_name] = param_value
        case _:
            raise ValueError(param_name)
    return COLLECTION_PARAMS[collection_name][param_name]


@observe()
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
    # AUTOINDEX is only supported through Zilliz, not standalone Milvus
    auto_index = {
        "index_type": "AUTOINDEX",
        "metric_type": "IP",
    }
    coll.create_index("vector", index_params=auto_index, index_name="auto_index")

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


@observe()
def query(
    collection_name: str,
    query: str,
    k: int = 4,
    expr: str = "",
    session_id: str = "",
) -> dict:
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
    coll = Collection(collection_name)
    encoder = load_vdb_param(collection_name, "encoder")
    data = embed_strs([query], encoder)
    search_params = {
        "anns_field": "vector",
        "param": {}, # can customize index params assuming you know index type
        "output_fields": ["text"],
        "data": data,
        "limit": k,
    }
    metadata_format = load_vdb_param(collection_name, "metadata_format")
    match metadata_format:
        case MilvusMetadataEnum.json:
            search_params["output_fields"] += ["metadata"]
        case MilvusMetadataEnum.field:
            search_params["output_fields"] += load_vdb_param(collection_name, "fields")
    if session_id:
        expr += (" and " if expr else "") + f"metadata[\"session_id\"]=='{session_id}'"
    if expr:
        search_params["expr"] = expr
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
            # format output for tracing
            pks = [str(hit["id"]) for hit in hits]
            langfuse_context.update_current_observation(output=pks)
            return {"message": "Success", "result": hits}
        return {"message": "Success", "result": res}
    return {"message": "Failure: unable to complete search"}


def fuzzy_keyword_query(keyword_query: str) -> str:
    """Create a fuzzy* version of a keyword query.

    *Replaces uppercase letters and the first letter of every word
    with a wildcard character (_). Mainly for case-insensitivity.

    Parameters
    ----------
    keyword_query : str
        the original keyword query

    Returns
    -------
    str
        A fuzzy version of the keyword query

    """
    min_length_fuzzy = 6
    keywords = keyword_query.split()
    fuzzy_keywords = [
        "".join([
            c if not c.isupper() or len(kw) < min_length_fuzzy else "_"
            for c in kw
        ])
        for kw in keywords
    ]
    fuzzy_keywords = [
        kw if len(kw) < min_length_fuzzy else "_" + kw[1:]
        for kw in fuzzy_keywords
    ]
    fuzzy_keywords_str = " ".join(fuzzy_keywords)
    fuzzy_keywords_str = fuzzy_keywords_str.replace("%", "\\\\%")
    fuzzy_keywords_str = fuzzy_keywords_str.replace('"', '\\"')
    return fuzzy_keywords_str.replace("'", "\\'")


@observe()
def source_exists(collection_name: str, url: str, bot_id: str, tool_name:str) -> bool:
    """Check if a url source exists in a collection, for a specific bot and tool.

    Parameters
    ----------
    collection_name : str
        name of collection
    url : str
        source url to check for
    bot_id : str
        bot id
    tool_name : str
        tool name

    Returns
    -------
    bool
        True if the url is found, False otherwise

    """
    collection = Collection(collection_name)
    q = collection.query(expr=f"metadata['url']=='{url}' && metadata['bot_id'] == '{bot_id}' && metadata['tool_name'] == '{tool_name}'")

    return len(q) > 0


@observe()
def upload_data(
    collection_name: str,
    data: list[dict],
    batch_size: int = 1000,
) -> dict[str, str]:
    """Upload data to a collection with json format.

    Parameters
    ----------
    collection_name : str
        The name of the Milvus collection.
    data : list[dict]
        The data to upload. Format should match collection schema.
        Example: `[{'vector': [], 'text': '', 'metadata': {}}]`
    batch_size : int, optional
        The number of records to be uploaded at a time, by default 1000.

    Returns
    -------
    dict[str, str]
        With a `message`, `insert_count` on success

    """
    data_count = len(data)
    langfuse_context.update_current_observation(
        input={
            "collection_name":collection_name,
            "data_count": data_count,
        },
    )
    collection = Collection(collection_name)
    pks = []
    insert_count = 0
    for i in range(0, data_count, batch_size):
        batch = data[i: i + batch_size]
        current_batch_size = len(batch)
        res = collection.insert(batch)
        pks += res.primary_keys
        insert_count += res.insert_count
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
    return {"message": "Success", "insert_count": insert_count}


def upload_data_firebase(
    coll_name: str,
    data: list[dict],
    source_ids: set | None = None,
) -> dict:
    """Insert chunk data into Firebase.

    Parameters
    ----------
    coll_name : str
        The name of the Firebase collection.
    data : list[dict]
        Each element must contain `text`, `chunk_index`, `source_id`.
    source_ids : set | None, optional
        A set of source_ids that have already been inserted into the database,
        by default None.

    Returns
    -------
    dict
        Containing a message and insert count

    """
    if source_ids is None:
        source_ids = set()
    db_batch = get_batch()
    insert_count = len(data)
    for i in range(insert_count):
        text = data[i]["text"]
        chunk_idx = data[i]["chunk_index"]
        source_id = data[i]["source_id"]
        if source_id not in source_ids:
            source_ids.add(source_id)
            db_source = load_vdb_source(coll_name, source_id)
            del data[i]["text"]
            del data[i]["chunk_index"]
            del data[i]["source_id"]
            db_batch.set(db_source, data[i])
        chunk_data = {"text": text, "chunk_index": chunk_idx}
        db_chunk = load_vdb_chunk(coll_name, source_id, data[i]["pk"])
        db_batch.set(db_chunk, chunk_data)
        if i % 1000 == 0:
            db_batch.commit()
            db_batch = get_batch()
    db_batch.commit()
    return {"message": "Success", "insert_count": insert_count}


@observe(capture_output=False)
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
    coll = Collection(collection_name)
    collection_format = load_vdb_param(collection_name, "metadata_format")
    output_fields = ["text", "vector"]
    match collection_format:
        case MilvusMetadataEnum.field:
            output_fields += [*load_vdb_param(collection_name, "fields")]
        case MilvusMetadataEnum.json:
            output_fields += ["metadata"]
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
    pks = [str(hit["pk"]) for hit in hits]
    langfuse_context.update_current_observation(output=pks)
    return {"message": "Success", "result": hits}


@observe()
def delete_expr(collection_name: str, expr: str) -> dict[str, str]:
    """Delete database entries according to a boolean expression.

    Parameters
    ----------
    collection_name : str
        The name of a pymilvus.Collection
    expr : str
        A boolean expression to filter database entries

    Returns
    -------
    dict[str, str]
        Contains `message`, `delete_count` if successful

    """
    coll = Collection(collection_name)
    ids = coll.delete(expr=expr)
    coll.flush()
    return {"message": "Success", "delete_count": ids.delete_count}


@observe()
def upsert_expr(
    collection_name: str,
    expr: str,
    upsert_data: list[dict],
) -> dict[str, str]:
    """Upsert database entries according to a boolean expression.

    Parameters
    ----------
    collection_name : str
        The name of a pymilvus.Collection
    expr : str
        A boolean expression to filter database entries
    upsert_data : list[dict]
        A list of dicts containing the data to be upserted into Milvus.

    Returns
    -------
    dict[str, str]
        Contains `message` and `insert_count` if successful.

    """
    langfuse_context.update_current_observation(
        input={
            "collection_name":collection_name,
            "expr": expr,
            "upsert_data_count": len(upsert_data),
        },
    )
    delete_result = delete_expr(collection_name, expr)
    if delete_result["message"] != "Success":
        return delete_result
    delete_count = delete_result["delete_count"]
    insert_result = upload_data(collection_name, upsert_data)
    return {"delete_count": delete_count, **insert_result}


def fields_to_json(fields_entry: dict) -> dict:
    """Convert a Collection entry from fields to json metadata format.

    Parameters
    ----------
    fields_entry : dict
        The entry from a Collection with fields metadata format

    Returns
    -------
    dict
        The entry with fields replaced with a metadata dictionary
        (json metadata format)

    """
    d = {k: v for k, v in fields_entry.items() if k != "entity"}
    d["entity"] = {
        "text": fields_entry["entity"]["text"],
        "metadata": {
            k: v for k, v in fields_entry["entity"].items() if k != "text"
        },
    }
    return d


def query_iterator(
    collection_name: str,
    expr: str,
    output_fields: list[str],
    batch_size: int,
) -> QueryIterator:
    """Get a query iterator for a collection.

    Parameters
    ----------
    collection_name : str
        The name of the collection
    expr : str
        The boolean expression to filter entities in the collection
    output_fields : list[str]
        the fields to return for each entity
    batch_size : int
        the number of entities to process at a time

    Returns
    -------
    QueryIterator
        An iterator over the chosen entities in the collection

    """
    coll = Collection(collection_name)
    return coll.query_iterator(
        expr=expr,
        output_fields=output_fields,
        batch_size=batch_size,
    )

# application level features

def crawl_upload_site(collection_name: str, description: str, url: str) -> list[str]:
    create_collection(collection_name, description=description)
    urls = [url]
    new_urls, prev_elements = scrape_with_links(url, urls)
    texts, metadatas = chunk_elements_by_title(prev_elements, 3000, 1000, 300)
    ai_summary = summarize(texts)
    for metadata in metadatas:
        metadata["ai_summary"] = ai_summary
    encoder = load_vdb_param(collection_name, "encoder")
    vectors = embed_strs(texts, encoder)
    data = [{
        "vector": vectors[i],
        "metadata": metadatas[i],
        "text": texts[i],
    } for i in range(len(texts))]
    upload_data(collection_name, data)
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
            ai_summary = summarize(strs)
            for metadata in metadatas:
                metadata["ai_summary"] = ai_summary
            vectors = embed_strs(strs, encoder)
            data = [{
                "vector": vectors[i],
                "metadata": metadatas[i],
                "text": texts[i],
            } for i in range(len(texts))]
            upload_data(collection_name, data)
            prev_elements = cur_elements
    print(urls)
    return urls


@observe(capture_output=False)
def upload_site(
    collection_name: str,
    url: str,
    bot_id: str,
    tool_name: str,
    max_chars: int = 10000,
    new_after_n_chars: int = 2500,
    overlap: int = 500,
) -> dict[str, str]:
    """Scrape, chunk, summarize, and upload a URLs contents to Milvus.

    Parameters
    ----------
    collection_name : str
        Where the chunks will be uploaded.
    url : str
        The site to scrape.
    bot_id : str
        The ID of the bot using this tool
    tool_name : str
        The name of the search tool
    max_chars : int, optional
        Maximum characters per chunk, by default 10000
    new_after_n_chars : int, optional
        Start a new chunk after this many characters, by default 2500
    overlap : int, optional
        Number of characters to overlap between chunks, by default 500

    Returns
    -------
    dict[str, str]
        With a `message` indicating success or failure

    """
    elements = scrape(url)
    if len(elements) == 0:
        return {"message": f"Failure: no elements found at {url}"}
    texts, metadatas = chunk_elements_by_title(
        elements,
        max_chars,
        new_after_n_chars,
        overlap,
    )
    vectors = embed_strs(texts, load_vdb_param(collection_name, "encoder"))
    ai_summary = summarize(texts)
    for metadata in metadatas:
        metadata["timestamp"] = str(time.time())
        metadata["url"] = url
        metadata["ai_summary"] = ai_summary
        metadata["bot_id"] = bot_id
        metadata["tool_name"] = tool_name
    data = [{
        "vector": vectors[i],
        "metadata": metadatas[i],
        "text": texts[i],
    } for i in range(len(texts))]
    return upload_data(collection_name, data)


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
    texts = chunk_str(reader, max_chunk_size, chunk_overlap)
    vectors = embed_strs(texts, load_vdb_param(SESSION_DATA, "encoder"))
    ai_summary = summarize(texts)
    metadata = {
        "session_id": session_id,
        "source": file.filename,
        "ai_summary": ai_summary,
    }
    if summary is not None:
        metadata["user_summary"] = summary
    # upload
    data = [{
        "vector": vectors[i],
        "metadata": metadata,
        "text": texts[i],
    } for i in range(len(texts))]
    return upload_data(SESSION_DATA, data)


def file_upload(
    file: UploadFile,
    session_id: str,
    summary: str | None = None,
    collection_name: str = SESSION_DATA,
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
    # chunk text
    texts, metadatas = chunk_elements_by_title(elements)
    vectors = embed_strs(texts, load_vdb_param(collection_name, "encoder"))
    # add session id to metadata
    for i in range(len(metadatas)):
        metadatas[i]["session_id"] = session_id
        if summary:
            metadatas[i]["user_summary"] = summary
    # upload
    data = [{
        "vector": vectors[i],
        "metadata": metadatas[i],
        "text": texts[i],
    } for i in range(len(texts))]
    return upload_data(collection_name, data)

def check_session_data(session_id: str) -> bool:
    """Check if user uploaded a file in a specific session.

    Parameters
    ----------
    session_id : str
        the session id

    Returns
    -------
    bool
        true if file was uploaded, false if it is empty

    """
    expr = f'metadata["session_id"] == "{session_id}"'
    data = get_expr(SESSION_DATA, expr, 1)
    return len(data["result"]) != 0

def fetch_session_data_files(
    session_id: str,
    batch_size: int = 1000,
) -> dict[str, dict[str, str]]:
    coll = Collection(SESSION_DATA)
    coll.load()
    query = coll.query(
        expr=f"metadata[\"session_id\"] in [\"{session_id}\"]",
        output_fields=["metadata"],
        batch_size=batch_size,
    )
    files = [data["metadata"]["filename"] for data in query]
    return set(files)
