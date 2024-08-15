"""Functions for processing bulk CourtListener data with OpenAI Batch API."""
from __future__ import annotations

import json
import os
import pathlib

from openai import OpenAI
from pymilvus import Collection

from app.loaders import (
    delete_completed_batches,
    download_batch_output,
    retry_failed_batches,
    wait_for_batches,
)
from app.milvusdb import MAX_K, query_iterator, upload_data


def load_cl_chunk_metadatas(basedir: str) -> dict:
    chunk_metadatas = {}
    with pathlib.Path(basedir + "chunks_metadata.jsonl").open("r") as f:
        for _, row in enumerate(f):
            metadata = json.loads(row)
            chunk_metadatas[metadata["id"]] = metadata
    return chunk_metadatas


def upload_cl_batch_files(basedir: str, chunk_metadatas: dict, coll_name: str) -> None:
    """Iterate batches in OpenAI API, upload to Milvus and Firebase."""
    client = OpenAI()
    openai_files = client.files.list()
    batches = client.batches.list()
    for page in batches.iter_pages():
        for batch in page.data:
            if batch.status != "completed":
                continue

            input_file = next(
                (f for f in openai_files if batch.input_file_id == f.id),
                None,
            )
            if input_file is None:
                print("input file not found in API for " + batch.input_file_id)
                continue

            input_filename = input_file.filename
            print(input_filename)

            if not pathlib.Path(basedir + input_filename).exists():
                print("input file not found locally for " + batch.input_file_id)
                continue

            output_filename = input_filename.split(".")[0] + "_out.jsonl"
            download_batch_output(client, batch, output_filename, basedir)

            upload_batch_milvus(
                basedir + input_filename,
                basedir + output_filename,
                chunk_metadatas,
                coll_name,
            )


def insert_data_milvus(
    coll_name: str,
    vectors: list,
    texts: list,
    chunk_idxs: list,
    metadatas: list,
    source_ids: list,
) -> dict:
    # insert to milvus
    data = [{
        "vector": vectors[k],
        "text": texts[k],
        "opinion_id": source_ids[k],
        "chunk_index": chunk_idxs[k],
        "metadata": metadatas[k],
    } for k in range(len(metadatas))]
    upload_result = upload_data(coll_name, data)
    if upload_result["insert_count"] != len(metadatas):
        print("error: bad upload")
    return upload_result


def upload_batch_milvus(
    input_filepath: str,
    output_filepath: str,
    chunk_metadatas: dict,
    uploaded_sources: set,
    coll_name: str,
) -> None:
    """Upload a batch file output to Milvus."""
    metadatas, texts, vectors, chunk_idxs, opinion_ids = [], [], [], [], []
    customid_inline = {}
    inline_text = {}
    with pathlib.Path(input_filepath).open("r") as in_f:
        # index input lines
        for j, line in enumerate(in_f, start=1):
            input_data = json.loads(line)
            customid_inline[input_data["custom_id"]] = j
            # just need the text from the input file
            inline_text[j] = input_data["body"]["input"]
    with pathlib.Path(output_filepath).open("r") as out_f:
        for j, line in enumerate(out_f, start=1):
            output_data = json.loads(line)
            # check output
            if output_data["response"]["status_code"] != 200:
                print(f"error: bad status code for {output_data['custom_id']}")
                continue
            # get vector
            vector = output_data["response"]["body"]["data"][0]["embedding"]
            # get text
            inline = customid_inline[output_data["custom_id"]]
            text = inline_text[inline]
            # get metadata
            custom_id_split = output_data["custom_id"].split("-")
            opinion_id = int(custom_id_split[1])
            # check if the opinion was already uploaded
            if opinion_id in uploaded_sources:
                continue
            chunk_idx = int(custom_id_split[2])
            opinion_ids.append(opinion_id)
            chunk_idxs.append(chunk_idx)
            metadata = chunk_metadatas[opinion_id]
            # add to batch
            metadatas.append(metadata)
            texts.append(text)
            vectors.append(vector)
            if len(metadatas) == 5000:
                print(f"j = {j}")
                upload_result = insert_data_milvus(
                    coll_name,
                    vectors,
                    texts,
                    chunk_idxs,
                    metadatas,
                    opinion_ids,
                )
                if upload_result["insert_count"] != 5000:
                    print("error: bad upload")
                    continue
                metadatas, texts, vectors,  = [], [], []
                chunk_idxs, opinion_ids = [], []
    # upload the last <1000 lines
    num_remaining = len(metadatas)
    if num_remaining > 0:
        upload_result = insert_data_milvus(
            coll_name,
            vectors,
            texts,
            chunk_idxs,
            metadatas,
            opinion_ids,
        )
        if upload_result["insert_count"] != num_remaining:
            print("error: bad upload")


def add_summaries_to_metadatas(chunk_metadatas: dict) -> None:
    """Get summaries from old collections."""
    for coll_name in ["courtlistener"]:
        q_iter = Collection(coll_name).query_iterator(
            expr="exists metadata['ai_summary']",
            output_fields=["metadata"],
            batch_size=1000,
        )
        res = q_iter.next()
        while len(res) > 0:
            for hit in res:
                if hit["metadata"]["id"] in chunk_metadatas:
                    matching_chunk = chunk_metadatas[hit["metadata"]["id"]]
                    matching_chunk["ai_summary"] = hit["metadata"]["ai_summary"]
            res = q_iter.next()
        q_iter.close()


def batch_main() -> None:
    coll_name = "test_firebase"
    basedir = str(pathlib.Path.cwd()) + "/data/courtlistener/"
    files = [f for f in os.listdir(basedir) if f.startswith("chunks_")]
    chunk_metadatas = load_cl_chunk_metadatas(basedir)
    add_summaries_to_metadatas(chunk_metadatas)
    while len(files) > 0:
        print(f"{len(files)} files remaining")
        print("uploading batches")
        upload_cl_batch_files(basedir, chunk_metadatas, coll_name)
        print("deleting completed batch files from OpenAI API")
        delete_completed_batches(basedir)
        print("retrying failed batches in OpenAI API")
        retry_failed_batches()
        print("waiting for batches to complete")
        wait_for_batches()
        files = [f for f in os.listdir(basedir) if f.startswith("chunks_")]


def finishfile() -> None:
    basedir = str(pathlib.Path.cwd()) + "/data/courtlistener/"
    chunk_metadatas = load_cl_chunk_metadatas(basedir)
    add_summaries_to_metadatas(chunk_metadatas)
    coll_name = "test_firebase"
    upload_batch_milvus(
        basedir + "chunks_122.jsonl",
        basedir + "chunks_122_out.jsonl",
        chunk_metadatas,
        coll_name,
    )


def upload_cl_completed_batches() -> None:
    """Upload CL data from locally downloaded batch files."""
    basedir = str(pathlib.Path.cwd()) + "/data/courtlistener/"
    chunk_metadatas = load_cl_chunk_metadatas(basedir)
    add_summaries_to_metadatas(chunk_metadatas)
    coll_name = "courtlistener_bulk"
    # get uploaded opinions
    q_iter = query_iterator(coll_name, "", ["opinion_id"], MAX_K)
    uploaded_sources = set()
    res = q_iter.next()
    while len(res) > 0:
        for hit in res:
            if hit["opinion_id"] not in uploaded_sources:
                uploaded_sources.add(hit["opinion_id"])
        res = q_iter.next()
    q_iter.close()
    # this was the last group of batches uploaded
    for i in range(50, 55):
        print(f"i = {i}")
        in_filename =  f"chunks_{i}.jsonl"
        out_filename = f"chunks_{i}_out.jsonl"
        upload_batch_milvus(
            basedir + "chunks/" + in_filename,
            basedir + "chunks/" + out_filename,
            chunk_metadatas,
            uploaded_sources,
            coll_name,
        )


def delete_batch() -> None:
    import json
    import pathlib

    from app.db import get_batch, load_vdb_source
    from app.milvusdb import delete_expr

    coll_name = "test_firebase"
    basedir = str(pathlib.Path.cwd()) + "/data/courtlistener/"
    with pathlib.Path(basedir + "chunks_104.jsonl").open("r") as f:
        lines = f.readlines()[30000:]
    opinion_ids = set()
    batch = get_batch()
    for i, line in enumerate(lines, start=1):
        req = json.loads(line)
        custom_id_split = req["custom_id"].split("-")
        opinion_id = int(custom_id_split[1])
        if opinion_id not in opinion_ids:
            opinion_ids.add(opinion_id)
            source = load_vdb_source(coll_name, opinion_id)
            chunks = source.collection("chunks").get()
            for doc in chunks:
                batch.delete(doc.reference)
        if i % 1000 == 0:
            print(delete_expr(coll_name, f"source_id in {list(opinion_ids)}"))
            batch.commit()
            batch = get_batch()
            opinion_ids = set()
    print(delete_expr(coll_name, f"source_id in {list(opinion_ids)}"))
    batch.commit()
