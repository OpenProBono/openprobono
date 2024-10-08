"""Tests for Milvus vector database."""
import warnings
from pathlib import Path

import pymilvus
from fastapi import UploadFile

from app import milvusdb

collection_name = "DevTest"
test_expr = "metadata['key']=='ocsmqjosfa'"

def test_connection() -> None:
    conns = pymilvus.connections.list_connections()
    assert len(conns) == 1


def test_firebase_config() -> None:
    from app.models import EncoderParams, MilvusMetadataEnum, OpenAIModelEnum

    collections = pymilvus.utility.list_collections()
    for coll in collections:
        # get params from firebase
        firebase_coll = milvusdb.load_vdb(coll)
        if firebase_coll is None:
            # the collection config is not in firebase, skip for now
            warnings.warn(f"collection {coll} not found in firebase", stacklevel=1)
            continue
        # test encoder
        encoder = milvusdb.load_vdb_param(coll, "encoder")
        assert isinstance(encoder, EncoderParams)
        if encoder.name != OpenAIModelEnum.embed_ada_2:
            # check the dimensions of the vector field
            milvus_coll = pymilvus.Collection(coll)
            vector_field = None
            for field in milvus_coll.schema.fields:
                if field.dtype == pymilvus.DataType.FLOAT_VECTOR:
                    vector_field = field
                    break
            assert vector_field is not None
            assert encoder.dim == vector_field.params["dim"]
        # get metadata fields from Milvus
        milvus_field_names = []
        for milvus_field in milvus_coll.schema.fields:
            if milvus_field.name in {"pk", "vector", "text"}:
                continue
            milvus_field_names.append(milvus_field.name)
        # test metadata
        metadata_format = milvusdb.load_vdb_param(coll, "metadata_format")
        assert isinstance(metadata_format, MilvusMetadataEnum)
        match metadata_format:
            case MilvusMetadataEnum.field:
                fields = milvusdb.load_vdb_param(coll, "fields")
                assert isinstance(fields, list)
                assert sorted(milvus_field_names) == sorted(fields)
            case MilvusMetadataEnum.json:
                assert milvus_field_names == ["metadata"]
            case MilvusMetadataEnum.no_field:
                assert milvus_field_names == []

def test_empty_session_data() -> None:
    session_id = "emptysession"
    result = milvusdb.check_session_data(session_id)
    assert result is False

def test_session_data_uploaded() -> None:
    session_id = "d42992b5-dfe6-4966-8e97-a16d6c8f7c7d"
    result = milvusdb.check_session_data(session_id)
    assert result is True

def test_get_expr() -> None:
    result = milvusdb.get_expr(collection_name, test_expr)
    assert result["message"] == "Success"
    assert "result" in result
    assert len(result["result"]) > 0

def test_file_upload() -> None:
    with Path("test_text.txt").open("w") as f:
        f.write("test text\n")
    f = UploadFile(file=Path("test_text.txt").open("rb"))
    result = milvusdb.file_upload(f, "test_session_id", collection_name=collection_name)
    assert result["message"] == "Success"
    assert result["insert_count"] > 0
