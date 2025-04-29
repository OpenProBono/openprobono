"""Tests for Milvus vector database."""
import warnings
from pathlib import Path

import pymilvus
import pytest
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
        assert isinstance(firebase_coll.encoder, EncoderParams)
        if firebase_coll.encoder.name != OpenAIModelEnum.embed_ada_2:
            # check the dimensions of the vector field
            milvus_coll = pymilvus.Collection(coll)
            vector_field = None
            for field in milvus_coll.schema.fields:
                if field.dtype == pymilvus.DataType.FLOAT_VECTOR:
                    vector_field = field
                    break
            assert vector_field is not None
            assert firebase_coll.encoder.dim == vector_field.params["dim"]
        # get metadata fields from Milvus
        milvus_field_names = []
        for milvus_field in milvus_coll.schema.fields:
            if milvus_field.name in {"pk", "vector", "text", "sparse"}:
                continue
            milvus_field_names.append(milvus_field.name)
        # test metadata
        match firebase_coll.metadata_format:
            case MilvusMetadataEnum.field:
                field_names = [f.name for f in firebase_coll.extra_fields]
                assert sorted(milvus_field_names) == sorted(field_names)
            case MilvusMetadataEnum.json:
                assert milvus_field_names == ["metadata"]
            case MilvusMetadataEnum.no_field:
                assert milvus_field_names == []


def test_get_expr() -> None:
    result = milvusdb.get_expr(collection_name, test_expr)
    assert result["message"] == "Success"
    assert "result" in result
    assert len(result["result"]) > 0


def test_upload_resource_file() -> None:
    fname = "test_text.txt"
    with Path(fname).open("w") as f:
        f.write("test text\n")
    with Path(fname).open("rb") as fp:
        f = UploadFile(file=fp, filename=fname)
        result = milvusdb.upload_resource(
            collection_name=collection_name,
            resource_type="file",
            resource=f,
            session_id="test_session_id",
        )
        assert result["message"] == "Success"
        assert result["insert_count"] > 0


def test_upload_resource_url() -> None:
    # Mock data for URL resource
    from app.models import SearchTool
    # Create a search result mock
    search_result = {
        "link": "https://example.com",
        "title": "Example Domain",
        "source": "Test Source",
    }
    # Create a test search tool
    search_tool = SearchTool(name="test_tool", prompt="")
    result = milvusdb.upload_resource(
        collection_name=collection_name,
        resource_type="url",
        resource=search_result,
        search_tool=search_tool,
        session_id="test_session_id",
    )
    assert result["message"] == "Success"
    assert result["insert_count"] > 0
