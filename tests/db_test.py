import os
import unittest

from app import db

TEST_API_KEY = os.environ["OPB_TEST_API_KEY"]

def test_validKey() -> None:
    key = TEST_API_KEY
    assert db.admin_check(key) is True
    assert db.api_key_check(key) is True

def test_invalidKey() -> None:
    key = "abc"
    assert db.admin_check(key) is False
    assert db.api_key_check(key) is False
