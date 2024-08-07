import unittest
from re import search

from app import milvusdb


class SearchToolTests(unittest.TestCase):

    def test_empty_session_data(self):
        session_id = "emptysession"
        result = milvusdb.check_session_data(session_id)
        assert result is False

    def test_session_data_uploaded(self):
        session_id = "d42992b5-dfe6-4966-8e97-a16d6c8f7c7d"
        result = milvusdb.check_session_data(session_id)
        assert result is True

# TODO: more unit tests
if __name__ == "__main__":
    unittest.main()
