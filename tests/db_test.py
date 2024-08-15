import os
import unittest

from app import db

TEST_API_KEY = os.environ["OPB_TEST_API_KEY"]
class ApiKeyTests(unittest.TestCase):
    def test_validKey(self):
        key = TEST_API_KEY
        self.assertFalse(db.admin_check(key))
        self.assertTrue(db.api_key_check(key))

    def test_invalidKey(self):
        key = "abc"
        self.assertFalse(db.admin_check(key))
        self.assertFalse(db.api_key_check(key))

if __name__ == "__main__":
    unittest.main()
