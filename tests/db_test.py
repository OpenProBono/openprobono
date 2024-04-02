import unittest

import db


class ApiKeyTests(unittest.TestCase):
    def test_anotherKey(self):
        key = "xyz"
        self.assertFalse(db.admin_check(key))
        self.assertTrue(db.api_key_check(key))

    def test_validKey(self):
        key = "gradio"
        self.assertFalse(db.admin_check(key))
        self.assertTrue(db.api_key_check(key))

    def test_invalidKey(self):
        key = "abc"
        self.assertFalse(db.admin_check(key))
        self.assertFalse(db.api_key_check(key))


if __name__ == "__main__":
    unittest.main()
