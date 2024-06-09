import unittest

import app.models as models

class UUIDTests(unittest.TestCase):
    uuid = models.get_uuid_id()
    def test_uuid_type(self):
        self.assertTrue(isinstance(self.uuid, str))

    def test_uuid_length(self):
        self.assertEqual(len(self.uuid), 36)


# TODO: more unit tests
if __name__ == "__main__":
    unittest.main()
