import unittest

import search_tools


class SearchToolTests(unittest.TestCase):

    def test_serpapi_search(self):
        qr = "Test Search"
        prf = ""
        results = search_tools.serpapi_tool(qr, prf)
        self.assertTrue(isinstance(results, dict))
        self.assertEqual(len(results.keys()) != 0, True)

    def test_whitelist_serpapi_search(self):
        qr = "Wyoming rule for designating email address for service in litigation"
        prf = "site:*.gov | site:*.edu | site:*scholar.google.com"
        results = search_tools.serpapi_tool(qr, prf)
        self.assertTrue(isinstance(results, dict))
        self.assertEqual(len(results.keys()) != 0, True)
# TODO: more unit tests
if __name__ == "__main__":
    unittest.main()
