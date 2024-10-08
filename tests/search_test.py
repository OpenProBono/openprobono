import unittest
from re import search

from app import search_tools
from app.models import SearchTool


class SearchToolTests(unittest.TestCase):

    def test_serpapi_search(self):
        qr = "Test Search"
        prf = ""
        results = search_tools.serpapi_tool(qr, prf)
        self.assertTrue(isinstance(results, dict))
        self.assertTrue(len(results.keys()) != 0)

    def test_whitelist_serpapi_search(self):
        qr = "Wyoming rule for designating email address for service in litigation"
        prf = "site:*.gov | site:*.edu | site:*scholar.google.com"
        results = search_tools.serpapi_tool(qr, prf)
        self.assertTrue(isinstance(results, dict))
        self.assertTrue(len(results.keys()) != 0)

    def test_dynamic_serpapi_search(self):
        qr = "Wyoming rule for designating email address for service in litigation"
        prf = "site:*.gov | site:*.edu | site:*scholar.google.com"
        st = SearchTool(name="test_tool", prompt="test_tool_prompt")
        results = search_tools.dynamic_serpapi_tool(qr, prf, st, num_results=1)
        self.assertTrue(isinstance(results, dict))
        self.assertTrue(len(results.keys()) != 0)

    def test_courtroom5_search(self):
        qr = "Wyoming rule for designating email address for service in litigation"
        results = search_tools.courtroom5_search_tool(qr)
        self.assertTrue(isinstance(results, str))
        self.assertTrue(len(results) != 0)

    def test_dynamic_courtroom5_search(self):
        qr = "Wyoming rule for designating email address for service in litigation"
        st = SearchTool(name="test_tool", prompt="test_tool_prompt")
        results = search_tools.dynamic_courtroom5_search_tool(qr, st)
        self.assertTrue(isinstance(results, dict))
        self.assertTrue(len(results.keys()) != 0)

# TODO: more unit tests
if __name__ == "__main__":
    unittest.main()
