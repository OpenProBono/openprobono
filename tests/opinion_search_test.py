"""Tests for opinion search."""
import pytest

from app.courtlistener import jurisdiction_codes
from app.opinion_search import opinion_search

test_query = "theme park"
test_jurisdiction = "wa"
test_after_date = "1989-12-31"
test_before_date = "1991-01-01"

def _test_results(results: list) -> None:
    """Test opinion search query results."""
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], dict)
    assert "distance" in results[0]
    assert "entity" in results[0]
    assert isinstance(results[0]["entity"], dict)
    assert "metadata" in results[0]["entity"]
    assert isinstance(results[0]["entity"]["metadata"], dict)
    assert "text" in results[0]["entity"]
    assert isinstance(results[0]["entity"]["text"], str)

@pytest.mark.parametrize("query", [test_query])
def test_query_unfiltered(query: str) -> None:
    """Run a query without jurisdiction and date filters."""
    results = opinion_search(query, None, None, None, 1)
    _test_results(results)

@pytest.mark.parametrize(("query", "jurisdiction"), [(test_query, test_jurisdiction)])
def test_query_jurisdiction(query: str, jurisdiction: str) -> None:
    """Run a query with a jurisdiction filter."""
    results = opinion_search(query, jurisdiction, None, None, 1)
    _test_results(results)
    query_juris_codes = jurisdiction_codes[jurisdiction].split(" ")
    assert "court_id" in results[0]["entity"]["metadata"]
    assert results[0]["entity"]["metadata"]["court_id"] in query_juris_codes

@pytest.mark.parametrize(("query", "after_date"), [(test_query, test_after_date)])
def test_query_after_date(query: str, after_date: str) -> None:
    """Run a query with an after date filter."""
    results = opinion_search(query, None, after_date, None, 1)
    _test_results(results)
    assert "date_filed" in results[0]["entity"]["metadata"]
    assert results[0]["entity"]["metadata"]["date_filed"] > after_date

@pytest.mark.parametrize(("query", "before_date"), [(test_query, test_before_date)])
def test_query_before_date(query: str, before_date: str) -> None:
    """Run a query with a before date filter."""
    results = opinion_search(query, None, None, before_date, 1)
    _test_results(results)
    assert "date_filed" in results[0]["entity"]["metadata"]
    assert results[0]["entity"]["metadata"]["date_filed"] < before_date

@pytest.mark.parametrize(
    ("query", "jurisdiction", "after_date", "before_date"),
    [(test_query, test_jurisdiction, test_after_date, test_before_date)],
)
def test_query_filtered(
    query: str,
    jurisdiction: str,
    after_date: str,
    before_date: str,
) -> None:
    """Run a query with all filters enabled."""
    results = opinion_search(query, jurisdiction, after_date, before_date, 1)
    _test_results(results)
    query_juris_codes = jurisdiction_codes[jurisdiction].split(" ")
    assert "court_id" in results[0]["entity"]["metadata"]
    assert results[0]["entity"]["metadata"]["court_id"] in query_juris_codes
    assert "date_filed" in results[0]["entity"]["metadata"]
    assert results[0]["entity"]["metadata"]["date_filed"] < before_date
    assert results[0]["entity"]["metadata"]["date_filed"] > after_date
