"""Tests for text loading functions."""

import pytest

from app import loaders
from app.logger import setup_logger

logger = setup_logger()
test_pdf_url = "https://s29.q4cdn.com/175625835/files/doc_downloads/test.pdf"
test_rtf_url = "https://raw.githubusercontent.com/bitfocus/rtf2text/master/sample.rtf"
test_url = "https://www.openprobono.com/"

@pytest.mark.parametrize("url", [test_pdf_url, test_rtf_url, test_url])
def test_scrape(url: str) -> None:
    elements = loaders.scrape(url)
    logger.info([element.text for element in elements])
    assert len(elements) > 0
