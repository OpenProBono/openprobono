"""Functions to search the British and Irish Legal Information Institute (BAILII)."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from threading import Lock

import requests
import urllib3
from bs4 import BeautifulSoup
from langfuse.decorators import langfuse_context, observe

from app.logger import setup_logger
from app.milvusdb import query, source_exists, upload_site
from app.models import SearchTool

BAILII_COLLECTION = "bailii"
BAILII_URL = "https://www.bailii.org/cgi-bin"
BAILII_SEARCH_PATH = "/lucy_search_1.cgi?datehigh=&method=boolean&query={query}&mask_path=/eu/cases+/ew/cases+/ie/cases+/nie/cases+/scot/cases+/uk/cases+/ae/cases+/qa/cases+/sh/cases+/je/cases+/ky/cases+/sg/cases&datelow=&sort=rank&highlight=1"
BAILII_RESULT_PATH = "/format.cgi?doc={doc}"
ADVANCED_SEARCH_DESC = """Advanced searches use CONNECTORS.

 - AND: all search terms in same document, e.g. customary AND right
 - OR: either or both search terms in the same document, e.g. customary OR ordinary
 - NOT: to exclude documents with a specific term, e.g. customary NOT statutory
 - " ": For phrase searches, e.g. "cruel and unusual punishment"
 - ( ): Terms placed within parentheses are processed as a unit and are processed before the terms and/or operators outside the parentheses, e.g. ("best interest" AND child) and ("foster care" OR adopt)"""


logger = setup_logger()

@observe()
def bailii_search(qr: str, tool: SearchTool) -> list[str]:
    """Query the BAILII search engine.

    Parameters
    ----------
    qr : str
        the query
    tool : SearchTool
        the tool using this search method

    Returns
    -------
    list[str]
        the list of search result URLs

    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:131.0) "
            "Gecko/20100101 Firefox/131.0"
        ),
    }
    adv_query = format_advanced_query(qr)
    url = BAILII_URL + BAILII_SEARCH_PATH.format(query=adv_query)
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
    except (
        requests.exceptions.Timeout,
        urllib3.exceptions.ConnectTimeoutError,
    ) as timeout_err:
        logger.exception("Timeout error: %s", url)
        langfuse_context.update_current_observation(
            level="ERROR",
            status_message=str(timeout_err),
        )
    except (requests.exceptions.SSLError, urllib3.exceptions.SSLError) as ssl_err:
        logger.exception("SSL error: %s", url)
        langfuse_context.update_current_observation(
            level="ERROR",
            status_message=str(ssl_err),
        )
    except (
        requests.exceptions.ConnectionError,
        urllib3.exceptions.ProtocolError,
    ) as conn_err:
        logger.exception("Connection error: %s", url)
        langfuse_context.update_current_observation(
            level="ERROR",
            status_message=str(conn_err),
        )
    except Exception as error:
        logger.exception("Unexpected error: %s", url)
        langfuse_context.update_current_observation(
            level="ERROR",
            status_message=str(error),
        )
    soup = BeautifulSoup(r.content)
    urls = get_result_links(soup)[:4]

    with ThreadPoolExecutor() as executor:
        futures = []
        for url in urls:
            ctx = copy_context()
            def task(u=url, t=tool, context=ctx):  # noqa: ANN001, ANN202
                return context.run(process_site, u, t)
            futures.append(executor.submit(task))

        for future in as_completed(futures):
            _ = future.result()

    expr = f"json_contains(metadata['bot_and_tool_id'], '{tool.bot_id + tool.name}')"
    res = query(BAILII_COLLECTION, qr, expr=expr)
    if "result" in res:
        pks = [str(hit["pk"]) for hit in res["result"]]
        langfuse_context.update_current_observation(output=pks)
    return res


# Global lock for URL processing
url_locks = {}
url_lock = Lock()

# Set to keep track of failed URLs
failed_urls = set()
failed_urls_lock = Lock()

def process_site(url: str, tool: SearchTool) -> None:
    num_attempts = 10
    with url_lock:
        if url not in url_locks:
            url_locks[url] = Lock()
    with url_locks[url]:
        logger.info("Site %s acquired lock", url)
        # Check if URL has already failed before processing
        with failed_urls_lock:
            if url in failed_urls:
                logger.info("Skipping previously failed URL: %s", url)
                return
        try:
            if not source_exists(BAILII_COLLECTION, url, tool.bot_id, tool.name):
                logger.info("Uploading site: %s", url)
                upload_site(BAILII_COLLECTION, url, tool)
                # check to ensure site appears in collection before releasing URL lock
                attempt = 0
                while not source_exists(BAILII_COLLECTION, url, tool.bot_id, tool.name) and attempt < num_attempts:
                    attempt += 1
                if attempt == num_attempts:
                    logger.error("Site not found in collection, add to failed URLs: %s", url)
                    with failed_urls_lock:
                        failed_urls.add(url)
            else:
                logger.info("Site already uploaded: %s", url)
        except Exception:
            logger.exception("Warning: Failed to upload site for dynamic serpapi: %s", url)
            with failed_urls_lock:
                failed_urls.add(url)
        logger.info("Site %s releasing lock", url)


def format_advanced_query(query: str) -> str:
    """Format an advanced query with Boolean operators and proximity connectors.

    Parameters
    ----------
    query : str
        The advanced search query

    Returns
    -------
    str
        Properly formatted query for the search engine

    """
    # Handle spaces within phrases first (preserve them)
    # Find all phrases in quotes and replace spaces with a temporary marker
    phrases = re.findall(r'"([^"]*)"', query)
    temp_query = query

    for phrase in phrases:
        phrase_with_markers = phrase.replace(" ", "_SPACE_")
        temp_query = temp_query.replace(f'"{phrase}"', f'"{phrase_with_markers}"')

    # Now replace all regular spaces with + signs
    temp_query = temp_query.replace(" ", "+")

    # Convert _SPACE_ markers back to spaces, but only within quoted phrases
    for phrase in phrases:
        phrase_with_markers = phrase.replace(" ", "_SPACE_")
        phrase_with_plus = phrase.replace(" ", "+")
        temp_query = temp_query.replace(f'"{phrase_with_markers}"', f'"{phrase_with_plus}"')

    # Your example shows that AND should be preserved, not converted to +
    # We'll keep all Boolean operators as they are

    # Final cleanup: ensure any extra + signs are removed
    return re.sub(r"\++", "+", temp_query)


def get_result_links(soup: BeautifulSoup) -> list[str]:
    """Extract the links from the search results page.

    Parameters
    ----------
    soup : BeautifulSoup
        The parsed HTML of the search results page.

    Returns
    -------
    list[str]
        A list of URLs found in the search results.

    """
    list_items = soup.find_all("li")
    docs = []
    for item in list_items:
        item_links = item.find_all("a")
        docs += [
            link["href"] for link in item_links
            if link and "href" in link.attrs and not link["href"].startswith("/cgi-bin")
        ]
    return [BAILII_URL + BAILII_RESULT_PATH.format(doc=doc) for doc in docs]
