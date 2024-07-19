"""A module for interacting with the CourtListener API. Written by Arman Aydemir."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from datetime import UTC, datetime, timedelta

import requests
from langfuse.decorators import observe

from app.milvusdb import fuzzy_keyword_query, query, upload_courtlistener

courtlistener_token = os.environ["COURTLISTENER_API_KEY"]
courtlistener_header = {"Authorization": "Token " + courtlistener_token}
base_url = "https://www.courtlistener.com/api/rest/v3"
search_url = base_url + "/search/?q="
opinion_url = base_url + "/opinions/?id="
cluster_url = base_url + "/clusters/?id="
docket_url = base_url + "/dockets/?id="
court_url = base_url + "/courts/?id="
people_url = base_url + "/people/?id="

courtlistener_collection = "courtlistener"

courtlistener_timeout = 10 #seconds

courtlistener_tool_args = {
    "jurisdiction": {
        "type": "string",
        "description": (
            "The two-letter abbreviation of a state or territory, e.g. 'NJ' or 'TX', "
            "to filter query results by jurisdiction. Use 'US' for federal courts."
        ),
    },
    "keyword-qr": {
        "type": "string",
        "description": (
            "A keyword query to search for exact names and terms."
        ),
    },
    "after-date": {
        "type": "string",
        "description": (
            "The after date for the query date range in YYYY-MM-DD "
            "format."
        ),
    },
    "before-date": {
        "type": "string",
        "description": (
            "The before date for the query date range in YYYY-MM-DD "
            "format."
        ),
    },
}

# https://github.com/freelawproject/courtlistener/discussions/3114
# manual mapping from two-letter state abbreviations to courtlistener court_id format
jurisdiction_codes = {
    "us-app": "ca1 ca2 ca3 ca4 ca5 ca6 ca7 ca8 ca9 ca10 ca11 cadc cafc",
    "us-dis": "dcd almd alnd alsd akd azd ared arwd cacd caed cand casd cod ctd ded flmd flnd flsd gamd gand gasd hid idd ilcd ilnd ilsd innd insd iand iasd ksd kyed kywd laed lamd lawd med mdd mad mied miwd mnd msnd mssd moed mowd mtd ned nvd nhd njd nmd nyed nynd nysd nywd nced ncmd ncwd ndd ohnd ohsd oked oknd okwd ord paed pamd pawd rid scd sdd tned tnmd tnwd txed txnd txsd txwd utd vtd vaed vawd waed wawd wvnd wvsd wied wiwd wyd gud nmid prd vid californiad illinoised illinoisd indianad orld ohiod pennsylvaniad southcarolinaed southcarolinawd tennessed canalzoned",
    "us-sup": "scotus",
    "us-misc": "bap1 bap2 bap6 bap8 bap9 bap10 ag afcca asbca armfor acca uscfc tax bia olc mc mspb nmcca cavc bva fiscr fisc cit usjc jpml cc com ccpa cusc bta eca tecoa reglrailreorgct kingsbench",
    "al": "almd alnd alsd almb alnb alsb ala alactapp alacrimapp alacivapp",
    "ak": "akd akb alaska alaskactapp",
    "az": "azd arb ariz arizctapp ariztaxct",
    "ar": "ared arwd areb arwb ark arkctapp arkworkcompcom arkag",
    "as": "amsamoa amsamoatc",
    "ca": "cacd caed cand casd californiad cacb caeb canb casb cal calctapp calappdeptsuper calag",
    "co": "cod cob colo coloctapp coloworkcompcom coloag",
    "ct": "ctd ctb conn connappct connsuperct connworkcompcom",
    "de": "ded deb del delch delorphct delsuperct delctcompl delfamct deljudct",
    "dc": "dcd dcb dc",
    "fl": "flmd flnd flsd flmb flnb flsb fla fladistctapp flaag",
    "ga": "gamd gand gasd gamb ganb gasb ga gactapp",
    "gu": "gud gub",
    "hi": "hid hib haw hawapp",
    "id": "idd idb idaho idahoctapp",
    "il": "ilcd ilnd ilsd illinoised illinoisd ilcb ilnb ilsb ill illappct",
    "in": "innd insd indianad innb insb ind indctapp indtc",
    "ia": "iand iasd ianb iasb iowa iowactapp",
    "ks": "ksd ksb kan kanctapp kanag",
    "ky": "kyed kywd kyeb kywb ky kyctapp kyctapphigh",
    "la": "laed lamd lawd laeb lamb lawb la lactapp laag",
    "me": "med bapme meb me mesuperct",
    "md": "mdd mdb md mdctspecapp mdag",
    "ma": "mad bapma mab mass massappct masssuperct massdistct masslandct maworkcompcom",
    "mi": "mied miwd mieb miwb mich michctapp",
    "mn": "mnd mnb minn minnctapp minnag",
    "ms": "msnd mssd msnb mssb miss missctapp",
    "mo": "moed mowd moeb mowb mo moctapp moag",
    "mt": "mtd mtb mont monttc montag",
    "ne": "ned nebraskab neb nebctapp nebag",
    "nv": "nvd nvb nev nevapp",
    "nh": "nhd nhb nh",
    "nj": "njd njb nj njsuperctappdiv njtaxct njch",
    "nm": "nmd nmb nm nmctapp",
    "ny": "nyed nynd nysd nywd nyeb nynb nysb nywb ny nyappdiv nyappterm nysupct nycountyctny nydistct nyjustct nyfamct nysurct nycivct nycrimct nyag",
    "nc": "nced ncmd ncwd nceb ncmb ncwb nc ncctapp ncsuperct ncworkcompcom",
    "nd": "ndd ndb nd ndctapp",
    "mp": "nmariana cnmisuperct cnmitrialct",
    "oh": "ohnd ohsd ohiod ohnb ohsb ohio ohioctapp ohioctcl",
    "ok": "oked oknd okwd okeb oknb okwb okla oklacivapp oklacrimapp oklajeap oklacoj oklaag",
    "or": "ord orb or orctapp ortc",
    "pa": "paed pamd pawd pennsylvaniad paeb pamb pawb pa pasuperct pacommwct cjdpa",
    "pr": "prsupreme prapp prd prb",
    "ri": "rid rib ri risuperct",
    "sc": "scd southcarolinaed southcarolinawd scb sc scctapp",
    "sd": "sdd sdb sd",
    "tn": "tned tnmd tnwd tennessed tneb tnmb tnwb tennesseeb tenn tennctapp tenncrimapp tennworkcompcl tennworkcompapp tennsuperct",
    "tx": "txed txnd txsd txwd txeb txnb txsb txwb tex texapp texcrimapp texreview texjpml texag",
    "ut": "utd utb utah utahctapp",
    "vt": "vtd vtb vt vtsuperct",
    "va": "vaed vawd vaeb vawb va vactapp",
    "vi": "vid vib",
    "wa": "waed wawd waeb wawb wash washctapp washag washterr",
    "wv": "wvnd wvsd wvnb wvsb wva",
    "wi": "wied wiwd wieb wiwb wis wisctapp wisag",
    "wy": "wyd wyb wyo",
}


def search(q: str) -> dict:
    """Call the general search api from courtlistener.

    Parameters
    ----------
    q : str
        the query

    Returns
    -------
    dict
        dict containing the results

    """
    max_retries = 5
    retry_delay = 1 # second
    for _ in range(max_retries):
        try:
            response = requests.get(
                search_url + q,
                headers=courtlistener_header,
                timeout=courtlistener_timeout,
            )
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.ReadTimeout:  # noqa: PERF203
            time.sleep(retry_delay)
    return {}

def get_opinion(result:dict) -> dict:
    """Get the full opinion info for a search result from search().

    Parameters
    ----------
    result : dict
        a single result from search()

    Returns
    -------
    dict
        dict containing the Opinion info

    """
    # get the opinion id
    opinion_id = str(result["id"])

    response = requests.get(opinion_url + opinion_id,
                            headers=courtlistener_header,
                            timeout=courtlistener_timeout)

    op = response.json()["results"][0]  # the actual opinion

    # getting the text, in the best format possible
    op["text"] = op["html_with_citations"]
    # these are backup formats
    backups = ["html", "plain_text", "html_lawbox", "html_columbia"]
    b_index = 0
    while op["text"] == "" and b_index < len(backups):
        op["text"] = op[backups[b_index]]
        b_index += 1
    return op


def get_cluster(result: dict) -> dict:
    """Get the full cluster info for a search result from search().

    Parameters
    ----------
    result : dict
        A single result from search()

    Returns
    -------
    dict
        dict containing the Cluster info

    """
    cid = str(result["cluster_id"])
    response = requests.get(cluster_url + cid,
                            headers=courtlistener_header,
                            timeout=courtlistener_timeout)

    return response.json()["results"][0]

def get_docket(result: dict) -> dict:
    """Get the full docket info for a search result from search().

    Parameters
    ----------
    result : dict
        A single result from search()

    Returns
    -------
    dict
        dict containing the Docket info

    """
    docket_id = str(result["docket_id"])
    response = requests.get(docket_url + docket_id,
                            headers=courtlistener_header,
                            timeout=courtlistener_timeout)

    return response.json()["results"][0]

def get_court(result: dict) -> dict:
    """Get the full court info for a search result from search().

    Parameters
    ----------
    result : dict
        A single result from search()

    Returns
    -------
    dict
        dict containing the Court info

    """
    court_id = str(result["court_id"])
    response = requests.get(court_url + court_id,
                            headers=courtlistener_header,
                            timeout=courtlistener_timeout)

    return response.json()["results"][0]

def get_person(result: dict) -> dict:
    """Get the full person info for a search result from search().

    Parameters
    ----------
    result : dict
        A single result from search()

    Returns
    -------
    dict
        dict containing the Person info

    """
    author_id = str(result["author_id"])
    response = requests.get(people_url + author_id,
                            headers=courtlistener_header,
                            timeout=courtlistener_timeout)

    return response.json()["results"][0]

def get_search_date(date_plus_timerange: str) -> str:
    split_date = date_plus_timerange.split("T")
    date_filed = split_date[0]
    split_time = split_date[1].split("-")
    len_split = 2
    if len(split_time) == len_split and split_time[0] > split_time[1]:
        # the time range goes overnight, so add one day to the date
        dt = datetime.strptime(date_filed, "%Y-%m-%d").replace(tzinfo=UTC)
        dt += timedelta(days=1)
        date_filed = dt.strftime("%Y-%m-%d")
    return date_filed

def get_case_name(result: dict) -> str:
    if result["caseName"]:
        return result["caseName"]
    # cluster level data: case name
    cluster = get_cluster(result)
    # prefer short name to full name
    if cluster["case_name"]:
        return cluster["case_name"]
    if cluster["case_name_short"]:
        return cluster["case_name_short"]
    return cluster["case_name_full"]

def get_author_name(result: dict) -> str:
    # person level data: author name
    author = get_person(result)
    full_name = ""
    if author["name_first"]:
        full_name += author["name_first"]
    if author["name_middle"]:
        full_name += (" " if full_name else "") + author["name_middle"]
    if author["name_last"]:
        full_name += (" " if full_name else "") + author["name_last"]
    return full_name

def upload_search_result(result: dict) -> None:
    opinion = get_opinion(result)
    opinion["cluster_id"] = result["cluster_id"]
    opinion["court_id"] = result["court_id"]
    opinion["date_filed"] = get_search_date(result["dateFiled"])
    opinion["docket_id"] = result["docket_id"]
    opinion["court_name"] = result["court"]
    opinion["citations"] = result["citation"]
    opinion["case_name"] = get_case_name(result)
    if result["author_id"]:
        opinion["author_id"] = result["author_id"]
        opinion["author_name"] = get_author_name(result)
    upload_courtlistener(courtlistener_collection, opinion)

@observe(capture_output=False)
def courtlistener_search(
    q: str,
    k: int = 3,
    jurisdictions: list[str] | None = None,
    keyword_query: str | None = None,
    after_date: str | None = None,
    before_date: str | None = None,
) -> dict:
    """Search courtlistener for a query.

    Search, get opinion text, upload the opinion data to milvus, and query it.

    Parameters
    ----------
    q : str
        The query
    k : int, optional
        The number of results to return, by default 3
    jurisdictions : list[str] | None, optional
        The two-letter abbreviations of a state or territory, e.g. 'NJ' or 'TX',
        to filter query results by state. Use 'us-app' for federal appellate,
        'us-dis' for federal district, 'us-sup' for supreme court, 'us-misc'
        for federal special. By default None.
    keyword_query: str | None, optional
        The users keyword query, by default None
    after_date : str | None, optional
        The after date for the query date range in YYYY-MM-DD format, by default None
    before_date : str | None, optional
        The before date for the query date range in YYYY-MM-DD format, by default None

    Returns
    -------
    dict
        the response with relevant info from courtlistener

    """
    # use semantic query if keyword not given
    query_str = '"' + keyword_query + '"' if keyword_query else q
    # add options to query string
    valid_jurisdics = []
    if jurisdictions:
        # look up each str in dictionary, append matches as lists
        for juris in jurisdictions:
            if juris in jurisdiction_codes:
                valid_jurisdics += jurisdiction_codes[juris].split(" ")
        # clear duplicate federal district jurisdictions if they exist
        valid_jurisdics = list(set(valid_jurisdics))
        query_str += f"&court={' '.join(valid_jurisdics)}"
    if after_date:
        dt = after_date.split("-")
        # needs to be in MM-DD-YYYY format
        query_str += f"&filed_after={dt[1]}-{dt[2]}-{dt[0]}"
    if before_date:
        dt = before_date.split("-")
        # needs to be in MM-DD-YYYY format
        query_str += f"&filed_before={dt[1]}-{dt[2]}-{dt[0]}"

    with ThreadPoolExecutor() as executor:
        futures = []
        for result in search(query_str)["results"][:k]:
            ctx = copy_context()
            def task(r=result, context=ctx):  # noqa: ANN001, ANN202
                return context.run(upload_search_result, r)
            futures.append(executor.submit(task))

        for future in as_completed(futures):
            _ = future.result()
    return courtlistener_query(q, k, valid_jurisdics, keyword_query, after_date, before_date)

@observe(capture_input=False, capture_output=False)
def courtlistener_query(
    q: str,
    k: int,
    jurisdictions: list[str] | None = None,
    keyword_query: str | None = None,
    after_date: str | None = None,
    before_date: str | None = None,
) -> dict:
    """Query Courtlistener data.

    Parameters
    ----------
    q : str
        The query text
    k : int
        How many chunks to return
    jurisdictions : str | None, optional
        The two-letter abbreviations of a state or territory, e.g. 'nj' or 'tx',
        to filter query results by state. Unlike `courtlistener_search`, these
        should already be in courtlistener court_id format. By default None.
    keyword_query: str | None, optional
        The users keyword query, by default None
    after_date : str | None, optional
        The after date for the query date range in YYYY-MM-DD format, by default None
    before_date : str | None, optional
        The before date for the query date range in YYYY-MM-DD format, by default None

    Returns
    -------
    dict
        Contains `message`, `result` list if successful

    """
    expr = ""
    # copy keyword query to semantic if not given
    q = q if q else keyword_query
    if jurisdictions:
        expr = f"metadata['court_id'] in {jurisdictions}"
    if after_date:
        expr += (" and " if expr else "") + f"metadata['date_filed']>'{after_date}'"
    if before_date:
        expr += (" and " if expr else "") + f"metadata['date_filed']<'{before_date}'"
    if keyword_query:
        keyword_query = fuzzy_keyword_query(keyword_query)
        expr += (" and " if expr else "") + f"text like '% {keyword_query} %'"
    return query(courtlistener_collection, q, k, expr)


def process_bulkdata() -> None:
    """Process CourtListener's bulk data file of opinions.

    Chunks opinions and makes a batch file for embedding with OpenAI.
    """
    import ast
    import bz2
    import io
    import json
    import pathlib

    import pandas as pd
    import requests
    from unstructured.partition.text import partition_text
    from unstructured.partition.xml import partition_xml

    from app.loaders import partition_html_str, partition_pdf
    from app.models import OpenAIModelEnum
    from app.splitters import chunk_elements_by_title

    def get_matching_cluster_ids(filename: str, ids_filename: str):
        """Get cluster IDs since 1980 with at least 1 citation."""
        matching_ids = set()
        total_rows = 0
        cutoff_date = "1979-12-31"

        with bz2.open(filename, "rt") as bz_file:
            for chunk in pd.read_csv(
                bz_file,
                chunksize=100000,
                usecols=["id", "date_filed", "citation_count"],
                parse_dates=["date_filed"],
            ):
                total_rows += len(chunk)

                # Filter rows that meet both conditions and add their ids to the set
                matches = chunk[(chunk["date_filed"] > cutoff_date) & (chunk["citation_count"] > 0)]
                matching_ids.update(matches["id"])

                if total_rows % 1000000 == 0:
                    print(f"Processed {total_rows} rows in cluster file...")

        print(f"\nTotal rows processed in cluster file: {total_rows}")
        print(f"Number of matching IDs: {len(matching_ids)}")
        with pathlib.Path(ids_filename).open("w") as f:
            f.write(str(matching_ids))

    def get_matching_opinion_ids(dirname: str, ids_filename: str):
        """Get the IDs of opinions that have already been chunked, for resuming."""
        matching_ids = set()

        for filename in os.listdir(dirname):
            print(filename)
            if not filename.endswith(".jsonl"):
                continue
            with pathlib.Path(dirname + filename).open("r") as f:
                lines = f.readlines()
            for line in lines:
                d = json.loads(line)
                opinion_id = d["custom_id"].split("-")[1]
                if opinion_id in matching_ids:
                    continue
                matching_ids.add(opinion_id)
        with pathlib.Path(ids_filename).open("w") as f:
            f.write(str(matching_ids))

    def process_opinion_file(filename: str, matching_clusters: set, matching_ids: set):
        """Chunk opinions and create jsonl batch files."""
        total_rows = 0
        req_input = {"custom_id": "", "method": "POST", "url": "/v1/embeddings",
                 "body": {"model": OpenAIModelEnum.embed_small, "input": ""}}
        file_number = 19
        current_file_size = 0
        current_file = None
        max_file_size_bytes = int(0.95 * 100 * 1024 * 1024)
        with bz2.open(filename, "rt") as bz_file:
            for chunk in pd.read_csv(bz_file, chunksize=10000):
                total_rows += len(chunk)
                matches = chunk[chunk["cluster_id"].isin(matching_clusters)]
                matches = matches[~matches["id"].isin(matching_ids)]
                for _, row in matches.iterrows():
                    val = None
                    if pd.notna(row["html_with_citations"]):
                        val = row["html_with_citations"]
                    elif pd.notna(row["html"]):
                        val = row["html"]
                    elif pd.notna(row["html_lawbox"]):
                        val = row["html_lawbox"]

                    elements = None
                    if val is not None:
                        # try parsing html
                        try:
                            elements = partition_html_str(val)
                        except Exception as e:
                            print(f"error parsing html for cluster {row['cluster_id']} opinion {row['id']}")
                            print(e)

                    # try parsing download_url (PDF)
                    if elements is None and pd.notna(row["download_url"]) and row["download_url"].endswith(".pdf"):
                        try:
                            r = requests.get(row["download_url"], timeout=60)
                            val = io.BytesIO(r.content)
                            elements = partition_pdf(file=val)
                            print("PDF downloaded")
                        except Exception as e:
                            print(f"error parsing PDF for cluster {row['cluster_id']} opinion {row['id']}")
                            print(e)

                    # try parsing xml
                    if elements is None and pd.notna(row["xml_harvard"]):
                        val = row["xml_harvard"]
                        try:
                            elements = partition_xml(text=val)
                        except Exception as e:
                            print(f"error parsing XML for cluster {row['cluster_id']} opinion {row['id']}")
                            print(e)

                    # try parsing plain_text
                    if elements is None and pd.notna(row["plain_text"]):
                        val = row["plain_text"]
                        try:
                            elements = partition_text(text=val)
                        except Exception as e:
                            print(f"error parsing txt for cluster {row['cluster_id']} opinion {row['id']}")
                            print(e)

                    if elements is None:
                        continue

                    # chunk
                    texts, _ = chunk_elements_by_title(
                        elements,
                        10000,
                        2500,
                        1000,
                    )
                    custom_id = 1
                    if current_file is None or current_file_size >= max_file_size_bytes:
                        if current_file:
                            current_file.close()
                        current_filename = f"/Users/njc/Documents/programming/opb/data/courtlistener_bulk/chunks_{file_number}.jsonl"
                        current_file = pathlib.Path(current_filename).open("w")
                        current_file_size = 0
                        file_number += 1
                        print(f"Started writing to new file: {current_filename}")
                    for text in texts:
                        req_input["custom_id"] = str(row["cluster_id"]) + "-" + str(row["id"]) + "-" + str(custom_id)
                        req_input["body"]["input"] = text
                        json_line = json.dumps(req_input) + "\n"
                        line_size = len(json_line.encode("utf-8"))
                        current_file.write(json_line)
                        custom_id += 1
                        current_file_size += line_size

                if total_rows % 100000 == 0:
                    print(f"Processed {total_rows} rows in opinion file...")

        if current_file:
            current_file.close()
        print(f"\nTotal rows processed in opinion file: {total_rows}")

    # Usage
    basedir = "/opb/data/courtlistener_bulk/"
    opinion_filename = basedir + "opinions-2024-05-06.csv.bz2"
    cluster_filename = basedir + "opinion-clusters-2024-05-06.csv.bz2"
    opinion_ids_filename = basedir + "opinion_ids"
    cluster_ids_filename = basedir + "cluster_ids"

    if not pathlib.Path(cluster_ids_filename).exists():
        get_matching_cluster_ids(cluster_filename, cluster_ids_filename)
    with pathlib.Path(cluster_ids_filename).open("r") as f:
        matching_clusters = ast.literal_eval(f.read())
    if not pathlib.Path(opinion_ids_filename).exists():
        get_matching_opinion_ids(basedir, opinion_ids_filename)
    with pathlib.Path(opinion_ids_filename).open("r") as f:
        matching_ids = ast.literal_eval(f.read())
    process_opinion_file(opinion_filename, matching_clusters, matching_ids)
