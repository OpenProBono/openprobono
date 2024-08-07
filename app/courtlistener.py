"""A module for interacting with the CourtListener API. Written by Arman Aydemir."""
from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta

import requests
from langfuse.decorators import observe

from app.milvusdb import (
    fuzzy_keyword_query,
    get_expr,
    query,
    upload_courtlistener,
    upload_data_json,
)

courtlistener_token = os.environ["COURTLISTENER_API_KEY"]
courtlistener_header = {"Authorization": "Token " + courtlistener_token}
base_url = "https://www.courtlistener.com/api/rest/v3"
courtlistener_collection = "courtlistener_bulk"
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
    "us-dis": (
        "dcd almd alnd alsd akd azd ared arwd cacd caed cand casd cod ctd ded flmd "
        "flnd flsd gamd gand gasd hid idd ilcd ilnd ilsd innd insd iand iasd ksd kyed "
        "kywd laed lamd lawd med mdd mad mied miwd mnd msnd mssd moed mowd mtd ned nvd "
        "nhd njd nmd nyed nynd nysd nywd nced ncmd ncwd ndd ohnd ohsd oked oknd okwd "
        "ord paed pamd pawd rid scd sdd tned tnmd tnwd txed txnd txsd txwd utd vtd "
        "vaed vawd waed wawd wvnd wvsd wied wiwd wyd gud nmid prd vid californiad "
        "illinoised illinoisd indianad orld ohiod pennsylvaniad southcarolinaed "
        "southcarolinawd tennessed canalzoned"
    ),
    "us-sup": "scotus",
    "us-misc": (
        "bap1 bap2 bap6 bap8 bap9 bap10 ag afcca asbca armfor acca uscfc tax bia olc "
        "mc mspb nmcca cavc bva fiscr fisc cit usjc jpml cc com ccpa cusc bta eca "
        "tecoa reglrailreorgct kingsbench"
    ),
    "al": "almd alnd alsd almb alnb alsb ala alactapp alacrimapp alacivapp",
    "ak": "akd akb alaska alaskactapp",
    "az": "azd arb ariz arizctapp ariztaxct",
    "ar": "ared arwd areb arwb ark arkctapp arkworkcompcom arkag",
    "as": "amsamoa amsamoatc",
    "ca": (
        "cacd caed cand casd californiad cacb caeb canb casb cal calctapp "
        "calappdeptsuper calag"
    ),
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
    "ny": (
        "nyed nynd nysd nywd nyeb nynb nysb nywb ny nyappdiv nyappterm nysupct "
        "nycountyctny nydistct nyjustct nyfamct nysurct nycivct nycrimct nyag"
    ),
    "nc": "nced ncmd ncwd nceb ncmb ncwb nc ncctapp ncsuperct ncworkcompcom",
    "nd": "ndd ndb nd ndctapp",
    "mp": "nmariana cnmisuperct cnmitrialct",
    "oh": "ohnd ohsd ohiod ohnb ohsb ohio ohioctapp ohioctcl",
    "ok": (
        "oked oknd okwd okeb oknb okwb okla oklacivapp oklacrimapp oklajeap "
        "oklacoj oklaag"
    ),
    "or": "ord orb or orctapp ortc",
    "pa": "paed pamd pawd pennsylvaniad paeb pamb pawb pa pasuperct pacommwct cjdpa",
    "pr": "prsupreme prapp prd prb",
    "ri": "rid rib ri risuperct",
    "sc": "scd southcarolinaed southcarolinawd scb sc scctapp",
    "sd": "sdd sdb sd",
    "tn": (
        "tned tnmd tnwd tennessed tneb tnmb tnwb tennesseeb tenn tennctapp "
        "tenncrimapp tennworkcompcl tennworkcompapp tennsuperct"
    ),
    "tx": (
        "txed txnd txsd txwd txeb txnb txsb txwb tex texapp texcrimapp "
        "texreview texjpml texag"
    ),
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
    http_code_ok = 200
    search_url = base_url + "/search/?q="
    max_retries = 5
    retry_delay = 1 # second
    for _ in range(max_retries):
        try:
            response = requests.get(
                search_url + q,
                headers=courtlistener_header,
                timeout=courtlistener_timeout,
            )
            if response.status_code == http_code_ok:
                return response.json()
        except requests.exceptions.ReadTimeout:  # noqa: PERF203
            time.sleep(retry_delay)
    return {}

def get_target(target_id: int, target_name: str) -> dict:
    """Get a courtlistener target object by id.

    Parameters
    ----------
    target_id : int
        The target object's id
    target_name : str
        The target object's name. Must be one of:
        "opinion", "cluster", "docket", "court", "people"

    Returns
    -------
    dict
        The target object with the matching id

    """
    url = None
    if target_name == "opinion":
        url = base_url + "/opinions/?id="
    elif target_name == "cluster":
        url = base_url + "/clusters/?id="
    elif target_name == "docket":
        url = base_url + "/dockets/?id="
    elif target_name == "court":
        url = base_url + "/courts/?id="
    elif target_name == "people":
        url = base_url + "/people/?id="
    if url is None:
        return {}
    response = requests.get(
        url + str(target_id),
        headers=courtlistener_header,
        timeout=courtlistener_timeout,
    )
    return response.json()["results"][0]

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
    # lookup the opinion
    op = get_target(result["id"], "opinion")
    # getting the text, in the best format possible
    op["text"] = op["xml_harvard"]
    # these are backup formats
    backups = ["html_with_citations", "html", "plain_text", "html_lawbox", "html_columbia"]
    b_index = 0
    while op["text"] == "" and b_index < len(backups):
        op["text"] = op[backups[b_index]]
        b_index += 1
    return op

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
    cluster = get_target(result["cluster_id"], "cluster")
    # prefer short name to full name
    if cluster["case_name"]:
        return cluster["case_name"]
    if cluster["case_name_short"]:
        return cluster["case_name_short"]
    return cluster["case_name_full"]

def get_author_name(result: dict) -> str:
    # person level data: author name
    author = get_target(result["author_id"], "people")
    full_name = ""
    if author["name_first"]:
        full_name += author["name_first"]
    if author["name_middle"]:
        full_name += (" " if full_name else "") + author["name_middle"]
    if author["name_last"]:
        full_name += (" " if full_name else "") + author["name_last"]
    return full_name

def upload_search_result(result: dict) -> None:
    # check if the opinion is already in the collection
    expr = f"metadata['id']=={result['id']}"
    hits = get_expr(courtlistener_collection, expr)
    if hits["result"] and len(hits["result"]) > 0:
        return
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
    valid_jurisdics = []
    if jurisdictions:
        # look up each str in dictionary, append matches as lists
        for juris in jurisdictions:
            if juris.lower() in jurisdiction_codes:
                valid_jurisdics += jurisdiction_codes[juris.lower()].split(" ")
        # clear duplicate federal district jurisdictions if they exist
        valid_jurisdics = list(set(valid_jurisdics))
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
        expr += (" and " if expr else "")
        keyword_expr = f"text like '% {keyword_query} %'"
        expr += keyword_expr
    return query(courtlistener_collection, q, k, expr)


def batch_metadata_files() -> None:
    import ast
    import bz2
    import json
    import pathlib

    import pandas as pd
    from openai import OpenAI

    from app.milvusdb import query_iterator

    def get_data_dictionary(data_filename: str, id_filename: str, chunksize: int, row_fn, people_filename: str | None = None, court_filename: str | None = None) -> dict:
        with pathlib.Path(id_filename).open("r") as f:
            matching_ids = ast.literal_eval(f.read())
        total_rows = 0
        data = {}
        if people_filename is not None:
            # opinion data
            people_df = None # pd.read_csv(people_filename)
            court_df = None
            cols = ["id", "type"]
        elif court_filename is not None:
            # docket data
            people_df = None
            court_df = pd.read_csv(court_filename)
            cols = ["id", "court_id"]
        else:
            # cluster data
            people_df, court_df = None, None
            cols = ["id", "slug"]

        with bz2.open(data_filename, "rt") as bz_file:
            for chunk in pd.read_csv(bz_file, chunksize=chunksize, low_memory=False, usecols=cols):
                total_rows += len(chunk)
                matches = chunk[chunk["id"].isin(matching_ids)]
                for _, row in matches.iterrows():
                    row_fn(row, data, people_df, court_df)

                if total_rows % (10 * chunksize) == 0:
                    print(f"Processed {total_rows} rows in {data_filename}...")
        return data

    def opinion_row_data(row, data, people_df, court_df) -> None:
        data[row["id"]] = {"id": row["id"], "type": row["type"]}
        # if pd.notna(row["author_str"]):
        #     data[row["id"]]["author_name"] = row["author_str"]
        # if pd.notna(row["author_id"]):
        #     # look up author_name in people db
        #     data[row["id"]]["author_id"] = row["author_id"]
        #     if "author_name" not in data[row["id"]]:
        #         author = people_df[people_df["id"] == row["author_id"]].iloc[0]
        #         author_name = author["name_first"]
        #         if pd.notna(author["name_middle"]):
        #             author_name += " " + author["name_middle"]
        #         author_name += " " + author["name_last"]
        #         if pd.notna(author["name_suffix"]):
        #             author_name += " " + author["name_suffix"]
        #         data[row["id"]]["author_name"] = author_name
        # if pd.notna(row["download_url"]):
        #     data[row["id"]]["download_url"] = row["download_url"]

    def cluster_row_data(row, data, people_df, court_df) -> None:
        data[row["id"]] = {"id": row["id"]}
        # if pd.notna(row["case_name"]):
        #     data[row["id"]]["case_name"] = row["case_name"]
        # elif pd.notna(row["case_name_short"]):
        #     data[row["id"]]["case_name"] = row["case_name_short"]
        # else:
        #     data[row["id"]]["case_name"] = row["case_name_full"]
        # if pd.notna(row["summary"]):
        #     data[row["id"]]["summary"] = row["summary"]
        # if pd.notna(row["precedential_status"]):
        #     data[row["id"]]["precedential_status"] = row["precedential_status"]
        # if pd.notna(row["other_dates"]):
        #     data[row["id"]]["other_dates"] = row["other_dates"]
        # if pd.notna(row["date_blocked"]):
        #     data[row["id"]]["date_blocked"] = row["date_blocked"]
        if pd.notna(row["slug"]):
            data[row["id"]]["slug"] = row["slug"]

    def docket_row_data(row, data, people_df, court_df) -> None:
        data[row["id"]] = {"court_id": row["court_id"]}
        # look up court_name in courts db
        court = court_df[court_df["id"] == row["court_id"]].iloc[0]
        data[row["id"]]["court_name"] = court["full_name"]

    basedir = "/Users/njc/Documents/programming/opb/data/courtlistener_bulk/"
    opinion_filename = basedir + "opinions-2024-05-06.csv.bz2"
    cluster_filename = basedir + "opinion-clusters-2024-05-06.csv.bz2"
    docket_filename = basedir + "dockets-2024-05-06.csv.bz2"
    people_filename = basedir + "people-db-people-2024-05-06.csv"
    court_filename = basedir + "courts-2024-05-06.csv.bz2"
    opinion_ids_filename = basedir + "opinion_ids"
    cluster_ids_filename = basedir + "cluster_ids"
    docket_ids_filename = basedir + "docket_ids"
    opinion_data_filename = basedir + "opinion_data"
    cluster_data_filename = basedir + "cluster_data"
    docket_data_filename = basedir + "docket_data"


    # docket_data = get_data_dictionary(docket_filename, docket_ids_filename, 100000, docket_row_data, court_filename=court_filename)
    # with pathlib.Path(docket_data_filename).open("w") as f:
    #     f.write(str(docket_data))
    # cluster_data = get_data_dictionary(cluster_filename, cluster_ids_filename, 100000, cluster_row_data)
    # with pathlib.Path(cluster_data_filename).open("w") as f:
    #     f.write(str(cluster_data))
    # opinion_data = get_data_dictionary(opinion_filename, opinion_ids_filename, 100000, opinion_row_data, people_filename=people_filename)
    # with pathlib.Path(opinion_data_filename).open("w") as f:
    #     f.write(str(opinion_data))

    with pathlib.Path(docket_data_filename).open("r") as f:
        docket_data = ast.literal_eval(f.read())
    with pathlib.Path(cluster_data_filename).open("r") as f:
        cluster_data = ast.literal_eval(f.read())
    with pathlib.Path(opinion_data_filename).open("r") as f:
        opinion_data = ast.literal_eval(f.read())

    q_iter = query_iterator("courtlistener", "", ["metadata"], 1000)
    res = q_iter.next()
    while len(res) > 0:
        for hit in res:
            if "ai_summary" in hit["metadata"] and hit["metadata"]["id"] in opinion_data:
                opinion_data[hit["metadata"]["id"]]["ai_summary"] = hit["metadata"]["ai_summary"]
        res = q_iter.next()
    q_iter.close()

    client = OpenAI()
    openai_files = client.files.list()
    batches = client.batches.list()
    upload_counter = 0
    for batch in batches.data:
        metadatas, texts, vectors = [], [], []
        if batch.status != "completed":
            continue

        input_file = next(
            (f for f in openai_files if batch.input_file_id == f.id),
            None,
        )
        if input_file is None:
            print("input file not found for " + batch.input_file_id)
            continue

        input_filename = input_file.filename
        print(input_filename)

        result_file_id = batch.output_file_id
        result_file_name = input_filename.split(".")[0] + "_out.jsonl"
        if not pathlib.Path(basedir + result_file_name).exists():
            result = client.files.content(result_file_id).content
            with pathlib.Path(basedir + result_file_name).open("wb") as f:
                f.write(result)
        with pathlib.Path(basedir + result_file_name).open("r") as f:
            for i, line in enumerate(f, start=1):
                output = json.loads(line)
                with pathlib.Path(basedir + input_filename).open("r") as in_f:
                    input_line = in_f.readline()
                    input_data = json.loads(input_line)
                    while input_line and input_data["custom_id"] != output["custom_id"]:
                        input_line = in_f.readline()
                        input_data = json.loads(input_line)
                custom_id_split = output["custom_id"].split("-")
                cluster_id = int(custom_id_split[0])
                opinion_id = int(custom_id_split[1])
                metadata = {}
                metadata.update(cluster_data[cluster_id])
                metadata.update(opinion_data[opinion_id])
                metadata.update(docket_data[metadata["docket_id"]])
                text = input_data["body"]["input"]
                vector = output["response"]["body"]["data"][0]["embedding"]
                metadatas.append(metadata)
                texts.append(text)
                vectors.append(vector)
                if i % 5000 == 0:
                    print(f"i = {i}")
                if len(metadatas) == 1000:
                    upload_result = upload_data_json("courtlistener_bulk", vectors, texts, metadatas)
                    if upload_result["message"] != "Success":
                        print(f"error: bad upload for batch {batch.id}")
                        continue
                    upload_counter += 1
                    if upload_counter % 10 == 0:
                        print("uploaded 10 batches")
                    metadatas, texts, vectors = [], [], []
        # upload the last <1000 lines
        if len(metadatas) > 0:
            upload_result = upload_data_json("courtlistener_bulk", vectors, texts, metadatas)
            if upload_result["message"] != "Success":
                print(f"error: bad upload for batch {batch.id}")
                continue
            upload_counter += 1
            if upload_counter % 10 == 0:
                print("uploaded 10 batches")




def process_batches() -> None:
    import ast
    import bz2
    import json
    import pathlib

    import pandas as pd
    from openai import OpenAI

    from app.milvusdb import query_iterator

    def get_data_dictionary(data_filename: str, id_filename: str, chunksize: int, row_fn, people_filename: str | None = None, court_filename: str | None = None) -> dict:
        with pathlib.Path(id_filename).open("r") as f:
            matching_ids = ast.literal_eval(f.read())
        total_rows = 0
        data = {}
        if people_filename is not None:
            # opinion data
            people_df = None # pd.read_csv(people_filename)
            court_df = None
            cols = ["id", "type"]
        elif court_filename is not None:
            # docket data
            people_df = None
            court_df = pd.read_csv(court_filename)
            cols = ["id", "court_id"]
        else:
            # cluster data
            people_df, court_df = None, None
            cols = ["id", "slug"]

        with bz2.open(data_filename, "rt") as bz_file:
            for chunk in pd.read_csv(bz_file, chunksize=chunksize, low_memory=False, usecols=cols):
                total_rows += len(chunk)
                matches = chunk[chunk["id"].isin(matching_ids)]
                for _, row in matches.iterrows():
                    row_fn(row, data, people_df, court_df)

                if total_rows % (10 * chunksize) == 0:
                    print(f"Processed {total_rows} rows in {data_filename}...")
        return data

    def opinion_row_data(row, data, people_df, court_df) -> None:
        data[row["id"]] = {"id": row["id"], "type": row["type"]}
        # if pd.notna(row["author_str"]):
        #     data[row["id"]]["author_name"] = row["author_str"]
        # if pd.notna(row["author_id"]):
        #     # look up author_name in people db
        #     data[row["id"]]["author_id"] = row["author_id"]
        #     if "author_name" not in data[row["id"]]:
        #         author = people_df[people_df["id"] == row["author_id"]].iloc[0]
        #         author_name = author["name_first"]
        #         if pd.notna(author["name_middle"]):
        #             author_name += " " + author["name_middle"]
        #         author_name += " " + author["name_last"]
        #         if pd.notna(author["name_suffix"]):
        #             author_name += " " + author["name_suffix"]
        #         data[row["id"]]["author_name"] = author_name
        # if pd.notna(row["download_url"]):
        #     data[row["id"]]["download_url"] = row["download_url"]

    def cluster_row_data(row, data, people_df, court_df) -> None:
        data[row["id"]] = {"id": row["id"]}
        # if pd.notna(row["case_name"]):
        #     data[row["id"]]["case_name"] = row["case_name"]
        # elif pd.notna(row["case_name_short"]):
        #     data[row["id"]]["case_name"] = row["case_name_short"]
        # else:
        #     data[row["id"]]["case_name"] = row["case_name_full"]
        # if pd.notna(row["summary"]):
        #     data[row["id"]]["summary"] = row["summary"]
        # if pd.notna(row["precedential_status"]):
        #     data[row["id"]]["precedential_status"] = row["precedential_status"]
        # if pd.notna(row["other_dates"]):
        #     data[row["id"]]["other_dates"] = row["other_dates"]
        # if pd.notna(row["date_blocked"]):
        #     data[row["id"]]["date_blocked"] = row["date_blocked"]
        if pd.notna(row["slug"]):
            data[row["id"]]["slug"] = row["slug"]

    def docket_row_data(row, data, people_df, court_df) -> None:
        data[row["id"]] = {"court_id": row["court_id"]}
        # look up court_name in courts db
        court = court_df[court_df["id"] == row["court_id"]].iloc[0]
        data[row["id"]]["court_name"] = court["full_name"]

    basedir = "/Users/njc/Documents/programming/opb/data/courtlistener_bulk/"
    opinion_filename = basedir + "opinions-2024-05-06.csv.bz2"
    cluster_filename = basedir + "opinion-clusters-2024-05-06.csv.bz2"
    docket_filename = basedir + "dockets-2024-05-06.csv.bz2"
    people_filename = basedir + "people-db-people-2024-05-06.csv"
    court_filename = basedir + "courts-2024-05-06.csv.bz2"
    opinion_ids_filename = basedir + "opinion_ids"
    cluster_ids_filename = basedir + "cluster_ids"
    docket_ids_filename = basedir + "docket_ids"
    opinion_data_filename = basedir + "opinion_data"
    cluster_data_filename = basedir + "cluster_data"
    docket_data_filename = basedir + "docket_data"


    # docket_data = get_data_dictionary(docket_filename, docket_ids_filename, 100000, docket_row_data, court_filename=court_filename)
    # with pathlib.Path(docket_data_filename).open("w") as f:
    #     f.write(str(docket_data))
    # cluster_data = get_data_dictionary(cluster_filename, cluster_ids_filename, 100000, cluster_row_data)
    # with pathlib.Path(cluster_data_filename).open("w") as f:
    #     f.write(str(cluster_data))
    # opinion_data = get_data_dictionary(opinion_filename, opinion_ids_filename, 100000, opinion_row_data, people_filename=people_filename)
    # with pathlib.Path(opinion_data_filename).open("w") as f:
    #     f.write(str(opinion_data))

    with pathlib.Path(docket_data_filename).open("r") as f:
        docket_data = ast.literal_eval(f.read())
    with pathlib.Path(cluster_data_filename).open("r") as f:
        cluster_data = ast.literal_eval(f.read())
    with pathlib.Path(opinion_data_filename).open("r") as f:
        opinion_data = ast.literal_eval(f.read())

    q_iter = query_iterator("courtlistener", "", ["metadata"], 1000)
    res = q_iter.next()
    while len(res) > 0:
        for hit in res:
            if "ai_summary" in hit["metadata"] and hit["metadata"]["id"] in opinion_data:
                opinion_data[hit["metadata"]["id"]]["ai_summary"] = hit["metadata"]["ai_summary"]
        res = q_iter.next()
    q_iter.close()

    client = OpenAI()
    openai_files = client.files.list()
    batches = client.batches.list()
    upload_counter = 0
    for batch in batches.data:
        metadatas, texts, vectors = [], [], []
        if batch.status != "completed":
            continue

        input_file = next(
            (f for f in openai_files if batch.input_file_id == f.id),
            None,
        )
        if input_file is None:
            print("input file not found for " + batch.input_file_id)
            continue

        input_filename = input_file.filename
        print(input_filename)

        result_file_id = batch.output_file_id
        result_file_name = input_filename.split(".")[0] + "_out.jsonl"
        if not pathlib.Path(basedir + result_file_name).exists():
            result = client.files.content(result_file_id).content
            with pathlib.Path(basedir + result_file_name).open("wb") as f:
                f.write(result)
        with pathlib.Path(basedir + result_file_name).open("r") as f:
            for i, line in enumerate(f, start=1):
                output = json.loads(line)
                with pathlib.Path(basedir + input_filename).open("r") as in_f:
                    input_line = in_f.readline()
                    input_data = json.loads(input_line)
                    while input_line and input_data["custom_id"] != output["custom_id"]:
                        input_line = in_f.readline()
                        input_data = json.loads(input_line)
                custom_id_split = output["custom_id"].split("-")
                cluster_id = int(custom_id_split[0])
                opinion_id = int(custom_id_split[1])
                metadata = {}
                metadata.update(cluster_data[cluster_id])
                metadata.update(opinion_data[opinion_id])
                metadata.update(docket_data[metadata["docket_id"]])
                text = input_data["body"]["input"]
                vector = output["response"]["body"]["data"][0]["embedding"]
                metadatas.append(metadata)
                texts.append(text)
                vectors.append(vector)
                if i % 5000 == 0:
                    print(f"i = {i}")
                if len(metadatas) == 1000:
                    upload_result = upload_data_json("courtlistener_bulk", vectors, texts, metadatas)
                    if upload_result["message"] != "Success":
                        print(f"error: bad upload for batch {batch.id}")
                        continue
                    upload_counter += 1
                    if upload_counter % 10 == 0:
                        print("uploaded 10 batches")
                    metadatas, texts, vectors = [], [], []
        # upload the last <1000 lines
        if len(metadatas) > 0:
            upload_result = upload_data_json("courtlistener_bulk", vectors, texts, metadatas)
            if upload_result["message"] != "Success":
                print(f"error: bad upload for batch {batch.id}")
                continue
            upload_counter += 1
            if upload_counter % 10 == 0:
                print("uploaded 10 batches")
