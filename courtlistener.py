"""A module for interacting with the CourtListener API. Written by Arman Aydemir."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import requests
from langchain.agents import Tool
from langfuse.decorators import observe

from milvusdb import query, upload_courtlistener

if TYPE_CHECKING:
    from models import SearchTool

courtlistener_token = os.environ["COURTLISTENER_API_KEY"]
courtlistener_header = {"Authorization": "Token " + courtlistener_token}
base_url = "https://www.courtlistener.com"
search_url = base_url + "/api/rest/v3/search/?q="
opinion_url = base_url + "/api/rest/v3/opinions/?id="
cluster_url = base_url + "/api/rest/v3/clusters/?id="
docket_url = base_url + "/api/rest/v3/dockets/?id="
court_url = base_url + "/api/rest/v3/courts/?id="

courtlistener_collection = "courtlistener"

courtlistener_timeout = 30 #seconds

courtlistener_tool_args = {
    "jurisdiction": {
        "type": "string",
        "description": (
            "The two-letter abbreviation of a state or territory, e.g. 'NJ' or 'TX', "
            "to filter query results by jurisdiction. Use 'US' for federal courts."
        ),
    },
    "from-date": {
        "type": "string",
        "description": (
            "The start date for the query date range in YYYY-MM-DD "
            "format."
        ),
    },
    "to-date": {
        "type": "string",
        "description": (
            "The end date for the query date range in YYYY-MM-DD "
            "format."
        ),
    },
}

# https://github.com/freelawproject/courtlistener/discussions/3114
# manual mapping from two-letter state abbreviations to court_id affixes
# for some ambiguous states
jurisdiction_codes = {
    "us": "scotus ca1 ca2 ca3 ca4 ca5 ca6 ca7 ca8 ca9 ca10 ca11 cadc cafc bap1 bap2 bap6 bap8 bap9 bap10 ag afcca asbca armfor acca uscfc tax bia olc mc mspb nmcca cavc bva fiscr fisc cit usjc jpml cc com ccpa cusc bta eca tecoa reglrailreorgct kingsbench",
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
    response = requests.get(search_url + q,
                            headers=courtlistener_header,
                            timeout=courtlistener_timeout)
    return response.json()

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

# TODO: Need to parallelize this
@observe()
def courtlistener_search(
    q: str,
    k: int = 3,
    jurisdiction: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> dict:
    """Search courtlistener for a query.

    Search, get opinion text, upload the opinion data to milvus, and query it.

    Parameters
    ----------
    q : str
        The query
    k : int, optional
        The number of results to return, by default 3
    jurisdiction : str | None, optional
        The two-letter abbreviation of a state or territory, e.g. 'NJ' or 'TX',
        to filter query results by state. Use 'US' for federal courts. By default None.
    from_date : str | None, optional
        The start date for the query date range in YYYY-MM-DD format, by default None
    to_date : str | None, optional
        The end date for the query date range in YYYY-MM-DD format, by default None

    Returns
    -------
    dict
        the response with relevant info from courtlistener

    """
    query_str = q
    # add options to query string
    if jurisdiction and jurisdiction in jurisdiction_codes:
        query_str += f"&court={jurisdiction_codes[jurisdiction]}"
    if from_date:
        dt = from_date.split("-")
        # needs to be in MM-DD-YYYY format
        query_str += f"&filed_after={dt[1]}-{dt[2]}-{dt[0]}"
    if to_date:
        dt = to_date.split("-")
        # needs to be in MM-DD-YYYY format
        query_str += f"&filed_before={dt[1]}-{dt[2]}-{dt[0]}"

    for result in search(query_str)["results"][:k]:
        opinion = get_opinion(result)
        opinion["cluster_id"] = result["cluster_id"]
        opinion["court_id"] = result["court_id"]
        opinion["date_filed"] = result["dateFiled"].split("T")[0]
        opinion["docket_id"] = result["docket_id"]
        upload_courtlistener(courtlistener_collection, opinion)

    return courtlistener_query(q, k, jurisdiction, from_date, to_date)

def courtlistener_query(
    q: str,
    k: int,
    jurisdiction: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> dict:
    """Query Courtlistener data.

    Parameters
    ----------
    q : str
        The query text
    k : int
        How many chunks to return
    jurisdiction : str | None, optional
        The two-letter abbreviation of a state or territory, e.g. 'NJ' or 'TX',
        to filter query results by state. Use 'US' for federal courts. By default None.
    from_date : str | None, optional
        The start date for the query date range in YYYY-MM-DD format, by default None
    to_date : str | None, optional
        The end date for the query date range in YYYY-MM-DD format, by default None

    Returns
    -------
    dict
        Contains `message`, `result` list if successful

    """
    expr = ""
    if jurisdiction and jurisdiction in jurisdiction_codes:
        code_list = jurisdiction_codes[jurisdiction].split(" ")
        expr = f"metadata['court_id'] in {code_list}"
    if from_date:
        if expr:
            expr += " and "
        expr += f"metadata['date_filed']>='{from_date}'"
    if to_date:
        if expr:
            expr += " and "
        expr += f"metadata['date_filed']<='{to_date}'"
    return query(courtlistener_collection, q, k, expr)

def courtlistener_tool_creator(t: SearchTool) -> Tool:
    """Create the courtlistener tool for agents to call.

    Parameters
    ----------
    t : SearchTool
        The search tool definition (we only use the name and prompt for now)

    Returns
    -------
    Tool
        The Tool object for the courtlistener search

    """
    def query_tool(q: str,
        jurisdiction: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict:
        return courtlistener_search(q, 3, jurisdiction, from_date, to_date)

    async def async_query_tool(q: str,
        jurisdiction: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict:
        return courtlistener_search(q, 3, jurisdiction, from_date, to_date)

    name = t.name
    prompt = t.prompt

    tool_func = lambda q: query_tool(q)
    co_func = lambda q: async_query_tool(q)
    return Tool(
        name=name,
        func=tool_func,
        coroutine=co_func,
        description=prompt,
    )
