"""A module for interacting with the CourtListener API. Written by Arman Aydemir."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import requests
import urllib3
from langfuse.decorators import langfuse_context, observe

from app.milvusdb import get_expr, query

if TYPE_CHECKING:
    from app.models import OpinionSearchRequest

courtlistener_token = os.environ["COURTLISTENER_API_KEY"]
courtlistener_header = {"Authorization": "Token " + courtlistener_token}
base_url = "https://www.courtlistener.com/api/rest/v3"
courtlistener_timeout = 10
courtlistener_collection = "courtlistener"
courtlistener_tool_args = {
    "jurisdictions": {
        "type": "array",
        "items": {
            "type": "string",
        },
        "description": (
            "The two-letter abbreviations of a state or territory, e.g. 'NJ' or 'TX', "
            "to filter query results by jurisdiction. Use 'US' for federal courts and "
            "'UK' for the United Kingdom. Leave empty to search all jurisdictions."
        ),
    },
    "keyword-qr": {
        "type": "string",
        "description": (
            "The keyword query to search for an exact name or term within the opinion "
            "text. Only results containing the entire keyword query as a substring "
            "will be returned, so it should be used precisely."
        ),
    },
    "after-date": {
        "type": "string",
        "description": (
            "The exclusive after date for the query date range in YYYY-MM-DD "
            "format. For example, if asked for cases since 2000, enter 1999-12-31."
        ),
    },
    "before-date": {
        "type": "string",
        "description": (
            "The exclusive before date for the query date range in YYYY-MM-DD "
            "format."
        ),
    },
}

us_federal_dict = {
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
}
# https://github.com/freelawproject/courtlistener/discussions/3114
# manual mapping from two-letter state abbreviations to courtlistener court_id format
jurisdiction_codes = us_federal_dict | {
    # gather all federal courts to a general us key
    "us": " ".join(us_federal_dict.values()),
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


@observe(capture_output=False)
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
    try:
        response = requests.get(
            search_url + q,
            headers=courtlistener_header,
            timeout=courtlistener_timeout,
        )
        if response.status_code == http_code_ok:
            return response.json()
    except (
        requests.exceptions.Timeout,
        urllib3.exceptions.ConnectTimeoutError,
    ) as timeout_err:
        print("Timeout error, skipping", timeout_err)
    return {}

@observe()
def get_opinion_ids(search_result: dict, num_cases: int = 5) -> list[int]:
    """Get opinion ids from search results."""
    return [
        result["id"]
        for result in search_result["results"]
        if "id" in result
    ][:num_cases]

@observe()
def courtlistener_get_opinion(opinion_id: int) -> dict:
    """Get the chunks in Milvus for a given opinion, if it's available."""
    expr = f"source_id=={opinion_id}"
    result = get_expr(courtlistener_collection, expr)
    if "result" in result and len(result["result"]) > 0:
        # the opinion is in the database
        metadata = result["result"][0]["metadata"]
        text = "\n\n".join([hit["text"] for hit in result["result"]])
        return {"text": text, "metadata": metadata}
    return {}

@observe(capture_output=False)
def courtlistener_query(request: OpinionSearchRequest) -> dict:
    """Query Courtlistener data.

    Parameters
    ----------
    request : OpinionSearchRequest
        The opinion search request object

    Returns
    -------
    dict
        result (Contains `message`, `result` list if successful)

    """
    expr = ""
    if request.keyword_query:
        expr += " and ".join([
            f"TEXT_MATCH(text, '{word}')"
            for word in request.keyword_query.split()
        ])
    if request.jurisdictions:
        valid_jurisdics = []
        # look up each str in dictionary, append matches as lists
        for juris in request.jurisdictions:
            if juris.lower() in jurisdiction_codes:
                valid_jurisdics += jurisdiction_codes[juris.lower()].split(" ")
        # clear duplicate federal district jurisdictions if they exist
        valid_jurisdics = list(set(valid_jurisdics))
        expr += (" and " if expr else "")
        expr += f"metadata['court_id'] in {valid_jurisdics}"
    if request.after_date:
        expr += (" and " if expr else "")
        expr += f"metadata['date_filed']>'{request.after_date}'"
    if request.before_date:
        expr += (" and " if expr else "")
        expr += f"metadata['date_filed']<'{request.before_date}'"
    result = query(courtlistener_collection, request.query, request.k, expr)
    if "result" in result:
        # tracing
        chunk_ids = [
            f"{hit['entity']['source_id']}-{hit['entity']['chunk_index']}"
            for hit in result["result"]
        ]
        langfuse_context.update_current_observation(output=chunk_ids)
    return result
