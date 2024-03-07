from email.mime import base
import re
import requests
from unstructured.partition.auto import partition


courtlistener_token = "fb5c522b46cece6589209b97c395a3bc75623039"
courtlistener_header = {"Authorization": "Token " + courtlistener_token}
base_url = "https://www.courtlistener.com"
search_url = base_url + "/api/rest/v3/search/?q="
opinion_url = base_url + "/api/rest/v3/opinions/?id="
cluster_url = base_url + "/api/rest/v3/clusters/?id="
docket_url = base_url + "/api/rest/v3/dockets/?id="
    
def search(query):
    response = requests.get(search_url + query, headers=courtlistener_header)
    return response.json()

def get_opinion(result):
    id = result['absolute_url'].split("/")[2] #grabbing the opinion id from the abs url
    response = requests.get(opinion_url + id, headers=courtlistener_header)

    op = response.json()["results"][0] #the actual opinion

    op["text"] = op["html_with_citations"]
    backups = ["html", "plain_text", "html_lawbox", "html_columbia"]
    b_index = 0
    while(op["text"] == "" and b_index < len(backups)):
        op["text"] = op[backups[b_index]]
        b_index += 1
    return op

def get_cluster(result):
    id = str(result["cluster_id"])
    response = requests.get(cluster_url + id, headers=courtlistener_header)

    return(response.json()["results"][0])

def get_docket(result):
    id = str(result["docket_id"])
    response = requests.get(docket_url + id, headers=courtlistener_header)

    return(response.json()["results"][0])

# def parse_opinion(site):
#     try:
#         elements = partition(url=site)
#     except:
#         elements = partition(url=site, content_type="text/html")
#     e_text = ""
#     for el in elements:
#         el = str(el)
#         e_text += el + "\n\n"
#     return e_text

def courtlistener_search(query):
    for result in [search(query)["results"][0]]:
        print(result)
        print("-")
        oo = get_opinion(result)
        cc = get_cluster(result)
        dd = get_docket(result)
        print(oo)
        print("^^ opinion")
        print(cc)
        print("^^ cluster")
        print(dd)
        print("^^ docket")
        print("----")

courtlistener_search("marijuana")