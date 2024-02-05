import requests


def search_tool_creator(name, txt, prompt):
    def search_tool(qr, txt, prompt):
        data = {"search": txt + " " + qr, 'prompt': prompt, 'timestamp': firestore.SERVER_TIMESTAMP}
        params = {
            'key': 'AIzaSyDjhu4Wl0tIKphT92wAgw78zV2AFCd8c_M',
            'cx': '31cf2b4b6383f4f33',
            'q': txt + " " + qr,
        }
        return str(requests.get('https://www.googleapis.com/customsearch/v1', params=params, headers=headers).json())[0:6400]
    
    async def async_search_tool(qr, txt, prompt):
        return search_tool(qr, txt, prompt)

    tool_func = lambda qr: search_tool(qr, txt, prompt)
    co_func = lambda qr: async_search_tool(qr, txt, prompt)
    return Tool(
                name = name,
                func = tool_func,
                coroutine = co_func,
                description = prompt
            )

def search_toolset_creator(r):
    toolset = []
    for t in r.tools:
        toolset.append(search_tool_creator(t['name'], t['txt'], t['prompt']))
    return toolset

def serpapi_tool_creator(name, txt, prompt):
    #Filter search results retured by serpapi to only include relavant results
    def filtered_search(results):
        new_dict = {}
        if('sports_results' in results):
            new_dict['sports_results'] = results['sports_results']
        if('organic_results' in results):
            new_dict['organic_results'] = results['organic_results']
            return new_dict

    def search_tool(qr, txt, prompt):
        return filtered_search(GoogleSearch({
                    'q': t2txt + " " + q,
                    'num': 5
                    }).get_dict())
    
    async def async_search_tool(qr, txt, prompt):
        return search_tool(qr, txt, prompt)

    tool_func = lambda qr: search_tool(qr, txt, prompt)
    co_func = lambda qr: async_search_tool(qr, txt, prompt)
    return Tool(
                name = name,
                func = tool_func,
                coroutine = co_func,
                description = prompt
            )

def serpapi_toolset_creator(r):
    toolset = []
    for t in r.tools:
        toolset.append(serpapi_tool_creator(t['name'], t['txt'], t['prompt']))
    return toolset