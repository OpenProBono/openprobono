#fastapi implementation of openprobono bot
from fastapi import FastAPI
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from pydantic import BaseModel
import uuid

cred = credentials.Certificate("../../creds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

root_path = "API"

def get_uuid_id():
    return str(uuid.uuid4())

def store_conversation(conversation, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session, api_key):
    (human, ai) = conversation[-1]
    data = {"human": human, "ai": ai, "t1name": t1name, 't1txt': t1txt, "t1prompt":t1prompt, "t2name": t2name, "t2txt":t2txt, "t2prompt":t2prompt, 'user_prompt': user_prompt, 'timestamp':  firestore.SERVER_TIMESTAMP, 'api_key': api_key}
    db.collection("API" + "conversations").document(session).collection('conversations').document("msg" + str(len(conversation))).set(data)

def create_bot(bot_id, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt):
    data = {"t1name": t1name, 't1txt': t1txt, "t1prompt":t1prompt, "t2name": t2name, "t2txt":t2txt, "t2prompt":t2prompt, 'user_prompt': user_prompt, 'timestamp':  firestore.SERVER_TIMESTAMP}
    db.collection("bots").document(bot_id).set(data)

def load_bot(bot_id):
    bot = db.collection("bots").document(bot_id).get()
    if(bot.exists):
        return bot.to_dict()
    else:
        return None
    

def openprobono_bot(history, 
    t1name = "government-search", 
    t1txt = "site:*.gov | site:*.edu | site:*scholar.google.com", 
    t1prompt = "Useful for when you need to answer questions or find resources about government and laws. Always cite your sources.", 
    t2name = "case-search", 
    t2txt = "site:*case.law | site:*.gov | site:*.edu | site:*courtlistener.com | site:*scholar.google.com", 
    t2prompt = "Use for finding case law. Always cite your sources.", 
    user_prompt = "", 
    session = ""):
    if(history[-1][0].strip() == ""):
        history[-1][1] = "Hi, how can I assist you today?"
        return history 
    else:
        history_langchain_format = ChatMessageHistory()
        for i in range(1, len(history)-1):
            (human, ai) = history[i]
            history_langchain_format.add_user_message(human)
            history_langchain_format.add_ai_message(ai)
        memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format, memory_key="memory")
        ##----------------------- tools -----------------------##

        def gov_search(q):
            data = {"search": t1txt + " " + q, 'prompt':t1prompt,'timestamp': firestore.SERVER_TIMESTAMP}
            db.collection(root_path + "search").document(session).collection('searches').document("search" + get_uuid_id()).set(data)
            return filtered_search(GoogleSearch({
                'q': t1txt + " " + q,
                'num': 5
                }).get_dict())

        def case_search(q):
            data = {"search": t2txt + " " + q, 'prompt': t2prompt, 'timestamp': firestore.SERVER_TIMESTAMP}
            db.collection(root_path + "search").document(session).collection('searches').document("search" + get_uuid_id()).set(data)
            return filtered_search(GoogleSearch({
                'q': t2txt + " " + q,
                'num': 5
                }).get_dict())

        async def async_gov_search(q):
            return gov_search(q)

        async def async_case_search(q):
            return case_search(q)

        #Filter search results retured by serpapi to only include relavant results
        def filtered_search(results):
            new_dict = {}
            if('sports_results' in results):
                new_dict['sports_results'] = results['sports_results']
            if('organic_results' in results):
                new_dict['organic_results'] = results['organic_results']
            return new_dict

        #Definition and descriptions of tools aviailable to the bot
        tools = [
            Tool(
                name=t1name,
                func=gov_search,
                coroutine=async_gov_search,
                description=t1prompt,
            ),
            Tool(
                name=t1name,
                func=case_search,
                coroutine=async_case_search,
                description=t2prompt,
            )
        ]
        ##----------------------- end of tools -----------------------##

        system_message = 'You are a helpful AI assistant. ALWAYS use tools to answer questions.'
        system_message += user_prompt
        system_message += '. If you used a tool, ALWAYS return a "SOURCES" part in your answer.'
        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        }
       
        #definition of llm used for bot
        prompt = "Using the tools at your disposal, answer the following question: " + prompt
        bot_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613', request_timeout=60*5)
        agent = initialize_agent(
            tools=tools,
            llm=bot_llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=False,
            agent_kwargs=agent_kwargs,
            memory=memory,
            #return_intermediate_steps=True
        )
        agent.agent.prompt.messages[0].content = system_message
        return agent.arun(prompt)


class BotRequest(BaseModel):
    history: list
    t1name: str = None
    t1txt: str = None
    t1prompt: str = None
    t2name: str = None
    t2txt: str = None
    t2prompt: str = None
    user_prompt: str = None
    session: str = None
    bot_id: str = None
    api_key: str = None

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "It's OpenProBono !!"}

@app.post("/bot")
def bot(request: BotRequest):
    request_dict = request.dict()
    history = request_dict['history']
    t1name = request_dict['t1name']
    t1txt = request_dict['t1txt']
    t1prompt = request_dict['t1prompt']
    t2name = request_dict['t2name']
    t2txt = request_dict['t2txt']
    t2prompt = request_dict['t2prompt']
    user_prompt = request_dict['user_prompt']
    session = request_dict['session']
    bot_id = request_dict['bot_id']
    api_key = request_dict['api_key']

    #if api key is valid (TODO: change this to a real api key)
    if(api_key == 'xyz'):
        #if bot_id is not provided, create a new bot id
        if bot_id is None or bot_id == "":
            bot_id = get_uuid_id()
            create_bot(bot_id, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt)
        #if bot_id is provided, load the bot
        else:
            bot = load_bot(bot_id)
            #if bot is not found, create a new bot
            if(bot is None):
                create_bot(bot_id, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt)
            else:
                t1name = bot['t1name']
                t1txt = bot['t1txt']
                t1prompt = bot['t1prompt']
                t2name = bot['t2name']
                t2txt = bot['t2txt']
                t2prompt = bot['t2prompt']
                user_prompt = bot['user_prompt']
        #get new response from ai
        chat = openprobono_bot(history, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session)
        #store conversation (log the api_key)
        store_conversation(chat, t1name, t1txt, t1prompt, t2name, t2txt, t2prompt, user_prompt, session, api_key)
        #return the chat and the bot_id
        return {"message": "Success", "chat": chat, "bot_id": bot_id}
    else:
        return {"message": "Invalid API Key"}


request = """
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"history":[["hi",""]],"api_key":"xyz"}' \
  http://35.232.62.221/bot
"""