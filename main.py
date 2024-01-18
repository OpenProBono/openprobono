#fastapi implementation of youtube bot
import uuid

import firebase_admin
import langchain
from typing import Annotated, List, Union, Tuple
from fastapi import Body, FastAPI
from firebase_admin import credentials, firestore
import gradio as gr
from langchain import PromptTemplate
from langchain.agents import (AgentExecutor, AgentOutputParser, AgentType,
                              LLMSingleActionAgent, Tool, ZeroShotAgent,
                              initialize_agent)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import \
    FinalStreamingStdOutCallbackHandler
from langchain.chains import LLMChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import (TextLoader, UnstructuredURLLoader,
                                        YoutubeLoader)
from langchain.document_loaders.blob_loaders.youtube_audio import \
    YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import (BaseChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain.schema import AgentAction, AgentFinish, AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from pydantic import BaseModel

from anyio.from_thread import start_blocking_portal
from queue import Queue
import re
from serpapi.google_search import GoogleSearch


cred = credentials.Certificate("../../creds.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
# opb bot db root path has api prefix
root_path = 'api_'
# manually set api key for now
GoogleSearch.SERP_API_KEY = 'e6e9a37144cdd3e3e40634f60ef69c1ea6e330dfa0d0cde58991aa2552fff980'

def get_uuid_id():
    return str(uuid.uuid4())

def store_conversation(conversation, user_prompt, youtube_urls, session, api_key):
    (human, ai) = conversation[-1]
    if(session is None or session == ""):
        session = get_uuid_id()
    data = {"human": human, "ai": ai, 'user_prompt': user_prompt, 'youtube_urls': youtube_urls, 'timestamp':  firestore.SERVER_TIMESTAMP, 'api_key': api_key}
    db.collection("API_youtube_" + "conversations").document(session).collection('conversations').document("msg" + str(len(conversation))).set(data)

def create_bot(bot_id, user_prompt, youtube_urls):
    data = {'user_prompt': user_prompt, 'youtube_urls': youtube_urls, 'timestamp':  firestore.SERVER_TIMESTAMP}
    db.collection("youtube_bots").document(bot_id).set(data)

def load_bot(bot_id):
    bot = db.collection("youtube_bots").document(bot_id).get()
    if(bot.exists):
        return bot.to_dict()
    else:
        return None

#TODO: cache vector db with bot_id
#TODO: do actual chat memory
#TODO: try cutting off intro and outro part of videos
def call_agent(
    history,
    bot_id,
    user_prompt = "",
    youtube_urls = [],
    session = ""):

    if(user_prompt is None or user_prompt == ""):
        user_prompt = "Respond in the same style as the youtuber in the context below."

    prompt_template = user_prompt + """
    \n\nContext: {context}
    \n\n\n\n
    Question: {question}
    Response:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}

    embeddings = OpenAIEmbeddings()
    bot_path = "./youtube_bots/" + bot_id
    try:
        vectordb = FAISS.load_local(bot_path, embeddings)
    except:
        text = ""
        for url in youtube_urls:
            try:
                # Load the audio
                loader = YoutubeLoader.from_youtube_url(
                    url, add_video_info=False
                )
                docs = loader.load()
                # Combine doc
                combined_docs = [doc.page_content for doc in docs]
                text += " ".join(combined_docs)
            except:
                print("Error occured while loading transcript from video with url: " + url)

        # Split them
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        splits = text_splitter.split_text(text)

        # Build an index
        vectordb = FAISS.from_texts(splits, embeddings)
        vectordb.save_local(bot_path)

    # Build a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
    )

    query = history[-1][0]
    #check for empty query
    if(query.strip() == ""):
        history[-1][1] = ""
    else:
        history[-1][1] = qa_chain.run(query)

    return history

def process(history, user_prompt, youtube_urls, session, bot_id, api_key):
    #if api key is valid (TODO: change this to a real api key check)
    if(api_key == 'xyz' or api_key == 'gradio'):
        try:
            warn = ""
            #if bot_id is not provided, create a new bot id
            if bot_id is None or bot_id == "":
                bot_id = get_uuid_id()
                create_bot(bot_id, user_prompt, youtube_urls)
            #if bot_id is provided, load the bot
            else:
                bot = load_bot(bot_id)
                #if bot is not found, create a new bot
                if(bot is None):
                    return {"message": "Failure: No bot found with bot id: " + bot_id}
                #else load bot settings
                else:
                    #if user_prompt or youtube_urls are provided, warn user that they are being ignored
                    if(user_prompt is not None and user_prompt != ""):
                        warn +=  " Warning: user_prompt is ignored because bot_id is provided\n"
                    if(youtube_urls is not None and youtube_urls != []):
                        warn +=  " Warning: youtube_urls is ignored because bot_id is provided\n"
                    user_prompt = bot['user_prompt']
                    youtube_urls = bot['youtube_urls']
            #get new response from ai
            chat = call_agent(history, bot_id, user_prompt, youtube_urls, session)
            #store conversation (log the api_key)
            store_conversation(chat, user_prompt, youtube_urls, session, api_key)
            #return the chat and the bot_id
            return {"message": "Success" + warn, "chat": chat, "bot_id": bot_id}
        except:
            return {"message": "Failure: Internal Error"}
    else:
        return {"message": "Invalid API Key"}
    
# OPB bot main function
def openai_bot(history, session):
    t1name = 'government-search'
    t1txt = 'site:*.gov | site:*.edu | site:*scholar.google.com'
    t1prompt = 'Useful for when you need to answer questions or find resources about government and laws. Always cite your sources.'
    t2name = 'case-search'
    t2txt = 'site:*case.law | site:*.gov | site:*.edu | site:*courtlistener.com | site:*scholar.google.com'
    t2prompt = 'Use for finding case law. Always cite your sources.'
    user_prompt = ''

    if(history[-1][0].strip() == ""):
        history[-1][1] = "Hi, how can I assist you today?"
        yield history 
    else:
        q = Queue()
        job_done = object()

        history_langchain_format = ChatMessageHistory()
        for i in range(1, len(history)-1):
            (human, ai) = history[i]
            if human:
                history_langchain_format.add_user_message(human)
            if ai:
                history_langchain_format.add_ai_message(ai)
        memory = ConversationBufferMemory(return_messages=True, chat_memory=history_langchain_format, memory_key="memory")
        ##----------------------- tools -----------------------##
        def gov_search(q):
            data = {"search": t1txt + " " + q, 'prompt': t1prompt,'timestamp': firestore.SERVER_TIMESTAMP}
            db.collection(root_path + "search").document(session).collection('searches').document("search" + get_uuid_id()).set(data)
            return process_search(GoogleSearch({
                'q': t1txt + " " + q,
                'num': 5
                }).get_dict())

        def case_search(q):
            data = {"search": t2txt + " " + q, 'prompt': t2prompt, 'timestamp': firestore.SERVER_TIMESTAMP}
            db.collection(root_path + "search").document(session).collection('searches').document("search" + get_uuid_id()).set(data)
            return process_search(GoogleSearch({
                'q': t2txt + " " + q,
                'num': 5
                }).get_dict())

        async def async_gov_search(q):
            return gov_search(q)

        async def async_case_search(q):
            return case_search(q)
        
        #Helper function for concurrent processing of search results, calls the summarizer llm
        def search_helper_summarizer(result):
            result.pop("displayed_link", None)
            result.pop("favicon", None)
            result.pop("about_page_link", None)
            result.pop("about_page_serpapi_link", None)
            result.pop("cached_page_link", None)
            result.pop("snippet_highlighted_words", None)

            summary_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-16k-0613')
            llm_input = """Summarize this web page in less than 50 words.

            Web Page:
            """
            llm_input += str(UnstructuredURLLoader(urls=[result["link"]]).load())[:16385]
            result["page_summary"] = summary_llm.predict(llm_input)
            return result

        #Filter search results retured by serpapi to only include relavant results
        def process_search(results):
            new_dict = {}
            # if('sports_results' in results):
            #     new_dict['sports_results'] = results['sports_results']
            if('organic_results' in results):
                new_dict['organic_results'] = [search_helper_summarizer(result) for result in results['organic_results']]

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
                name=t2name,
                func=case_search,
                coroutine=async_case_search,
                description=t2prompt,
            )
        ]
        tool_names = [tool.name for tool in tools]
        ##----------------------- end of tools -----------------------##
        #------- agent definition -------#
        # Set up the base template
        template = user_prompt + """Respond the user as best you can. You have access to the following tools:

        {tools}

        The following is the chat history so far:
        {memory}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question, including your sources.

        These were previous tasks you completed:



        Begin!

        {input}
        {agent_scratchpad}"""

        # Set up a prompt template
        class CustomPromptTemplate(BaseChatPromptTemplate):
            # The template to use
            template: str
            # The list of tools available
            tools: List[Tool]

            def format_messages(self, **kwargs) -> str:
                # Get the intermediate steps (AgentAction, Observation tuples)
                # Format them in a particular way
                intermediate_steps = kwargs.pop("intermediate_steps")
                thoughts = ""
                for action, observation in intermediate_steps:
                    thoughts += action.log
                    thoughts += f"\nObservation: {observation}\nThought: "
                # Set the agent_scratchpad variable to that value
                kwargs["agent_scratchpad"] = thoughts
                # Create a tools variable from the list of tools provided
                kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
                # Create a list of tool names for the tools provided
                kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
                formatted = self.template.format(**kwargs)
                return [HumanMessage(content=formatted)]
            
        class CustomOutputParser(AgentOutputParser):
            def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
                print(llm_output)
                print('inside parse')
                llm_output = '\n' + llm_output
                q.put(llm_output)
                # Check if agent should finish
                if "Final Answer:" in llm_output:
                    print('inside final answer')
                    # q.put(llm_output.split("Final Answer:")[-1])
                    return AgentFinish(
                        # Return values is generally always a dictionary with a single `output` key
                        # It is not recommended to try anything else at the moment :)
                        return_values={"output": llm_output.split("Final Answer:")[-1]},
                        log=llm_output,
                    )
                # Parse out the action and action input
                regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
                match = re.search(regex, llm_output, re.DOTALL)
                if not match:
                    print('inside no match')
                    # q.put(llm_output) #.split("Question:")[-1].split("\n")[0])
                    # raise ValueError(f"Could not parse LLM output: `{llm_output}`")
                    return AgentFinish(
                        # Return values is generally always a dictionary with a single `output` key
                        # It is not recommended to try anything else at the moment :)
                        return_values={"output": llm_output}, #.split("Question:")[-1].split("\n")[0]},
                        log=llm_output,
                    )
                action = match.group(1).strip()
                action_input = match.group(2)
                # Return the action and action input
                # q.put("Processing...\n")
                return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

        prompt_template = CustomPromptTemplate(
            template=template,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps", "memory"]
        )

        output_parser = CustomOutputParser()
        #------- end of agent definition -------#
        async def task(prompt):
            #definition of llm used for bot
            bot_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613', request_timeout=60*5)
            agent_kwargs = {
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            }
            llm_chain = LLMChain(llm=bot_llm, prompt=prompt_template)
            agent = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names
            )
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)
            ret = await agent_executor.arun(prompt)
            q.put(job_done)
            return ret

        with start_blocking_portal() as portal:
            portal.start_task_soon(task, history[-1][0])

            content = ""
            while True:
                next_token = q.get(True)
                if next_token is job_done:
                    break
                content += next_token
                history[-1] = (history[-1][0], content)

                yield history

# FastAPI 

class YoutubeRequest(BaseModel):
    history: list
    user_prompt: str = ""
    youtube_urls: list = []
    session: str = None
    bot_id: str = None
    api_key: str = None

api = FastAPI()

@api.get("/")
def read_root():
    return {"message": "API is alive"}

helper = """
This is an description of all the parameters that can be used. \n\n history: a list of messages in the conversation. (currently chat history is not working, ignores everything but last user message)
\n\n user_prompt: prompt to use for the bot, will use default if empty. \n\n session: session id, used for analytics/logging conversations, not necessary
\n\n youtube_urls: a list of youtube urls used to create a new bot. \n\n bot_id: a bot id used to call previously created bots \n\n api_key: api key necessary for auth
\n\n
Keys to good response:
- Can use this tool to grab videos from playlist https://www.thetubelab.com/get-all-urls-of-youtube-playlist-channel/
- Make sure videos includes only the youtuber talking, because we are grabbing the youtube generated captions, there is no way to differenciate between voices or background game audio which got captioned
- There maybe mispellings / mistakes in the captions which cannot be avoided, espeically with foreign names/words
- Include many / longer videos to get better results
- BotID saves the parameters for the bot, so you can use the same bot multiple times
    - the two parameters saved are user_prompt and youtube_urls
    - if you pass in a bot_id, it will ignore the both of these parameters
"""

@api.post("/youtube")
def youtube_bot_request(request: Annotated[
        YoutubeRequest,
        Body(
            openapi_examples={
                "create new youtube bot": {
                    "summary": "create new youtube bot",
                    "description": "Returns: {message: 'Success', chat: [[user message, ai reply]], bot_id: the new bot_id which was created}",
                    "value": {
                        "history": [["hi", ""]],
                        "youtube_urls":["https://youtu.be/6XEOVaL5a1Q", "https://youtu.be/5Qu-TCVCO3Q"],
                        "api_key":"xyz",
                    },
                },
                "create new youtube bot with custom prompt": {
                    "summary": "create new youtube bot with a custom prompt",
                    "description": "Returns: {message: 'Success', chat: [[user message, ai reply]], bot_id: the new bot_id which was created}",
                    "value": {
                        "history": [["hi", ""]],
                        "user_prompt": "Respond like the youtuber in the context below.",
                        "youtube_urls":["https://youtu.be/6XEOVaL5a1Q", "https://youtu.be/5Qu-TCVCO3Q"],
                        "api_key":"xyz",
                    },
                },
                "call the zealand bot": {
                    "summary": "call the zealand bot",
                    "description": "Use a bot_id to call a bot that has already been created for the youtuber zealand. \n\n  Returns: {message: 'Success', chat: [[user message, ai reply]], bot_id: the bot id}",
                    "value": {
                        "history": [["hello there", ""]],
                        "bot_id": "6e39115b-c771-49af-bb12-4cef3d072b45",
                        "api_key":"xyz",
                    },
                },
                "call the sirlarr bot": {
                    "summary": "call the sirlarr bot",
                    "description": "Use a bot_id to call a bot that has already been created for the youtuber sirlarr. \n\n  Returns: {message: 'Success', chat: [[user message, ai reply]], bot_id: the bot id}",
                    "value": {
                        "history": [["hello there", ""]],
                        "bot_id": "6cd7e23f-8be1-4eb4-b18c-55795eb1aca1",
                        "api_key":"xyz",
                    },
                },
                "call the offhand disney bot": {
                    "summary": "call the offhand disney bot",
                    "description": "Use a bot_id to call a bot that has already been created for the youtuber offhand disney. \n\n  Returns: {message: 'Success', chat: [[user message, ai reply]], bot_id: the bot id}",
                    "value": {
                        "history": [["hello there", ""]],
                        "bot_id": "8368890b-a45e-4dd3-a0ba-03250ea0cf30",
                        "api_key":"xyz",
                    },
                },
                "full descriptions of every parameter": {
                    "summary": "Description and Tips",
                    "description": helper,
                    "value": {
                        "history": [["user message 1", "ai replay 1"], ["user message 2", "ai replay 2"], ["user message 3", "ai replay 3"]],
                        "user_prompt": "prompt to use for the bot, will use the default of \"Respond in the same style as the youtuber in the context below.\" if empty",
                        "session": "session id, used for analytics/logging conversations, not necessary",
                        "youtube_urls":["url of youtube video", "url of youtube video", "url of youtube video"],
                        "bot_id": "id of bot previously created, if bot_id is passed then youtube_urls and user_prompt are ignored",
                        "api_key": "api key necessary for auth",
                    },
                },
            },
        )]):
    request_dict = request.dict()
    history = request_dict['history']
    user_prompt = request_dict['user_prompt']
    youtube_urls = request_dict['youtube_urls']
    session = request_dict['session']
    bot_id = request_dict['bot_id']
    api_key = request_dict['api_key']
    return process(history, user_prompt, youtube_urls, session, bot_id, api_key)

# request_OPB = """
# curl --header "Content-Type: application/json" \
#   --request POST \
#   --data '{"history":[["hi",""]],"api_key":"xyz"}' \
#   http://35.232.62.221/bot
# """

# request_youtube = """
# curl --header "Content-Type: application/json" \
#   --request POST \
#   --data '{"history":[["whats up lawerence?",""]],"youtube_urls":["https://www.youtube.com/watch?v=wnRTpHKTJgM", "https://www.youtube.com/watch?v=QHjuFAbUkg0"], "api_key":"xyz"}' \
#   http://35.232.62.221/youtube
# """

# request_youtube = """
# curl --header "Content-Type: application/json" \
#   --request POST \
#   --data '{"history":[["whats up?",""]],"bot_id":"8e35157b-9717-4f7d-bc34-e3365ea98673", "api_key":"xyz"}' \
#   http://35.232.62.221/youtube
# """


# #8e35157b-9717-4f7d-bc34-e3365ea98673

# OPB bot chat methods

class InputPrompt(BaseModel):
    prompt: str

class InputChat(BaseModel):
    chat: List[Tuple[Union[str, None], Union[str, None]]]

@api.post("/handle-prompt/")
def read_item(data: InputPrompt):
    return list(openai_bot([(data.prompt, None)], None))[-1][0][1]

@api.post("/handle-chat/")
def read_item(data: InputChat):
    return openai_bot(data.chat, None)
