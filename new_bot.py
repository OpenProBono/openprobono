import json
from multiprocessing import Pool

from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatOpenAI

from bot import opb_bot
from milvusdb import session_source_summaries
from models import BotRequest, ChatRequest
from search_tools import search_toolset_creator, serpapi_toolset_creator
from vdb_tools import session_query_tool, vdb_toolset_creator


def call_gpt4(input):
    model = ChatOpenAI(model="gpt-4", temperature=0.0)
    return model([HumanMessage(content=input)])

def tool_selector(input, tools):
    tool_template = """GENERAL INSTRUCTIONS

    As a legal expert, your role is to strategically select up to 8 tools to address a user's question. Optimize the template for clarity and flexibility in utilizing the available tools.

    AVAILABLE TOOLS
    {tools}

    TEMPLATE FOR TOOL SELECTION
    {{"tool": "name of the tool", "input": "input provided for the tool"}}

    USER QUESTION
    {input}

    ANSWER FORMAT
    {{"tools": ["<FILL>"]}}"""
    return call_gpt4(tool_template.format(input=input, tools=tools))

def decompisition_bot(input, memory):
    decomp_template = """GENERAL INSTRUCTIONS
        You are a legal expert. Your task is to break down a legal question into simpler sub-parts. Consider different possibilites of interpretations and alternative ways to approach the question, but stay focused on the user's request. Generate a maximum of 8 sub-questions.
        
        CHAT CONTEXT
        {memory}

        USER QUESTION
        {input}
        
        ANSWER FORMAT
        {{"sub-questions":["<FILL>"]}}"""
    return call_gpt4(decomp_template.format(input=input, memory=memory))

def further_aspects_bot(input, response, context, memory):
    further_template = """GENERAL INSTRUCTIONS
        You are a legal expert. Your task is to determine if any further research into certain aspects could enhance the AI's response to the query. If so, provide a list of topics that could be researched further.

        RESPONSE FORMAT INSTRUCTIONS
        ----------------------------

        When responding to me, please output a response in one of two formats:

        **Option 1:**
        Use this if further research is needed.
        Markdown code snippet formatted in the following schema:
        {{"sub-questions":["<FILL>"]}}

        **Option #2:**
        Use this if the answer fulfills the users query in the most optimal manner. 
        Markdown code snippet formatted in the following schema:
        {{"action": "Final Answer"}}

        SUB QUESTION CONTEXT
        {context}

        CHAT CONTEXT
        {memory}

        USER QUESTION
        {input}
        
        AI RESPONSE
        {response}
        """
    return call_gpt4(further_template.format(input=input, memory=memory, response=response, context=context))

def recompisition_bot(input, context, memory):
    recomp_template = """GENERAL INSTRUCTIONS
        You are a legal expert. Your task is to re-compose a legal question from simpler sub-parts. Consider different possibilites of interpretations and alternative ways to approach the question. Try to provide a clear and concise answer, that is easy to understand for non-legal experts. Always include SOURCES section with references to the legal sources you used in the context.
        
        SUB QUESTION CONTEXT
        {context}

        CHAT CONTEXT
        {memory}

        USER QUESTION
        {input}"""
    return call_gpt4(recomp_template.format(input=input, context=context, memory=memory))

#research a specific aspect (q) or 'sub question' of the user's original request
def research_aspects(input):
    q, r, bot = input
    cr = ChatRequest(history = [[q,""]], api_key = r.api_key, bot_id = r.bot_id, session_id = "")
    return opb_bot(cr, bot)

NUM_PROCESSES = 8
def flow(r: ChatRequest, bot: BotRequest):
    
    memory_llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-1106')
    memory = ConversationSummaryBufferMemory(llm=memory_llm, max_token_limit=2000, memory_key="memory", return_messages=True)
    for i in range(0, len(r.history)-1):
        memory.save_context({'input': r.history[i][0]}, {'output': r.history[i][1]})
    input = r.history[-1][0]

    toolset = []
    if(bot.search_tool_method == "google_search"):
        toolset += search_toolset_creator(bot)
    else:
        toolset += serpapi_toolset_creator(bot)
    toolset += vdb_toolset_creator(bot.vdb_tools)
    source_summaries = session_source_summaries(r.session_id)
    if source_summaries:
        toolset.append(session_query_tool(r.session_id, source_summaries))
    
    print("hiii")
    print(tool_selector(input, toolset))
    decomp = decompisition_bot(input, str(memory.chat_memory))
    #add try / retry here if json.loads fails
    comp = json.loads(decomp.content)
    context = ""
    response = ""
    while('sub-questions' in comp):
        zipped = list(zip(comp['sub-questions'], [toolset]*len(comp['sub-questions'])))
        results = Pool(NUM_PROCESSES).map(tool_selector, zipped)
        context += str(results)
        print(context)
        # response = recompisition_bot(input, context, str(memory.chat_memory)).content
        # comp = json.loads(further_aspects_bot(input, response, context, str(memory.chat_memory)).content)

    return response