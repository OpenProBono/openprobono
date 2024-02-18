import json
from multiprocessing import Pool

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from bot import opb_bot
from models import BotRequest, ChatRequest


def decompisition_bot(input):
    decomp_template = """GENERAL INSTRUCTIONS
        You are a legal expert. Your task is to break down a legal question into simpler sub-parts. Consider different possibilites of interpretations and alternative ways to approach the question, but stay focused on the user's request.
        
        USER QUESTION
        {input}
        
        ANSWER FORMAT
        {{"sub-questions":["<FILL>"]}}"""
    messages = [
        HumanMessage(
            content=decomp_template.format(input=input)
        ),
    ]
    model = ChatOpenAI(model="gpt-4", temperature=0.0)
    return model(messages)


def recompisition_bot(input, context):
    recomp_template = """GENERAL INSTRUCTIONS
        You are a legal expert. Your task is to re-compose a legal question from simpler sub-parts. Consider different possibilites of interpretations and alternative ways to approach the question. Try to provide a clear and concise answer, that is easy to understand for non-legal experts. Always include SOURCES section with references to the legal sources you used in the context.
        
        SUB QUESTION CONTEXT
        {context}


        USER QUESTION
        {input}"""
    messages = [
        HumanMessage(
            content=recomp_template.format(input=input, context=context)
        ),
    ]
    model = ChatOpenAI(model="gpt-4", temperature=0.0)
    return model(messages)

#research a specific aspect (q) or 'sub question' of the user's original request
def research_aspect(input):
    q, r, bot = input
    cr = ChatRequest(history = [[q,""]], api_key = r.api_key, bot_id = r.bot_id, session = r.session if r.session is not None else " ")
    return opb_bot(cr, bot)
    

NUM_PROCESSES = 4
def flow(r: ChatRequest, bot: BotRequest):
    #add try / retry here if json.loads fails
    input = r.history[-1][0]
    decomp = decompisition_bot(input)
    comp = json.loads(decomp.content)
    context = ""
    
    if('sub-questions' in comp):
        zipped = list(zip(comp['sub-questions'], [r]*len(comp['sub-questions']), [bot]*len(comp['sub-questions'])))
        results = Pool(NUM_PROCESSES).map(research_aspect, zipped)
        context += str(results)

    return recompisition_bot(input, context).content