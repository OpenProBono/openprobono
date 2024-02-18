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

def research_question(input_tuple):
    q, bot = input_tuple
    cr = ChatRequest(history=[[q,""]], api_key="xyz", bot_id="216bec82-f063-4f48-897b-8bf1e77e24ef")
    return opb_bot(cr, bot)
    

NUM_PROCESSES = 4
def flow(r, bot):
    #add try / retry here if json.loads fails
    input = r.history[-1][0]
    decomp = decompisition_bot(input)
    comp = json.loads(decomp.content)
    context = ""
    
    # def research_question(q):
    #     cr = ChatRequest(history=[[q,""]], api_key="xyz", bot_id=r.bot_id)
    #     return opb_bot(r, bot)
    
    if('sub-questions' in comp):
        print(comp['sub-questions'])
        zipped = list(zip(comp['sub-questions'], [bot]*len(comp['sub-questions'])))
        print(zipped)
        print("000")
        # print(zipped.__dict__)
        print(zipped[0])
        results = Pool(NUM_PROCESSES).map(research_question, zipped)
        context += str(results)

    return recompisition_bot(input, context).content