"""Run evaluation methods on agents and chains."""

import pandas as pd

from bot import opb_bot
from models import BotRequest, ChatRequest


def legalbench_ruleqa():
    # load ruleqa
    ruleqa = pd.read_csv("data/legalbench-ruleqa.csv")

    # load responses from competitors
    #gpt = pd.read_csv("data/gpt_responses.csv")
    #perplexity = pd.read_csv("data/perplexity_responses.csv")
    #gemini = pd.read_csv("data/gemini_responses.csv")

    # get OPB_bot responses
    bot_dict = {
        "search_tools": [
            {
                "name": "government-search",
                "method": "serpapi",
                "prefix": "site:*.gov | site:*.edu | site:*scholar.google.com",
                "prompt": "Useful for when you need to answer questions or find resources about "  # noqa: E501
                            "government and laws.",
            },
            {
                "name": "case-search",
                "method": "courtlistener",
                "prompt": "Use for finding case law.",
            },
        ],
        "vdb_tools": [
            {
                "collection_name": "USCode",
                "k": 4,
                "prompt": "Useful for finding information about US Code",
            },
        ],
        "engine": "langchain",
        "model": "gpt-3.5-turbo-0125",
        "api_key": "xyz",
    }
    bot = BotRequest(**bot_dict)
    for _, row in ruleqa.iterrows():
        question = row["text"]
        chat = ChatRequest(
            history=[[question, None]],
            bot_id="",
            api_key="",
        )
        answer = opb_bot(chat, bot)
        print(f"question: {question}\nanswer: {answer}")
        break
    # evaluate answers
    # plot evaluation results

def evaluate_answer(question: str, answer: str, true_answer: str) -> int:
    pass
