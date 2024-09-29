"""Functions for testing summarization."""

from app.models import ChatModelParams, EngineEnum, GoogleModelEnum, SummaryMethodEnum
from app.summarization import summarize
import logging
import json_log_formatter

formatter = json_log_formatter.JSONFormatter()

json_handler = logging.StreamHandler()
json_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(json_handler)
logger.setLevel(logging.INFO)

test_documents = [
    "My name is John Doe. I am from New York City and I live on the Upper West Side.",
    "I have a cat named Tiger who is very cute. He likes to play with my dog, but he does not like to be petted by me.",
]

def test_summarize_stuff_reduce() -> None:
    summary = summarize(test_documents, method=SummaryMethodEnum.stuff_reduce)
    print(summary)
    assert isinstance(summary, str)
    assert len(summary) > 0

def test_summarize_stuffing() -> None:
    summary = summarize(test_documents, method=SummaryMethodEnum.stuffing)
    print(summary)
    assert isinstance(summary, str)
    assert len(summary) > 0

def test_summarize_gemini_full() -> None:
    chatmodel = ChatModelParams(engine=EngineEnum.google, model=GoogleModelEnum.gemini_1_5_flash)
    summary = summarize(test_documents, method=SummaryMethodEnum.stuff_reduce, chat_model=chatmodel)
    logger.info(summary)
    assert isinstance(summary, str)
    assert len(summary) > 0

def test_summarize_mapreduce() -> None:
    summary = summarize(test_documents, method=SummaryMethodEnum.map_reduce)
    print(summary)
    assert isinstance(summary, str)
    assert len(summary) > 0

def test_summarize_refine() -> None:
    summary = summarize(test_documents, method=SummaryMethodEnum.refine)
    print(summary)
    assert isinstance(summary, str)
    assert len(summary) > 0
