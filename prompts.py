from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

MAX_NUM_TOOLS = 8
MULTIPLE_TOOLS_TEMPLATE = """You are a legal expert, tasked with answering any questions about law. ALWAYS use tools to answer questions.

Combine tool results into a coherent answer. If you used a tool, ALWAYS return a "SOURCES" part in your answer.
"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a legal expert. Given a question and chunks of legal text, answer the user question. If none of the chunks answer the question, just say you don't know.\n\nHere are the chunks:{context}",
        ),
        ("human", "{question}"),
    ]
)

# https://docs.anthropic.com/claude/docs/tool-use-examples
# To prompt Sonnet or Haiku to better assess the user query before making tool calls,
# the following prompt can be used:"
ANTHROPIC_SONNET_TOOL_PROMPT = (
    "Answer the user's request using relevant tools (if they are available). "
    "Before calling a tool, do some analysis within <thinking></thinking> tags. "
    "First, think about which of the provided tools is the relevant tool to answer the "
    "user's request. Second, go through each of the required parameters of the "
    " relevant tool and determine if the user has directly provided or given enough "
    "information to infer a value. When deciding if the parameter can be inferred, "
    "carefully consider all the context to see if it supports a specific value. If all "
    "of the required parameters are present or can be reasonably inferred, close the "
    "thinking tag and proceed with the tool call. BUT, if one of the values for a "
    "required parameter is missing, DO NOT invoke the function (not even with fillers "
    "for the missing params) and instead, ask the user to provide the missing "
    "parameters. DO NOT ask for more information on optional parameters if it is not "
    "provided."
)

# based on moderation prompt from Anthropic's API:
# https://docs.anthropic.com/claude/docs/content-moderation
MODERATION_PROMPT = (
    "A human user is in dialogue with an AI. The human is asking the AI questions "
    "about legal information and resources. Here is the most recent request from "
    "the user:<user query>{user_input}</user query>\n\n"
    "If the user's request refers to harmful, pornographic, or illegal activities, "
    "reply with (Y). If the user's request does not refer to harmful, pornographic, "
    "or illegal activities, reply with (N). Reply with nothing else other than (Y) or "
    "(N)."
)

HIVE_QA_PROMPT = (
    "You are a legal assistant that provides factual information about "
    "legal code based on the following prompts.  After using information "
    "from a selected prompt, please attempt to provide a reference to the "
    "section that the information was pulled from. If a question does not "
    "make sense or is not factually coherent, explain why instead of "
    "answering something not correct. Please respond clearly and concisely "
    "as if talking to the customer directly."
)
class CitedAnswer(BaseModel):
    """Answer the user question based only on the given chunks, and cite the chunks used."""

    answer: str = Field(description="The answer to the user question, which is based only on the given chunks.")
    citations: List[int] = Field(description="The integer IDs of the SPECIFIC chunks which justify the answer.")
