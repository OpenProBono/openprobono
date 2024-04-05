"""Prompts used all across our system."""
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a legal expert. Given a question and chunks of legal text, answer the user question. If none of the chunks answer the question, just say you don't know.\n\nHere are the chunks:{context}",
        ),
        ("human", "{question}"),
    ]
)

class CitedAnswer(BaseModel):
    """Answer the user question based only on the given chunks, and cite the chunks used."""

    answer: str = Field(description="The answer to the user question, which is based only on the given chunks.")
    citations: List[int] = Field(description="The integer IDs of the SPECIFIC chunks which justify the answer.")



#for bot.py
COMBINE_TOOL_OUTPUTS_TEMPLATE = """You are a legal expert, tasked with answering any questions about law. ALWAYS use tools to answer questions.

Combine tool results into a coherent answer. If you used a tool, ALWAYS return a "SOURCES" part in your answer.
"""


#these are not used currently
ANSWER_TEMPLATE = """GENERAL INSTRUCTIONS
    You are a legal expert. Your task is to compose a response to the user's question using the information in the given context.

    CONTEXT:
    {context}

    USER QUESTION
    {input}"""

TOOLS_TEMPLATE = """GENERAL INSTRUCTIONS
    You are a legal expert. Your task is to decide which tools to use to answer a user's question. You can use up to X tools, and you can use tools multiple times with different inputs as well.

    These are the tools which are at your disposal
    {tools}

    When choosing tools, use this template:
    {{"tool": "name of the tool", "input": "input given to the tool"}}

    USER QUESTION
    {input}

    ANSWER FORMAT
    {{"tools":["<FILL>"]}}"""