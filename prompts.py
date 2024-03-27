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
