from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a helpful AI assistant. Given a user question and some legal text snippets, answer the user question. If none of the texts answer the question, just say you don't know.\n\nHere are the legal texts:{context}",
        ),
        ("human", "{question}"),
    ]
)

class CitedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""
    answer: str = Field(description="The answer to the user question, which is based only on the given sources.")
    citations: List[int] = Field(description="The integer IDs of the SPECIFIC sources which justify the answer.")