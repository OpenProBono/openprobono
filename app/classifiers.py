"""LLM classification tasks."""

from __future__ import annotations

import csv
import json
import pathlib
import re

from langfuse.decorators import observe

from app.chat_models import chat_str
from app.logger import setup_logger
from app.models import ChatModelParams, LISTTerm, LISTTermProb, OpenAIModelEnum
from app.prompts import (
    ISSUE_CLASSIFER_PROMPT,
    JURISDICTION_PROMPT,
    JURISDICTION_SUMMARY_PROMPT,
)

logger = setup_logger()

def parse_bracketed_items(text: str):
    # Use regex to find all items enclosed in square brackets
    return re.findall(r"\[([^\]]+)\]", text)

def read_taxonomy() -> dict:
    """Read the LIST taxonomy from a CSV file.

    Returns
    -------
    dict
        The LIST taxonomy

    """
    taxonomy = {}
    with (pathlib.Path(__file__).parent.parent / "data/list-taxonomy.csv").open("r") as f:
        reader = csv.reader(f, quotechar='"', delimiter=",")
        next(reader)  # Skip header row if present
        for row in reader:
            code, title, definition, parent_terms, taxonomies = row
            term = LISTTerm(code=code, title=title, definition=definition)
            term.parent_codes = parse_bracketed_items(parent_terms)
            term.taxonomies = [t.strip() for t in taxonomies.split(",") if t]
            taxonomy[code] = term
    return taxonomy


def build_hierarchy(taxonomy: dict[str, LISTTerm]) -> LISTTerm:
    root = LISTTerm(code="ROOT", title="Root", definition="Root of the taxonomy")
    for term in taxonomy.values():
        if not term.parent_codes:
            root.children.append(term)
        else:
            for parent_code in term.parent_codes:
                if parent_code in taxonomy:
                    taxonomy[parent_code].children.append(term)
    return root

def print_hierarchy(term: LISTTerm, level: int = 0) -> None:
    logger.info("%s%s: %s", "  " * level, term.code, term.title)
    for child in term.children:
        print_hierarchy(child, level + 1)

@observe()
def get_probs(
    message: str,
    terms: list[LISTTerm],
    chatmodel: ChatModelParams | None = None,
    **kwargs: dict,
) -> list[LISTTermProb]:
    """Get LIST classification probabilities for a message.

    Parameters
    ----------
    message : str
        The message to be classified.
    terms : list[LISTTerm]
        The LIST term classes to be assigned probabilities.
    chatmodel : ChatModelParams, optional
        The engine and model to use for classification, by default
        openai and gpt-4o.
    client : OpenAI | anthropic.Anthropic, optional
        The client to use for the classification request. If not specified,
        one will be created.
    kwargs : dict, optional
        For the LLM. By default, temperature = 0 and max_tokens = 1000.

    Returns
    -------
    list[LISTTermProb]
        with `title` and `probability` fields, ordered by descending probability

    Raises
    ------
    ValueError
        If chatmodel.engine is not `openai` or `anthropic`.

    """
    if chatmodel is None:
        chatmodel = ChatModelParams(model=OpenAIModelEnum.gpt_4o)
    terms_str = "\n".join([
        f"{i}. {t.title}: {t.definition}"
        for i, t in enumerate(terms, start=1)
    ])
    msg = {
        "role": "user",
        "content": ISSUE_CLASSIFER_PROMPT.format(terms=terms_str, message=message),
    }
    response: str = chat_str([msg], chatmodel, **kwargs)
    response = response.replace("json", "")
    response = response.replace("`", "")
    response_json: dict = json.loads(response)
    categories = response_json.get("categories", [])
    return [
        LISTTermProb(title=cat["title"], probability=cat["probability"])
        for cat in categories
    ]


def tree_search(
    root: LISTTerm,
    message: str,
    llm: ChatModelParams | None = None,
    threshold: float = 0.5,
    max_depth: int | None = None,
) -> list[LISTTerm]:
    current_terms = [root]
    selected_terms = []
    depth = 0

    while current_terms and (max_depth is None or depth < max_depth):
        # Get all children of current terms
        next_level_terms: list[LISTTerm] = []
        for term in current_terms:
            next_level_terms.extend(term.children)

        if not next_level_terms:
            break  # We've reached the bottom of the tree

        # Rank the next level terms
        ranked_terms = get_probs(message, next_level_terms, llm)

        # Select terms above the threshold
        next_selected_terms = [
            next((nlt for nlt in next_level_terms if nlt.title == pred.title), None)
            for pred in ranked_terms
            if pred.probability > threshold
        ]

        # Add selected terms to the result
        selected_terms.extend([term for term in next_selected_terms if term])

        # Prepare for next iteration
        current_terms = next_selected_terms
        depth += 1

    return selected_terms


def url_jurisdictions(
    url: str,
    chatmodel: ChatModelParams | None = None,
    summary: str = "",
) -> list[dict[str, str]]:
    """Get jurisdiction(s) from a URL.

    Parameters
    ----------
    url : str
        URL to search for jurisdictions.
    chatmodel : ChatModelParams | None, optional
        Chat model to use for classification, by default None
    summary : str, optional
        Summary of the URL contents to aid classification, by default ""

    Returns
    -------
    list[dict[str, str]]
        List of dicts containing predicted jurisdiction `name` and `confidence`.

    """
    if summary:
        summary_prompt = JURISDICTION_SUMMARY_PROMPT.format(ai_summary=summary)
    else:
        summary_prompt = ""
    sys_prompt = JURISDICTION_PROMPT.format(optional_summary_prompt=summary_prompt)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": url},
    ]
    kwargs = {"max_tokens": 250}
    if chatmodel is None:
        chatmodel = ChatModelParams(model=OpenAIModelEnum.gpt_4o)
    response = chat_str(messages, chatmodel, **kwargs)
    response = response.replace("json", "")
    response = response.replace("`", "")
    return json.loads(response)


# usage
# taxonomy = read_taxonomy()
# root = build_hierarchy(taxonomy)

# example = "My landlord refuses to fix the heat in my apartment."
# result = tree_search(root, example)
# print("Selected terms:", [term.title for term in result])
