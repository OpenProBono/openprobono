"""Classifies a chat request into LIST categories."""

from __future__ import annotations

import json

from langfuse.decorators import observe

from app.chat_models import chat_str_fn
from app.models import ChatModelParams, OpenAIModelEnum
from app.prompts import ISSUE_CLASSIFER_PROMPT


@observe()
def get_probs(
    message: str,
    chatmodel: ChatModelParams | None = None,
    **kwargs: dict,
) -> list:
    """Get LIST classification probabilities for a message.

    Parameters
    ----------
    message : str
        The message to be moderated.
    chatmodel : ChatModelParams, optional
        The engine and model to use for moderation, by default
        openai and gpt-4o.
    client : OpenAI | anthropic.Anthropic, optional
        The client to use for the moderation request. If not specified,
        one will be created.
    kwargs : dict, optional
        For the LLM. By default, temperature = 0 and max_tokens = 1000.

    Returns
    -------
    list
        of dictionaries of the form: {'title': '', 'probability': 0.0}

    Raises
    ------
    ValueError
        If chatmodel.engine is not `openai` or `anthropic`.

    """
    if chatmodel is None:
        chatmodel = ChatModelParams(model=OpenAIModelEnum.gpt_4o)
    chat_fn = chat_str_fn(chatmodel)
    msg = {"role": "user", "content":ISSUE_CLASSIFER_PROMPT.format(message=message)}
    response: str = chat_fn([msg], chatmodel.model, **kwargs)
    if "json" in response:
        response = response.replace("json", "")
    if "`" in response:
        response = response.replace("`", "")
    try:
        response_json = json.loads(response)
        return response_json["categories"]
    except Exception as e:
        print(e)
