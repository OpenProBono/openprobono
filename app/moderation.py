"""Moderate messages with moderation models/prompts."""

from __future__ import annotations

from langfuse.decorators import observe

from app.chat_models import ANTHROPIC_CLIENT, OPENAI_CLIENT
from app.models import ChatModelParams, EngineEnum, OpenAIModelEnum
from app.prompts import MODERATION_PROMPT

DEFAULT_MODERATOR = ChatModelParams(
    engine=EngineEnum.openai,
    model=OpenAIModelEnum.mod_latest,
)

def moderate(message: str, chatmodel: ChatModelParams = DEFAULT_MODERATOR) -> bool:
    """Moderate a message using the specified engine and model.

    Parameters
    ----------
    message : str
        The message to be moderated.
    chatmodel : ChatModelParams, optional
        The engine and model to use for moderation, by default
        openai and text-moderation-latest.

    Returns
    -------
    bool
        True if flagged; False otherwise.

    Raises
    ------
    ValueError
        If chatmodel.engine is not `openai` or `anthropic`.

    """
    match chatmodel.engine:
        case EngineEnum.openai:
            moderators = {OpenAIModelEnum.mod_latest, OpenAIModelEnum.mod_stable}
            if chatmodel.model not in moderators:
                msg = f"Unsupported moderation model: {chatmodel.model}"
                raise ValueError(msg)
            return moderate_openai(message, chatmodel.model)
        case EngineEnum.anthropic:
            return moderate_anthropic(message, chatmodel.model)
    msg = f"Unsupported engine: {chatmodel.engine}"
    raise ValueError(msg)


@observe()
def moderate_openai(message: str, model: str) -> bool:
    """Moderate a message using OpenAI's Moderation API.

    Parameters
    ----------
    message : str
        The message to be moderated.
    model : str
        The model to use for moderation.

    Returns
    -------
    bool
        True if flagged; False otherwise.

    """
    response = OPENAI_CLIENT.moderations.create(model=model, input=message)
    return response.results[0].flagged


@observe()
def moderate_anthropic(message: str, model: str) -> bool:
    """Moderate a message using an Anthropic model.

    Parameters
    ----------
    message : str
        The message to be moderated.
    model : str
        The model to use for moderation.

    Returns
    -------
    bool
        True if flagged; False otherwise.

    """
    moderation_msg = {
        "role": "user",
        "content": MODERATION_PROMPT.format(user_input=message),
    }
    response = ANTHROPIC_CLIENT.messages.create(
        model=model,
        max_tokens=10,
        temperature=0,
        messages=[moderation_msg],
    )
    return "Y" in response.content[-1].text.strip()
