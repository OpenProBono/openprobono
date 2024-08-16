"""Tests for bot methods."""

from app.bot import anthropic_bot, openai_bot
from app.models import (
    AnthropicModelEnum,
    BotRequest,
    ChatModelParams,
    ChatRequest,
    EngineEnum,
)


def test_openai_bot() -> None:
    search_tool = {
        "name": "filtered-case-search",
        "method": "courtlistener",
        "prompt": "",
    }
    bot = BotRequest(search_tools=[search_tool])
    sys_msg = {"role": "system", "content": bot.system_prompt}
    user_msg = {
        "role": "user",
        "content": "Tell me about cases related to copyright in New York since 2000.",
    }
    cr = ChatRequest(history=[sys_msg, user_msg], bot_id = "")
    bot_msg_content = openai_bot(cr, bot)
    assert isinstance(bot_msg_content, str)
    assert len(cr.history) > 2

def test_anthropic_bot() -> None:
    search_tool = {
        "name": "filtered-case-search",
        "method": "courtlistener",
        "prompt": "",
    }
    chat_model = ChatModelParams(
        engine=EngineEnum.anthropic,
        model=AnthropicModelEnum.claude_3_5_sonnet,
    )
    bot = BotRequest(search_tools=[search_tool], chat_model=chat_model)
    sys_msg = {"role": "system", "content": bot.system_prompt}
    user_msg = {
        "role": "user",
        "content": "Tell me about cases related to copyright in New York since 2000.",
    }
    cr = ChatRequest(history=[sys_msg, user_msg], bot_id = "")
    bot_msg_content = anthropic_bot(cr, bot)
    assert isinstance(bot_msg_content, str)
    assert len(cr.history) > 2
