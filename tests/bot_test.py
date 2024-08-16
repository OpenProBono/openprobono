"""Tests for bot methods."""

from app.bot import anthropic_bot, anthropic_bot_stream, openai_bot, openai_bot_stream
from app.models import (
    AnthropicModelEnum,
    BotRequest,
    ChatModelParams,
    ChatRequest,
    EngineEnum,
    OpenAIModelEnum,
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

def test_openai_bot_stream() -> None:
    search_tool = {
        "name": "filtered-case-search",
        "method": "courtlistener",
        "prompt": "",
    }
    chat_model = ChatModelParams(
        engine=EngineEnum.openai,
        model=OpenAIModelEnum.gpt_3_5,
    )
    bot = BotRequest(search_tools=[search_tool], chat_model=chat_model)
    sys_msg = {"role": "system", "content": bot.system_prompt}
    user_msg = {
        "role": "user",
        "content": "Tell me about cases related to copyright in New York since 2000.",
    }
    cr = ChatRequest(history=[sys_msg, user_msg], bot_id = "")
    for chunk in openai_bot_stream(cr, bot):
        print(chunk, end="", flush=True)


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
    assert len(bot_msg_content) > 2 #this is temporary because currently in anthropic bot we are not editing
                                        #ChatRequest history itself, but a copy of it

def test_anthropic_bot_stream() -> None:
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
    for chunk in anthropic_bot_stream(cr, bot):
        print(chunk, end="", flush=True)
