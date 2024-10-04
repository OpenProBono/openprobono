
from app.chat_models import chat_gemini, chat_single_gemini
from app.models import (
   ChatModelParams,
   EngineEnum,
   GoogleModelEnum,
)


def test_hi_chat_single_gemini() -> None:
   chatmodel = ChatModelParams(engine=EngineEnum.google, model=GoogleModelEnum.gemini_1_5_flash)
   msg = "Hi"
   output = (chat_single_gemini(msg, chatmodel.model))


def test_chat_gemini() -> None:
   chatmodel = ChatModelParams(engine=EngineEnum.google, model=GoogleModelEnum.gemini_1_5_flash)
   msg = [{"role":"user", "content":"Hi"}]
   output = (chat_gemini(msg, chatmodel.model))