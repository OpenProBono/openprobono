import pytest

from app.classifiers import get_probs
from app.models import AnthropicModelEnum, ChatModelParams, EngineEnum, OpenAIModelEnum

test_msg = "I am a new tenant and I'm having trouble with my utilities."

def category_test(category) -> None:
    assert isinstance(category, dict)
    assert "title" in category
    assert "probability" in category
    assert isinstance(category["title"], str)
    assert isinstance(category["probability"], float)
    assert category["probability"] >= 0.0
    assert category["probability"] <= 1.0


@pytest.mark.parametrize(("test_msg"), [(test_msg)])
def test_get_probs_openai(test_msg: str) -> None:
    chat_model = ChatModelParams(engine=EngineEnum.openai, model=OpenAIModelEnum.gpt_4o)
    probs = get_probs(test_msg, chat_model)
    assert isinstance(probs, list)
    num_categories = len(probs)
    assert num_categories > 0
    assert num_categories <= 20
    for category in probs:
        category_test(category)


@pytest.mark.parametrize(("test_msg"), [(test_msg)])
def test_get_probs_anthropic(test_msg: str) -> None:
    chat_model = ChatModelParams(
        engine=EngineEnum.anthropic,
        model=AnthropicModelEnum.claude_3_5_sonnet,
    )
    probs = get_probs(test_msg, chat_model)
    assert isinstance(probs, list)
    num_categories = len(probs)
    assert num_categories > 0
    assert num_categories <= 20
    for category in probs:
        category_test(category)
