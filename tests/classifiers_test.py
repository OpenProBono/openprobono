import pytest

from app.classifiers import tree_search
from app.models import (
    AnthropicModelEnum,
    ChatModelParams,
    EngineEnum,
    LISTTerm,
    OpenAIModelEnum,
)

test_msg = "I am a new tenant and I'm having trouble with my utilities."

def category_test(category: LISTTerm) -> None:
    assert isinstance(category, LISTTerm)
    assert category.code
    assert category.title
    assert category.definition
    for parent_code in category.parent_codes:
        assert isinstance(parent_code, str)
    for taxonomy in category.taxonomies:
        assert isinstance(taxonomy, str)
    for child in category.children:
        assert isinstance(child, LISTTerm)


@pytest.mark.parametrize(("test_msg"), [(test_msg)])
def test_get_probs_openai(test_msg: str) -> None:
    chat_model = ChatModelParams(engine=EngineEnum.openai, model=OpenAIModelEnum.gpt_4o)
    probs = tree_search(test_msg, chat_model, 0.2, 7)
    assert isinstance(probs, list)
    num_categories = len(probs)
    assert num_categories > 0
    for category in probs:
        category_test(category)


@pytest.mark.parametrize(("test_msg"), [(test_msg)])
def test_get_probs_anthropic(test_msg: str) -> None:
    chat_model = ChatModelParams(
        engine=EngineEnum.anthropic,
        model=AnthropicModelEnum.claude_3_5_sonnet,
    )
    probs = tree_search(test_msg, chat_model, 0.2, 7)
    assert isinstance(probs, list)
    num_categories = len(probs)
    assert num_categories > 0
    for category in probs:
        category_test(category)
