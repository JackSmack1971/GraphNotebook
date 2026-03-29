"""Tests for graphnotebook.llm.gateway.LLMGateway."""
import json

from unittest.mock import MagicMock, patch

import pytest

from graphnotebook.llm.gateway import LLMGateway


@pytest.fixture()
def gateway():
    return LLMGateway("synthesis")

class MockMessage:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)

class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]


@patch("graphnotebook.llm.gateway.completion")
def test_invoke_returns_string(mock_completion, gateway):
    mock_completion.return_value = MockResponse("hello")
    result = gateway.invoke("Say hello")
    assert result == "hello"
    mock_completion.assert_called_once()


@patch("graphnotebook.llm.gateway.completion")
def test_invoke_json_valid(mock_completion, gateway):
    mock_completion.return_value = MockResponse('{"mode": "global"}')
    result = gateway.invoke_json("classify")
    assert result == {"mode": "global"}


@patch("graphnotebook.llm.gateway.completion")
def test_invoke_json_strips_markdown_fences(mock_completion, gateway):
    """gateway must strip ```json ...``` fences before JSON parsing."""
    mock_completion.return_value = MockResponse('```json\n{"key": "val"}\n```')
    result = gateway.invoke_json("test")
    assert result["key"] == "val"


@patch("graphnotebook.llm.gateway.completion")
def test_invoke_json_malformed_raises(mock_completion, gateway):
    mock_completion.return_value = MockResponse("not json at all")
    with pytest.raises((json.JSONDecodeError, ValueError)):
        gateway.invoke_json("bad response")


@patch("graphnotebook.llm.gateway.completion")
def test_invoke_json_pin_fence_stripping(mock_completion, gateway):
    """Ensure exact fence stripping for marginal cases."""
    # JSON within a single-line block
    mock_completion.return_value = MockResponse('```json {"a":1} ```')
    result = gateway.invoke_json("test")
    assert result == {"a": 1}


@patch("graphnotebook.llm.gateway.completion")
def test_invoke_uses_task_model(mock_completion, gateway):
    from graphnotebook.llm.models import MODEL_REGISTRY
    mock_completion.return_value = MockResponse("ok")
    gateway.invoke("test")
    call_kwargs = mock_completion.call_args[1]
    expected_model = MODEL_REGISTRY["synthesis"]["primary"]
    assert call_kwargs.get("model") == expected_model, (
        f"Gateway used wrong model. Expected '{expected_model}', "
        f"got '{call_kwargs.get('model')}'"
    )
