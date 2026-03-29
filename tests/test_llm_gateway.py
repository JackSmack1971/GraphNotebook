"""Tests for graphnotebook.llm.gateway.LLMGateway.

Mutation targets addressed:
  M8a — StringMethodsMutator: remove all fence stripping → killed by
        test_invoke_json_strips_markdown_fences (existing).
  M8b — StringMethodsMutator: single-line / no-newline fence variants not
        stripped. Killed by:
          test_invoke_json_single_line_fence
          test_invoke_json_no_trailing_newline_fence
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from graphnotebook.llm.gateway import LLMGateway


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


@pytest.fixture()
def gateway():
    return LLMGateway("synthesis")


# ---------------------------------------------------------------------------
# invoke
# ---------------------------------------------------------------------------


@patch("graphnotebook.llm.gateway.completion")
def test_invoke_returns_string(mock_completion, gateway):
    mock_completion.return_value = _Response("hello")
    result = gateway.invoke("Say hello")
    assert result == "hello"
    mock_completion.assert_called_once()


@patch("graphnotebook.llm.gateway.completion")
def test_invoke_uses_task_model(mock_completion, gateway):
    """The model kwarg must match MODEL_REGISTRY['synthesis']['primary']."""
    from graphnotebook.llm.models import MODEL_REGISTRY

    mock_completion.return_value = _Response("ok")
    gateway.invoke("test")

    call_kwargs = mock_completion.call_args[1]
    expected_model = MODEL_REGISTRY["synthesis"]["primary"]
    assert call_kwargs.get("model") == expected_model, (
        f"Gateway used wrong model: expected '{expected_model}', "
        f"got '{call_kwargs.get('model')}'"
    )


# ---------------------------------------------------------------------------
# invoke_json — happy paths
# ---------------------------------------------------------------------------


@patch("graphnotebook.llm.gateway.completion")
def test_invoke_json_valid_plain(mock_completion, gateway):
    mock_completion.return_value = _Response('{"mode": "global"}')
    result = gateway.invoke_json("classify")
    assert result == {"mode": "global"}


@patch("graphnotebook.llm.gateway.completion")
def test_invoke_json_strips_multiline_fence(mock_completion, gateway):
    """Standard multi-line ```json\\n...\\n``` fence must be stripped."""
    mock_completion.return_value = _Response('```json\n{"key": "val"}\n```')
    result = gateway.invoke_json("test")
    assert result["key"] == "val"


# ---------------------------------------------------------------------------
# M8b — StringMethodsMutator kill: fence variant coverage
# ---------------------------------------------------------------------------


@patch("graphnotebook.llm.gateway.completion")
def test_invoke_json_single_line_fence(mock_completion, gateway):
    """Single-line ```json {...} ``` (no surrounding newlines) must be stripped.

    Kills M8b: a stripping implementation that only handles \\n-delimited fences
    will leave this variant intact, causing json.loads to raise.
    """
    mock_completion.return_value = _Response('```json {"a": 1} ```')
    result = gateway.invoke_json("test")
    assert result == {"a": 1}, (
        "Single-line ```json ... ``` fence was not stripped before JSON parsing"
    )


@patch("graphnotebook.llm.gateway.completion")
def test_invoke_json_no_trailing_newline_fence(mock_completion, gateway):
    """```json\\n{...}``` with no trailing newline before closing fence."""
    mock_completion.return_value = _Response('```json\n{"b": 2}```')
    result = gateway.invoke_json("test")
    assert result == {"b": 2}, (
        "Fence without trailing newline before closing ``` was not stripped"
    )


@patch("graphnotebook.llm.gateway.completion")
def test_invoke_json_plain_fence_no_lang_tag(mock_completion, gateway):
    """Plain ``` fence (no 'json' tag) must also be stripped."""
    mock_completion.return_value = _Response('```\n{"c": 3}\n```')
    result = gateway.invoke_json("test")
    assert result == {"c": 3}


# ---------------------------------------------------------------------------
# invoke_json — error paths
# ---------------------------------------------------------------------------


@patch("graphnotebook.llm.gateway.completion")
def test_invoke_json_malformed_raises(mock_completion, gateway):
    """Unparseable content must raise JSONDecodeError or ValueError."""
    mock_completion.return_value = _Response("not json at all")
    with pytest.raises((json.JSONDecodeError, ValueError)):
        gateway.invoke_json("bad response")


@patch("graphnotebook.llm.gateway.completion")
def test_invoke_json_empty_braces_returns_dict(mock_completion, gateway):
    """Minimal valid JSON `{}` must parse without error."""
    mock_completion.return_value = _Response("{}")
    result = gateway.invoke_json("empty")
    assert result == {}
