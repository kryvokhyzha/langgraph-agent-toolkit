import pytest
from langchain_core.messages import AIMessage

from langgraph_agent_toolkit.helper.utils import create_ai_message


@pytest.mark.parametrize(
    "parts, expected",
    [
        # 1) Basic content + tool_calls
        (
            {"content": "Hello", "tool_calls": []},
            {"content": "Hello", "tool_calls": []},
        ),
        # 2) Unknown keys are ignored
        (
            {"content": "Test", "foobar": 123, "tool_calls": []},
            {"content": "Test", "tool_calls": []},
        ),
        # 3) Extra valid AIMessage params (id, type) pass through
        (
            {
                "content": "Hey",
                "id": "abc-123",
                "type": "ai",
                "tool_calls": [],
            },
            {"content": "Hey", "id": "abc-123", "type": "ai", "tool_calls": []},
        ),
    ],
)
def test_create_ai_message_filters_and_passes_through(parts, expected):
    """create_ai_message should:
    - Drop unknown keys ("foobar")
    - Preserve keys that match AIMessage signature
    - Use the final value for duplicate keys in the parts dict.
    """  # noqa: D205
    msg: AIMessage = create_ai_message(parts)
    for key, val in expected.items():
        assert getattr(msg, key) == val


def test_create_ai_message_missing_content_defaults_to_empty():
    """AIMessage requires 'content'; create_ai_message defaults it to '' so field-only parts don't crash."""
    msg = create_ai_message({"tool_calls": []})
    assert msg.content == ""


def test_create_ai_message_empty_dict_defaults_to_empty():
    """Completely empty parts construct an AIMessage with empty content (no crash)."""
    msg = create_ai_message({})
    assert msg.content == ""
