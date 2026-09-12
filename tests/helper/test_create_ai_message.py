import pytest
from langchain_core.messages import AIMessage

from langgraph_agent_toolkit.helper.utils import create_ai_message


def test_create_ai_message_preserves_fields_and_drops_unknown_keys():
    """Keep message fields and remove fields outside the LangChain schema."""
    message = AIMessage(
        content="Search for this value.",
        id="message-1",
        tool_calls=[{"name": "search", "args": {"query": "one"}, "id": "call-1", "type": "tool_call"}],
        response_metadata={"source": "test"},
    )

    result = create_ai_message({**message.model_dump(), "unknown": "drop this"})

    assert result.model_dump() == message.model_dump()
    assert not hasattr(result, "unknown")


@pytest.mark.parametrize("parts", [{}, {"tool_calls": []}], ids=["empty", "fields-without-content"])
def test_create_ai_message_defaults_missing_content(parts):
    assert create_ai_message(parts).content == ""
