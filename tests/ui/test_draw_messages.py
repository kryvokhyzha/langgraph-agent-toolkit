"""Tests for the Streamlit draw_messages renderer (streamlit mocked)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from langgraph_agent_toolkit.schema import ChatMessage


async def _agen(items):
    for item in items:
        yield item


def _mock_st():
    st = MagicMock()
    st.session_state = SimpleNamespace(messages=[], last_message=None, display_tools_execution=False)
    return st


def _tool_call_msg():
    return ChatMessage(
        type="ai",
        content="",
        tool_calls=[{"name": "send_email", "args": {"to": "x"}, "id": "c1", "type": "tool_call"}],
    )


@pytest.mark.asyncio
async def test_draw_messages_renders_hitl_interrupt_without_error():
    """A HITL interrupt (an 'ai' message) following a tool call must render, not error."""
    from langgraph_agent_toolkit.ui.components import draw_message as dm

    st = _mock_st()
    interrupt = ChatMessage(type="ai", content="Approve sending email? Reply 'approve'.")
    with patch.object(dm, "st", st):
        await dm.draw_messages(_agen([_tool_call_msg(), interrupt]), is_new=True)

    st.error.assert_not_called()
    st.stop.assert_not_called()
    st.warning.assert_called_once()
    assert "Approve" in st.warning.call_args.args[0]


@pytest.mark.asyncio
async def test_draw_messages_normal_tool_result_still_works():
    """A normal tool call -> tool result -> answer flow renders without error."""
    from langgraph_agent_toolkit.ui.components import draw_message as dm

    st = _mock_st()
    tool_result = ChatMessage(type="tool", content="sent", tool_call_id="c1")
    final = ChatMessage(type="ai", content="done")
    with patch.object(dm, "st", st):
        await dm.draw_messages(_agen([_tool_call_msg(), tool_result, final]), is_new=True)

    st.error.assert_not_called()
    st.stop.assert_not_called()
