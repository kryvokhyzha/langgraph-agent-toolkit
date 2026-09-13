"""Test the mocked Streamlit `draw_messages` renderer."""

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
    """Render an AI HITL interrupt after a tool call without an error."""
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
    """Render a tool call, tool result, and answer without an error."""
    from langgraph_agent_toolkit.ui.components import draw_message as dm

    st = _mock_st()
    tool_result = ChatMessage(type="tool", content="sent", tool_call_id="c1")
    final = ChatMessage(type="ai", content="done")
    with patch.object(dm, "st", st):
        await dm.draw_messages(_agen([_tool_call_msg(), tool_result, final]), is_new=True)

    st.error.assert_not_called()
    st.stop.assert_not_called()


@pytest.mark.parametrize("show_tools", [False, True])
async def test_tool_results_can_follow_interleaved_progress_without_losing_messages(show_tools):
    """Consume each event once when tools emit progress before their results."""
    from langgraph_agent_toolkit.ui.components import draw_message as dm

    st = _mock_st()
    st.session_state.display_tools_execution = show_tools
    calls = ChatMessage(
        type="ai",
        content="",
        tool_calls=[
            {"name": "lookup", "args": {}, "id": "c1"},
            {"name": "lookup", "args": {}, "id": "c2"},
        ],
    )
    progress = ChatMessage(type="custom", content="", custom_data={"name": "lookup", "state": "running"})
    second_result = ChatMessage(type="tool", content="second result", tool_call_id="c2")
    first_result = ChatMessage(type="tool", content="first result", tool_call_id="c1")
    final = ChatMessage(type="ai", content="complete answer")
    messages = [calls, progress, second_result, first_result, final]
    with patch.object(dm, "st", st), patch.object(dm, "TaskDataStatus") as task_status:
        await dm.draw_messages(
            _agen([calls, progress, "progress token", second_result, first_result, final]), is_new=True
        )

    st.error.assert_not_called()
    st.stop.assert_not_called()
    assert st.session_state.messages == messages
    task_status.return_value.add_and_draw_task_data.assert_called_once()
    if show_tools:
        assert st.status.return_value.code.call_count == 2


@pytest.mark.parametrize("custom_data", [{"state": "paused"}, {"progress": 50}])
async def test_generic_custom_payload_does_not_stop_tool_results_or_final_answer(custom_data):
    """Keep custom schemas and continue consuming the stream."""
    from langgraph_agent_toolkit.ui.components import draw_message as dm

    st = _mock_st()
    progress = ChatMessage(type="custom", content="", custom_data=custom_data)
    result = ChatMessage(type="tool", content="tool result", tool_call_id="c1")
    final = ChatMessage(type="ai", content="complete answer")
    messages = [_tool_call_msg(), progress, result, final]
    with patch.object(dm, "st", st), patch.object(dm, "TaskDataStatus") as task_status:
        await dm.draw_messages(_agen(messages), is_new=True)

    st.error.assert_not_called()
    st.stop.assert_not_called()
    task_status.assert_not_called()
    st.write.assert_any_call(custom_data)
    assert st.session_state.messages == messages
