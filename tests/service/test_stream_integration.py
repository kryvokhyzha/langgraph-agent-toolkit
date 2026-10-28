"""Check message conversion through the graph, HTTP route, and client parser."""

import sys

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import StreamWriter

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.agents.blueprints.bg_task_agent.utils import CustomData
from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.schema.schema import ChatMessage


@pytest.mark.parametrize("protocol", ["stream", "stream_jsonl"])
@pytest.mark.parametrize("stream_tokens", [True, False])
def test_custom_and_tool_messages_cross_http_boundary(app, monkeypatch, protocol, stream_tokens):
    call = ToolCall(name="test_tool", args={"arg1": "value1"}, id="test_call_id")

    async def reply(state: MessagesState, writer: StreamWriter):
        CustomData(type="start", data={"key1": "value1", "key2": 123}).dispatch(writer)
        response = await FakeListChatModel(responses=["The answer is 42"]).ainvoke(state["messages"])
        return {
            "messages": [
                AIMessage(content="", tool_calls=[call]),
                ToolMessage(content="42", tool_call_id="test_call_id"),
                response,
                CustomData(type="end", data={"time": "end"}).to_langchain(),
            ]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.set_entry_point("reply")
    builder.add_edge("reply", END)
    agent = Agent(name="static-agent", description="A static agent.", graph=builder.compile(checkpointer=MemorySaver()))
    monkeypatch.setattr(sys.modules[__name__], "static_agent", agent, raising=False)
    monkeypatch.setattr("langgraph_agent_toolkit.helper.constants._runtime_default_agent", agent.name)
    app.state.agent_executor = AgentExecutor(f"{__name__}:static_agent")

    # Inject the executor. The separate process tests cover service lifespan.
    http = TestClient(app, raise_server_exceptions=False)
    try:
        with AgentClient(agent=agent.name, get_info=False, http_client=http) as client:
            events = list(
                getattr(client, protocol)(
                    {"message": "Test message"}, thread_id="stream-contract", stream_tokens=stream_tokens
                )
            )
    finally:
        http.close()

    messages = [event for event in events if isinstance(event, ChatMessage)]
    tokens = "".join(event for event in events if isinstance(event, str))
    assert tokens == ("The answer is 42" if stream_tokens else "")
    assert len(messages) == 5
    assert all(message.run_id for message in messages)
    assert {message.thread_id for message in messages} == {"stream-contract"}
    assert [message.model_copy(update={"run_id": None, "thread_id": None}) for message in messages] == [
        ChatMessage(type="custom", content="", custom_data={"key1": "value1", "key2": 123}),
        ChatMessage(type="ai", content="", tool_calls=[{**call, "type": "tool_call"}]),
        ChatMessage(type="tool", content="42", tool_call_id="test_call_id", tool_status="success"),
        ChatMessage(type="ai", content="The answer is 42"),
        ChatMessage(type="custom", content="", custom_data={"time": "end"}),
    ]
