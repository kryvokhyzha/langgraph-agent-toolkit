from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from pydantic import BaseModel

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.schema import ChatMessage


class MockInput(BaseModel):
    """Define test input."""

    message: str


@pytest.fixture
def mock_agent(mock_state_snapshot):
    """Create a mock agent."""
    agent = Mock(spec=Agent)
    agent.name = "test-agent"
    agent.description = "A test agent"

    graph = AsyncMock()
    graph.ainvoke = AsyncMock()
    graph.astream = AsyncMock()
    graph.aget_state = AsyncMock(return_value=mock_state_snapshot(values={"messages": []}, tasks=[]))
    agent.graph = graph

    agent.observability = Mock()
    agent.observability.get_callback_handler = Mock(return_value=None)

    @contextmanager
    def mock_trace_context(*args, **kwargs):
        yield MagicMock()

    agent.observability.trace_context = mock_trace_context

    return agent


@pytest.fixture
def agent_executor(mock_agent):
    """Create an `AgentExecutor` with a mock agent."""
    default_agent = Mock(spec=Agent)
    default_agent.name = settings.DEFAULT_AGENT
    default_agent.description = "Default test agent"

    with patch.object(AgentExecutor, "load_agents_from_imports"):
        with patch.object(AgentExecutor, "_validate_default_agent_loaded"):
            executor = AgentExecutor("dummy_import:dummy_agent")
            executor.agents = {
                "test-agent": mock_agent,
                settings.DEFAULT_AGENT: default_agent,
            }
            return executor


@pytest.mark.asyncio
async def test_invoke_basic_flow(agent_executor, mock_agent):
    """Verify a successful `invoke` response."""
    mock_response = [("values", {"messages": [AIMessage(content="Test response")]})]
    mock_agent.graph.astream = Mock(side_effect=_astream(mock_response))

    input_obj = MockInput(message="Hello, agent!")
    result = await agent_executor.invoke(
        agent_id="test-agent",
        input=input_obj,
        thread_id="test-thread",
        user_id="test-user",
    )

    assert isinstance(result, ChatMessage)
    assert result.type == "ai"
    assert result.content == "Test response"
    assert result.run_id is not None

    call_args = mock_agent.graph.astream.call_args[1]
    config = call_args["config"]
    assert config["configurable"]["thread_id"] == "test-thread"
    assert config["configurable"]["user_id"] == "test-user"


@pytest.mark.asyncio
async def test_invoke_with_interrupt_handling(agent_executor, mock_agent, mock_state_snapshot):
    """Verify that `invoke` handles interrupts."""
    with patch.object(settings, "CHECK_INTERRUPTS", True):
        mock_agent.graph.checkpointer = Mock()

        interrupt_task = Mock()
        interrupt_task.interrupts = [Mock()]
        mock_agent.graph.aget_state.return_value = mock_state_snapshot(values={"messages": []}, tasks=[interrupt_task])

        mock_response = [("values", {"__interrupt__": [Mock(value="Need more info")]})]
        mock_agent.graph.astream = Mock(side_effect=_astream(mock_response))

        user_input = MockInput(message="Continue")
        result = await agent_executor.invoke(agent_id="test-agent", input=user_input)

        assert result.content == "Need more info"

        call_args = mock_agent.graph.astream.call_args[1]
        assert isinstance(call_args["input"], Command)
        assert call_args["input"].resume == user_input.model_dump()


def _real_executor(agents: dict) -> AgentExecutor:
    """Build an `AgentExecutor` with real agents."""
    with (
        patch.object(AgentExecutor, "load_agents_from_imports"),
        patch.object(AgentExecutor, "_validate_default_agent_loaded"),
    ):
        ex = AgentExecutor("dummy:dummy")
    ex.agents = agents
    return ex


def _real_agent(name: str, graph):
    from langgraph_agent_toolkit.agents.agent import Agent
    from langgraph_agent_toolkit.core.observability.empty import EmptyObservability

    agent = Agent(name=name, description="d", graph=graph)
    agent.observability = EmptyObservability()
    return agent


@pytest.mark.asyncio
async def test_invoke_interrupt_then_resume_real_graph():
    """Verify that an interrupting graph resumes on the next call."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.types import interrupt

    from langgraph_agent_toolkit.schema.schema import UserComplexInput

    def ask(state):
        reply = interrupt("What is your birthdate?")
        return {"messages": [AIMessage(f"Got it: {reply['message']}")]}

    g = StateGraph(MessagesState)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    ex = _real_executor({"int": _real_agent("int", g.compile(checkpointer=MemorySaver()))})

    with patch.object(settings, "CHECK_INTERRUPTS", True):
        r1 = await ex.invoke(agent_id="int", input=UserComplexInput(message="my sign?"), thread_id="th")
        assert r1.content == "What is your birthdate?"
        r2 = await ex.invoke(agent_id="int", input=UserComplexInput(message="1990-05-15"), thread_id="th")
        assert r2.content == "Got it: 1990-05-15"


@pytest.mark.asyncio
async def test_stream_surfaces_structured_response():
    """Verify that streaming returns `structured_response`."""
    from typing import Annotated, TypedDict

    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    from langgraph_agent_toolkit.schema.schema import UserComplexInput

    class SO(BaseModel):
        answer: str

    class SOState(TypedDict):
        messages: Annotated[list, add_messages]
        structured_response: SO

    def node(state):
        return {"messages": [AIMessage("text")], "structured_response": SO(answer="42")}

    g = StateGraph(SOState)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    ex = _real_executor({"so": _real_agent("so", g.compile())})

    out = [
        m
        async for m in ex.stream(agent_id="so", input=UserComplexInput(message="q"), thread_id="t", stream_tokens=False)
    ]
    contents = [m.content for m in out]
    assert any(isinstance(c, dict) and c.get("answer") == "42" for c in contents), contents


@pytest.mark.asyncio
async def test_error_handling_with_recursion_error(agent_executor, mock_agent):
    """Verify that the error handler raises `GraphRecursionError`."""
    mock_agent.graph.astream = Mock(side_effect=GraphRecursionError("Recursion limit exceeded"))

    with pytest.raises(GraphRecursionError, match="Recursion limit exceeded"):
        await agent_executor.invoke(agent_id="test-agent", input=MockInput(message="Test"))


@pytest.mark.asyncio
async def test_setup_agent_execution_configuration(agent_executor, mock_agent):
    """Verify that `_setup_agent_execution` configures the agent."""
    input_obj = MockInput(message="Hello")

    agent, input_data, config, run_id = await agent_executor._setup_agent_execution(
        agent_id="test-agent",
        input=input_obj,
        thread_id="test-thread",
        user_id="test-user",
        model_name="test-model",
        agent_config={"temperature": 0.7},
        recursion_limit=50,
    )

    assert agent is mock_agent
    assert isinstance(input_data, dict)
    assert "messages" in input_data
    assert isinstance(input_data["messages"][0], HumanMessage)

    assert config["configurable"]["thread_id"] == "test-thread"
    assert config["configurable"]["user_id"] == "test-user"
    assert config["configurable"]["model_name"] == "test-model"
    assert config["configurable"]["temperature"] == 0.7
    assert config["recursion_limit"] == 50
    assert isinstance(run_id, UUID)


@pytest.mark.asyncio
async def test_setup_agent_execution_multimodal_message(agent_executor, mock_agent):
    """Verify that content blocks become a `HumanMessage` list."""
    from langgraph_agent_toolkit.schema.schema import UserComplexInput

    blocks = [
        {"type": "text", "text": "Describe this image."},
        {"type": "image", "url": "https://example.com/cat.jpg"},
        {"type": "file", "base64": "QUJD", "mime_type": "application/pdf"},
    ]
    _, input_data, _, _ = await agent_executor._setup_agent_execution(
        agent_id="test-agent",
        input=UserComplexInput(message=blocks),
        thread_id="t",
        user_id="u",
        model_name="m",
        agent_config={},
        recursion_limit=50,
    )

    human = input_data["messages"][0]
    assert isinstance(human, HumanMessage)
    assert isinstance(human.content, list)
    assert [b["type"] for b in human.content_blocks] == ["text", "image", "file"]


def test_user_complex_input_accepts_and_validates_content_blocks():
    """Verify that `UserComplexInput` validates content blocks."""
    from pydantic import ValidationError

    from langgraph_agent_toolkit.schema.schema import UserComplexInput

    assert UserComplexInput(message="hi").message == "hi"
    assert len(UserComplexInput(message=[{"type": "image", "url": "https://x/y.jpg"}]).message) == 1
    assert UserComplexInput(message=[{"type": "image", "file_id": "file-abc"}]).message
    assert UserComplexInput(message=[{"type": "text", "text": "hello"}]).message
    assert UserComplexInput(message=[{"type": "file", "base64": "QUJD", "mime_type": "application/pdf"}]).message

    for bad in (
        [{"type": "hologram"}],
        [{"text": "no type"}],
        ["not-a-dict"],
        [{"type": "image"}],
        [{"type": "text"}],
        [{"type": "image", "base64": "QUJD"}],
    ):
        with pytest.raises(ValidationError):
            UserComplexInput(message=bad)


def test_user_complex_input_enforces_attachment_limit():
    """Reject excess attachments when `MULTIMODAL_MAX_ATTACHMENTS` is set."""
    from pydantic import ValidationError

    from langgraph_agent_toolkit.schema.schema import UserComplexInput

    three_images = [{"type": "text", "text": "look"}]
    three_images += [{"type": "image", "url": f"https://x/{i}.jpg"} for i in range(3)]
    with patch.object(settings, "MULTIMODAL_MAX_ATTACHMENTS", 2):
        with pytest.raises(ValidationError):
            UserComplexInput(message=three_images)
    with patch.object(settings, "MULTIMODAL_MAX_ATTACHMENTS", 5):
        assert len(UserComplexInput(message=three_images).message) == 4


@pytest.mark.asyncio
async def test_trace_context_integration(agent_executor, mock_agent):
    """Verify that `invoke` uses `trace_context`."""
    mock_response = [("values", {"messages": [AIMessage(content="Response")]})]
    mock_agent.graph.astream = Mock(side_effect=_astream(mock_response))

    trace_context_called = False
    original_trace_context = mock_agent.observability.trace_context

    @contextmanager
    def tracked_trace_context(*args, **kwargs):
        nonlocal trace_context_called
        trace_context_called = True
        with original_trace_context(*args, **kwargs) as span:
            yield span

    mock_agent.observability.trace_context = tracked_trace_context

    input_obj = MockInput(message="Test")
    await agent_executor.invoke(
        agent_id="test-agent",
        input=input_obj,
        thread_id="test-thread",
        user_id="test-user",
    )

    assert trace_context_called


def test_agent_management_operations(mock_agent):
    """Verify agent add and get operations."""
    with patch.object(AgentExecutor, "load_agents_from_imports"):
        with patch.object(AgentExecutor, "_validate_default_agent_loaded"):
            executor = AgentExecutor("dummy_import:dummy_agent")

            default_agent = Mock(spec=Agent)
            default_agent.name = settings.DEFAULT_AGENT
            default_agent.description = "Default agent"
            executor.agents = {settings.DEFAULT_AGENT: default_agent}

            executor.add_agent("test-agent", mock_agent)
            assert "test-agent" in executor.agents

            agent = executor.get_agent("test-agent")
            assert agent is mock_agent

            agent_info = executor.get_all_agent_info()
            agent_keys = [info.key for info in agent_info]
            assert "test-agent" in agent_keys
            assert settings.DEFAULT_AGENT in agent_keys

            with pytest.raises(KeyError):
                executor.get_agent("nonexistent-agent")


def _astream(events):
    """Build an async generator that yields the specified events."""

    async def _gen(*args, **kwargs):
        for event in events:
            yield event

    return _gen


async def test_stream_updates_emits_message(agent_executor, mock_agent):
    """Verify that an `updates` event yields a `ChatMessage`."""
    mock_agent.graph.astream = _astream([("updates", {"agent": {"messages": [AIMessage(content="hello")]}})])

    out = [m async for m in agent_executor.stream(agent_id="test-agent", input=MockInput(message="hi"))]

    assert len(out) == 1
    assert out[0].type == "ai"
    assert out[0].content == "hello"


async def test_stream_supervisor_keeps_only_last_ai_message(agent_executor, mock_agent):
    """Verify that the supervisor emits its last `AIMessage`."""
    mock_agent.graph.astream = _astream(
        [("updates", {"supervisor": {"messages": [AIMessage(content="a"), AIMessage(content="b")]}})]
    )

    out = [m async for m in agent_executor.stream(agent_id="test-agent", input=MockInput(message="hi"))]

    assert [m.content for m in out] == ["b"]


async def test_stream_expert_node_becomes_tool_message(agent_executor, mock_agent):
    """Verify that expert updates become tool messages."""
    mock_agent.graph.astream = _astream(
        [("updates", {"research_expert": {"messages": [AIMessage(content="research result")]}})]
    )

    out = [m async for m in agent_executor.stream(agent_id="test-agent", input=MockInput(message="hi"))]

    assert len(out) == 1
    assert out[0].type == "tool"
    assert out[0].content == "research result"


async def test_stream_interrupt_yields_ai_message(agent_executor, mock_agent):
    """Verify that an interrupt update yields an `AIMessage`."""
    mock_agent.graph.astream = _astream([("updates", {"__interrupt__": [Mock(value="need input")]})])

    out = [m async for m in agent_executor.stream(agent_id="test-agent", input=MockInput(message="hi"))]

    assert len(out) == 1
    assert out[0].content == "need input"


async def test_stream_tokens_filters_skip_stream_and_non_chunks(agent_executor, mock_agent):
    """Verify that `messages` mode yields valid token strings."""
    mock_agent.graph.astream = _astream(
        [
            ("messages", (AIMessageChunk(content="hi"), {"tags": []})),
            ("messages", (AIMessageChunk(content="skip"), {"tags": ["skip_stream"]})),
            ("messages", (HumanMessage(content="ignored"), {"tags": []})),
        ]
    )

    out = [m async for m in agent_executor.stream(agent_id="test-agent", input=MockInput(message="hi"))]

    assert out == ["hi"]


async def test_stream_reassembles_tuple_parts_into_message(agent_executor, mock_agent):
    """Verify that tuple parts form one `AIMessage`."""
    mock_agent.graph.astream = _astream([("updates", {"agent": {"messages": [("content", "assembled")]}})])

    out = [m async for m in agent_executor.stream(agent_id="test-agent", input=MockInput(message="hi"))]

    assert len(out) == 1
    assert out[0].type == "ai"
    assert out[0].content == "assembled"
