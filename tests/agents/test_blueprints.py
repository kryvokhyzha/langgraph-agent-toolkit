import importlib

import pytest
from langchain_core.messages import HumanMessage
from langgraph.pregel import Pregel

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core.settings import settings


# These blueprints build an OpenAI model at import time, so they can only be smoke-tested when a
# model name is configured (e.g. via .env). In CI without OPENAI_MODEL_NAME they are skipped.
_needs_openai_model = pytest.mark.skipif(
    not settings.OPENAI_MODEL_NAME,
    reason="set OPENAI_MODEL_NAME to smoke-test blueprints that build an OpenAI model at import",
)

# The `react` blueprint hard-pins Langfuse and pushes a prompt at import time (needs LANGFUSE_*
# credentials), so it is not import-smoke-testable here.
IMPORTABLE_BLUEPRINTS = [
    "chatbot",
    "command_agent",
    "interrupt_agent",
    "bg_task_agent",
    "knowledge_base_agent",
    pytest.param("supervisor_agent", marks=_needs_openai_model),
    pytest.param("create_agent", marks=_needs_openai_model),
    pytest.param("create_agent_structured", marks=_needs_openai_model),
    pytest.param("hitl_agent", marks=_needs_openai_model),
]


@pytest.mark.parametrize("blueprint", IMPORTABLE_BLUEPRINTS)
def test_blueprint_module_imports_and_exposes_agent(blueprint):
    """Each importable blueprint compiles its graph and exposes an Agent at import time."""
    module = importlib.import_module(f"langgraph_agent_toolkit.agents.blueprints.{blueprint}.agent")

    agents = [v for v in vars(module).values() if isinstance(v, Agent)]
    assert agents, f"{blueprint} exposes no Agent instance"
    assert agents[0].graph is not None


async def test_command_agent_routes_and_produces_messages():
    """command_agent needs no model: node_a routes to node_b/node_c, each emitting a message."""
    from langgraph_agent_toolkit.agents.blueprints.command_agent.agent import command_agent

    result = await command_agent.graph.ainvoke({"messages": []})

    contents = [m.content for m in result["messages"]]
    assert len(contents) >= 2
    assert all("Hello" in c for c in contents)


async def test_chatbot_invokes_with_fake_model():
    """The chatbot entrypoint builds its model from config; model_provider='fake' yields the canned reply."""
    from langgraph_agent_toolkit.agents.blueprints.chatbot.agent import chatbot

    config = {"configurable": {"model_provider": "fake", "thread_id": "t1"}}
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="hi")]}, config=config)

    assert result["messages"][-1].content == "This is a test response from the fake model."


def test_create_react_agent_builds_compiled_graph():
    """The custom create_react_agent factory returns a compiled (Pregel) graph."""
    from langgraph_agent_toolkit.agents.components.checkpoint.empty import NoOpSaver
    from langgraph_agent_toolkit.agents.components.creators.create_react_agent import create_react_agent
    from langgraph_agent_toolkit.agents.components.tools import add, multiply
    from langgraph_agent_toolkit.agents.components.utils import AgentStateWithRemainingSteps
    from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory
    from langgraph_agent_toolkit.schema.models import ModelProvider

    model = CompletionModelFactory.create(ModelProvider.FAKE)
    graph = create_react_agent(
        model=model,
        tools=[add, multiply],
        state_schema=AgentStateWithRemainingSteps,
        checkpointer=NoOpSaver(),
    )

    assert isinstance(graph, Pregel)


async def test_create_react_agent_invokes_with_fake_model():
    """A FAKE model returns no tool calls, so the react loop ends with the canned reply."""
    from langgraph_agent_toolkit.agents.components.checkpoint.empty import NoOpSaver
    from langgraph_agent_toolkit.agents.components.creators.create_react_agent import create_react_agent
    from langgraph_agent_toolkit.agents.components.tools import add, multiply
    from langgraph_agent_toolkit.agents.components.utils import AgentStateWithRemainingSteps
    from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory
    from langgraph_agent_toolkit.schema.models import ModelProvider

    model = CompletionModelFactory.create(ModelProvider.FAKE)
    graph = create_react_agent(
        model=model,
        tools=[add, multiply],
        state_schema=AgentStateWithRemainingSteps,
        checkpointer=NoOpSaver(),
    )

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "t1"}},
    )

    assert result["messages"][-1].content == "This is a test response from the fake model."
