from langchain_core.messages import HumanMessage


async def test_command_agent_routes_and_produces_messages():
    """Verify that `command_agent` emits a message."""
    from langgraph_agent_toolkit.agents.blueprints.command_agent.agent import command_agent

    result = await command_agent.graph.ainvoke({"messages": []})

    contents = [m.content for m in result["messages"]]
    assert len(contents) >= 2
    assert all("Hello" in c for c in contents)


async def test_chatbot_invokes_with_fake_model():
    """Verify that the chatbot uses the configured fake model."""
    from langgraph_agent_toolkit.agents.blueprints.chatbot.agent import chatbot

    config = {"configurable": {"model_provider": "fake", "thread_id": "t1"}}
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="hi")]}, config=config)

    assert result["messages"][-1].content == "This is a test response from the fake model."


async def test_create_react_agent_invokes_with_fake_model():
    """Verify that a fake model ends the ReAct loop without tool calls."""
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
