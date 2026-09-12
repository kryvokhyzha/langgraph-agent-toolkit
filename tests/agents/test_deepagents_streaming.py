"""Keep internal summaries and retained history out of the response stream."""

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware as LangChainSummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Overwrite

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.memory.concurrency import ConversationCoordinator
from langgraph_agent_toolkit.core.models.fake import FakeToolModel
from langgraph_agent_toolkit.schema import ChatMessage


def executor_for(graph):
    executor = object.__new__(AgentExecutor)
    executor.agents = {"summary-test": Agent("summary-test", "A summary test agent.", graph)}
    executor.concurrency = ConversationCoordinator()
    return executor


class OverwriteSummary(LangChainSummarizationMiddleware):
    """Use LangGraph's replacement value for the same upstream summary update."""

    @property
    def name(self):
        return "SummarizationMiddleware"

    async def abefore_model(self, state, runtime):
        update = await super().abefore_model(state, runtime)
        if update is not None:
            update["messages"] = Overwrite(
                [message for message in update["messages"] if not isinstance(message, RemoveMessage)]
            )
        return update


@pytest.mark.parametrize("kind", ["deepagents", "langchain", "deepagents-langchain", "langchain-overwrite"])
@pytest.mark.parametrize("stream_tokens", [True, False])
async def test_summary_model_output_and_retained_answers_stay_internal(kind, stream_tokens):
    model = FakeToolModel(responses=["PUBLIC ANSWER"])
    summary_model = FakeToolModel(responses=["INTERNAL SUMMARY"])
    arguments = {"model": summary_model, "trigger": ("messages", 4), "keep": ("messages", 3)}
    if kind.startswith("deepagents"):
        pytest.importorskip("deepagents")
        from deepagents import create_deep_agent
        from deepagents.backends import StateBackend
        from deepagents.middleware.summarization import SummarizationMiddleware

        backend = StateBackend()
        summary = (
            SummarizationMiddleware(backend=backend, **arguments)
            if kind == "deepagents"
            else LangChainSummarizationMiddleware(**arguments)
        )
        graph = create_deep_agent(
            model=model,
            backend=backend,
            middleware=[summary],
            checkpointer=MemorySaver(),
        )
    else:
        graph = create_agent(
            model=model,
            middleware=[
                (OverwriteSummary if kind == "langchain-overwrite" else LangChainSummarizationMiddleware)(**arguments)
            ],
            checkpointer=MemorySaver(),
        )
    config = {"configurable": {"thread_id": "summary-thread"}}
    history = [
        HumanMessage("first question", id="human-1"),
        AIMessage("first old answer", id="ai-1"),
        HumanMessage("second question", id="human-2"),
        AIMessage("second old answer", id="ai-2"),
    ]
    await graph.aupdate_state(config, {"messages": history}, as_node="model")

    events = [
        event
        async for event in executor_for(graph).stream(
            "summary-test", {"message": "current question"}, thread_id="summary-thread", stream_tokens=stream_tokens
        )
    ]

    state = await graph.aget_state(config)
    if kind == "deepagents":
        assert "INTERNAL SUMMARY" in state.values["_summarization_event"]["summary_message"].content
        assert [message.content for message in state.values["messages"][:4]] == [message.content for message in history]
    else:
        assert "INTERNAL SUMMARY" in state.values["messages"][0].content
    assert [event.content for event in events if isinstance(event, ChatMessage)] == ["PUBLIC ANSWER"]
    assert "".join(event for event in events if isinstance(event, str)) == ("PUBLIC ANSWER" if stream_tokens else "")


async def test_removing_a_message_does_not_hide_a_new_answer_from_another_node():
    async def replace_answer(state):
        return {"messages": [RemoveMessage(id="old-answer"), AIMessage("NEW PUBLIC ANSWER")]}

    builder = StateGraph(MessagesState)
    builder.add_node("replace_answer", replace_answer)
    builder.add_edge(START, "replace_answer")
    builder.add_edge("replace_answer", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "replace-thread"}}
    await graph.aupdate_state(config, {"messages": [AIMessage("old", id="old-answer")]}, as_node="replace_answer")

    events = [
        event
        async for event in executor_for(graph).stream(
            "summary-test", {"message": "replace it"}, thread_id="replace-thread", stream_tokens=False
        )
    ]

    assert [event.content for event in events] == ["NEW PUBLIC ANSWER"]
    state = await graph.aget_state(config)
    assert [message.content for message in state.values["messages"]] == ["replace it", "NEW PUBLIC ANSWER"]
