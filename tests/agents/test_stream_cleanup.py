"""Check graph cleanup and state retention through the real executor."""

import asyncio
import gc
import weakref
from contextlib import aclosing, suppress
from typing import TypedDict

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.execution import execution_timeout, request_deadline_scope
from langgraph_agent_toolkit.core.memory.concurrency import ConversationCoordinator
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.core.settings import settings


def executor_for(graph):
    executor = AgentExecutor.__new__(AgentExecutor)
    executor.agents = {"test": Agent("test", "Local cleanup test", graph, EmptyObservability())}
    executor.concurrency = ConversationCoordinator(timeout=2)
    return executor


@pytest.mark.parametrize("stop", ["close", "cancel"])
async def test_stream_keeps_conversation_locked_until_graph_and_checkpoint_cleanup(stop, monkeypatch):
    # Async durability allows a checkpoint write to overlap node cleanup.
    monkeypatch.setattr(settings, "CHECKPOINT_DURABILITY", "async")
    first_received = asyncio.Event()
    close_requested = asyncio.Event()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()
    checkpoint_started = asyncio.Event()
    finish_checkpoint = asyncio.Event()
    checkpoint_finished = asyncio.Event()
    second_started = asyncio.Event()

    class GatedSaver(MemorySaver):
        async def aput(self, config, checkpoint, metadata, new_versions):
            if not checkpoint_started.is_set():
                checkpoint_started.set()
                await finish_checkpoint.wait()
                result = await super().aput(config, checkpoint, metadata, new_versions)
                checkpoint_finished.set()
                return result
            return await super().aput(config, checkpoint, metadata, new_versions)

    async def reply(state):
        if state["messages"][-1].content == "second":
            second_started.set()
            assert cleanup_finished.is_set()
            assert checkpoint_finished.is_set()
            return {"messages": [AIMessage("second finished")]}
        get_stream_writer()(AIMessage("partial answer"))
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await finish_cleanup.wait()
            cleanup_finished.set()

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    graph = builder.compile(checkpointer=GatedSaver())
    executor = executor_for(graph)

    async def consume():
        async with aclosing(executor.stream("test", {"message": "first"}, thread_id="shared")) as stream:
            assert (await anext(stream)).content == "partial answer"
            first_received.set()
            if stop == "close":
                await close_requested.wait()
            else:
                await anext(stream)

    owner = asyncio.create_task(consume())
    follower = None
    try:
        async with asyncio.timeout(5):
            await first_received.wait()
            await checkpoint_started.wait()
            if stop == "close":
                close_requested.set()
            else:
                owner.cancel()
            await cleanup_started.wait()
            follower = asyncio.create_task(executor.invoke("test", {"message": "second"}, thread_id="shared"))
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(follower), timeout=0.03)
            assert not owner.done()
            assert not second_started.is_set()

            finish_cleanup.set()
            await cleanup_finished.wait()
            assert not owner.done()
            assert not second_started.is_set()
            finish_checkpoint.set()
            if stop == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await owner
            else:
                await owner
            assert (await follower).content == "second finished"
            state = await graph.aget_state({"configurable": {"thread_id": "shared"}})
            assert state.values["messages"][-1].content == "second finished"
    finally:
        finish_cleanup.set()
        finish_checkpoint.set()
        for task in (owner, follower):
            if task is not None:
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


async def test_invoke_releases_replaced_state_before_the_graph_finishes():
    class State(TypedDict):
        messages: list
        step: int

    first_answer = None
    checked_retention = False

    async def replace(state):
        nonlocal first_answer, checked_retention
        step = state.get("step", 0) + 1
        if step == 6:
            gc.collect()
            assert first_answer() is None, "Invoke retained a replaced state until the end of the run."
            checked_retention = True
        message = AIMessage(f"answer {step}")
        if step == 1:
            first_answer = weakref.ref(message)
        return {"messages": [message], "step": step}

    builder = StateGraph(State)
    builder.add_node("replace", replace)
    builder.add_edge(START, "replace")
    builder.add_conditional_edges("replace", lambda state: END if state["step"] == 8 else "replace")

    result = await executor_for(builder.compile()).invoke("test", {"message": "start"})

    assert checked_retention
    assert result.content == "answer 8"


async def test_invoke_uses_the_declared_output_schema():
    class Output(TypedDict):
        messages: list

    class State(Output):
        structured_response: dict

    builder = StateGraph(State, output_schema=Output)
    builder.add_node(
        "reply", lambda state: {"messages": [AIMessage("public answer")], "structured_response": {"private": "data"}}
    )
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)

    result = await executor_for(builder.compile()).invoke("test", {"message": "start"})

    assert result.content == "public answer"


async def test_invoke_returns_interrupts_after_multiple_state_updates_and_resumes():
    builder = StateGraph(MessagesState)
    builder.add_node("draft", lambda state: {"messages": [AIMessage("draft answer")]})
    builder.add_node("review", lambda state: {"messages": [AIMessage("reviewed draft")]})

    def approve(state):
        response = interrupt("Approve the draft?")
        return {"messages": [AIMessage(f"Final answer: {response['message']}")]}

    builder.add_node("approve", approve)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "review")
    builder.add_edge("review", "approve")
    builder.add_edge("approve", END)
    executor = executor_for(builder.compile(checkpointer=MemorySaver()))

    paused = await executor.invoke("test", {"message": "start"}, thread_id="approval")
    assert paused.content == "Approve the draft?"
    assert len(paused.custom_data["interrupts"]) == 1
    assert paused.custom_data["interrupts"][0]["id"]

    completed = await executor.invoke("test", {"message": "approved"}, thread_id="approval")
    assert completed.content == "Final answer: approved"
    assert not completed.custom_data.get("interrupts")


@pytest.mark.parametrize("streaming", [False, True])
async def test_request_cancellation_does_not_trigger_a_second_executor_deadline(streaming, monkeypatch):
    monkeypatch.setattr(settings, "REQUEST_TIMEOUT", 0.03)
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def reply(state):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await finish_cleanup.wait()
            cleanup_finished.set()

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    executor = executor_for(builder.compile())

    async def request():
        with request_deadline_scope():
            if streaming:
                async with aclosing(executor.stream("test", {"message": "start"})) as stream:
                    await anext(stream)
            else:
                await executor.invoke("test", {"message": "start"})

    task = asyncio.create_task(request())
    try:
        async with asyncio.timeout(2):
            await started.wait()
            task.cancel()
            await cleanup_started.wait()
            # Cross the direct-executor deadline while graph cleanup remains active.
            await asyncio.sleep(0.06)
            assert not task.done()
            finish_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cleanup_finished.is_set()
    finally:
        finish_cleanup.set()
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def test_expired_request_scope_does_not_disable_a_child_execution_deadline():
    begin = asyncio.Event()

    async def delayed_child():
        await begin.wait()
        async with execution_timeout(0.01):
            await asyncio.Event().wait()

    with request_deadline_scope():
        task = asyncio.create_task(delayed_child())
    begin.set()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(task, timeout=1)
    assert not task.cancelled()
