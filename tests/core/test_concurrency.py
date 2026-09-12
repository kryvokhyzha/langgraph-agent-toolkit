"""Verify conversation locks and SQLite coordination without external services."""

import asyncio
import sys
from contextlib import suppress

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from pydantic import BaseModel

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.memory.concurrency import (
    ConversationBusyError,
    ConversationCoordinator,
    SQLiteConversationCoordinator,
)
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.core.settings import settings


class Input(BaseModel):
    message: str


def make_executor(graph, coordinator):
    executor = AgentExecutor.__new__(AgentExecutor)
    executor.agents = {"test": Agent("test", "Local test", graph, EmptyObservability())}
    executor.concurrency = coordinator
    return executor


def message_graph(saver, node):
    builder = StateGraph(MessagesState)
    builder.add_node("reply", node)
    builder.set_entry_point("reply")
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=saver)


@pytest.mark.asyncio
async def test_same_key_is_serialized_and_different_keys_can_overlap():
    coordinator = ConversationCoordinator(timeout=1)
    entered = asyncio.Event()
    release = asyncio.Event()
    order = []

    async def first():
        async with coordinator.lock("shared"):
            order.append("first")
            entered.set()
            await release.wait()

    async def second():
        async with coordinator.lock("shared"):
            order.append("second")

    task = asyncio.create_task(first())
    await entered.wait()
    waiter = asyncio.create_task(second())
    async with coordinator.lock("independent"):
        await asyncio.sleep(0)
        assert order == ["first"]
    release.set()
    await asyncio.gather(task, waiter)
    assert order == ["first", "second"]
    assert coordinator._entries == {}


@pytest.mark.asyncio
async def test_cancelled_holder_releases_lock_and_removes_queue_entry():
    coordinator = ConversationCoordinator(timeout=1)
    entered = asyncio.Event()

    async def hold():
        async with coordinator.lock("shared"):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with coordinator.lock("shared"):
        pass
    assert coordinator._entries == {}


@pytest.mark.asyncio
async def test_queue_limit_and_wait_timeout_do_not_leak_entries():
    coordinator = ConversationCoordinator(timeout=0.03, max_waiters=1)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold():
        async with coordinator.lock("shared"):
            entered.set()
            await release.wait()

    holder = asyncio.create_task(hold())
    await entered.wait()
    try:
        waiter = asyncio.create_task(coordinator.lock("shared").__aenter__())
        await asyncio.sleep(0)
        with pytest.raises(ConversationBusyError, match="full"):
            async with coordinator.lock("shared"):
                pass
        with pytest.raises(ConversationBusyError, match="expired"):
            await waiter
    finally:
        release.set()
        await holder
    assert coordinator._entries == {}


@pytest.mark.asyncio
async def test_sqlite_workers_keep_both_conversation_turns(tmp_path):
    path = str(tmp_path / "checkpoints.sqlite")

    async def reply(state):
        await asyncio.sleep(0.04)
        return {"messages": [AIMessage("reply to " + state["messages"][-1].content)]}

    async with AsyncSqliteSaver.from_conn_string(path) as saver_one:
        await saver_one.setup()
        async with AsyncSqliteSaver.from_conn_string(path) as saver_two:
            first = make_executor(message_graph(saver_one, reply), SQLiteConversationCoordinator(path, timeout=2))
            second = make_executor(message_graph(saver_two, reply), SQLiteConversationCoordinator(path, timeout=2))
            await asyncio.gather(
                first.invoke("test", Input(message="one"), thread_id="shared"),
                second.invoke("test", Input(message="two"), thread_id="shared"),
            )
            state = await first.agents["test"].graph.aget_state({"configurable": {"thread_id": "shared"}})
            assert {m.content for m in state.values["messages"]} == {"one", "two", "reply to one", "reply to two"}


@pytest.mark.asyncio
async def test_sqlite_process_termination_releases_file_lock(tmp_path):
    path = str(tmp_path / "process.sqlite")
    code = """
import asyncio, sys, dotenv
dotenv.find_dotenv = lambda *args, **kwargs: ''
from langgraph_agent_toolkit.core.memory.concurrency import SQLiteConversationCoordinator
async def main():
    async with SQLiteConversationCoordinator(sys.argv[1], timeout=3).lock('shared'):
        print('READY', flush=True)
        await asyncio.Event().wait()
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c", code, path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        assert await asyncio.wait_for(process.stdout.readline(), timeout=10) == b"READY\n"
        coordinator = SQLiteConversationCoordinator(path, timeout=0.05)
        with pytest.raises(ConversationBusyError):
            async with coordinator.lock("shared"):
                pass
        process.kill()
        await asyncio.wait_for(process.wait(), timeout=5)
        async with SQLiteConversationCoordinator(path, timeout=1).lock("shared"):
            pass
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_sqlite_cancellation_releases_file_lock(tmp_path):
    path = str(tmp_path / "cancel.sqlite")
    coordinator = SQLiteConversationCoordinator(path, timeout=1)
    entered = asyncio.Event()

    async def hold():
        async with coordinator.lock("shared"):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with SQLiteConversationCoordinator(path, timeout=1).lock("shared"):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_executor_timeout_stops_node_and_releases_conversation(monkeypatch, stream):
    monkeypatch.setattr(settings, "REQUEST_TIMEOUT", 0.5)
    stopped = asyncio.Event()

    async def reply(state):
        if state["messages"][-1].content == "slow":
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        return {"messages": [AIMessage("done")]}

    coordinator = ConversationCoordinator(timeout=1)
    executor = make_executor(message_graph(MemorySaver(), reply), coordinator)
    with pytest.raises(TimeoutError):
        if stream:
            async for _ in executor.stream("test", Input(message="slow"), thread_id="shared"):
                pass
        else:
            await executor.invoke("test", Input(message="slow"), thread_id="shared")
    assert stopped.is_set()
    assert coordinator._entries == {}
    # The second call checks recovery, not execution speed under coverage.
    monkeypatch.setattr(settings, "REQUEST_TIMEOUT", 5)
    result = await executor.invoke("test", Input(message="fast"), thread_id="shared")
    assert result.content == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_executor_cancellation_stops_node_and_releases_conversation(stream):
    entered = asyncio.Event()
    stopped = asyncio.Event()

    async def reply(state):
        if state["messages"][-1].content == "slow":
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        return {"messages": [AIMessage("done")]}

    coordinator = ConversationCoordinator(timeout=1)
    executor = make_executor(message_graph(MemorySaver(), reply), coordinator)

    async def run():
        if stream:
            async for _ in executor.stream("test", Input(message="slow"), thread_id="shared"):
                pass
        else:
            await executor.invoke("test", Input(message="slow"), thread_id="shared")

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
    assert coordinator._entries == {}
    result = await executor.invoke("test", Input(message="fast"), thread_id="shared")
    assert result.content == "done"
