"""Check durable state and failure reporting with real graphs and SQLite."""

import asyncio
from contextlib import aclosing, suppress

import pytest
from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor, get_graph_history
from langgraph_agent_toolkit.core.memory.concurrency import ConversationCoordinator
from langgraph_agent_toolkit.core.memory.coordinated_sqlite import CoordinatedSqliteSaver
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.exceptions import UnsupportedMessageTypeError


def executor_for(graph):
    executor = AgentExecutor.__new__(AgentExecutor)
    executor.agents = {"test": Agent("test", "A checkpoint test agent.", graph)}
    executor.concurrency = ConversationCoordinator(timeout=1)
    return executor


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("write_fails", [False, True])
@pytest.mark.parametrize("durability", ["sync", "async"])
async def test_checkpoint_durability_controls_step_progress_and_waits_for_pending_writes(
    tmp_path, monkeypatch, streaming, write_fails, durability
):
    monkeypatch.setattr(settings, "CHECKPOINT_DURABILITY", durability)
    write_started = asyncio.Event()
    allow_write = asyncio.Event()
    write_finished = asyncio.Event()
    next_node_started = asyncio.Event()

    class GatedSaver(CoordinatedSqliteSaver):
        async def aput(self, config, checkpoint, metadata, new_versions):
            if metadata["step"] == 1:
                write_started.set()
                await allow_write.wait()
                if write_fails:
                    raise OSError("Synthetic checkpoint write failure")
                result = await super().aput(config, checkpoint, metadata, new_versions)
                write_finished.set()
                return result
            return await super().aput(config, checkpoint, metadata, new_versions)

    async def finish(state):
        next_node_started.set()
        if durability == "sync":
            assert write_finished.is_set(), "The next node started before the previous checkpoint was saved."
        return {"messages": [AIMessage("finished")]}

    builder = StateGraph(MessagesState)
    builder.add_node("draft", lambda state: {"messages": [AIMessage("draft")]})
    builder.add_node("finish", finish)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "finish")
    builder.add_edge("finish", END)
    database = str(tmp_path / "durability.sqlite")
    config = {"configurable": {"thread_id": "durability"}}
    async with GatedSaver.from_conn_string(database) as saver:
        graph = builder.compile(checkpointer=saver)
        executor = executor_for(graph)

        async def consume():
            if streaming:
                async with aclosing(executor.stream("test", {"message": "start"}, thread_id="durability")) as events:
                    return [event async for event in events]
            return await executor.invoke("test", {"message": "start"}, thread_id="durability")

        task = asyncio.create_task(consume())
        try:
            async with asyncio.timeout(3):
                await write_started.wait()
                if durability == "sync":
                    with pytest.raises(TimeoutError):
                        await asyncio.wait_for(next_node_started.wait(), timeout=0.03)
                else:
                    await next_node_started.wait()
                    assert not write_finished.is_set()
                    with pytest.raises(TimeoutError):
                        await asyncio.wait_for(asyncio.shield(task), timeout=0.03)
                assert not task.done()
                allow_write.set()
                if write_fails:
                    with pytest.raises(OSError, match="Synthetic checkpoint write failure"):
                        await task
                    assert next_node_started.is_set() == (durability == "async")
                else:
                    result = await task
                    assert (result[-1] if streaming else result).content == "finished"
        finally:
            allow_write.set()
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError, OSError, AssertionError):
                await task

    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        graph = builder.compile(checkpointer=saver)
        contents = [message.content for message in await get_graph_history(graph, config)]
        assert contents[0] == "start"
        if write_fails and durability == "sync":
            assert "finished" not in contents
        elif not write_fails:
            assert contents == ["start", "draft", "finished"]


async def test_invalid_custom_message_fails_the_stream_and_releases_the_conversation(tmp_path):
    async def reply(state):
        if state["messages"][-1].content == "invalid":
            get_stream_writer()(object())
        return {"messages": [AIMessage("valid answer")]}

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    async with CoordinatedSqliteSaver.from_conn_string(str(tmp_path / "stream.sqlite")) as saver:
        executor = executor_for(builder.compile(checkpointer=saver))
        with pytest.raises(UnsupportedMessageTypeError):
            async with aclosing(executor.stream("test", {"message": "invalid"}, thread_id="conversation")) as events:
                async for _ in events:
                    pass
        result = await executor.invoke("test", {"message": "valid"}, thread_id="conversation")
        assert result.content == "valid answer"


@pytest.mark.parametrize("streaming", [False, True])
async def test_cancelled_run_keeps_completed_steps_after_sqlite_reopens(tmp_path, streaming):
    node_started = asyncio.Event()
    node_stopped = asyncio.Event()

    async def wait_for_model(state):
        node_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            node_stopped.set()

    builder = StateGraph(MessagesState)
    builder.add_node("draft", lambda state: {"messages": [AIMessage("saved draft")]})
    builder.add_node("wait_for_model", wait_for_model)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "wait_for_model")
    builder.add_edge("wait_for_model", END)
    database = str(tmp_path / "cancelled.sqlite")
    config = {"configurable": {"thread_id": "cancelled"}}
    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        executor = executor_for(builder.compile(checkpointer=saver))

        async def consume():
            if streaming:
                async with aclosing(executor.stream("test", {"message": "start"}, thread_id="cancelled")) as events:
                    async for _ in events:
                        pass
            else:
                await executor.invoke("test", {"message": "start"}, thread_id="cancelled")

        task = asyncio.create_task(consume())
        try:
            async with asyncio.timeout(3):
                await node_started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert node_stopped.is_set()
                assert not executor.concurrency._entries
        finally:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        graph = builder.compile(checkpointer=saver)
        assert [message.content for message in await get_graph_history(graph, config)] == ["start", "saved draft"]
        state = await graph.aget_state(config)
        assert state.next == ("wait_for_model",)
