"""Verify graph cleanup when a transport stops accepting stream output."""

import asyncio
from contextlib import contextmanager, suppress
from contextvars import ContextVar

import pytest
from fastapi import FastAPI, Request
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from starlette.requests import ClientDisconnect

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.memory.concurrency import ConversationCoordinator
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.schema import StreamInput
from langgraph_agent_toolkit.service import routes
from langgraph_agent_toolkit.service.auth import Principal, storage_thread_id


@pytest.mark.parametrize("route", [routes.stream, routes.stream_jsonl], ids=["sse", "jsonl"])
@pytest.mark.parametrize(
    ("stop", "stage"),
    [("disconnect", "send"), ("cancel", "send"), ("send_error", "send"), ("disconnect", "graph"), ("cancel", "graph")],
)
async def test_transport_stop_closes_graph_before_releasing_the_conversation(route, stop, stage):
    before = asyncio.all_tasks()
    send_started = asyncio.Event()
    stop_transport = asyncio.Event()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()
    followup_started = asyncio.Event()
    trace = ContextVar("transport_test_trace", default=None)

    class ContextObservability(EmptyObservability):
        @contextmanager
        def trace_context(self, run_id, **kwargs):
            token = trace.set(run_id)
            try:
                yield None
            finally:
                trace.reset(token)

    async def reply(state):
        if state["messages"][-1].content == "followup":
            followup_started.set()
            assert cleanup_finished.is_set()
            return {"messages": [AIMessage("followup finished")]}
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
    executor = AgentExecutor.__new__(AgentExecutor)
    executor.agents = {
        "local": Agent(
            "local", "Transport cleanup test", builder.compile(checkpointer=MemorySaver()), ContextObservability()
        )
    }
    executor.concurrency = ConversationCoordinator(timeout=2)
    app = FastAPI()
    app.state.agent_executor = executor
    scope = {
        "type": "http",
        "app": app,
        "method": "POST",
        "path": "/local/stream" if route is routes.stream else "/local/stream/jsonl",
        "headers": [],
        "state": {"principal": Principal("owner")},
        "asgi": {"spec_version": "2.0" if stop == "disconnect" else "2.4"},
    }
    response = await route(StreamInput(input={"message": "first"}, thread_id="shared"), "local", Request(scope))
    sent = []

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            assert b"partial answer" in message["body"]
            send_started.set()
            if stage == "graph":
                return
            await stop_transport.wait()
            if stop == "send_error":
                raise OSError("The client connection closed.")
            await asyncio.Event().wait()

    async def receive():
        await stop_transport.wait()
        return {"type": "http.disconnect"}

    request_task = asyncio.create_task(response(scope, receive, send))
    follower = None
    try:
        async with asyncio.timeout(5):
            await send_started.wait()
            if stop == "cancel":
                request_task.cancel()
            else:
                stop_transport.set()
            await cleanup_started.wait()
            follower = asyncio.create_task(
                executor.invoke(
                    "local", {"message": "followup"}, thread_id=storage_thread_id("owner", "local", "shared")
                )
            )
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(follower), timeout=0.03)
            assert not request_task.done()
            assert not followup_started.is_set()
            finish_cleanup.set()
            if stop == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await request_task
            elif stop == "send_error":
                with pytest.raises(ClientDisconnect):
                    await request_task
            else:
                await request_task
            assert cleanup_finished.is_set()
            assert (await follower).content == "followup finished"
            assert sent[0]["status"] == 200
            await asyncio.sleep(0)
            assert not (asyncio.all_tasks() - before)
    finally:
        finish_cleanup.set()
        for task in (request_task, follower):
            if task is not None:
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError, ClientDisconnect):
                    await task
