"""Test real MCP tools with the built-in graph factories."""

import asyncio
import importlib
import json
import os
import sys
import textwrap
from contextlib import asynccontextmanager

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_agent_toolkit.core.settings import settings


fastmcp = pytest.importorskip("fastmcp")
MCPAdapter = pytest.importorskip("langchain.mcp").MCPAdapter


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def build_graph(monkeypatch, blueprint, tools, tool_name, arguments):
    """Build a graph that calls one supplied tool."""
    monkeypatch.setattr(settings, "USE_FAKE_MODEL", True)
    module = importlib.import_module(f"langgraph_agent_toolkit.agents.blueprints.{blueprint}.agent")
    model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": arguments, "id": "mcp-call", "type": "tool_call"}],
            ),
            AIMessage(content="Complete."),
        ]
    )
    if blueprint == "react":
        monkeypatch.setattr(module.CompletionModelFactory, "create", lambda **kwargs: model)
    else:
        monkeypatch.setattr(module, "model", model)
    return module.build_graph(tools)


@pytest.mark.parametrize("blueprint", ["react", "create_agent"])
async def test_blueprint_calls_real_mcp_tool_after_discovery_closes(monkeypatch, blueprint):
    calls = []
    sessions = {"started": 0, "closed": 0}

    @asynccontextmanager
    async def lifespan(server):
        sessions["started"] += 1
        try:
            yield {}
        finally:
            sessions["closed"] += 1

    server = fastmcp.FastMCP("local-math", lifespan=lifespan)

    @server.tool
    async def remote_sum(left: int, right: int) -> dict[str, int]:
        """Add two integers."""
        calls.append((left, right))
        return {"total": left + right}

    adapter = MCPAdapter(fastmcp.Client(server, timeout=5, init_timeout=5))
    tools = await adapter.list_tools()
    assert not adapter.client.is_connected()
    graph = build_graph(monkeypatch, blueprint, tools, "remote_sum", {"left": 2, "right": 3})

    result = await asyncio.wait_for(graph.ainvoke({"messages": [HumanMessage("Add two and three.")]}), timeout=10)

    messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(messages) == 1
    assert messages[0].name == "remote_sum"
    assert messages[0].status == "success"
    assert messages[0].artifact == {"structured_content": {"total": 5}}
    assert result["messages"][-1].content == "Complete."
    assert calls == [(2, 3)]
    assert sessions["started"] > 0
    assert sessions["started"] == sessions["closed"]
    assert not adapter.client.is_connected()


@pytest.mark.parametrize("blueprint", ["react", "create_agent"])
async def test_mcp_error_reaches_model_without_repeating_write(monkeypatch, blueprint):
    writes = []
    server = fastmcp.FastMCP("local-writes")

    @server.tool
    async def remote_write(value: str) -> str:
        """Record a test value and report an error."""
        writes.append(value)
        raise RuntimeError("The test write completed. Its response failed.")

    adapter = MCPAdapter(fastmcp.Client(server, timeout=5, init_timeout=5))
    tools = await adapter.list_tools()
    graph = build_graph(monkeypatch, blueprint, tools, "remote_write", {"value": "one"})

    result = await asyncio.wait_for(graph.ainvoke({"messages": [HumanMessage("Record one.")]}), timeout=10)

    messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(messages) == 1
    assert messages[0].status == "error"
    assert "response failed" in str(messages[0].content)
    assert writes == ["one"]
    assert not adapter.client.is_connected()


async def test_mcp_tools_run_concurrently_and_release_shared_session():
    entered = set()
    both_entered = asyncio.Event()
    server = fastmcp.FastMCP("local-concurrency")

    @server.tool
    async def remote_echo(value: str) -> str:
        """Return a value after both calls start."""
        entered.add(value)
        if len(entered) == 2:
            both_entered.set()
        await both_entered.wait()
        return value

    adapter = MCPAdapter(fastmcp.Client(server, timeout=5, init_timeout=5))
    [tool] = await adapter.list_tools()

    results = await asyncio.wait_for(
        asyncio.gather(tool.ainvoke({"value": "one"}), tool.ainvoke({"value": "two"})),
        timeout=10,
    )

    assert entered == {"one", "two"}
    assert [result[0]["text"] for result in results] == ["one", "two"]
    assert not adapter.client.is_connected()


async def test_cancelling_mcp_tool_stops_server_call_and_closes_session():
    entered = asyncio.Event()
    stopped = asyncio.Event()
    never = asyncio.Event()
    server = fastmcp.FastMCP("local-cancellation")

    @server.tool
    async def remote_wait() -> str:
        """Wait until the caller cancels the request."""
        entered.set()
        try:
            await never.wait()
            return "Complete."
        finally:
            stopped.set()

    adapter = MCPAdapter(fastmcp.Client(server, timeout=5, init_timeout=5))
    [tool] = await adapter.list_tools()
    task = asyncio.create_task(tool.ainvoke({}))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        await asyncio.wait_for(stopped.wait(), timeout=5)
        assert not adapter.client.is_connected()
    finally:
        never.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_configured_stdio_tool_keeps_upstream_name_and_stops_process(monkeypatch, tmp_path, mode):
    from langgraph_agent_toolkit.core.mcp import MCPServerConfig, load_mcp_tools

    lifecycle = tmp_path / "lifecycle.jsonl"
    script = tmp_path / "local_mcp_server.py"
    script.write_text(
        textwrap.dedent(
            '''
            import json
            import os
            import sys
            from contextlib import asynccontextmanager
            from pathlib import Path

            from fastmcp import FastMCP

            lifecycle = Path(sys.argv[1])

            def record(event):
                with lifecycle.open("a") as stream:
                    stream.write(json.dumps({"event": event, "pid": os.getpid()}) + "\\n")

            @asynccontextmanager
            async def lifespan(server):
                record("started")
                try:
                    yield {}
                finally:
                    record("closed")

            server = FastMCP("local-stdio", lifespan=lifespan)

            @server.tool
            async def remote_echo(value: str) -> dict[str, str | int]:
                """Return the value and process ID."""
                return {"value": value, "pid": os.getpid()}

            @server.tool
            async def hidden_tool() -> str:
                """Return a value from a tool outside the allowlist."""
                return "hidden"

            server.run(show_banner=False)
            '''
        )
    )
    loaded = await load_mcp_tools(
        {
            "local": MCPServerConfig(
                transport="stdio",
                command=sys.executable,
                args=[str(script), str(lifecycle)],
                mode=mode,
                timeout=10,
                tool_allowlist=["remote_echo"],
            )
        }
    )
    assert [tool.name for tool in loaded["local"]] == ["local_remote_echo"]
    graph = build_graph(monkeypatch, "create_agent", loaded["local"], "local_remote_echo", {"value": "hello"})

    result = await asyncio.wait_for(graph.ainvoke({"messages": [HumanMessage("Echo hello.")]}), timeout=15)

    [message] = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert message.name == "local_remote_echo"
    assert message.status == "success"
    assert message.artifact["structured_content"]["value"] == "hello"
    events = [json.loads(line) for line in lifecycle.read_text().splitlines()]
    assert [event["event"] for event in events] == ["started", "closed", "started", "closed"]
    for pid in {event["pid"] for event in events}:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


async def test_cancelling_discovery_stops_stdio_process_during_handshake(tmp_path):
    from langgraph_agent_toolkit.core.mcp import MCPServerConfig, load_mcp_tools

    pid_file = tmp_path / "pid"
    script = tmp_path / "stalled_mcp_server.py"
    script.write_text(
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    task = asyncio.create_task(
        load_mcp_tools(
            {
                "stalled": MCPServerConfig(
                    transport="stdio",
                    command=sys.executable,
                    args=[str(script), str(pid_file)],
                    timeout=20,
                )
            }
        )
    )
    try:
        async with asyncio.timeout(5):
            while not pid_file.exists():
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
