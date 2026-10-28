"""Select tools through the real MCP protocol after discovery closes."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import Field

from langgraph_agent_toolkit.core.tool_selection import ToolSelectionConfig


fastmcp = pytest.importorskip("fastmcp")
MCPAdapter = pytest.importorskip("langchain.mcp").MCPAdapter


class ScriptedToolModel(FakeMessagesListChatModel):
    visible_toolsets: list[list[str]] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.visible_toolsets.append([item.name if hasattr(item, "name") else item["title"] for item in tools])
        return self


@pytest.mark.parametrize("builder", ["native", "deepagents"])
@pytest.mark.parametrize("fails", [False, True])
async def test_selected_remote_tool_runs_once_and_closes_its_session(builder, fails):
    calls, unrelated_writes = [], []
    sessions = {"started": 0, "closed": 0}

    @asynccontextmanager
    async def lifespan(server):
        sessions["started"] += 1
        try:
            yield {}
        finally:
            sessions["closed"] += 1

    server = fastmcp.FastMCP("selection-math", lifespan=lifespan)

    @server.tool
    async def math_remote_sum(left: int, right: int) -> dict[str, int]:
        """Add two integers in the remote service."""
        calls.append((left, right))
        if fails:
            raise RuntimeError("The remote operation completed. Its response failed.")
        return {"total": left + right}

    @server.tool
    async def math_remote_write(value: str) -> str:
        """Write an unrelated value in the remote service."""
        unrelated_writes.append(value)
        raise AssertionError("An unselected remote write must not run")

    adapter = MCPAdapter(fastmcp.Client(server, timeout=5, init_timeout=5))
    tools = await adapter.list_tools()
    assert {item.name for item in tools} == {"math_remote_sum", "math_remote_write"}
    assert not adapter.client.is_connected()
    assert sessions["started"] == sessions["closed"]
    selector = ScriptedToolModel(
        responses=[
            AIMessage(
                "",
                tool_calls=[
                    {"name": "ToolSelectionResponse", "args": {"tools": ["math_remote_sum"]}, "id": "selection"}
                ],
            )
        ]
    )
    answer = "Tool operation failed." if fails else "Total: 5."
    model = ScriptedToolModel(
        responses=[
            AIMessage(
                "",
                tool_calls=[{"name": "math_remote_sum", "args": {"left": 2, "right": 3}, "id": "remote-call"}],
            ),
            AIMessage(answer),
        ]
    )
    configuration = ToolSelectionConfig(mode="llm", max_tools=1)
    if builder == "native":
        from langgraph_agent_toolkit.agents.blueprints.create_agent._shared import build_tool_graph

        graph = build_tool_graph(model, tools, tool_selection=configuration, selector_model=selector)
    else:
        pytest.importorskip("deepagents")
        from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph

        graph = build_graph(tools, model=model, tool_selection=configuration, selector_model=selector)

    result = await asyncio.wait_for(graph.ainvoke({"messages": [HumanMessage("Add two and three.")]}), timeout=10)

    messages = [item for item in result["messages"] if isinstance(item, ToolMessage)]
    assert len(messages) == 1
    assert messages[0].name == "math_remote_sum"
    assert messages[0].status == ("error" if fails else "success")
    if fails:
        assert "response failed" in str(messages[0].content)
    else:
        assert messages[0].artifact == {"structured_content": {"total": 5}}
    assert result["messages"][-1].content == answer
    assert calls == [(2, 3)]
    assert unrelated_writes == []
    assert len(selector.visible_toolsets) == 2
    assert all("math_remote_sum" in names and "math_remote_write" not in names for names in model.visible_toolsets)
    if builder == "deepagents":
        assert all({"task", "write_todos", "read_file", "write_file"} <= set(names) for names in model.visible_toolsets)
    assert sessions["started"] >= 2
    assert sessions["started"] == sessions["closed"]
    assert not adapter.client.is_connected()
