"""Configure MCP tools without importing the optional client at module load."""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, StringConstraints, TypeAdapter, model_validator


if TYPE_CHECKING:
    from fastmcp import Client
    from langchain_core.tools import BaseTool

    from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
    from langgraph_agent_toolkit.core._base_settings import Settings


ServerName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")]


class MCPServerConfig(BaseModel):
    """Describe one operator-configured MCP server and its credentials."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    transport: Literal["http", "stdio"] = "http"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    headers: dict[str, SecretStr] = Field(default_factory=dict)
    headers_env: dict[str, str] = Field(default_factory=dict)
    env: dict[str, SecretStr] = Field(default_factory=dict)
    env_env: dict[str, str] = Field(default_factory=dict)
    mode: Literal["auto", "legacy"] = "auto"
    timeout: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    tool_allowlist: list[str] | None = None

    @model_validator(mode="after")
    def validate_transport(self) -> MCPServerConfig:
        if self.transport == "http":
            if not self.url or self.command is not None or self.args or self.env or self.env_env:
                raise ValueError("An HTTP MCP server needs url and cannot use command, args, env, or env_env.")
            parsed = TypeAdapter(HttpUrl).validate_python(self.url)
            if parsed.username is not None or parsed.password is not None:
                raise ValueError("Use MCP headers or headers_env for credentials, not URL user information.")
            header_names = [name.lower() for name in [*self.headers, *self.headers_env]]
            if len(set(header_names)) != len(header_names):
                raise ValueError("MCP header names must be unique across headers and headers_env, ignoring case.")
            if any(not re.fullmatch(r"[!#$%&'*+.^_`|~0-9a-z-]+", name) for name in header_names):
                raise ValueError("MCP headers must use valid HTTP header names.")
        else:
            if not self.command or not self.command.strip() or self.url is not None or self.headers or self.headers_env:
                raise ValueError("A stdio MCP server needs command and cannot use url, headers, or headers_env.")
            if set(self.env) & set(self.env_env):
                raise ValueError("MCP env and env_env must use different variable names.")
        for mapping in (self.headers_env, self.env_env):
            if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) for value in mapping.values()):
                raise ValueError("MCP environment references must contain valid environment variable names.")
        if self.tool_allowlist is not None and (
            any(not name for name in self.tool_allowlist) or len(set(self.tool_allowlist)) != len(self.tool_allowlist)
        ):
            raise ValueError("MCP tool_allowlist must contain distinct, nonempty upstream tool names.")
        return self


def merge_tools(local_tools: Sequence[BaseTool], extra_tools: Sequence[BaseTool]) -> list[BaseTool]:
    """Reject duplicate names before a graph can replace a tool silently."""
    tools = [*local_tools, *extra_tools]
    names: set[str] = set()
    for tool in tools:
        if tool.name in names:
            raise ValueError(f"Duplicate agent tool name: {tool.name!r}")
        names.add(tool.name)
    return tools


def _resolve_values(values: dict[str, SecretStr], references: dict[str, str]) -> dict[str, str]:
    resolved = {name: value.get_secret_value() for name, value in values.items()}
    for name, variable in references.items():
        value = os.environ.get(variable)
        if not value:
            raise ValueError(f"MCP environment variable {variable!r} is missing or empty.")
        resolved[name] = value
    return resolved


def _create_client(name: str, config: MCPServerConfig) -> Client:
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport, StreamableHttpTransport

    if config.transport == "http":
        headers = _resolve_values(config.headers, config.headers_env)
        if any(not key or any(char in key + value for char in "\r\n\x00") for key, value in headers.items()):
            raise ValueError("MCP headers cannot contain empty names, line breaks, or NUL characters.")
        transport = StreamableHttpTransport(config.url, headers=headers)
    else:
        transport = StdioTransport(
            command=config.command,
            args=config.args,
            env=_resolve_values(config.env, config.env_env),
            # The default keeps subprocesses alive after a tool or interrupt exits.
            keep_alive=False,
        )
    return Client(transport, name=name, mode=config.mode, timeout=config.timeout, init_timeout=config.timeout)


async def load_mcp_tools(
    servers: dict[str, MCPServerConfig], *, discovery_timeout: float = 30.0
) -> dict[str, list[BaseTool]]:
    """Discover namespaced tools. Each tool opens and closes its own connection context.

    The SDK can share a connection between overlapping calls. No connection stays
    open after the last call exits. Restart the worker to refresh tool schemas.
    """
    servers = TypeAdapter(dict[ServerName, MCPServerConfig]).validate_python(servers)
    if not servers:
        return {}
    try:
        from langchain.mcp import MCPAdapter
    except ImportError:
        raise RuntimeError("MCP servers are configured. Install langgraph-agent-toolkit[mcp] to use them.") from None

    # Resolve every credential before starting any connection or subprocess.
    clients = {name: _create_client(name, config) for name, config in servers.items()}

    async def discover(name: str, config: MCPServerConfig) -> list[BaseTool]:
        try:
            tools = await MCPAdapter(clients[name]).list_tools()
        except Exception as exc:
            # Transport errors can include request headers or credential-bearing URLs.
            raise RuntimeError(f"MCP discovery failed for server {name!r} ({type(exc).__name__}).") from None
        if config.tool_allowlist is not None:
            missing = set(config.tool_allowlist) - {tool.name for tool in tools}
            if missing:
                raise ValueError(f"MCP server {name!r} does not provide these tools: {sorted(missing)!r}")
            tools = [tool for tool in tools if tool.name in config.tool_allowlist]
        namespaced = []
        for tool in tools:
            public_name = f"{name}_{tool.name}"
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", public_name):
                raise ValueError(f"MCP server {name!r} has a tool name that exceeds the supported name format.")
            # Copy the LangChain wrapper. Its callback keeps the upstream name.
            # A ClientGroup would also refresh its routing catalog on each call.
            namespaced.append(tool.model_copy(update={"name": public_name}))
        return merge_tools([], namespaced)

    async with asyncio.timeout(discovery_timeout):
        # Wait for all contexts to close before reporting a discovery failure.
        results = await asyncio.gather(
            *(discover(name, config) for name, config in servers.items()), return_exceptions=True
        )
    for result in results:
        if isinstance(result, BaseException):
            raise result
    loaded = dict(zip(servers, results, strict=True))
    merge_tools([], [tool for tools in loaded.values() for tool in tools])
    return loaded


async def configure_mcp_agents(executor: AgentExecutor, settings: Settings, *, rebuild_all: bool = False) -> None:
    """Build MCP graphs and optionally refresh all graphs for a new service lifespan."""
    if not settings.MCP_SERVERS and not settings.MCP_AGENT_SERVERS and not rebuild_all:
        return

    from langgraph_agent_toolkit.helper.constants import get_default_agent

    selections = settings.MCP_AGENT_SERVERS or (
        {get_default_agent(): list(settings.MCP_SERVERS)} if settings.MCP_SERVERS else {}
    )
    agents = {info.key: executor.get_agent(info.key) for info in executor.get_all_agent_info()}
    for agent_name, server_names in selections.items():
        if agent_name not in agents:
            raise ValueError(f"MCP_AGENT_SERVERS refers to an unloaded agent: {agent_name!r}")
        if agents[agent_name].graph_factory is None:
            raise ValueError(f"Agent {agent_name!r} needs a graph_factory to use configured MCP tools.")
        if len(server_names) != len(set(server_names)):
            raise ValueError(f"MCP_AGENT_SERVERS has duplicate servers for agent {agent_name!r}.")
        if set(server_names) - settings.MCP_SERVERS.keys():
            raise ValueError(f"MCP_AGENT_SERVERS refers to an unknown server for agent {agent_name!r}.")

    needed = {name for names in selections.values() for name in names}
    loaded = await load_mcp_tools(
        {name: config for name, config in settings.MCP_SERVERS.items() if name in needed},
        discovery_timeout=settings.MCP_DISCOVERY_TIMEOUT,
    )
    # Build all graphs before replacing any executor-local graph.
    graphs = {
        name: agent.graph_factory([tool for server in selections.get(name, []) for tool in loaded[server]])
        for name, agent in agents.items()
        if agent.graph_factory is not None and (rebuild_all or name in selections)
    }
    for name, graph in graphs.items():
        agents[name].graph = graph


__all__ = ["MCPServerConfig", "configure_mcp_agents", "load_mcp_tools", "merge_tools"]
