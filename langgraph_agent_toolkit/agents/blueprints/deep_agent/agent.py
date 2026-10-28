"""Build a Deep Agent with thread-scoped virtual files and optional MCP tools."""

from collections.abc import Sequence

from langchain.agents.middleware import AgentMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore


try:
    from deepagents import AsyncSubAgent, CompiledSubAgent, FilesystemPermission, SubAgent, create_deep_agent
    from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.filesystem import FilesystemMiddleware
except ModuleNotFoundError as exc:
    if exc.name != "deepagents":
        raise
    raise ImportError("Install langgraph-agent-toolkit[deepagents] to use the Deep Agents blueprint.") from None

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.components.tools import add, multiply
from langgraph_agent_toolkit.core.mcp import merge_tools
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.core.tool_selection import ToolSelectionConfig, create_tool_selection_middleware
from langgraph_agent_toolkit.schema.models import ModelProvider


_FILE_TOOLS = ("ls", "read_file", "write_file", "edit_file", "glob", "grep")
_RESERVED_TOOLS = {
    *_FILE_TOOLS,
    "execute",
    "delete",
    "delete_file",
    "task",
    "write_todos",
    "start_async_task",
    "check_async_task",
    "update_async_task",
    "cancel_async_task",
    "list_async_tasks",
}


def _binary_storage_backend(backend: BackendProtocol, path: str) -> BackendProtocol:
    """Check the storage route for binary content."""
    if isinstance(backend, CompositeBackend):
        for prefix, routed_backend in sorted(backend.routes.items(), key=lambda item: len(item[0]), reverse=True):
            prefix = prefix.rstrip("/") + "/"
            if path.startswith(prefix):
                return _binary_storage_backend(routed_backend, "/" + path[len(prefix) :])
        return _binary_storage_backend(backend.default, path)
    return backend


def create_model() -> BaseChatModel:
    """Select the model at worker startup without changing shared settings."""
    if settings.USE_FAKE_MODEL:
        return CompletionModelFactory.create(ModelProvider.FAKE)
    configured = settings.get_model_config("deep_agent")
    if configured is not None:
        return CompletionModelFactory.get_model_from_config(configured, configurable_fields=(), config_prefix="")
    return CompletionModelFactory.create(
        ModelProvider.OPENAI,
        settings.OPENAI_MODEL_NAME,
        configurable_fields=(),
        config_prefix="",
        model_parameter_values=(),
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_API_BASE_URL,
    )


def build_graph(
    extra_tools: Sequence[BaseTool] = (),
    *,
    model: BaseChatModel | None = None,
    subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    backend: BackendProtocol | None = None,
    offload_binary_content: bool = False,
    store: BaseStore | None = None,
    permissions: list[FilesystemPermission] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    tool_selection: ToolSelectionConfig | None = None,
    selector_model: BaseChatModel | None = None,
    extra_middleware: Sequence[AgentMiddleware] = (),
) -> CompiledStateGraph:
    """Create a graph for the service or an embedded application.

    The service assigns its configured checkpointer after MCP discovery.
    The default backend keeps virtual files in that conversation. It does not
    open host files, create a shell, or configure a cross-thread memory store.
    Custom tools and compiled subagents retain their operator-defined access.
    An injected backend retains its operator-defined access and storage scope.
    Binary offloading requires external storage at `/blobs/`. The operator must
    keep that storage available for checkpoint restoration. Earlier checkpoints
    can still contain the original media. Checkpoint deletion does not delete
    external blobs.
    The caller manages the lifecycle of an injected store.
    Pass `store` when binary content uses a `StoreBackend`.
    Permissions apply to built-in file tools. They do not restrict custom tools.
    Skills and memory paths refer to trusted files in the configured backend.
    Async subagents require a separate Agent Protocol server.
    """
    tools = merge_tools([add, multiply], extra_tools)
    collisions = sorted(tool.name for tool in tools if tool.name in _RESERVED_TOOLS)
    if collisions:
        raise ValueError(f"Tools cannot replace Deep Agents built-ins: {', '.join(collisions)}")
    backend = backend if backend is not None else StateBackend()
    artifacts_root = backend.artifacts_root if isinstance(backend, CompositeBackend) else "/"
    blob_path = f"{artifacts_root.rstrip('/')}/blobs/"
    if offload_binary_content:
        blob_backend = _binary_storage_backend(backend, blob_path)
        if isinstance(blob_backend, StateBackend):
            raise ValueError(f"Binary offloading requires a backend that stores {blob_path} outside checkpoint state.")
        if isinstance(blob_backend, StoreBackend) and store is None:
            raise ValueError("Pass store= to build_graph when binary offloading uses StoreBackend.")
    return create_deep_agent(
        model=model if model is not None else create_model(),
        tools=tools,
        system_prompt=(
            "You are a task assistant. Use a plan for complex work. Keep working notes and reports in virtual files. "
            "Use tools for evidence and calculations. Delegate independent work when useful. "
            "Give the user a concise final answer and describe any files you created."
        ),
        backend=backend,
        middleware=[
            TodoListMiddleware(),
            FilesystemMiddleware(
                backend=backend,
                tools=list(_FILE_TOOLS),
                offload_binary_content=offload_binary_content,
            ),
            *extra_middleware,
            *create_tool_selection_middleware(
                tool_selection, selector_model=selector_model, protected_tools=tuple(_RESERVED_TOOLS)
            ),
        ],
        subagents=subagents,
        permissions=permissions,
        skills=skills,
        memory=memory,
        interrupt_on=interrupt_on,
        checkpointer=None,
        store=store,
        name="deep-agent",
    )


deep_agent = Agent(
    name="deep-agent",
    description="A Deep Agent with planning, virtual files, context management, and delegated tools.",
    graph=build_graph(),
    graph_factory=build_graph,
)

__all__ = ["build_graph", "create_model", "deep_agent"]
