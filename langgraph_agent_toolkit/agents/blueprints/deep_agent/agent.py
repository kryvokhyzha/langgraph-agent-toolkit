"""Build a Deep Agent with thread-scoped virtual files and optional MCP tools."""

from collections.abc import Sequence

from langchain.agents.middleware import InterruptOnConfig, TodoListMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph


try:
    from deepagents import CompiledSubAgent, SubAgent, create_deep_agent
    from deepagents.backends import StateBackend
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
from langgraph_agent_toolkit.schema.models import ModelProvider


_FILE_TOOLS = ("ls", "read_file", "write_file", "edit_file", "glob", "grep")
_RESERVED_TOOLS = {*_FILE_TOOLS, "execute", "delete_file", "task", "write_todos"}


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
    subagents: Sequence[SubAgent | CompiledSubAgent] | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
) -> CompiledStateGraph:
    """Create a graph for the service or an embedded application.

    The service assigns its configured checkpointer after MCP discovery.
    Virtual files belong to that conversation. This blueprint does not open
    host files, create a shell, or configure a cross-thread memory store.
    Custom tools and compiled subagents retain their operator-defined access.
    """
    tools = merge_tools([add, multiply], extra_tools)
    collisions = sorted(tool.name for tool in tools if tool.name in _RESERVED_TOOLS)
    if collisions:
        raise ValueError(f"Tools cannot replace Deep Agents built-ins: {', '.join(collisions)}")
    backend = StateBackend()
    return create_deep_agent(
        model=model if model is not None else create_model(),
        tools=tools,
        system_prompt=(
            "You are a task assistant. Use a plan for complex work. Keep working notes and reports in virtual files. "
            "Use tools for evidence and calculations. Delegate independent work when useful. "
            "Give the user a concise final answer and describe any files you created."
        ),
        backend=backend,
        middleware=[TodoListMiddleware(), FilesystemMiddleware(backend=backend, tools=list(_FILE_TOOLS))],
        subagents=subagents,
        interrupt_on=interrupt_on,
        checkpointer=None,
        name="deep-agent",
    )


deep_agent = Agent(
    name="deep-agent",
    description="A Deep Agent with planning, virtual files, context management, and delegated tools.",
    graph=build_graph(),
    graph_factory=build_graph,
)

__all__ = ["build_graph", "create_model", "deep_agent"]
