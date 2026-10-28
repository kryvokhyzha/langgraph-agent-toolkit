"""Configure optional LangChain tool selection without shared request state."""

from collections.abc import Awaitable, Callable, Sequence

from langchain.agents.middleware import AgentMiddleware, LLMToolSelectorMiddleware, ProviderToolSearchMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from langgraph_agent_toolkit.core.tool_selection_config import ToolSelectionConfig


class ToolSelectionMiddleware(AgentMiddleware):
    """Adapt native selectors to the tools in each model request.

    Each call creates a native middleware instance. Required Deep Agents tools
    stay visible when they exist in that request. Concurrent calls do not change
    one another's tool lists or selection configuration.
    """

    transformers = LLMToolSelectorMiddleware.transformers

    def __init__(
        self,
        config: ToolSelectionConfig,
        *,
        selector_model: BaseChatModel | None = None,
        protected_tools: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self.config = config.model_copy(deep=True)
        self.selector_model = selector_model
        self.protected_tools = frozenset(protected_tools)

    def _prepare(self, request: ModelRequest) -> tuple[ModelRequest, AgentMiddleware | None]:
        if self.config.mode == "off" or not request.tools:
            return request, None
        base_tools = [item for item in request.tools if isinstance(item, BaseTool)]
        available = {item.name for item in base_tools}
        missing = set(self.config.always_include) - available
        if missing:
            raise ValueError(f"Tool selection always_include names are not available: {', '.join(sorted(missing))}")
        required = set(self.config.always_include) | (available & self.protected_tools)
        candidates = [item for item in base_tools if item.name not in required]
        if self.config.mode == "provider":
            tools = [
                item.model_copy(update={"extras": {**item.extras, "defer_loading": False}})
                if isinstance(item, BaseTool)
                and item.name in required
                and isinstance(item.extras, dict)
                and item.extras.get("defer_loading") is True
                else item
                for item in request.tools
            ]
            middleware = ProviderToolSearchMiddleware(searchable_tools=candidates) if candidates else None
            return request.override(tools=tools), middleware
        if len(candidates) <= self.config.max_tools:
            return request, None
        return request, LLMToolSelectorMiddleware(
            model=self.selector_model,
            max_tools=self.config.max_tools,
            always_include=sorted(required),
            max_retries=self.config.max_retries,
            on_parsing_failure=self.config.on_parsing_failure,
        )

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        """Select tools for one synchronous model request."""
        request, middleware = self._prepare(request)
        return middleware.wrap_model_call(request, handler) if middleware is not None else handler(request)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        """Select tools without blocking the event loop."""
        request, middleware = self._prepare(request)
        if middleware is None:
            return await handler(request)
        return await middleware.awrap_model_call(request, handler)


def create_tool_selection_middleware(
    config: ToolSelectionConfig | None = None,
    *,
    selector_model: BaseChatModel | None = None,
    protected_tools: Sequence[str] = (),
) -> list[AgentMiddleware]:
    """Resolve settings and optional managed model configuration at graph setup."""
    from langgraph_agent_toolkit.core.settings import settings

    config = config if config is not None else settings.TOOL_SELECTION
    if config.mode == "off":
        return []
    if config.mode == "llm" and config.model_config_key is not None:
        if selector_model is not None:
            raise ValueError("Pass selector_model or model_config_key, not both")
        configured = settings.get_model_config(config.model_config_key)
        if configured is None:
            raise ValueError(f"Tool selection model_config_key is not in MODEL_CONFIGS: {config.model_config_key!r}")
        from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory

        selector_model = CompletionModelFactory.get_model_from_config(
            configured, configurable_fields=(), config_prefix=""
        )
    return [ToolSelectionMiddleware(config, selector_model=selector_model, protected_tools=protected_tools)]


__all__ = ["ToolSelectionConfig", "ToolSelectionMiddleware", "create_tool_selection_middleware"]
