"""Middleware that trims the model's input to a bounded token budget (view-only)."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from langgraph_agent_toolkit.agents.components.middlewares._history import keep_latest_turn_if_emptied
from langgraph_agent_toolkit.core.settings import settings


class TokenTrimMiddleware(AgentMiddleware):
    """Bound the model's input to a token budget, non-destructively.

    The token-counting companion to ``TrimMessagesMiddleware`` (which bounds by *message* count). Each
    model call sees at most ``max_tokens`` tokens of recent history (trimmed to start on a human turn
    and to keep the system prompt), while the full history stays in state. Compose the two to cap both
    message count and token size, as the custom ``create_react_agent``'s ``pre_model_hook`` did.

    Args:
        max_tokens: Token budget for the model's message view. Defaults to
            ``settings.DEFAULT_MAX_TOKENS_HISTORY_LENGTH``. ``None`` (the toolkit default) disables
            trimming — set it, or pass ``max_tokens``, to a budget appropriate for your model.
        token_counter: How to count tokens — a per-message or per-list callable, or a chat model.
            Defaults to ``count_tokens_approximately`` (fast, model-agnostic, no extra dependency).
            Pass a ``tiktoken``-backed counter (or the model itself) for an exact count.

    Note:
        On ``create_agent`` the system prompt is never part of ``request.messages`` (it lives on
        ``request.system_message`` and is prepended at call time), so it is always preserved
        regardless of the budget.

    """

    def __init__(
        self,
        max_tokens: int | None = None,
        token_counter: (
            Callable[[list[BaseMessage]], int] | Callable[[BaseMessage], int] | BaseLanguageModel
        ) = count_tokens_approximately,
    ) -> None:
        super().__init__()
        resolved = max_tokens if max_tokens is not None else settings.DEFAULT_MAX_TOKENS_HISTORY_LENGTH
        if resolved is not None and resolved < 1:
            raise ValueError("max_tokens must be >= 1 (or None to disable)")
        self.max_tokens = resolved
        self.token_counter = token_counter

    def _trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        if not self.max_tokens:
            return messages
        trimmed = trim_messages(
            messages,
            token_counter=self.token_counter,
            max_tokens=self.max_tokens,
            strategy="last",
            start_on="human",
            end_on=("human", "tool"),
            include_system=True,
            allow_partial=False,
        )
        # A latest turn larger than the budget collapses to []; keep that turn (over budget but
        # answerable) instead of invoking the model with no user content.
        return keep_latest_turn_if_emptied(messages, trimmed)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(request.override(messages=self._trim(request.messages)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(request.override(messages=self._trim(request.messages)))
