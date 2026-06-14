"""Middleware that trims the model's input to a bounded number of recent messages (view-only)."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import trim_messages

from langgraph_agent_toolkit.agents.components.middlewares._history import keep_latest_turn_if_emptied
from langgraph_agent_toolkit.core.settings import settings


class TrimMessagesMiddleware(AgentMiddleware):
    """Bound the model's input to the last ``max_messages`` messages, non-destructively.

    Ports the custom ``create_react_agent``'s ``pre_model_hook`` trimming to native ``create_agent``:
    each model call sees at most ``max_messages`` recent messages (trimmed to start on a human turn
    and to keep the system prompt), while the full history stays in state. Use it to bound *text*
    growth in long conversations — ``ContextEditingMiddleware`` only bounds tool output.

    ``max_messages`` defaults to ``settings.DEFAULT_MAX_MESSAGE_HISTORY_LENGTH`` and can be overridden
    per agent. It counts messages (not tokens). Best for conversational / text-heavy agents; for a
    single turn that makes very many tool calls, prefer a larger value (or compose with
    ``SanitizeHistoryMiddleware``) so the current turn's tool results are not cut.
    """

    def __init__(self, max_messages: int | None = None) -> None:
        super().__init__()
        resolved = max_messages if max_messages is not None else settings.DEFAULT_MAX_MESSAGE_HISTORY_LENGTH
        if resolved < 1:
            raise ValueError("max_messages must be >= 1")
        self.max_messages = resolved

    def _trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        trimmed = trim_messages(
            messages,
            token_counter=len,  # count messages, not tokens
            max_tokens=self.max_messages,
            strategy="last",
            start_on="human",
            end_on=("human", "tool"),
            include_system=True,
            allow_partial=False,
        )
        # A current turn longer than max_messages collapses to []; keep that turn instead of
        # invoking the model with no user content.
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
