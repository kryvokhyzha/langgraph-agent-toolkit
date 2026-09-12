"""Middleware that limits the model input to a token budget."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from langgraph_agent_toolkit.agents.components.middlewares._history import keep_latest_turn_if_emptied
from langgraph_agent_toolkit.core.settings import settings


class TokenTrimMiddleware(AgentMiddleware):
    """Limit the model input to a token budget.

    Each model call receives at most ``max_tokens`` from recent history. The
    middleware keeps the full history in state.

    Args:
        max_tokens: Token budget for the message view. `None` disables trimming.
        token_counter: Token-counting callable or chat model.

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
