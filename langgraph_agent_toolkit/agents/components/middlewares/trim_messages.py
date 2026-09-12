"""Middleware that limits the model input to recent messages."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import trim_messages

from langgraph_agent_toolkit.agents.components.middlewares._history import keep_latest_turn_if_emptied
from langgraph_agent_toolkit.core.settings import settings


class TrimMessagesMiddleware(AgentMiddleware):
    """Limit the model input to the last ``max_messages`` messages.

    Each model call receives at most ``max_messages`` recent messages. The
    middleware keeps the full history in state. It counts messages, not tokens.
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
            token_counter=len,
            max_tokens=self.max_messages,
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
