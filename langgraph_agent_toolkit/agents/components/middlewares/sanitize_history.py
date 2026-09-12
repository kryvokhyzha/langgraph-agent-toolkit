"""Middleware that removes incomplete tool calls before model calls."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from langgraph_agent_toolkit.helper.utils import sanitize_chat_history


class SanitizeHistoryMiddleware(AgentMiddleware):
    """Repair tool-call and tool-result pairs before each model call.

    Remove tool calls without results. Remove tool results without matching calls.
    The middleware changes only the messages sent to the model.
    """

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(request.override(messages=sanitize_chat_history(request.messages)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(request.override(messages=sanitize_chat_history(request.messages)))
