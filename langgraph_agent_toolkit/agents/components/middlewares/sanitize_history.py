"""Middleware that removes incomplete tool calls from the messages sent to the model."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from langgraph_agent_toolkit.helper.utils import sanitize_chat_history


class SanitizeHistoryMiddleware(AgentMiddleware):
    """Repair broken tool-call/tool-result pairing in the messages before each model call.

    Mirrors the custom ``create_react_agent``'s ``sanitize_chat_history``, fixing both directions
    that providers reject: an AIMessage tool call with no ToolMessage (call without result) has the
    dangling call stripped, and a ToolMessage with no requesting AIMessage tool call (orphaned
    result, e.g. after trimming/summarization) is dropped. Only the messages sent to the model are
    sanitized; persisted state is left untouched.
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
