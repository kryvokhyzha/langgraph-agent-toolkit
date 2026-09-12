"""Middleware that requests a direct answer near the model-call limit."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langgraph_agent_toolkit.core.settings import settings


DEFAULT_IMMEDIATE_INSTRUCTION = (
    "You have gathered enough information and reached the tool-use budget for this task. "
    "Provide your best final answer now, using the information already collected in this "
    "conversation, including any tool results above. Do not request additional tools; if some "
    "details are missing, answer with what you have and clearly note any gaps."
)


class ImmediateGenerationMiddleware(AgentMiddleware):
    """Request a direct answer on the last allowed model call.

    The middleware removes tools and adds an instruction to the system message.
    It keeps tool results so the model can use collected data. The budget counts
    AI messages after the latest human message.
    """

    def __init__(self, model_call_limit: int | None = None, instruction: str | None = None) -> None:
        super().__init__()
        if model_call_limit is None:
            model_call_limit = max(1, settings.DEFAULT_RECURSION_LIMIT // 2)
        if model_call_limit < 1:
            raise ValueError("model_call_limit must be >= 1")
        self.model_call_limit = model_call_limit
        self.instruction = instruction or DEFAULT_IMMEDIATE_INSTRUCTION

    def _calls_made_this_run(self, messages: list[BaseMessage]) -> int:
        """Count AI messages after the latest human message."""
        count = 0
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break
            if isinstance(message, AIMessage):
                count += 1
        return count

    def _force_immediate(self, request: ModelRequest) -> ModelRequest:
        """Remove tools and add the direct-answer instruction."""
        existing = request.system_message.content if request.system_message else ""
        merged = f"{existing}\n\n{self.instruction}".strip() if existing else self.instruction
        return request.override(tools=[], system_message=SystemMessage(content=merged))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if self._calls_made_this_run(request.messages) >= self.model_call_limit - 1:
            request = self._force_immediate(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if self._calls_made_this_run(request.messages) >= self.model_call_limit - 1:
            request = self._force_immediate(request)
        return await handler(request)
