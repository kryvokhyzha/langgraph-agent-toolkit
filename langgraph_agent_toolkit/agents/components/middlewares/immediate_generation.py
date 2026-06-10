"""Middleware that forces a direct, tool-free answer when the model-call budget is nearly used up."""

from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langgraph_agent_toolkit.core.settings import settings


# Worded to make the model *answer now from what it already has* — not as a prohibition. Phrasing it
# as "DO NOT make tool calls" makes some models reply "I can't search", refusing instead of
# synthesizing even when the tool results are right there in the context.
DEFAULT_IMMEDIATE_INSTRUCTION = (
    "You have gathered enough information and reached the tool-use budget for this task. "
    "Provide your best final answer now, using the information already collected in this "
    "conversation, including any tool results above. Do not request additional tools; if some "
    "details are missing, answer with what you have and clearly note any gaps."
)


class ImmediateGenerationMiddleware(AgentMiddleware):
    """Force a final, tool-free answer on the last allowed model call of a run.

    Mirrors the custom ``create_react_agent``'s ``immediate_generation`` router: rather than running
    out of steps mid-tool-loop (returning "need more steps" or hitting the recursion limit), the model
    is asked to synthesize a direct answer from what it already has once it reaches the budget. The
    tool *results* are always kept — only ``tools`` and the system message are changed — so the model
    answers from the data it already collected.

    The budget is the number of model calls already made in the current run (AI messages since the
    latest human message), so the middleware needs no extra state. It should fire only as a graceful
    fallback *near the recursion limit*, not cut off legitimate multi-step work, so ``model_call_limit``
    defaults to about half the recursion limit (a model→tools cycle is ~2 steps): with the default
    recursion limit of 64 that is 32 model calls.
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
        """Count AI messages since the latest human message (= model calls already made this run)."""
        count = 0
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break
            if isinstance(message, AIMessage):
                count += 1
        return count

    def _force_immediate(self, request: ModelRequest) -> ModelRequest:
        """Strip tools and append the synthesize-now instruction to the system message."""
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
