"""Middleware that drops repeated/intermediate tool calls from earlier turns to save tokens."""

import json
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Collection
from typing import Literal

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from langgraph_agent_toolkit.core.settings import settings


def _args_key(args: object) -> str:
    """Stable string key for a tool call's arguments (order-independent)."""
    if not args:
        return ""
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except TypeError:
        return str(args)


def _compute_kept_ids(
    messages: list[BaseMessage],
    by: str,
    keep_last_n: int,
    exclude_tools: Collection[str],
) -> set[str]:
    """Decide which tool_call_ids to keep: the last ``keep_last_n`` per dedup key, plus excluded tools."""
    # tool_call_id -> (name, args_key) from the AIMessage tool calls (where the args live)
    call_meta: dict[str, tuple[str | None, str]] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                cid = call.get("id")
                if cid:
                    call_meta[cid] = (call.get("name"), _args_key(call.get("args")))

    kept: set[str] = set()
    groups: "OrderedDict[object, list[str]]" = OrderedDict()
    for msg in messages:
        if not (isinstance(msg, ToolMessage) and msg.tool_call_id):
            continue
        name, args_key = call_meta.get(msg.tool_call_id, (None, ""))
        name = msg.name or name
        if name in exclude_tools:
            kept.add(msg.tool_call_id)  # protected tool: never deduped
            continue
        key: object = name if by == "name" else (name, args_key)
        groups.setdefault(key, []).append(msg.tool_call_id)

    for ids in groups.values():
        kept.update(ids[-keep_last_n:])
    return kept


def _dedup_previous_turns(
    messages: list[BaseMessage],
    by: str,
    keep_last_n: int,
    exclude_tools: Collection[str],
) -> list[BaseMessage]:
    """Keep only the last ``keep_last_n`` result(s) of each tool key; drop earlier call/result pairs.

    Pairing stays consistent: a tool call is kept iff its result is kept, so the output never contains
    an orphan in either direction. An AIMessage that loses all of its tool calls is kept only if it
    still has textual content.
    """
    kept_ids = _compute_kept_ids(messages, by, keep_last_n, exclude_tools)

    result: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            if msg.tool_call_id in kept_ids:
                result.append(msg)
            # else: drop intermediate tool result
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            kept_calls = [call for call in msg.tool_calls if call.get("id") in kept_ids]
            if len(kept_calls) == len(msg.tool_calls):
                result.append(msg)  # nothing to drop
            elif kept_calls or (msg.content and str(msg.content).strip()):
                result.append(
                    AIMessage(
                        content=msg.content,
                        id=msg.id,
                        name=msg.name,
                        tool_calls=kept_calls,
                        response_metadata=msg.response_metadata,
                    )
                )
            # else: AIMessage had only dropped tool calls and no content -> drop entirely
        else:
            result.append(msg)
    return result


class ClearIntermediateToolCallsMiddleware(AgentMiddleware):
    """Reduce tokens by keeping only the most recent result(s) of each tool from *previous* turns.

    Ports the ``_clean_intermediate_tool_calls`` history-stripping used in downstream agents: the
    current turn (everything from the latest human message onward) is preserved in full, while in
    earlier turns each tool keeps only its last ``keep_last_n`` call/result pair(s).

    Defaults come from ``Settings`` (``CLEAR_INTERMEDIATE_TOOL_CALLS*``) and can be overridden per
    agent. With ``by="name_args"`` (the default) only genuinely identical repeat calls are collapsed,
    so distinct-argument calls are preserved; ``by="name"`` is the aggressive variant that collapses
    all repeats of a tool regardless of arguments. ``exclude_tools`` are never deduped — use it for
    stateful tools whose every call matters. Only the messages sent to the model are affected;
    persisted state is left untouched.
    """

    def __init__(
        self,
        by: Literal["name_args", "name"] | None = None,
        keep_last_n: int | None = None,
        exclude_tools: Collection[str] | None = None,
        enabled: bool | None = None,
    ) -> None:
        super().__init__()
        self.enabled = settings.CLEAR_INTERMEDIATE_TOOL_CALLS if enabled is None else enabled
        self.by = by if by is not None else settings.CLEAR_INTERMEDIATE_TOOL_CALLS_BY
        self.keep_last_n = (
            keep_last_n if keep_last_n is not None else settings.CLEAR_INTERMEDIATE_TOOL_CALLS_KEEP_LAST_N
        )
        self.exclude_tools = set(exclude_tools or ())
        if self.by not in ("name_args", "name"):
            raise ValueError(f"by must be 'name_args' or 'name', got {self.by!r}")
        if self.keep_last_n < 1:
            raise ValueError("keep_last_n must be >= 1")

    def _process(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        if not self.enabled:
            return messages
        last_human = next(
            (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)),
            None,
        )
        if last_human is None:
            return messages
        previous, current = messages[:last_human], messages[last_human:]
        return _dedup_previous_turns(previous, self.by, self.keep_last_n, self.exclude_tools) + current

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(request.override(messages=self._process(request.messages)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(request.override(messages=self._process(request.messages)))
