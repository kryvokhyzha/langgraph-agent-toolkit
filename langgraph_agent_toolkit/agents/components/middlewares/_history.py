"""Internal helper shared by the history-trimming middleware (``TrimMessages`` / ``TokenTrim``)."""

from langchain_core.messages import BaseMessage


def keep_latest_turn_if_emptied(
    original: list[BaseMessage],
    trimmed: list[BaseMessage],
) -> list[BaseMessage]:
    """Guard against ``trim_messages`` destroying the model's view.

    ``trim_messages(strategy="last", start_on="human", allow_partial=False)`` returns an empty (or
    system-only) list whenever the latest human turn does not fit the budget — e.g. a single user
    message larger than the budget, or a current turn with more tool calls than the budget allows. The
    model would then be invoked with no user content and answer from the system prompt alone.

    When trimming leaves no human/tool message, fall back to the latest turn (from the last
    ``HumanMessage`` to the end, plus any leading system messages) verbatim, bypassing the budget for
    that turn: a too-long-but-answerable prompt beats a hallucinated answer to nothing, and an
    over-budget turn surfaces as a visible model context error rather than silent corruption.

    Args:
        original: The messages before trimming.
        trimmed: The result of ``trim_messages``.

    Returns:
        ``trimmed`` when it still carries user content, else the latest turn from ``original``.

    """
    if any(m.type in ("human", "tool") for m in trimmed):
        return trimmed

    last_human = next((i for i in range(len(original) - 1, -1, -1) if original[i].type == "human"), None)
    if last_human is None:
        # No human turn to anchor on (pathological); preserve everything rather than wipe the view.
        return original

    leading_system = [m for m in original[:last_human] if m.type == "system"]
    return leading_system + original[last_human:]
