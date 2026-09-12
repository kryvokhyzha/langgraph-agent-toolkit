"""Internal helper for history-trimming middleware."""

from langchain_core.messages import BaseMessage


def keep_latest_turn_if_emptied(
    original: list[BaseMessage],
    trimmed: list[BaseMessage],
) -> list[BaseMessage]:
    """Keep the latest user turn when ``trim_messages`` removes it.

    ``trim_messages`` can remove the latest user turn when it exceeds the
    budget. In that case, preserve the latest turn and leading system messages.

    Args:
        original: Messages before trimming.
        trimmed: Result from ``trim_messages``.

    Returns:
        ``trimmed`` when it contains user content. Otherwise, the latest turn
        from ``original``.

    """
    if any(m.type in ("human", "tool") for m in trimmed):
        return trimmed

    last_human = next((i for i in range(len(original) - 1, -1, -1) if original[i].type == "human"), None)
    if last_human is None:
        return original

    leading_system = [m for m in original[:last_human] if m.type == "system"]
    return leading_system + original[last_human:]
