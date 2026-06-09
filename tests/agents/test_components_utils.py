from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph_agent_toolkit.agents.components.utils import (
    default_pre_model_hook,
    pre_model_hook_standard,
    trim_messages_wrapper,
)


def _make_messages(pairs: int):
    """Build system + `pairs` human/ai turns + a trailing human message."""
    msgs = [SystemMessage(content="system")]
    for i in range(pairs):
        msgs.append(HumanMessage(content=f"h{i}"))
        msgs.append(AIMessage(content=f"a{i}"))
    msgs.append(HumanMessage(content="last"))
    return msgs


def test_default_pre_model_hook_is_identity():
    state = {"messages": [HumanMessage(content="hi")]}
    assert default_pre_model_hook(state, {}) is state


def test_pre_model_hook_standard_trims_to_k():
    messages = _make_messages(10)  # 22 messages, well over both the k and the default cap
    small = pre_model_hook_standard({"messages": messages}, {"configurable": {"checkpointer_params": {"k": 3}}})
    default = pre_model_hook_standard({"messages": messages}, {})

    small_msgs = small["llm_input_messages"]
    default_msgs = default["llm_input_messages"]

    # A small k trims more aggressively than the default cap, and both shrink the input.
    assert len(small_msgs) < len(messages)
    assert len(small_msgs) <= len(default_msgs)
    # The trim window must end on a human or tool message (end_on constraint).
    assert small_msgs[-1].type in ("human", "tool")


def test_trim_messages_wrapper_respects_k_override():
    messages = _make_messages(10)
    trimmed = trim_messages_wrapper(messages, {"configurable": {"checkpointer_params": {"k": 2}}})

    assert len(trimmed) < len(messages)
    assert trimmed[-1].type in ("human", "tool")
