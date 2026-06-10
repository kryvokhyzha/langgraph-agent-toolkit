"""Tests for HumanInTheLoopMiddleware support: the executor resume bridge + an interrupt/resume flow."""

from types import SimpleNamespace

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgraph_agent_toolkit.agents.agent_executor import build_resume_command, interrupt_value_to_content
from langgraph_agent_toolkit.helper.utils import langchain_to_chat_message


def _hitl_task(n: int = 1):
    """Build a pending task whose interrupt is a HITL request with n action_requests."""
    value = {"action_requests": [{"name": "send_email"}] * n}
    return SimpleNamespace(interrupts=[SimpleNamespace(value=value)])


def _custom_interrupt_task(value: str = "Please tell me your birthdate?"):
    """Build a pending task from a raw interrupt() blueprint (string payload)."""
    return SimpleNamespace(interrupts=[SimpleNamespace(value=value)])


# --- build_resume_command (the bridge) ---


def test_bridge_approve():
    cmd = build_resume_command([_hitl_task()], {"message": "approve"})
    assert isinstance(cmd, Command)
    assert cmd.resume == {"decisions": [{"type": "approve"}]}


def test_bridge_reject_with_reason():
    cmd = build_resume_command([_hitl_task()], {"message": "reject: too risky"})
    assert cmd.resume == {"decisions": [{"type": "reject", "message": "too risky"}]}


def test_bridge_reject_plain():
    cmd = build_resume_command([_hitl_task()], {"message": "no"})
    assert cmd.resume["decisions"][0]["type"] == "reject"


def test_bridge_freetext_becomes_respond():
    cmd = build_resume_command([_hitl_task()], {"message": "actually send it to alice"})
    assert cmd.resume["decisions"][0] == {"type": "respond", "message": "actually send it to alice"}


def test_bridge_one_decision_per_action_request():
    cmd = build_resume_command([_hitl_task(n=3)], {"message": "approve"})
    assert cmd.resume["decisions"] == [{"type": "approve"}] * 3


def test_bridge_non_hitl_passes_raw_input():
    """A custom interrupt() blueprint still receives the raw input dict (unchanged behavior)."""
    user_input = {"message": "1990-05-15", "user_id": "u1"}
    cmd = build_resume_command([_custom_interrupt_task()], user_input)
    assert cmd.resume == user_input


# --- interrupt_value_to_content (surfacing a HITL interrupt to the client) ---


def test_interrupt_value_to_content_renders_hitl_request():
    value = {"action_requests": [{"name": "send_email", "args": {"to": "x"}, "description": "Approve sending email?"}]}
    content = interrupt_value_to_content(value)
    assert isinstance(content, str)
    assert "Approve sending email?" in content
    assert "approve" in content.lower()  # reply hint included


def test_interrupt_value_to_content_passes_strings_through():
    assert interrupt_value_to_content("Tell me your birthdate?") == "Tell me your birthdate?"


def test_hitl_interrupt_surfaces_as_valid_chat_message():
    """Regression: a HITL interrupt dict must not break AIMessage(content=...) / ChatMessage."""
    value = {"action_requests": [{"name": "send_email", "args": {"to": "x"}, "description": "Approve?"}]}
    msg = AIMessage(content=interrupt_value_to_content(value))  # previously raised a validation error
    cm = langchain_to_chat_message(msg)
    assert cm.type == "ai"
    assert "Approve?" in cm.content


# --- end-to-end interrupt/resume with HumanInTheLoopMiddleware (deterministic fake model) ---


class _ToolScriptModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"SENT to {recipient}"


def _hitl_graph():
    model = _ToolScriptModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "send_email",
                        "args": {"recipient": "bob", "subject": "Hi", "body": "yo"},
                        "id": "e1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    return create_agent(
        model=model,
        tools=[send_email],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"send_email": True})],
        checkpointer=MemorySaver(),
    )


def test_hitl_pauses_before_tool_then_approve_runs_it():
    graph = _hitl_graph()
    cfg = {"configurable": {"thread_id": "approve"}}
    graph.invoke({"messages": [HumanMessage("email bob")]}, config=cfg)

    tasks = [t for t in graph.get_state(cfg).tasks if getattr(t, "interrupts", None)]
    assert tasks  # paused before executing the tool

    out = graph.invoke(build_resume_command(tasks, {"message": "approve"}), config=cfg)
    assert any(isinstance(m, ToolMessage) and "SENT" in str(m.content) for m in out["messages"])


def test_hitl_reject_does_not_run_tool():
    graph = _hitl_graph()
    cfg = {"configurable": {"thread_id": "reject"}}
    graph.invoke({"messages": [HumanMessage("email bob")]}, config=cfg)
    tasks = [t for t in graph.get_state(cfg).tasks if getattr(t, "interrupts", None)]

    out = graph.invoke(build_resume_command(tasks, {"message": "reject: not now"}), config=cfg)
    # the real tool never executed (no "SENT" tool result)
    assert not any(isinstance(m, ToolMessage) and "SENT" in str(m.content) for m in out["messages"])
