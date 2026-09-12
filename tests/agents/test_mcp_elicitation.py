"""Test MCP interruption responses and resume values."""

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Interrupt, interrupt

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor, build_resume_command
from langgraph_agent_toolkit.core.memory.concurrency import ConversationCoordinator
from langgraph_agent_toolkit.helper.exceptions import InputValidationError
from langgraph_agent_toolkit.schema import ChatMessage


def payload(label="one", mode="form"):
    request = {"key": "details", "message": f"Answer for {label}.", "mode": mode}
    if mode == "form":
        request["requested_schema"] = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
    else:
        request["url"] = "https://example.com/authorize"
    return {"type": "mcp_elicitation", "tool_name": f"server_{label}", "requests": [request]}


def tasks(*values):
    return [
        SimpleNamespace(interrupts=[Interrupt(value=value, id=f"interrupt-{index}")])
        for index, value in enumerate(values)
    ]


def answer(value="accepted", action="accept"):
    response = {"action": action}
    if action == "accept":
        response["content"] = {"answer": value}
    return {"responses": {"details": response}}


@pytest.mark.parametrize("action", ["accept", "decline", "cancel"])
def test_single_mcp_interrupt_uses_responses(action):
    resume = answer(action=action)

    command = build_resume_command(tasks(payload()), resume)

    assert command.resume == resume


def test_url_mcp_interrupt_accepts_an_answer_without_content():
    resume = {"responses": {"details": {"action": "accept"}}}

    assert build_resume_command(tasks(payload(mode="url")), resume).resume == resume


def test_parallel_mcp_interrupts_keep_answers_separate():
    resume = {"interrupt-0": answer("first"), "interrupt-1": answer("second")}

    command = build_resume_command(tasks(payload("one"), payload("two")), {"resume": resume})

    assert command.resume == resume


def test_resume_map_can_include_a_generic_interrupt():
    resume = {"interrupt-0": answer("first"), "interrupt-1": {"message": "second"}}

    command = build_resume_command(tasks(payload(), "A custom question."), {"resume": resume})

    assert command.resume == resume


@pytest.mark.parametrize(
    "user_input",
    [
        {},
        {"message": "approve"},
        {"responses": []},
        {"responses": {}},
        {"responses": {"wrong": {"action": "cancel"}}},
        {"responses": {"details": {"action": "cancel"}, "extra": {"action": "cancel"}}},
        {"responses": {"details": "accept"}},
        {"responses": {"details": {"action": "approve"}}},
        {"responses": {"details": {"action": []}}},
        {"responses": {"details": {"content": {}}}},
        {"responses": {"details": {"action": "accept"}}},
        {"responses": {"details": {"action": "accept", "content": []}}},
        {"responses": {"details": {"action": "accept", "content": {"nested": {"invalid": True}}}}},
        {"responses": {"details": {"action": "cancel", "content": {"answer": "ignored"}}}},
        {"responses": {"details": {"action": "decline", "extra": "invalid"}}},
        {"resume": []},
        {"resume": {}},
        {"resume": {"wrong-id": answer()}},
        {"resume": {"interrupt-0": answer()}, "responses": answer()["responses"]},
        {"resume": {"interrupt-0": {"responses": answer()["responses"], "extra": True}}},
    ],
)
def test_malformed_mcp_answers_raise_input_validation_error(user_input):
    with pytest.raises(InputValidationError):
        build_resume_command(tasks(payload()), user_input)


@pytest.mark.parametrize("user_input", [answer(), {"resume": {"interrupt-0": answer()}}])
def test_parallel_mcp_interrupts_require_every_interrupt_id(user_input):
    with pytest.raises(InputValidationError):
        build_resume_command(tasks(payload("one"), payload("two")), user_input)


def executor_for(graph):
    executor = object.__new__(AgentExecutor)
    executor.agents = {"mcp-test": Agent("mcp-test", "MCP test agent.", graph)}
    executor.concurrency = ConversationCoordinator()
    return executor


def graph_with_interrupts(count):
    graph = StateGraph(MessagesState)
    for index in range(count):
        label = str(index)

        def node(state, label=label):
            response = interrupt(payload(label))
            return {"messages": [AIMessage(f"{label}: {response['responses']['details']['content']['answer']}")]}

        graph.add_node(label, node)
        graph.add_edge(START, label)
        graph.add_edge(label, END)
    return graph.compile(checkpointer=MemorySaver())


async def call_executor(executor, streaming, user_input):
    if streaming:
        return [
            message
            async for message in executor.stream(
                "mcp-test", input=user_input, thread_id="mcp-thread", stream_tokens=False
            )
            if isinstance(message, ChatMessage)
        ]
    return [await executor.invoke("mcp-test", input=user_input, thread_id="mcp-thread")]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("count", [1, 2])
async def test_executor_preserves_interrupt_ids_and_resumes(streaming, count):
    executor = executor_for(graph_with_interrupts(count))

    paused = await call_executor(executor, streaming, {"message": "Begin."})
    pending = [item for message in paused for item in message.custom_data.get("interrupts", [])]

    assert len([message for message in paused if message.custom_data.get("interrupts")]) == 1
    assert len(pending) == count
    assert len({item["id"] for item in pending}) == count
    assert all(item["id"] for item in pending)
    assert {item["value"]["tool_name"] for item in pending} == {f"server_{index}" for index in range(count)}
    assert all("Answer for" in message.content for message in paused)
    user_input = answer("single") if count == 1 else {"resume": {item["id"]: answer(item["id"]) for item in pending}}

    resumed = await call_executor(executor, streaming, user_input)

    assert resumed
    assert not any(message.custom_data.get("interrupts") for message in resumed)
    state = await executor.agents["mcp-test"].graph.aget_state({"configurable": {"thread_id": "mcp-thread"}})
    assert not state.next
    final_answers = {message.content for message in state.values["messages"] if isinstance(message, AIMessage)}
    for item in pending:
        label = item["value"]["tool_name"].removeprefix("server_")
        expected = item["id"] if count > 1 else "single"
        assert f"{label}: {expected}" in final_answers


@pytest.mark.parametrize("streaming", [False, True])
async def test_executor_leaves_an_interrupt_pending_after_invalid_input(streaming):
    executor = executor_for(graph_with_interrupts(1))
    await call_executor(executor, streaming, {"message": "Begin."})

    with pytest.raises(InputValidationError):
        await call_executor(executor, streaming, {"responses": {"wrong": {"action": "accept"}}})

    resumed = await call_executor(executor, streaming, answer("corrected"))

    assert resumed[-1].content == "0: corrected"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("count", [1, 2])
async def test_real_mcp_protocol_elicitation_round_trip(streaming, count):
    fastmcp = pytest.importorskip("fastmcp")
    from fastmcp import Context
    from langchain.mcp import MCPAdapter
    from mcp.types import ElicitRequest, ElicitRequestFormParams, InputRequiredResult

    server = fastmcp.FastMCP("Elicitation test")
    completed = []

    @server.tool
    async def request_answer(label: str, ctx: Context) -> Any:
        """Request an answer before the tool does its work."""
        if ctx.input_responses is None:
            return InputRequiredResult(
                input_requests={
                    "details": ElicitRequest(
                        params=ElicitRequestFormParams(
                            message=f"Answer for {label}.",
                            requested_schema=payload()["requests"][0]["requested_schema"],
                        )
                    )
                },
                request_state=label,
            )
        assert ctx.request_state == label
        response = ctx.input_responses["details"]
        completed.append((label, response.content["answer"]))
        return f"{label}: {response.content['answer']}"

    tools = await MCPAdapter(server).list_tools()
    graph = StateGraph(MessagesState)
    for index in range(count):
        label = str(index)

        async def node(state, label=label):
            result = await tools[0].ainvoke({"label": label})
            return {"messages": [AIMessage(content=result)]}

        graph.add_node(label, node)
        graph.add_edge(START, label)
        graph.add_edge(label, END)
    executor = executor_for(graph.compile(checkpointer=MemorySaver()))

    paused = await call_executor(executor, streaming, {"message": "Begin."})
    pending = [item for message in paused for item in message.custom_data.get("interrupts", [])]

    assert len(pending) == count
    assert completed == []
    resume = {
        "resume": {
            item["id"]: answer(item["value"]["requests"][0]["message"].removeprefix("Answer for ").removesuffix("."))
            for item in pending
        }
    }

    resumed = await call_executor(executor, streaming, resume)

    assert resumed
    assert sorted(completed) == [(str(index), str(index)) for index in range(count)]
