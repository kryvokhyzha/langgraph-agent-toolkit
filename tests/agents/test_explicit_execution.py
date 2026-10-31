"""Exercise explicit execution and recovery with real graph checkpoints."""

import asyncio
from typing import TypedDict

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Send, interrupt
from pydantic import BaseModel, Field, ValidationError

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.agents.components.checkpoint.empty import NoOpSaver
from langgraph_agent_toolkit.core.execution import ExecutionConflictError
from langgraph_agent_toolkit.core.identity import storage_thread_id
from langgraph_agent_toolkit.core.memory.concurrency import ConversationCoordinator
from langgraph_agent_toolkit.core.memory.coordinated_sqlite import CoordinatedSqliteSaver
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.exceptions import InputValidationError
from langgraph_agent_toolkit.schema.execution import ExecutionRequest


class Counter(TypedDict):
    value: int


class Decision(BaseModel):
    quantity: int = Field(ge=1, le=5, strict=True)


def executor_for(graph):
    executor = AgentExecutor.__new__(AgentExecutor)
    executor.agents = {"test": Agent("test", "An explicit execution test agent.", graph)}
    executor.concurrency = ConversationCoordinator(timeout=1)
    return executor


def counter_graph(saver, calls):
    def increment(state):
        calls.append(state["value"])
        return {"value": state["value"] + 1}

    builder = StateGraph(Counter)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile(checkpointer=saver)


@pytest.mark.parametrize(
    "fields",
    [
        {"operation": "resume", "resume": {"question": True}},
        {"operation": "recover"},
        {"operation": "resume", "expected_checkpoint_id": "version"},
        {"operation": "input", "resume": {"question": True}},
        {"operation": "recover", "expected_checkpoint_id": "version", "input": {"value": 1}},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": float("nan")},
        {"thread_id": "bad\x00identity"},
    ],
)
def test_execution_request_rejects_ambiguous_or_invalid_operations(fields):
    with pytest.raises(ValidationError):
        ExecutionRequest.model_validate({"thread_id": "work", **fields})


@pytest.mark.parametrize(
    "field",
    [
        "thread_id",
        "user_id",
        "checkpoint_id",
        "checkpoint_ns",
        "lat_user_id",
        "lat_agent_id",
        "lat_thread_id",
        "lat_operation_id",
        "lat_operation_fingerprint",
        "lat_execution_run_id",
        "lat_execution_triggers",
        "__pregel_runtime",
    ],
)
def test_execution_request_rejects_reserved_config_before_job_storage(field):
    with pytest.raises(ValidationError, match="cannot override"):
        ExecutionRequest(thread_id="work", agent_config={field: "spoofed"})


async def test_structured_result_operation_reentry_and_owner_isolation():
    calls = []
    executor = executor_for(counter_graph(InMemorySaver(), calls))
    request = ExecutionRequest(thread_id="work", input={"value": 4})
    baselines = []

    async def started(checkpoint_id):
        baselines.append(checkpoint_id)

    result = await executor.execute("test", request, owner_id="owner", operation_id="first", on_start=started)
    assert result.status == "completed"
    assert result.output == {"value": 5}
    assert result.checkpoint_id
    assert baselines == [None]
    assert calls == [4]
    repeated = await executor.execute("test", request, owner_id="owner", operation_id="first", retry=True)
    assert repeated == result
    assert calls == [4]
    state = await executor.inspect_execution("test", "work", owner_id="owner")
    assert state.output == result.output
    assert state.operation_id == "first"
    assert state.checkpoint_id == result.checkpoint_id
    assert (await executor.inspect_execution("test", "work", owner_id="other")).status == "empty"
    with pytest.raises(ExecutionConflictError, match="different input"):
        await executor.execute(
            "test", request.model_copy(update={"input": {"value": 7}}), owner_id="owner", operation_id="first"
        )
    with pytest.raises(InputValidationError, match="execution owner"):
        await executor.execute("test", request.model_copy(update={"user_id": "other"}), owner_id="owner")


@pytest.mark.parametrize("saver", [None, False, NoOpSaver()])
async def test_explicit_stateless_execution_keeps_supported_saver_choices(saver):
    executor = executor_for(counter_graph(saver, []))
    result = await executor.execute("test", ExecutionRequest(thread_id="work", input={"value": 2}), owner_id="owner")
    assert result.output == {"value": 3}
    assert result.checkpoint_id is None


@pytest.mark.parametrize("saver_factory", [lambda: None, lambda: False, NoOpSaver, InMemorySaver])
async def test_retry_refuses_to_replay_after_a_nonpersistent_worker_lost_its_state(saver_factory):
    calls = []
    request = ExecutionRequest(thread_id="work", input={"value": 2})
    executor = executor_for(counter_graph(saver_factory(), calls))
    result = await executor.execute("test", request, owner_id="owner", operation_id="job")
    assert result.output == {"value": 3}
    restarted = executor_for(counter_graph(saver_factory(), calls))
    with pytest.raises(ExecutionConflictError, match="cannot prove"):
        await restarted.execute("test", request, owner_id="owner", operation_id="job", retry=True)
    assert calls == [2]


async def test_functional_output_recovery_keeps_public_output_separate_from_saved_state():
    calls = []

    @entrypoint(checkpointer=InMemorySaver())
    async def workflow(inputs):
        calls.append(inputs)
        return entrypoint.final(value={"answer": 42}, save={"private_context": "retained"})

    executor = executor_for(workflow)
    request = ExecutionRequest(thread_id="work", input={"question": "calculate"})
    result = await executor.execute("test", request, owner_id="owner", operation_id="functional")
    assert result.output == {"answer": 42}
    assert (await executor.execute("test", request, owner_id="owner", operation_id="functional", retry=True)) == result
    assert len(calls) == 1


async def test_explicit_message_execution_uses_the_same_scoped_history_as_http():
    builder = StateGraph(MessagesState)
    builder.add_node("reply", lambda state: {"messages": [AIMessage("ready")]})
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    executor = executor_for(graph)
    result = await executor.execute(
        "test",
        ExecutionRequest(thread_id="public", input={"messages": [{"role": "user", "content": "start"}]}),
        owner_id="owner",
    )
    assert [item["content"] for item in result.output["messages"]] == ["start", "ready"]
    state = await graph.aget_state({"configurable": {"thread_id": storage_thread_id("owner", "test", "public")}})
    assert state.metadata["lat_thread_id"] == "public"
    assert state.metadata["lat_user_id"] == "owner"


async def test_typed_resume_survives_restart_and_rejects_new_input_and_stale_answers(tmp_path):
    prepared = []
    completed = []

    def prepare(state):
        prepared.append(True)
        return {}

    def choose(state):
        answer = interrupt({"question": "Choose a quantity."}, response_schema=Decision)
        completed.append(answer.quantity)
        return {"value": answer.quantity}

    builder = StateGraph(Counter)
    builder.add_node("prepare", prepare)
    builder.add_node("choose", choose)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "choose")
    builder.add_edge("choose", END)
    database = str(tmp_path / "questions.sqlite")
    request = ExecutionRequest(thread_id="work", input={"value": 0})
    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        executor = executor_for(builder.compile(checkpointer=saver))
        result = await executor.execute("test", request, owner_id="owner", operation_id="question")
        assert result.status == "interrupted"
        assert result.interrupts[0]["response_schema"] == Decision.model_json_schema()
        assert await executor.execute("test", request, owner_id="owner", operation_id="question", retry=True) == result
        with pytest.raises(ExecutionConflictError, match="explicit resume"):
            await executor.execute("test", request, owner_id="owner")
        with pytest.raises(ExecutionConflictError, match="no unfinished"):
            await executor.execute(
                "test",
                ExecutionRequest(thread_id="work", operation="recover", expected_checkpoint_id=result.checkpoint_id),
                owner_id="owner",
            )

    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        executor = executor_for(builder.compile(checkpointer=saver))
        answers = {result.interrupts[0]["id"]: {"quantity": 3}}
        resume = ExecutionRequest(
            thread_id="work", operation="resume", expected_checkpoint_id=result.checkpoint_id, resume=answers
        )
        with pytest.raises(ExecutionConflictError, match="no longer current"):
            await executor.execute(
                "test", resume.model_copy(update={"expected_checkpoint_id": "stale"}), owner_id="owner"
            )
        assert completed == []
        with pytest.raises(InputValidationError, match="each pending interrupt"):
            await executor.execute("test", resume.model_copy(update={"resume": {"wrong-id": True}}), owner_id="owner")
        resumed = await executor.execute("test", resume, owner_id="owner", operation_id="answer")
        assert resumed.status == "completed"
        assert resumed.output == {"value": 3}
        assert not resumed.interrupts
        with pytest.raises(ExecutionConflictError, match="no longer current"):
            await executor.execute("test", resume, owner_id="owner")
    assert prepared == [True]
    assert completed == [3]


@pytest.mark.parametrize("explicit_recovery", [False, True])
async def test_cancelled_execution_recovers_without_repeating_input_or_completed_nodes(tmp_path, explicit_recovery):
    entered = asyncio.Event()
    release = asyncio.Event()
    prepared = []
    baselines = []

    def prepare(state):
        prepared.append(state["value"])
        return {"value": state["value"] + 1}

    async def finish(state):
        entered.set()
        await release.wait()
        return {"value": state["value"] + 10}

    builder = StateGraph(Counter)
    builder.add_node("prepare", prepare)
    builder.add_node("finish", finish)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "finish")
    builder.add_edge("finish", END)
    database = str(tmp_path / "recovery.sqlite")
    request = ExecutionRequest(thread_id="work", input={"value": 0})

    async def started(checkpoint_id):
        baselines.append(checkpoint_id)

    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        executor = executor_for(builder.compile(checkpointer=saver))
        task = asyncio.create_task(
            executor.execute("test", request, owner_id="owner", operation_id="job", on_start=started)
        )
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not executor.concurrency._entries
    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        executor = executor_for(builder.compile(checkpointer=saver))
        snapshot = await executor.inspect_execution("test", "work", owner_id="owner")
        assert snapshot.status == "unfinished"
        assert snapshot.operation_id == "job"
        with pytest.raises(ExecutionConflictError, match="explicit resume or recover"):
            await executor.execute("test", request, owner_id="owner")
        release.set()
        if explicit_recovery:
            result = await executor.execute(
                "test",
                ExecutionRequest(operation="recover", thread_id="work", expected_checkpoint_id=snapshot.checkpoint_id),
                owner_id="owner",
            )
        else:
            result = await executor.execute(
                "test", request, owner_id="owner", operation_id="job", retry=True, baseline_checkpoint_id=baselines[0]
            )
        assert result.output == {"value": 11}
        assert prepared == [0]


@pytest.mark.parametrize("successor", [False, True])
async def test_retry_commits_pending_task_writes_before_reporting_completion(tmp_path, monkeypatch, successor):
    calls = []

    def increment(state):
        calls.append("increment")
        return {"value": state["value"] + 1}

    def finish(state):
        calls.append("finish")
        return {"value": state["value"] + 10}

    builder = StateGraph(Counter)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    if successor:
        builder.add_node("finish", finish)
        builder.add_edge("increment", "finish")
        builder.add_edge("finish", END)
    else:
        builder.add_edge("increment", END)
    database = str(tmp_path / "pending-writes.sqlite")
    request = ExecutionRequest(thread_id="work", input={"value": 0})
    config = {"configurable": {"thread_id": storage_thread_id("owner", "test", "work")}}
    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        original_aput = saver.aput

        async def fail_after_task(config, checkpoint, metadata, new_versions):
            if metadata["step"] == 1:
                raise OSError("Worker lost before the completed task checkpoint")
            return await original_aput(config, checkpoint, metadata, new_versions)

        monkeypatch.setattr(saver, "aput", fail_after_task)
        executor = executor_for(builder.compile(checkpointer=saver))
        with pytest.raises(OSError, match="Worker lost"):
            await executor.execute("test", request, owner_id="owner", operation_id="job")
        saved = await saver.aget_tuple(config)
        assert saved.checkpoint["channel_values"]["value"] == 0
        assert any(channel == "value" and value == 1 for _, channel, value in saved.pending_writes)

    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        executor = executor_for(builder.compile(checkpointer=saver))
        snapshot = await executor.inspect_execution("test", "work", owner_id="owner")
        result = await executor.execute("test", request, owner_id="owner", operation_id="job", retry=True)
        assert result.status == "completed"
        assert result.output == {"value": 11 if successor else 1}
        assert snapshot.status == "unfinished"
        assert result.checkpoint_id != snapshot.checkpoint_id
        assert calls == (["increment", "finish"] if successor else ["increment"])
        assert await executor.execute("test", request, owner_id="owner", operation_id="job", retry=True) == result


async def test_functional_retry_commits_pending_public_output_without_repeating_work(tmp_path, monkeypatch):
    calls = []
    database = str(tmp_path / "functional-pending-writes.sqlite")
    request = ExecutionRequest(thread_id="work", input={"question": "calculate"})

    def workflow_for(saver):
        @entrypoint(checkpointer=saver)
        async def workflow(inputs):
            calls.append(inputs)
            return entrypoint.final(value={"answer": 42}, save={"private_context": "retained"})

        return workflow

    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        original_aput = saver.aput

        async def fail_after_output(config, checkpoint, metadata, new_versions):
            if checkpoint["channel_values"].get("__end__") == {"answer": 42}:
                raise OSError("Worker lost before the functional output checkpoint")
            return await original_aput(config, checkpoint, metadata, new_versions)

        monkeypatch.setattr(saver, "aput", fail_after_output)
        executor = executor_for(workflow_for(saver))
        with pytest.raises(OSError, match="Worker lost"):
            await executor.execute("test", request, owner_id="owner", operation_id="job")

    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        executor = executor_for(workflow_for(saver))
        snapshot = await executor.inspect_execution("test", "work", owner_id="owner")
        assert snapshot.status == "unfinished"
        result = await executor.execute("test", request, owner_id="owner", operation_id="job", retry=True)
        assert result.status == "completed"
        assert result.output == {"answer": 42}
        assert calls == [{"question": "calculate"}]
        assert (await executor.inspect_execution("test", "work", owner_id="owner")).output == result.output
        assert await executor.execute("test", request, owner_id="owner", operation_id="job", retry=True) == result


@pytest.mark.parametrize("send", [False, True])
@pytest.mark.parametrize("operation_marker", [False, True])
async def test_graph_upgrade_rejects_a_removed_pending_node(tmp_path, send, operation_marker):
    entered = asyncio.Event()
    calls = []

    async def unfinished(state):
        entered.set()
        await asyncio.Event().wait()

    builder = StateGraph(Counter)
    builder.add_node("old", unfinished)
    if send:
        builder.add_conditional_edges(START, lambda state: [Send("old", state)])
    else:
        builder.add_edge(START, "old")
    builder.add_edge("old", END)
    request = ExecutionRequest(thread_id="work", input={"value": 1})
    database = str(tmp_path / "graph-upgrade.sqlite")
    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        graph = builder.compile(checkpointer=saver)
        executor = executor_for(graph)
        if operation_marker:
            task = asyncio.create_task(executor.execute("test", request, owner_id="owner", operation_id="job"))
        else:
            # Checkpoints written before toolkit operation metadata remain protected.
            config = {"configurable": {"thread_id": storage_thread_id("owner", "test", "work")}}
            task = asyncio.create_task(graph.ainvoke(request.input, config, durability="sync"))
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    replacement = StateGraph(Counter)
    replacement.add_node("new", lambda state: calls.append("new") or {"value": 99})
    replacement.add_edge(START, "new")
    replacement.add_edge("new", END)
    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        executor = executor_for(replacement.compile(checkpointer=saver))
        with pytest.raises(ExecutionConflictError, match="incompatible with this graph"):
            await executor.inspect_execution("test", "work", owner_id="owner")
        with pytest.raises(ExecutionConflictError, match="incompatible with this graph"):
            await executor.execute("test", request, owner_id="owner", operation_id="job", retry=True)
    assert calls == []


async def test_functional_upgrade_rejects_a_renamed_unfinished_entrypoint(tmp_path):
    entered = asyncio.Event()
    request = ExecutionRequest(thread_id="work", input={"value": 1})
    database = str(tmp_path / "functional-upgrade.sqlite")
    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:

        @entrypoint(checkpointer=saver)
        async def old(inputs):
            entered.set()
            await asyncio.Event().wait()

        executor = executor_for(old)
        task = asyncio.create_task(executor.execute("test", request, owner_id="owner", operation_id="job"))
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    calls = []
    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:

        @entrypoint(checkpointer=saver)
        async def replacement(inputs):
            calls.append(inputs)
            return {"value": 99}

        executor = executor_for(replacement)
        with pytest.raises(ExecutionConflictError, match="incompatible with this graph"):
            await executor.execute("test", request, owner_id="owner", operation_id="job", retry=True)
    assert calls == []


@pytest.mark.parametrize("send", [False, True])
async def test_graph_upgrade_rejects_pending_writes_to_a_removed_node(tmp_path, monkeypatch, send):
    calls = []

    def graph_for(saver, name):
        builder = StateGraph(Counter)
        builder.add_node("route", lambda state: {"value": state["value"] + 1})
        builder.add_node(name, lambda state: calls.append(name) or {"value": 99})
        builder.add_edge(START, "route")
        if send:
            builder.add_conditional_edges("route", lambda state: [Send(name, state)])
        else:
            builder.add_edge("route", name)
        builder.add_edge(name, END)
        return builder.compile(checkpointer=saver)

    database = str(tmp_path / "pending-upgrade.sqlite")
    request = ExecutionRequest(thread_id="work", input={"value": 1})
    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        original = saver.aput

        async def fail_after_route(config, checkpoint, metadata, new_versions):
            if metadata["step"] == 1:
                raise OSError("Lost worker before scheduling the next node")
            return await original(config, checkpoint, metadata, new_versions)

        monkeypatch.setattr(saver, "aput", fail_after_route)
        executor = executor_for(graph_for(saver, "old"))
        with pytest.raises(OSError, match="Lost worker"):
            await executor.execute("test", request, owner_id="owner", operation_id="job")
    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        executor = executor_for(graph_for(saver, "replacement"))
        with pytest.raises(ExecutionConflictError, match="incompatible with this graph"):
            await executor.execute("test", request, owner_id="owner", operation_id="job", retry=True)
    assert calls == []


@pytest.mark.parametrize("functional", [False, True])
async def test_completed_checkpoint_allows_node_rename_and_new_work(tmp_path, functional):
    calls = []
    request = ExecutionRequest(thread_id="work", input={"value": 1})

    def graph_for(saver, name):
        async def step(inputs):
            calls.append(name)
            return {"value": inputs["value"] + 1}

        step.__name__ = name
        if functional:
            return entrypoint(checkpointer=saver)(step)
        builder = StateGraph(Counter)
        builder.add_node(name, step)
        builder.add_edge(START, name)
        builder.add_edge(name, END)
        return builder.compile(checkpointer=saver)

    async with CoordinatedSqliteSaver.from_conn_string(str(tmp_path / "completed-upgrade.sqlite")) as saver:
        old = executor_for(graph_for(saver, "old"))
        result = await old.execute("test", request, owner_id="owner", operation_id="first")
        current = executor_for(graph_for(saver, "current"))
        assert await current.execute("test", request, owner_id="owner", operation_id="first", retry=True) == result
        new = await current.execute("test", request, owner_id="owner", operation_id="second")
        assert new.status == "completed"
        assert new.output == {"value": 2}
    assert calls == ["old", "current"]


async def test_retry_before_input_checkpoint_rejects_an_intervening_operation(tmp_path):
    calls = []
    request = ExecutionRequest(thread_id="work", input={"value": 1})

    async def fail_before_execution(checkpoint_id):
        assert checkpoint_id is None
        raise OSError("Queue acknowledgement failed")

    async with CoordinatedSqliteSaver.from_conn_string(str(tmp_path / "before-input.sqlite")) as saver:
        executor = executor_for(counter_graph(saver, calls))
        with pytest.raises(OSError):
            await executor.execute(
                "test", request, owner_id="owner", operation_id="first", on_start=fail_before_execution
            )
        assert calls == []
        result = await executor.execute("test", request, owner_id="owner", operation_id="first", retry=True)
        assert result.output == {"value": 2}
        await executor.execute("test", request, owner_id="owner", operation_id="later")
        with pytest.raises(ExecutionConflictError, match="Another operation"):
            await executor.execute("test", request, owner_id="owner", operation_id="first", retry=True)
    assert calls == [1, 1]


async def test_resume_retry_recovers_after_restart_without_reapplying_the_answer(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()
    accepted = []
    baselines = []

    def approve(state):
        answer = interrupt({"question": "Choose a value."})
        accepted.append(answer)
        return {"value": answer}

    async def finish(state):
        entered.set()
        await release.wait()
        return {"value": state["value"] + 1}

    async def started(checkpoint_id):
        baselines.append(checkpoint_id)

    builder = StateGraph(Counter)
    builder.add_node("approve", approve)
    builder.add_node("finish", finish)
    builder.add_edge(START, "approve")
    builder.add_edge("approve", "finish")
    builder.add_edge("finish", END)
    database = str(tmp_path / "resume-recovery.sqlite")
    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        executor = executor_for(builder.compile(checkpointer=saver))
        question = await executor.execute(
            "test", ExecutionRequest(thread_id="work", input={"value": 0}), owner_id="owner"
        )
        request = ExecutionRequest(
            thread_id="work",
            operation="resume",
            expected_checkpoint_id=question.checkpoint_id,
            resume={question.interrupts[0]["id"]: 4},
        )
        task = asyncio.create_task(
            executor.execute("test", request, owner_id="owner", operation_id="answer", on_start=started)
        )
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async with CoordinatedSqliteSaver.from_conn_string(database) as saver:
        executor = executor_for(builder.compile(checkpointer=saver))
        snapshot = await executor.inspect_execution("test", "work", owner_id="owner")
        assert snapshot.checkpoint_id != question.checkpoint_id
        assert snapshot.operation_id == "answer"
        assert baselines == [question.checkpoint_id]
        release.set()
        result = await executor.execute(
            "test",
            request,
            owner_id="owner",
            operation_id="answer",
            retry=True,
            baseline_checkpoint_id=baselines[0],
        )
        assert result.output == {"value": 5}
        assert result.status == "completed"
    assert accepted == [4]


@pytest.mark.parametrize("legacy_invoke", [False, True])
async def test_retry_rejects_a_later_input_without_an_operation_marker(tmp_path, legacy_invoke):
    calls = []

    def reply(state):
        calls.append(state["messages"][-1].content)
        return {"messages": [AIMessage("ready")]}

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    async with CoordinatedSqliteSaver.from_conn_string(str(tmp_path / "mixed-input.sqlite")) as saver:
        executor = executor_for(builder.compile(checkpointer=saver))
        request = ExecutionRequest(thread_id="work", input={"messages": [{"role": "user", "content": "first"}]})
        await executor.execute("test", request, owner_id="owner", operation_id="first")
        if legacy_invoke:
            await executor.invoke(
                "test",
                {"message": "later"},
                thread_id=storage_thread_id("owner", "test", "work"),
                user_id="owner",
                trace_session_id="work",
            )
        else:
            await executor.execute(
                "test",
                ExecutionRequest(thread_id="work", input={"messages": [{"role": "user", "content": "later"}]}),
                owner_id="owner",
            )
        state = await executor.inspect_execution("test", "work", owner_id="owner")
        assert state.operation_id is None
        with pytest.raises(ExecutionConflictError, match="Another operation"):
            await executor.execute("test", request, owner_id="owner", operation_id="first", retry=True)
    assert calls == ["first", "later"]


async def test_worker_deadline_is_independent_from_request_timeout(monkeypatch):
    monkeypatch.setattr(settings, "REQUEST_TIMEOUT", 0.001)

    async def step(state):
        await asyncio.sleep(0.03)
        return {"value": 7}

    builder = StateGraph(Counter)
    builder.add_node("step", step)
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    executor = executor_for(builder.compile(checkpointer=InMemorySaver()))
    result = await executor.execute(
        "test", ExecutionRequest(thread_id="work", input={"value": 0}, timeout_seconds=1), owner_id="owner"
    )
    assert result.output == {"value": 7}
    with pytest.raises(TimeoutError):
        await executor.execute(
            "test", ExecutionRequest(thread_id="slow", input={"value": 0}, timeout_seconds=0.001), owner_id="owner"
        )
    assert not executor.concurrency._entries


@pytest.mark.parametrize("configured_durability", ["async", "exit"])
async def test_operation_marker_commits_before_the_first_node(monkeypatch, configured_durability):
    monkeypatch.setattr(settings, "CHECKPOINT_DURABILITY", configured_durability)

    class SlowSaver(InMemorySaver):
        async def aput(self, *args, **kwargs):
            await asyncio.sleep(0.01)
            return await super().aput(*args, **kwargs)

    saver = SlowSaver()
    key = storage_thread_id("owner", "test", "work")

    async def step(state):
        saved = await saver.aget_tuple({"configurable": {"thread_id": key}})
        assert saved is not None
        assert saved.metadata["lat_operation_id"] == "job"
        assert saved.checkpoint["channel_values"]["value"] == 2
        return {"value": 3}

    builder = StateGraph(Counter)
    builder.add_node("step", step)
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    executor = executor_for(builder.compile(checkpointer=saver))
    result = await executor.execute(
        "test", ExecutionRequest(thread_id="work", input={"value": 2}), owner_id="owner", operation_id="job"
    )
    assert result.output == {"value": 3}


async def test_mcp_resume_requires_the_existing_answer_envelope():
    def question(state):
        answer = interrupt({"type": "mcp_elicitation", "requests": [{"key": "answer", "mode": "form"}]})
        return {"value": answer["responses"]["answer"]["content"]["quantity"]}

    builder = StateGraph(Counter)
    builder.add_node("question", question)
    builder.add_edge(START, "question")
    builder.add_edge("question", END)
    executor = executor_for(builder.compile(checkpointer=InMemorySaver()))
    initial = await executor.execute("test", ExecutionRequest(thread_id="work", input={"value": 0}), owner_id="owner")
    interrupt_id = initial.interrupts[0]["id"]
    resume = ExecutionRequest(
        thread_id="work",
        operation="resume",
        expected_checkpoint_id=initial.checkpoint_id,
        resume={interrupt_id: {"quantity": 2}},
    )
    with pytest.raises(InputValidationError, match="responses object"):
        await executor.execute("test", resume, owner_id="owner")
    answer = {"responses": {"answer": {"action": "accept", "content": {"quantity": 2}}}}
    result = await executor.execute(
        "test", resume.model_copy(update={"resume": {interrupt_id: answer}}), owner_id="owner"
    )
    assert result.output == {"value": 2}
