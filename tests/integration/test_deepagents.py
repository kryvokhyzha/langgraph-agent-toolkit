"""Run DeepAgents tools through real graphs and the toolkit service."""

import base64
import hashlib
import importlib
import sys
from types import ModuleType
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field, SecretStr

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core.memory.types import MemoryBackends
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import constants
from langgraph_agent_toolkit.helper.types import EnvironmentMode
from langgraph_agent_toolkit.service.auth import storage_thread_id
from langgraph_agent_toolkit.service.handler import create_app


pytest.importorskip("deepagents")
blueprint = importlib.import_module("langgraph_agent_toolkit.agents.blueprints.deep_agent.agent")
backends = importlib.import_module("deepagents.backends")
filesystem_middleware = importlib.import_module("deepagents.middleware.filesystem").FilesystemMiddleware
HEADERS = {"Authorization": "Bearer deepagents-test-credential"}


class ScriptedModel(FakeMessagesListChatModel):
    seen: list = Field(default_factory=list)
    bound_tool_names: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names.append({tool.name for tool in tools})
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

    def get_num_tokens(self, text):
        return len(text.split())

    def get_num_tokens_from_messages(self, messages, tools=None):
        return sum(self.get_num_tokens(str(message.content)) for message in messages)


def call(name, **arguments):
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": uuid4().hex}])


def tool_results(messages, name):
    return [message for message in messages if isinstance(message, ToolMessage) and message.name == name]


def configure_service(monkeypatch, tmp_path, graph):
    module = ModuleType("lat_deepagents_test_" + uuid4().hex)
    module.agent = Agent("deep-agent", "A DeepAgents integration test.", graph)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for key, value in {
        "ENV_MODE": EnvironmentMode.PRODUCTION,
        "AUTH_MODE": "trusted",
        "AUTH_SECRET": SecretStr("deepagents-test-credential"),
        "AUTH_USERS": {},
        "MEMORY_BACKEND": MemoryBackends.SQLITE,
        "SQLITE_DB_PATH": str(tmp_path / "deepagents.sqlite"),
        "OBSERVABILITY_BACKEND": ObservabilityBackend.EMPTY,
        "AGENT_PATHS": [f"{module.__name__}:agent"],
        "DEFAULT_AGENT": "deep-agent",
        "MCP_SERVERS": {},
        "MCP_AGENT_SERVERS": {},
    }.items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(constants, "_runtime_default_agent", "deep-agent")
    return module


def invoke(client, message, *, thread="first-thread", user="user-1"):
    response = client.post(
        "/deep-agent/invoke",
        headers=HEADERS,
        json={"input": {"message": message}, "thread_id": thread, "user_id": user},
    )
    assert response.status_code == 200, response.text
    return response.json()


async def test_planning_and_virtual_files_do_not_write_to_the_host(tmp_path):
    host_path = tmp_path / "host-must-not-change.txt"
    model = ScriptedModel(
        responses=[
            call("write_todos", todos=[{"content": "Prepare note", "status": "in_progress"}]),
            call("write_file", file_path=str(host_path), content="The virtual note."),
            call("read_file", file_path=str(host_path)),
            call("write_todos", todos=[{"content": "Prepare note", "status": "completed"}]),
            AIMessage(content="The note is ready."),
        ]
    )
    graph = blueprint.build_graph(model=model)
    result = await graph.ainvoke({"messages": [HumanMessage("Prepare a note.")]})

    assert result["messages"][-1].content == "The note is ready."
    assert "The virtual note." in tool_results(result["messages"], "read_file")[0].content
    assert result["todos"] == [{"content": "Prepare note", "status": "completed"}]
    assert not host_path.exists()
    assert model.bound_tool_names
    assert all("execute" not in names for names in model.bound_tool_names)
    assert graph.checkpointer is None


def test_virtual_files_survive_service_restart_and_remain_private(monkeypatch, tmp_path):
    writer = ScriptedModel(
        responses=[
            call("write_file", file_path="/private-note.txt", content="User one private note."),
            AIMessage("Saved."),
        ]
    )
    module = configure_service(monkeypatch, tmp_path, blueprint.build_graph(model=writer))
    with TestClient(create_app()) as client:
        assert invoke(client, "Save a private note.")["content"] == "Saved."

    reader = ScriptedModel(responses=[call("read_file", file_path="/private-note.txt"), AIMessage("Read complete.")])
    module.agent.graph = blueprint.build_graph(model=reader)
    with TestClient(create_app()) as client:
        invoke(client, "Read my note.")
        recovered = tool_results(reader.seen[-1], "read_file")[-1]
        assert recovered.status == "success"
        assert "User one private note." in recovered.content

        invoke(client, "Read that path.", thread="second-thread")
        separate_thread = tool_results(reader.seen[-1], "read_file")[-1]
        assert separate_thread.status == "error"
        assert all("User one private note." not in message.model_dump_json() for message in reader.seen[-1])

        invoke(client, "Read that path.", user="user-2")
        separate_user = tool_results(reader.seen[-1], "read_file")[-1]
        assert separate_user.status == "error"
        assert all("User one private note." not in message.model_dump_json() for message in reader.seen[-1])


async def test_subagent_returns_a_file_without_exposing_its_private_conversation():
    specialist = ScriptedModel(
        responses=[
            call("write_file", file_path="/result.txt", content="The specialist result."),
            AIMessage(content="The result is in /result.txt."),
        ]
    )
    parent = ScriptedModel(
        responses=[
            call("task", subagent_type="specialist", description="Write the requested result."),
            call("read_file", file_path="/result.txt"),
            AIMessage(content="The delegated task is complete."),
        ]
    )
    graph = blueprint.build_graph(
        model=parent,
        subagents=[{"name": "specialist", "description": "Write a result file.", "model": specialist}],
    )
    result = await graph.ainvoke({"messages": [HumanMessage("Parent-only detail: private planning context.")]})

    delegated = tool_results(result["messages"], "task")
    assert len(delegated) == 1
    assert "The result is in /result.txt." in delegated[0].content
    assert "The specialist result." in tool_results(result["messages"], "read_file")[0].content
    assert not tool_results(result["messages"], "write_file")
    assert all("Parent-only detail" not in str(message.content) for message in specialist.seen[0])
    assert result["messages"][-1].content == "The delegated task is complete."


def test_file_approval_survives_restart_and_executes_only_after_resume(monkeypatch, tmp_path):
    model = ScriptedModel(
        responses=[call("write_file", file_path="/approved.txt", content="Approved content."), AIMessage("Written.")]
    )
    module = configure_service(
        monkeypatch, tmp_path, blueprint.build_graph(model=model, interrupt_on={"write_file": True})
    )
    with TestClient(create_app()) as client:
        paused = invoke(client, "Write the file after approval.")
        requests = paused["custom_data"]["interrupts"][0]["value"]["action_requests"]
        assert requests[0]["name"] == "write_file"
        graph = client.app.state.agent_executor.get_agent("deep-agent").graph
        config = {"configurable": {"thread_id": storage_thread_id("user-1", "deep-agent", "first-thread")}}
        state = client.portal.call(graph.aget_state, config)
        assert "/approved.txt" not in state.values.get("files", {})

    resumed_model = ScriptedModel(
        responses=[call("read_file", file_path="/approved.txt"), AIMessage("Approved and read.")]
    )
    module.agent.graph = blueprint.build_graph(model=resumed_model, interrupt_on={"write_file": True})
    with TestClient(create_app()) as client:
        completed = invoke(client, "approve")
        assert completed["content"] == "Approved and read."
        writes = tool_results(resumed_model.seen[-1], "write_file")
        assert len(writes) == 1
        assert writes[0].status == "success"
        assert "Approved content." in tool_results(resumed_model.seen[-1], "read_file")[0].content


@pytest.mark.parametrize("fail_after_write", [False, True])
async def test_mcp_tool_runs_once_and_closes_its_connection(monkeypatch, fail_after_write):
    fastmcp = pytest.importorskip("fastmcp")
    adapter_class = pytest.importorskip("langchain.mcp").MCPAdapter
    writes = []
    server = fastmcp.FastMCP("deepagents-local-test")

    @server.tool
    async def remote_write(value: str) -> dict:
        """Store one value in the test server."""
        writes.append(value)
        if fail_after_write:
            raise RuntimeError("The write completed but the response failed.")
        return {"saved": value}

    adapter = adapter_class(fastmcp.Client(server, timeout=5, init_timeout=5))
    tools = await adapter.list_tools()
    assert not adapter.client.is_connected()
    model = ScriptedModel(responses=[call("remote_write", value="one"), AIMessage("Tool attempt complete.")])
    monkeypatch.setattr(blueprint, "create_model", lambda: model)
    graph = blueprint.deep_agent.graph_factory(tools)

    result = await graph.ainvoke({"messages": [HumanMessage("Store one value.")]})

    assert writes == ["one"]
    messages = tool_results(result["messages"], "remote_write")
    assert len(messages) == 1
    assert messages[0].status == ("error" if fail_after_write else "success")
    if not fail_after_write:
        assert messages[0].artifact == {"structured_content": {"saved": "one"}}
    assert not adapter.client.is_connected()


def binary_message(payload: bytes):
    return HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image."},
            {"type": "image", "base64": base64.b64encode(payload).decode("ascii"), "mime_type": "image/png"},
        ]
    )


def binary_backend(directory):
    return backends.CompositeBackend(
        default=backends.StateBackend(),
        routes={"/blobs/": backends.FilesystemBackend(root_dir=directory, virtual_mode=True)},
    )


async def test_binary_content_stays_inline_by_default():
    payload = b"synthetic image bytes"
    model = ScriptedModel(responses=[AIMessage("Image received.")])
    result = await blueprint.build_graph(model=model).ainvoke({"messages": [binary_message(payload)]})
    assert result["messages"][0].content[1]["base64"] == base64.b64encode(payload).decode("ascii")
    assert "deepagents_blob" not in result["messages"][0].content[1]


@pytest.mark.parametrize("blob_state", ["present", "missing", "corrupt"])
async def test_binary_references_survive_restart_and_handle_unavailable_blobs(tmp_path, blob_state):
    payload = b"synthetic image bytes" * 200
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    config = {"configurable": {"thread_id": "binary-restart"}}
    directory = tmp_path / "blobs"
    directory.mkdir()
    database = str(tmp_path / "checkpoints.sqlite")
    model = ScriptedModel(responses=[AIMessage("Image received.")])
    graph = blueprint.build_graph(model=model, backend=binary_backend(directory), offload_binary_content=True)
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        graph.checkpointer = saver
        await graph.ainvoke({"messages": [binary_message(payload)]}, config, durability="sync")
        checkpoint = await saver.aget_tuple(config)
        state = await graph.aget_state(config)
        saved_image = state.values["messages"][0].content[1]
        assert saved_image == {"type": "image", "mime_type": "image/png", "deepagents_blob": digest}
        assert encoded not in str(checkpoint.checkpoint)
        assert encoded not in str(state.values)
        assert "_blob_payloads" not in checkpoint.checkpoint["channel_values"]
        history = [state async for state in graph.aget_state_history(config)]
        assert any(encoded in str(state.values.get("messages")) for state in history)
        assert model.seen[-1][1].content[1]["base64"] == encoded
    assert (directory / digest).read_bytes() == payload

    if blob_state == "missing":
        (directory / digest).unlink()
    elif blob_state == "corrupt":
        (directory / digest).write_bytes(b"wrong content")

    restored_model = ScriptedModel(responses=[AIMessage("Conversation restored.")])
    restored = blueprint.build_graph(
        model=restored_model, backend=binary_backend(directory), offload_binary_content=True
    )
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        restored.checkpointer = saver
        result = await restored.ainvoke({"messages": [HumanMessage("Continue.")]}, config, durability="sync")
        assert result["messages"][-1].content == "Conversation restored."
        image = next(message for message in restored_model.seen[-1] if isinstance(message, HumanMessage)).content[1]
        if blob_state == "present":
            assert image["base64"] == encoded
            assert "deepagents_blob" not in image
        else:
            assert image["type"] == "text"
            assert "no longer available" in image["text"]
            assert encoded not in str(restored_model.seen[-1])
        saved = await restored.aget_state(config)
        assert saved.values["messages"][0].content[1]["deepagents_blob"] == digest
        await saver.adelete_thread(config["configurable"]["thread_id"])
        assert await saver.aget_tuple(config) is None
    if blob_state == "present":
        assert (directory / digest).read_bytes() == payload


def test_service_binary_storage_is_scoped_to_user_and_thread(monkeypatch, tmp_path):
    store = InMemoryStore()
    namespace = lambda runtime: ("binary", runtime.execution_info.thread_id)
    backend = backends.CompositeBackend(
        default=backends.StateBackend(), routes={"/blobs/": backends.StoreBackend(namespace=namespace)}
    )
    model = ScriptedModel(responses=[AIMessage("Image saved.")])
    graph = blueprint.build_graph(model=model, backend=backend, store=store, offload_binary_content=True)
    configure_service(monkeypatch, tmp_path, graph)
    payload = b"private synthetic image"
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    message = binary_message(payload).content
    with TestClient(create_app()) as client:
        for thread, user in [("first-thread", "user-1"), ("second-thread", "user-1"), ("first-thread", "user-2")]:
            invoke(client, message, thread=thread, user=user)
            key = storage_thread_id(user, "deep-agent", thread)
            assert store.get(("binary", key), "/" + digest) is not None
        assert len(store.list_namespaces()) == 3
        model.responses = [call("read_file", file_path="/blobs/" + digest), AIMessage("Read attempt complete.")]
        model.i = 0
        for thread, user in [("third-thread", "user-1"), ("first-thread", "user-3")]:
            invoke(client, "Read that blob.", thread=thread, user=user)
            assert tool_results(model.seen[-1], "read_file")[0].status == "error"
            assert all(encoded not in item.model_dump_json() for item in model.seen[-1])
        response = client.request(
            "DELETE",
            "/deep-agent/history/clear",
            headers=HEADERS,
            json={"thread_id": "first-thread", "user_id": "user-1"},
        )
        assert response.status_code == 200, response.text
        key = storage_thread_id("user-1", "deep-agent", "first-thread")
        assert store.get(("binary", key), "/" + digest) is not None
        model.responses = [AIMessage("The cleared conversation is empty.")]
        model.i = 0
        invoke(client, "The checkpoints were cleared.")
        assert all(digest not in item.model_dump_json() for item in model.seen[-1])
        assert all(encoded not in item.model_dump_json() for item in model.seen[-1])


async def test_binary_read_results_are_offloaded_and_restored_for_the_model(tmp_path):
    payload = b"synthetic image read result"
    encoded = base64.b64encode(payload).decode("ascii")
    backend = backends.FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    uploaded = await backend.aupload_files([("/image.png", payload)])
    assert uploaded[0].error is None
    model = ScriptedModel(responses=[call("read_file", file_path="/image.png"), AIMessage("Image read.")])
    graph = blueprint.build_graph(model=model, backend=backend, offload_binary_content=True)
    result = await graph.ainvoke({"messages": [HumanMessage("Read /image.png.")]})
    read = tool_results(result["messages"], "read_file")[0]
    assert read.content[0]["deepagents_blob"] == hashlib.sha256(payload).hexdigest()
    assert "base64" not in read.content[0]
    seen = tool_results(model.seen[-1], "read_file")[0]
    assert seen.content[0]["base64"] == encoded


async def test_failed_blob_upload_keeps_the_original_media(monkeypatch, tmp_path):
    backend = backends.FilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    async def unavailable(files):
        raise OSError("The blob store is unavailable.")

    monkeypatch.setattr(backend, "aupload_files", unavailable)
    payload = b"media that must stay available"
    encoded = base64.b64encode(payload).decode("ascii")
    model = ScriptedModel(responses=[AIMessage("Image received.")])
    graph = blueprint.build_graph(model=model, backend=backend, offload_binary_content=True)
    result = await graph.ainvoke({"messages": [binary_message(payload)]})
    assert result["messages"][0].content[1]["base64"] == encoded
    assert model.seen[-1][1].content[1]["base64"] == encoded


@pytest.mark.parametrize("offload_subagent", [False, True])
async def test_named_subagents_control_their_own_binary_offloading(tmp_path, offload_subagent):
    payload = b"synthetic specialist image"
    backend = backends.FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    await backend.aupload_files([("/image.png", payload)])
    specialist = ScriptedModel(responses=[call("read_file", file_path="/image.png"), AIMessage("Image inspected.")])
    parent = ScriptedModel(
        responses=[
            call("task", subagent_type="image-specialist", description="Read /image.png."),
            AIMessage("Task complete."),
        ]
    )
    subagent = {"name": "image-specialist", "description": "Inspect an image.", "model": specialist}
    if offload_subagent:
        subagent["middleware"] = [
            filesystem_middleware(backend=backend, tools=["read_file"], offload_binary_content=True)
        ]
    graph = blueprint.build_graph(model=parent, backend=backend, offload_binary_content=True, subagents=[subagent])
    result = await graph.ainvoke({"messages": [HumanMessage("Delegate the image task.")]})
    assert result["messages"][-1].content == "Task complete."
    assert tool_results(specialist.seen[-1], "read_file")[0].content[0]["base64"] == base64.b64encode(payload).decode(
        "ascii"
    )
    blob = tmp_path / "blobs" / hashlib.sha256(payload).hexdigest()
    assert blob.exists() is offload_subagent
    assert not tool_results(result["messages"], "read_file")
