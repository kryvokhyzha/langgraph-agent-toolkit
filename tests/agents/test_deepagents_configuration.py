"""Check Deep Agents startup configuration and tool naming contracts."""

import importlib

import pytest
from langchain_core.tools import tool
from langgraph.store.memory import InMemoryStore

from langgraph_agent_toolkit.core.settings import settings


pytest.importorskip("deepagents")
backends = importlib.import_module("deepagents.backends")

build_graph = importlib.import_module("langgraph_agent_toolkit.agents.blueprints.deep_agent.agent").build_graph


@pytest.mark.parametrize("duplicate", [False, True])
def test_supplied_tools_cannot_replace_existing_tools(duplicate):
    @tool("external_read" if duplicate else "read_file")
    def external_read(path: str) -> str:
        """Read one external document."""
        raise AssertionError("A rejected tool must not run.")

    with pytest.raises(ValueError, match="Duplicate agent tool name|cannot replace Deep Agents built-ins"):
        build_graph([external_read] * (2 if duplicate else 1))


@pytest.mark.parametrize(
    "name",
    ["delete", "start_async_task", "check_async_task", "update_async_task", "cancel_async_task", "list_async_tasks"],
)
def test_supplied_tools_cannot_replace_async_or_delete_tools(name):
    @tool(name)
    def external_action() -> str:
        """Run one external action."""
        raise AssertionError("A rejected tool must not run.")

    with pytest.raises(ValueError, match="cannot replace Deep Agents built-ins"):
        build_graph([external_action])


def test_startup_uses_the_named_model_configuration(monkeypatch):
    monkeypatch.setattr(settings, "USE_FAKE_MODEL", False)
    monkeypatch.setattr(settings, "MODEL_CONFIGS", {"deep_agent": {"provider": "fake", "name": "configured-demo"}})
    monkeypatch.setattr(settings, "OPENAI_MODEL_NAME", None)
    result = build_graph().invoke({"messages": [{"role": "user", "content": "Check the configured model."}]})
    assert result["messages"][-1].content == "This is a test response from the fake model."


@pytest.mark.parametrize("configuration", ["default", "state", "unrouted", "state-route", "nested-state"])
def test_binary_offloading_rejects_checkpoint_storage(tmp_path, configuration):
    external = backends.FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    state = backends.StateBackend()
    candidates = {
        "default": None,
        "state": state,
        "unrouted": backends.CompositeBackend(default=state, routes={"/documents/": external}),
        "state-route": backends.CompositeBackend(default=external, routes={"/blobs/": state}),
        "nested-state": backends.CompositeBackend(
            default=external,
            routes={"/blobs/": backends.CompositeBackend(default=state, routes={})},
        ),
    }
    with pytest.raises(ValueError, match="stores /blobs/ outside checkpoint state"):
        build_graph(backend=candidates[configuration], offload_binary_content=True)


@pytest.mark.parametrize("route", ["/blobs", "/blobs/"])
def test_binary_offloading_accepts_external_blob_route(tmp_path, route):
    backend = backends.CompositeBackend(
        default=backends.StateBackend(),
        routes={route: backends.FilesystemBackend(root_dir=tmp_path, virtual_mode=True)},
    )
    graph = build_graph(backend=backend, offload_binary_content=True)
    assert graph.checkpointer is None


def test_binary_offloading_checks_the_configured_artifact_route(tmp_path):
    external = backends.FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    backend = backends.CompositeBackend(
        default=backends.StateBackend(), routes={"/artifacts/": external}, artifacts_root="/artifacts"
    )
    assert build_graph(backend=backend, offload_binary_content=True).checkpointer is None
    backend = backends.CompositeBackend(
        default=backends.StateBackend(), routes={"/blobs/": external}, artifacts_root="/artifacts"
    )
    with pytest.raises(ValueError, match="stores /artifacts/blobs/ outside checkpoint state"):
        build_graph(backend=backend, offload_binary_content=True)


@pytest.mark.parametrize("explicit_backend_store", [False, True])
def test_binary_offloading_requires_an_explicit_store_argument(explicit_backend_store):
    store = InMemoryStore()
    backend = backends.CompositeBackend(
        default=backends.StateBackend(),
        routes={
            "/blobs/": backends.StoreBackend(
                namespace=lambda runtime: ("binary", runtime.execution_info.thread_id),
                store=store if explicit_backend_store else None,
            )
        },
    )
    with pytest.raises(ValueError, match="Pass store= to build_graph"):
        build_graph(backend=backend, offload_binary_content=True)
    assert build_graph(backend=backend, offload_binary_content=True, store=store).store is store
