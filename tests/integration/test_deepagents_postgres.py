"""Verify binary offloading with the toolkit's asynchronous PostgreSQL store."""

import asyncio
import base64
import hashlib

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_agent_toolkit.service.auth import storage_thread_id


pytest.importorskip("deepagents")

import test_postgres_reliability as postgres_tests
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from test_deepagents import ScriptedModel

from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph


pytestmark = [
    pytest.mark.postgres,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not postgres_tests.TEST_DSN, reason="Set LAT_TEST_POSTGRES_DSN to a disposable PostgreSQL database"
    ),
]
postgres_backend = postgres_tests.backend


def blob_namespace(runtime):
    """Use the scoped conversation key for external binary storage."""
    return ("document-assistant", "blobs", runtime.execution_info.thread_id)


def media_graph(model, store, saver):
    """Use the public APIs from the durable offloading example."""
    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/blobs/": StoreBackend(namespace=blob_namespace)},
    )
    graph = build_graph(model=model, backend=backend, store=store, offload_binary_content=True)
    graph.checkpointer = saver
    return graph


async def test_postgres_media_restores_after_reopening_store_and_saver(postgres_backend):
    """Restore media from a new pool and keep blobs after checkpoint deletion."""
    payload = b"\x89PNG\xffsynthetic PostgreSQL image"
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    owner = "user with spaces and \u041a\u0438\u0457\u0432"
    scoped_id = storage_thread_id(owner, "document-assistant", "public thread")
    config = {"configurable": {"thread_id": scoped_id, "user_id": owner}}
    namespace = ("document-assistant", "blobs", scoped_id)
    content = [
        {"type": "text", "text": "Read this image."},
        {"type": "image", "base64": encoded, "mime_type": "image/png"},
    ]
    model = ScriptedModel(profile={"image_inputs": True}, responses=[AIMessage("Image received.")])

    async with asyncio.timeout(20):
        async with postgres_backend.get_memory_store() as store, postgres_backend.get_checkpoint_saver() as saver:
            await store.setup()
            await saver.setup()
            graph = media_graph(model, store, saver)
            await graph.ainvoke({"messages": [HumanMessage(content)]}, config, durability="sync")
            snapshot = await graph.aget_state(config)
            assert snapshot.values["messages"][0].content[1] == {
                "type": "image",
                "mime_type": "image/png",
                "deepagents_blob": digest,
            }
            assert encoded not in str(snapshot.values)
            item = await store.aget(namespace, "/" + digest)
            assert item.value["encoding"] == "base64"
            assert base64.b64decode(item.value["content"]) == payload
            other_owner = storage_thread_id("different-owner", "document-assistant", "public thread")
            assert await store.aget(("document-assistant", "blobs", other_owner), "/" + digest) is None
        assert store.conn.closed
        assert saver.conn.closed

        restored_model = ScriptedModel(profile={"image_inputs": True}, responses=[AIMessage("Media restored.")])
        async with postgres_backend.get_memory_store() as store, postgres_backend.get_checkpoint_saver() as saver:
            await store.setup()
            await saver.setup()
            graph = media_graph(restored_model, store, saver)
            result = await graph.ainvoke({"messages": [HumanMessage("Continue.")]}, config, durability="sync")
            assert result["messages"][-1].content == "Media restored."
            restored_image = next(
                message for message in restored_model.seen[-1] if isinstance(message, HumanMessage)
            ).content[1]
            assert restored_image["base64"] == encoded
            assert "deepagents_blob" not in restored_image
            await saver.adelete_thread(scoped_id)
            assert await saver.aget_tuple(config) is None
            assert await store.aget(namespace, "/" + digest) is not None
        assert store.conn.closed
        assert saver.conn.closed
