"""Check real local HTTP faults through the model factory and OpenAI SDK."""

import asyncio
import json
import socket
import sys
import threading
import time
import types
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import httpx2
import pytest
from langchain.agents import create_agent
from langchain_core.exceptions import ModelAPIError, ModelAuthenticationError, ModelConnectionError, ModelTimeoutError
from pydantic import SecretStr

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.core.models import (
    CompletionModelFactory,
    EmbeddingModelFactory,
    LLMTransportConfig,
    LLMTransportManager,
)
from langgraph_agent_toolkit.core.models.transport import _CONNECTION_ENV, current_llm_transport_manager
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import constants
from langgraph_agent_toolkit.service.handler import create_app


@pytest.fixture(autouse=True)
def disable_remote_tracing(monkeypatch):
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.setenv(name, "false")


class LocalModelServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self):
        super().__init__(("127.0.0.1", 0), ModelHandler)
        self.records = []
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.started = threading.Event()
        self.behavior = lambda handler, body: handler.completion(body)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class ModelHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            # Closing a client can reset a persistent connection between requests.
            pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.records.append(
                {
                    "path": self.path,
                    "port": self.client_address[1],
                    "authorization": self.headers.get("Authorization"),
                    "api_key": self.headers.get("api-key"),
                    "project": self.headers.get("OpenAI-Project"),
                    "cookie": self.headers.get("Cookie"),
                    "body": body,
                }
            )
            self.server.active += 1
            self.server.peak = max(self.server.peak, self.server.active)
        self.server.started.set()
        try:
            self.server.behavior(self, body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self.server.lock:
                self.server.active -= 1

    def send_json(self, body, status=200, headers=None):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def completion(self, body, headers=None):
        self.send_json(
            {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "local answer"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
            headers=headers,
        )

    def start_stream(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(
            b"data: "
            + json.dumps(
                {
                    "id": "chatcmpl-local",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": "first"}, "finish_reason": None}
                    ],
                }
            ).encode()
            + b"\n\n"
        )
        self.wfile.flush()


@pytest.fixture
def model_server(monkeypatch):
    for name in _CONNECTION_ENV:
        monkeypatch.delenv(name, raising=False)
    server = LocalModelServer()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def model(server, provider="openai", **kwargs):
    params = {"api_key": "test-only-key", "streaming": False}
    if provider == "azure_openai":
        params.update(azure_endpoint=server.url, azure_deployment="local-deployment", api_version="2025-01-01-preview")
    else:
        params["base_url"] = server.url + "/v1"
    params.update(kwargs)
    return CompletionModelFactory.create(provider, "local-model", model_parameter_values=(), **params)


@pytest.mark.parametrize("transport", ["httpx", "aiohttp"])
@pytest.mark.parametrize("provider", ["openai", "azure_openai"])
async def test_reuses_bounded_connections_for_prebuilt_models_and_sync_calls(model_server, transport, provider):
    def respond(handler, body):
        time.sleep(0.04)
        handler.completion(body)

    model_server.behavior = respond
    # Built-in configurable models can be imported before the service starts.
    configurable = model(model_server, provider)
    config = LLMTransportConfig(async_transport=transport, max_connections=2, max_keepalive_connections=2)
    async with LLMTransportManager(config) as manager:
        with manager.bind():
            results = await asyncio.gather(
                *[
                    configurable.ainvoke("hello", config={"configurable": {"agent_temperature": index / 100}})
                    for index in range(10)
                ]
            )
            assert all(result.content == "local answer" for result in results)
            assert 1 < model_server.peak <= 2
            assert len({record["port"] for record in model_server.records}) == 2
            sync_result = await asyncio.to_thread(configurable.invoke, "sync hello")
            assert sync_result.content == "local answer"
            assert len(manager._pools) == 1
            pair = next(iter(manager._pools.values()))
            assert not any(client.is_closed for client in pair)
            assert {record["body"].get("temperature") for record in model_server.records[:10]} == {
                index / 100 for index in range(10)
            }
            expected_path = (
                "/openai/deployments/local-deployment/chat/completions?api-version=2025-01-01-preview"
                if (provider == "azure_openai")
                else "/v1/chat/completions"
            )
            assert {record["path"] for record in model_server.records} == {expected_path}
            credential_field = "api_key" if provider == "azure_openai" else "authorization"
            expected_credential = "test-only-key" if provider == "azure_openai" else "Bearer test-only-key"
            assert {record[credential_field] for record in model_server.records} == {expected_credential}
    assert all(client.is_closed for client in pair)
    assert current_llm_transport_manager() is None


@pytest.mark.parametrize("transport", ["httpx", "aiohttp"])
async def test_retries_are_bounded_and_authentication_errors_are_not_retried(model_server, transport):
    counts = Counter()

    def respond(handler, body):
        prompt = body["messages"][0]["content"]
        counts[prompt] += 1
        if prompt == "transient" and counts[prompt] > 1:
            handler.completion(body)
        else:
            handler.send_json(
                {"error": {"message": "local fault", "type": "test_error"}},
                status=401 if prompt == "unauthorized" else 503,
                headers={"retry-after-ms": "1"},
            )

    model_server.behavior = respond
    async with LLMTransportManager(LLMTransportConfig(async_transport=transport, max_retries=1)) as manager:
        with manager.bind():
            configurable = model(model_server)
            assert (await configurable.ainvoke("transient")).content == "local answer"
            with pytest.raises(ModelAPIError):
                await configurable.ainvoke("persistent")
            with pytest.raises(ModelAuthenticationError):
                await configurable.ainvoke("unauthorized")
    assert counts == {"transient": 2, "persistent": 2, "unauthorized": 1}


@pytest.mark.parametrize("transport", ["httpx", "aiohttp"])
@pytest.mark.parametrize("fault", ["disconnect", "read_timeout"])
async def test_network_faults_stop_after_retry_limit_and_next_call_recovers(model_server, transport, fault):
    def respond(handler, body):
        if body["messages"][0]["content"] == "fail":
            if fault == "disconnect":
                handler.close_connection = True
                handler.connection.shutdown(socket.SHUT_RDWR)
                handler.connection.close()
                return
            model_server.release.wait(timeout=5)
        handler.completion(body)

    model_server.behavior = respond
    config = LLMTransportConfig(async_transport=transport, read_timeout=0.1, max_retries=1)
    async with LLMTransportManager(config) as manager:
        with manager.bind():
            configurable = model(model_server)
            # The SDK aiohttp adapter maps a server disconnect to ConnectTimeout.
            expected_error = (
                ModelConnectionError if fault == "disconnect" and transport == "httpx" else ModelTimeoutError
            )
            with pytest.raises(expected_error):
                await asyncio.wait_for(configurable.ainvoke("fail"), timeout=3)
            assert len(model_server.records) == 2
            model_server.release.set()
            assert (await configurable.ainvoke("recover")).content == "local answer"
    assert len(model_server.records) == 3


@pytest.mark.parametrize("transport", ["httpx", "aiohttp"])
async def test_pool_timeout_and_cancelled_request_release_capacity(model_server, transport):
    def respond(handler, body):
        if body["messages"][0]["content"] == "hold":
            assert model_server.release.wait(timeout=5)
        handler.completion(body)

    model_server.behavior = respond
    config = LLMTransportConfig(
        async_transport=transport, max_connections=1, max_keepalive_connections=1, max_retries=0, pool_timeout=0.1
    )
    async with LLMTransportManager(config) as manager:
        with manager.bind():
            configurable = model(model_server)
            pending = asyncio.create_task(configurable.ainvoke("hold"))
            try:
                assert await asyncio.to_thread(model_server.started.wait, 3)
                with pytest.raises(ModelTimeoutError):
                    await asyncio.wait_for(configurable.ainvoke("queue full"), timeout=2)
                assert len(model_server.records) == 1
            finally:
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                model_server.release.set()
            assert (await configurable.ainvoke("recovered")).content == "local answer"
    assert [record["body"]["messages"][0]["content"] for record in model_server.records] == ["hold", "recovered"]


@pytest.mark.parametrize("transport", ["httpx", "aiohttp"])
async def test_stream_read_timeout_does_not_replay_received_tokens(model_server, transport):
    def respond(handler, body):
        handler.start_stream(body)
        model_server.release.wait(timeout=5)

    model_server.behavior = respond
    config = LLMTransportConfig(async_transport=transport, max_retries=2, read_timeout=0.15)
    async with LLMTransportManager(config) as manager:
        with manager.bind():
            chunks = []
            with pytest.raises(ModelTimeoutError) as error:
                async for chunk in model(model_server, streaming=True).astream("stall after first token"):
                    chunks.append(chunk.content)
            assert isinstance(error.value.__cause__, httpx2.ReadTimeout)
            assert "".join(chunks) == "first"
            assert len(model_server.records) == 1
    model_server.release.set()


@pytest.mark.parametrize("transport", ["httpx", "aiohttp"])
async def test_proxy_route_credentials_and_cookies_remain_isolated(model_server, transport, monkeypatch):
    def respond(handler, body):
        handler.completion(body, headers={"Set-Cookie": "session=first-credential; Path=/"})

    model_server.behavior = respond
    async with LLMTransportManager(LLMTransportConfig(async_transport=transport)) as manager:
        with manager.bind():
            first = model(model_server, base_url="http://model.invalid/v1", openai_proxy=model_server.url)
            second = model(
                model_server,
                base_url="http://model.invalid/v1",
                openai_proxy=model_server.url,
                api_key="different-test-key",
            )
            await first.ainvoke("first")
            await first.ainvoke("second")
            await second.ainvoke("third")
            monkeypatch.setenv("OPENAI_PROJECT_ID", "other-test-project")
            await first.ainvoke("different project")
            assert len(manager._pools) == 3
    records = model_server.records
    assert all(record["path"] == "http://model.invalid/v1/chat/completions" for record in records)
    assert [record["cookie"] for record in records] == [None, "session=first-credential", None, None]
    assert [record["project"] for record in records] == [None, None, None, "other-test-project"]
    assert [record["authorization"] for record in records] == [
        "Bearer test-only-key",
        "Bearer test-only-key",
        "Bearer different-test-key",
        "Bearer test-only-key",
    ]


async def test_explicit_client_ownership_and_options_are_unchanged(model_server):
    async with httpx.AsyncClient() as caller_client:
        async with LLMTransportManager() as manager:
            with manager.bind():
                configurable = model(model_server, http_async_client=caller_client, timeout=7.5, max_retries=0)
                concrete = configurable._model()
                assert concrete.http_async_client is caller_client
                assert concrete.request_timeout == 7.5
                assert concrete.max_retries == 0
                assert (await configurable.ainvoke("custom client")).content == "local answer"
                assert not manager._pools
        assert not caller_client.is_closed
    assert caller_client.is_closed


async def test_explicit_model_timeouts_and_retries_override_manager_defaults(model_server):
    async with LLMTransportManager() as manager:
        with manager.bind():
            concrete = model(model_server, timeout=0.5, max_retries=0)._model()
            assert concrete.request_timeout == 0.5
            assert concrete.max_retries == 0
            assert (await concrete.ainvoke("explicit options")).content == "local answer"


@pytest.mark.parametrize("provider", ["openai", "azure_openai"])
@pytest.mark.parametrize("transport", ["httpx", "aiohttp"])
async def test_embedding_calls_share_owned_connections_and_close(model_server, provider, transport):
    def respond(handler, body):
        handler.send_json(
            {
                "object": "list",
                "model": body["model"],
                "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        )

    model_server.behavior = respond
    params = {"api_key": "test-only-key", "check_embedding_ctx_length": False}
    if provider == "azure_openai":
        params.update(
            azure_endpoint=model_server.url, azure_deployment="local-embedding", api_version="2025-01-01-preview"
        )
    else:
        params["base_url"] = model_server.url + "/v1"
    async with LLMTransportManager(LLMTransportConfig(async_transport=transport)) as manager:
        with manager.bind():
            for _ in range(2):
                embeddings = EmbeddingModelFactory.create(provider, "local-embedding", **params)
                assert await embeddings.aembed_query("hello") == [0.25, 0.5]
            assert len({record["port"] for record in model_server.records}) == 1
            assert await asyncio.to_thread(embeddings.embed_query, "sync hello") == [0.25, 0.5]
            assert len(manager._pools) == 1
            pair = next(iter(manager._pools.values()))
    assert all(client.is_closed for client in pair)
    expected_path = (
        "/openai/deployments/local-embedding/embeddings?api-version=2025-01-01-preview"
        if (provider == "azure_openai")
        else "/v1/embeddings"
    )
    assert {record["path"] for record in model_server.records} == {expected_path}


async def test_pool_limit_and_cross_loop_use_fail_without_closing_active_clients(model_server):
    async with LLMTransportManager(LLMTransportConfig(max_pools=1)) as manager:
        with manager.bind():
            configurable = model(model_server)
            assert (await configurable.ainvoke("first")).content == "local answer"
            with pytest.raises(RuntimeError, match="pool limit"):
                model(model_server, api_key="different-test-key")._model()

            async def wrong_loop():
                with manager.bind():
                    await configurable.ainvoke("wrong loop")

            with pytest.raises(RuntimeError, match="another event loop"):
                await asyncio.to_thread(asyncio.run, wrong_loop())
            assert (await configurable.ainvoke("still open")).content == "local answer"
    with pytest.raises(RuntimeError, match="not open"):
        with manager.bind():
            pass


async def test_stalled_sync_close_has_bounded_shutdown_and_does_not_block_next_manager(model_server, monkeypatch):
    manager = await LLMTransportManager(LLMTransportConfig(shutdown_timeout=0.1)).__aenter__()
    with manager.bind():
        assert (await model(model_server).ainvoke("first manager")).content == "local answer"
    synchronous, asynchronous = next(iter(manager._pools.values()))
    original_close = synchronous.close
    close_started = threading.Event()
    close_release = threading.Event()
    close_finished = threading.Event()
    close_threads = []

    def stalled_close():
        close_threads.append(threading.current_thread())
        close_started.set()
        try:
            close_release.wait(timeout=5)
            original_close()
        finally:
            close_finished.set()

    monkeypatch.setattr(synchronous, "close", stalled_close)
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(manager.aclose(), timeout=1)
        assert close_started.is_set()
        assert not close_finished.is_set()
        assert close_threads[0].daemon
        assert asynchronous.is_closed
        async with LLMTransportManager() as replacement:
            with replacement.bind():
                assert (await model(model_server).ainvoke("replacement manager")).content == "local answer"
    finally:
        close_release.set()
        async with asyncio.timeout(2):
            while not close_finished.is_set():
                await asyncio.sleep(0.01)
        close_threads[0].join(timeout=1)
        assert not close_threads[0].is_alive()
    assert synchronous.is_closed


async def test_sync_close_failure_does_not_skip_other_client_cleanup(model_server, monkeypatch):
    manager = await LLMTransportManager().__aenter__()
    with manager.bind():
        concrete = model(model_server)._model()
    synchronous, asynchronous = next(iter(manager._pools.values()))
    original_close = synchronous.close

    def failed_close():
        original_close()
        raise RuntimeError("local close fault")

    monkeypatch.setattr(synchronous, "close", failed_close)
    with pytest.raises(RuntimeError, match="did not close"):
        await manager.aclose()
    assert synchronous.is_closed and asynchronous.is_closed
    assert concrete.http_client is synchronous


@pytest.mark.parametrize("transport", ["httpx", "aiohttp"])
async def test_two_service_lifespans_rebuild_preimported_concrete_model_with_fresh_clients(
    model_server, monkeypatch, transport
):
    from openai import DefaultAioHttpClient, DefaultAsyncHttpxClient

    created = []

    def build_graph(extra_tools=()):
        concrete = model(model_server, configurable_fields=(), config_prefix="")
        created.append(concrete)
        return create_agent(concrete, tools=list(extra_tools))

    module = types.ModuleType("local_transport_lifecycle_agent")
    module.agent = Agent(
        name="transport-agent", description="Local transport test", graph=build_graph(), graph_factory=build_graph
    )
    original_graph = module.agent.graph
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(constants, "_runtime_default_agent", None)
    overrides = {
        "AGENT_PATHS": [f"{module.__name__}:agent"],
        "DEFAULT_AGENT": "transport-agent",
        "MEMORY_BACKEND": None,
        "MCP_SERVERS": {},
        "MCP_AGENT_SERVERS": {},
        "OBSERVABILITY_BACKEND": None,
        "AUTH_MODE": "trusted",
        "AUTH_SECRET": SecretStr("test-service-token"),
        "AUTH_USERS": {},
        "LLM_HTTP_ASYNC_TRANSPORT": transport,
    }
    for name, value in overrides.items():
        monkeypatch.setattr(settings, name, value)
    app = create_app()
    old_pair = None
    try:
        for iteration in range(2):
            async with app.router.lifespan_context(app):
                manager = app.state.llm_transport_manager
                assert len(manager._pools) == 1
                pair = next(iter(manager._pools.values()))
                client_type = DefaultAioHttpClient if transport == "aiohttp" else DefaultAsyncHttpxClient
                assert isinstance(pair[1], client_type)
                if old_pair is not None:
                    assert pair[0] is not old_pair[0] and pair[1] is not old_pair[1]
                    assert all(client.is_closed for client in old_pair)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app)) as http_client:
                    async with AgentClient(
                        base_url="http://toolkit.invalid",
                        agent="transport-agent",
                        get_info=False,
                        async_http_client=http_client,
                        auth_secret="test-service-token",
                    ) as client:
                        answer = await client.ainvoke(
                            {"message": "hello"}, thread_id=f"thread-{iteration}", user_id="test-user"
                        )
                assert answer.content == "local answer"
                assert module.agent.graph is original_graph
                assert not created[0].http_async_client
            assert all(client.is_closed for client in pair)
            assert app.state.llm_transport_manager is None
            old_pair = pair
        assert len(created) == 3
        assert len(model_server.records) == 2
    finally:
        # The pre-imported standalone model remains caller-owned.
        created[0].root_client.close()
        await created[0].root_async_client.close()


@pytest.mark.parametrize("provider", ["openai", "azure_openai"])
@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"LANGSMITH_GATEWAY": "true"},
        {"OPENAI_API_BASE": ""},
        {"OPENAI_API_BASE": "https://gateway.invalid"},
        {"OPENAI_PROXY": ""},
    ],
)
async def test_managed_transport_preserves_automatic_stream_usage(monkeypatch, provider, environment):
    for name in _CONNECTION_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    params = {"api_key": "test-only-key", "configurable_fields": (), "config_prefix": "", "model_parameter_values": ()}
    if provider == "azure_openai":
        params.update(azure_endpoint="https://azure.invalid", api_version="2025-01-01-preview")
    baseline = CompletionModelFactory.create(provider, "local-model", **params)
    async with LLMTransportManager() as manager:
        with manager.bind():
            managed = CompletionModelFactory.create(provider, "local-model", **params)
            assert managed.stream_usage == baseline.stream_usage
    # Default Azure clients are separate SDK resources. No requests were made.
    if provider == "azure_openai":
        baseline.root_client.close()
        await baseline.root_async_client.close()
