"""Check model-provider failures through the installed SDK and HTTP service."""

import json

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import SecretStr

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core.memory.concurrency import ConversationCoordinator
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.logging import logger
from langgraph_agent_toolkit.helper.types import EnvironmentMode
from langgraph_agent_toolkit.service.exception_handlers import register_exception_handlers
from langgraph_agent_toolkit.service.handler import create_app


def service_for_node(node):
    builder = StateGraph(MessagesState)
    builder.add_node("model", node)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    executor = AgentExecutor.__new__(AgentExecutor)
    executor.agents = {"provider": Agent("provider", "Provider error test", builder.compile(), EmptyObservability())}
    executor.concurrency = ConversationCoordinator()
    app = create_app()
    app.state.agent_executor = executor
    return app


@pytest.fixture
def production_settings(monkeypatch):
    monkeypatch.setattr(settings, "ENV_MODE", EnvironmentMode.PRODUCTION)
    monkeypatch.setattr(settings, "AUTH_USERS", {})
    monkeypatch.setattr(settings, "AUTH_MODE", "token")
    monkeypatch.setattr(settings, "AUTH_SECRET", SecretStr("local-provider-test"))
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.setenv(name, "false")


@pytest.mark.parametrize(
    ("fault", "error_type", "status", "error_code", "detail"),
    [
        ("rate", ModelRateLimitError, 429, "model_rate_limit", "The model provider rate limit was exceeded"),
        ("connection", ModelConnectionError, 503, "model_unavailable", "The model provider is unavailable"),
        ("timeout", ModelTimeoutError, 504, "model_timeout", "The model provider request timed out"),
    ],
)
async def test_sdk_failures_keep_their_public_type_and_return_safe_status(
    production_settings, fault, error_type, status, error_code, detail
):
    ChatOpenAI = pytest.importorskip("langchain_openai").ChatOpenAI
    calls = []
    errors = []

    async def provider(request):
        calls.append(request)
        if fault == "rate":
            return httpx.Response(
                429,
                json={"error": {"message": "private provider body", "type": "rate_limit_error"}},
                headers={"retry-after": "private provider header"},
            )
        if fault == "connection":
            raise httpx.ConnectError("private connection detail", request=request)
        raise httpx.ReadTimeout("private timeout detail", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as provider_client:
        model = ChatOpenAI(
            model="local-model",
            api_key="local-provider-key",
            base_url="https://provider.invalid/v1",
            max_retries=0,
            disable_streaming=True,
            http_async_client=provider_client,
        )

        async def invoke_model(state):
            try:
                return {"messages": [await model.ainvoke(state["messages"])]}
            except Exception as exc:
                errors.append(exc)
                raise

        app = service_for_node(invoke_model)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app, raise_app_exceptions=False),
            base_url="http://test",
            headers={"Authorization": "Bearer local-provider-test"},
        ) as client:
            response = await client.post("/provider/invoke", json={"input": {"message": "test failure"}})

    assert len(calls) == len(errors) == 1
    assert isinstance(errors[0], error_type)
    assert response.status_code == status
    assert response.json() == {"detail": detail, "error_code": error_code}
    assert "private" not in response.text
    assert "provider.invalid" not in response.text
    assert "retry-after" not in response.headers


@pytest.mark.parametrize("provider_name", ["openai", "azure_openai"])
@pytest.mark.parametrize("mode", [EnvironmentMode.PRODUCTION, EnvironmentMode.DEVELOPMENT])
@pytest.mark.parametrize("route", ["/provider/invoke", "/provider/stream", "/provider/stream/jsonl"])
async def test_rejected_provider_credentials_return_safe_service_error(
    production_settings, monkeypatch, provider_name, mode, route
):
    monkeypatch.setattr(settings, "ENV_MODE", mode)
    calls = []
    errors = []
    toolkit_logs = []

    async def provider(request):
        calls.append(request)
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "private provider body: rejected local-provider-key at provider.invalid",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
            headers={"WWW-Authenticate": "Bearer private provider realm", "Retry-After": "10"},
        )

    sink = logger.add(
        lambda message: toolkit_logs.append(message.record),
        filter=lambda record: record["name"].startswith("langgraph_agent_toolkit."),
    )
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as provider_client:
            model = streaming_model(provider_name, http_async_client=provider_client)

            async def invoke_model(state):
                try:
                    return {"messages": [await model.ainvoke(state["messages"])]}
                except Exception as exc:
                    errors.append(exc)
                    raise

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(service_for_node(invoke_model), raise_app_exceptions=False),
                base_url="http://test",
                headers={"Authorization": "Bearer local-provider-test"},
            ) as client:
                response = await client.post(route, json={"input": {"message": "test authentication"}})
    finally:
        logger.remove(sink)

    assert len(calls) == len(errors) == 1
    assert isinstance(errors[0], ModelAuthenticationError)
    if route.endswith("/invoke"):
        assert response.status_code == 503
        assert response.json() == {
            "detail": "The model provider credentials were rejected",
            "error_code": "model_authentication_failed",
        }
    else:
        assert response.status_code == 200
        lines = [line.removeprefix("data: ") for line in response.text.splitlines() if line and line != "data: [DONE]"]
        assert [json.loads(line) for line in lines] == [
            {"type": "error", "content": "The model provider credentials were rejected"}
        ]
    assert "www-authenticate" not in response.headers
    assert "retry-after" not in response.headers
    assert "private" not in response.text
    assert "local-provider-key" not in response.text
    assert "provider.invalid" not in response.text
    warnings = [record["message"] for record in toolkit_logs if record["level"].no >= 30]
    assert warnings
    assert all(message == "The model provider credentials were rejected" for message in warnings)
    assert all(
        value not in record["message"]
        for record in toolkit_logs
        for value in ("private", "local-provider-key", "provider.invalid")
    )
    assert all(record["exception"] is None for record in toolkit_logs)


@pytest.mark.parametrize("route", ["/provider/stream", "/provider/stream/jsonl"])
async def test_provider_failure_after_stream_headers_remains_a_safe_error_chunk(production_settings, route):
    async def fail(state):
        raise ModelRateLimitError("private provider body")

    app = service_for_node(fail)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app),
        base_url="http://test",
        headers={"Authorization": "Bearer local-provider-test"},
    ) as client:
        response = await client.post(route, json={"input": {"message": "test failure"}})

    assert response.status_code == 200
    lines = [line.removeprefix("data: ") for line in response.text.splitlines() if line and line != "data: [DONE]"]
    assert [json.loads(line) for line in lines] == [{"type": "error", "content": "Internal server error"}]


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (RuntimeError("429 private error"), 500),
        (OSError("private error"), 500),
    ],
)
async def test_unrelated_errors_are_not_reclassified(production_settings, error, status):
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/failure")
    async def fail():
        raise error

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.get("/failure")

    assert response.status_code == status
    assert response.json() == {"detail": "Internal server error"}


def first_token():
    return (
        b"data: "
        + json.dumps(
            {
                "id": "local-stream",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "local-model",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "partial"}, "finish_reason": None}],
            }
        ).encode()
        + b"\n\n"
    )


def streaming_model(provider, **clients):
    from langgraph_agent_toolkit.core.models import CompletionModelFactory

    options = {"api_key": "local-provider-key", "streaming": True, "max_retries": 2, "stream_usage": False}
    if provider == "azure_openai":
        options.update(
            azure_endpoint="https://provider.invalid", azure_deployment="local-model", api_version="2025-01-01-preview"
        )
    else:
        options["base_url"] = "https://provider.invalid/v1"
    return CompletionModelFactory.create(provider, "local-model", model_parameter_values=(), **options, **clients)


@pytest.mark.parametrize("provider_name", ["openai", "azure_openai"])
@pytest.mark.parametrize("http_library", ["httpx", "httpx2"])
@pytest.mark.parametrize(("fault", "status"), [("timeout", 504), ("disconnect", 503)])
async def test_stream_transport_failure_maps_invoke_without_replaying_partial_output(
    production_settings, provider_name, http_library, fault, status
):
    pytest.importorskip("langchain_openai")
    library = pytest.importorskip(http_library)
    calls = []
    streams = []
    errors = []
    raw_type = library.ReadTimeout if fault == "timeout" else library.RemoteProtocolError

    class FailingStream(library.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield first_token()
            raise raw_type("private stream failure", request=calls[-1])

        async def aclose(self):
            self.closed = True

    async def provider(request):
        calls.append(request)
        stream = FailingStream()
        streams.append(stream)
        return library.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with library.AsyncClient(transport=library.MockTransport(provider)) as provider_client:
        model = streaming_model(provider_name, http_async_client=provider_client)

        async def invoke_model(state):
            try:
                return {"messages": [await model.ainvoke(state["messages"])]}
            except Exception as exc:
                errors.append(exc)
                raise

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(service_for_node(invoke_model), raise_app_exceptions=False),
            base_url="http://test",
            headers={"Authorization": "Bearer local-provider-test"},
        ) as client:
            response = await client.post("/provider/invoke", json={"input": {"message": "test stream failure"}})

    assert response.status_code == status
    assert len(calls) == len(streams) == len(errors) == 1
    assert streams[0].closed
    assert isinstance(errors[0], ModelTimeoutError if fault == "timeout" else ModelConnectionError)
    assert isinstance(errors[0].__cause__, raw_type)
    assert response.json()["error_code"] == ("model_timeout" if fault == "timeout" else "model_unavailable")
    assert "private" not in response.text
    assert "partial" not in response.text


@pytest.mark.parametrize("provider_name", ["openai", "azure_openai"])
def test_sync_model_stream_keeps_received_token_and_normalizes_disconnect(production_settings, provider_name):
    pytest.importorskip("langchain_openai")
    calls = []
    streams = []

    class FailingStream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            yield first_token()
            raise httpx.RemoteProtocolError("private stream failure", request=calls[-1])

        def close(self):
            self.closed = True

    def provider(request):
        calls.append(request)
        stream = FailingStream()
        streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    with httpx.Client(transport=httpx.MockTransport(provider)) as provider_client:
        model = streaming_model(provider_name, http_client=provider_client)
        chunks = []
        with pytest.raises(ModelConnectionError) as error:
            for chunk in model.stream("test stream failure"):
                chunks.append(chunk.content)
    assert "".join(chunks) == "partial"
    assert len(calls) == 1
    assert streams[0].closed
    assert isinstance(error.value.__cause__, httpx.RemoteProtocolError)
