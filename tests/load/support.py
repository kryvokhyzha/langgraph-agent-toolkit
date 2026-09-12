"""Run a local model simulator and the real toolkit for load tests.

This module does not load an environment file. It does not call a remote model.
Use a separate process for ``model_app`` and ``create_service_app``.
"""

import asyncio
import ipaddress
import json
import os
import resource
import sys
import time
from contextlib import asynccontextmanager, suppress
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send


MODEL_API_KEY = "lat-load-test-only"


def _disable_external_configuration() -> None:
    """Keep local tests separate from environment files and trace exporters."""
    import dotenv

    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    dotenv.find_dotenv = lambda *args, **kwargs: ""
    dotenv.load_dotenv = lambda *args, **kwargs: False
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2", "LANGFUSE_TRACING_ENABLED"):
        os.environ[name] = "false"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "OPENAI_PROXY"):
        os.environ.pop(name, None)
        os.environ.pop(name.lower(), None)


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def local_model_url() -> str:
    """Require a literal loopback address and the simulator's API path."""
    value = os.environ.get("LAT_LOAD_MODEL_URL", "")
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme == "http"
            and parsed.hostname is not None
            and _is_loopback(parsed.hostname)
            and parsed.port is not None
            and 0 < parsed.port < 65536
            and parsed.path.rstrip("/") == "/v1"
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("LAT_LOAD_MODEL_URL must be http://<literal-loopback>:<port>/v1")
    return value.rstrip("/")


class ModelControl(BaseModel):
    """Set bounded delays, response sizes, and failure modes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["none", "429", "stall", "disconnect"] = "none"
    remaining: int = Field(default=-1, ge=-1)
    delay_ms: float = Field(default=100, ge=0, le=300000, allow_inf_nan=False)
    first_token_ms: float = Field(default=50, ge=0, le=300000, allow_inf_nan=False)
    chunk_interval_ms: float = Field(default=10, ge=0, le=300000, allow_inf_nan=False)
    stream_chunk_count: int = Field(default=5, ge=1, le=65536)
    stream_chunk_bytes: int = Field(default=0, ge=0, le=1024 * 1024)
    stall_seconds: float = Field(default=30, gt=0, le=300, allow_inf_nan=False)
    retry_after: float = Field(default=0.05, gt=0, le=60, allow_inf_nan=False)

    @model_validator(mode="after")
    def bound_response(self):
        if self.stream_chunk_bytes * self.stream_chunk_count > 64 * 1024 * 1024:
            raise ValueError("The configured stream must not exceed 64 MiB")
        return self


async def _read_json(receive: Receive, limit: int = 1024 * 1024) -> dict[str, Any]:
    body = bytearray()
    while True:
        event = await receive()
        if event["type"] == "http.disconnect":
            raise ConnectionError("The caller disconnected")
        body.extend(event.get("body", b""))
        if len(body) > limit:
            raise ValueError("The request body is too large")
        if not event.get("more_body", False):
            value = json.loads(body)
            if not isinstance(value, dict):
                raise ValueError("The request body must be an object")
            return value


class ModelSimulator:
    """Implement the OpenAI chat wire protocol on a loopback interface."""

    def __init__(self):
        self.control = ModelControl()
        self.active = 0
        self.peak = 0
        self._epoch = 0
        self._connections: set[tuple[str, int]] = set()
        self._reset_counts()

    def _reset_counts(self) -> None:
        self._epoch += 1
        self.peak = self.active
        self.counts = dict.fromkeys(
            (
                "requests",
                "completed",
                "cancelled",
                "errors",
                "rate_limited",
                "stalled",
                "disconnected",
                "stream_requests",
                "nonstream_requests",
                "chunks_sent",
                "bytes_sent",
            ),
            0,
        )
        self._connections.clear()

    def metrics(self) -> dict[str, Any]:
        return {
            **self.counts,
            "pid": os.getpid(),
            "active": self.active,
            "peak": self.peak,
            "distinct_connections": len(self._connections),
            "control": self.control.model_dump(),
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif event["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        if not _is_loopback((scope.get("client") or ("", 0))[0]):
            await JSONResponse({"detail": "Loopback access is required"}, status_code=403)(scope, receive, send)
            return
        path, method = scope["path"], scope["method"]
        if path in {"/health", "/metrics"} and method == "GET":
            result = {"status": "ready"} if path == "/health" else self.metrics()
            await JSONResponse(result)(scope, receive, send)
            return
        if path == "/control" and method == "POST":
            try:
                value = await _read_json(receive, limit=8192)
                reset = value.pop("reset_metrics", False)
                if not isinstance(reset, bool):
                    raise ValueError("reset_metrics must be a boolean")
                self.control = ModelControl.model_validate({**self.control.model_dump(), **value})
                if reset:
                    self._reset_counts()
            except (ValueError, ValidationError):
                await JSONResponse({"detail": "Invalid model control"}, status_code=422)(scope, receive, send)
                return
            await JSONResponse(self.metrics())(scope, receive, send)
            return
        if path != "/v1/chat/completions" or method != "POST":
            await JSONResponse({"detail": "Not found"}, status_code=404)(scope, receive, send)
            return
        headers = dict(scope["headers"])
        if headers.get(b"authorization") != f"Bearer {MODEL_API_KEY}".encode():
            await JSONResponse({"error": {"message": "Invalid local test key"}}, status_code=401)(scope, receive, send)
            return
        try:
            body = await _read_json(receive)
            payload = json.loads(body["messages"][-1]["content"])
            request_id = payload["request_id"]
            if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
                raise ValueError("Invalid request ID")
        except (KeyError, IndexError, TypeError, ValueError):
            await JSONResponse({"error": {"message": "Invalid local model request"}}, status_code=400)(
                scope, receive, send
            )
            return

        self.active += 1
        self.peak = max(self.peak, self.active)
        self.counts["requests"] += 1
        self.counts["stream_requests" if body.get("stream") else "nonstream_requests"] += 1
        self._connections.add(tuple(scope["client"]))
        epoch, control = self._epoch, self.control
        mode = control.mode if control.remaining != 0 else "none"
        if mode != "none" and control.remaining > 0:
            self.control = control.model_copy(update={"remaining": control.remaining - 1})

        response = asyncio.create_task(self._respond(body, request_id, control, mode, scope, receive, send))

        async def watch_disconnect() -> None:
            while True:
                if (await receive())["type"] == "http.disconnect":
                    return

        watcher = asyncio.create_task(watch_disconnect())
        try:
            done, _ = await asyncio.wait({response, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if response not in done:
                response.cancel()
                with suppress(asyncio.CancelledError):
                    await response
                if epoch == self._epoch:
                    self.counts["cancelled"] += 1
                return
            await response
            if epoch == self._epoch:
                self.counts["completed"] += 1
        except asyncio.CancelledError:
            if epoch == self._epoch:
                self.counts["cancelled"] += 1
            raise
        except Exception:
            if epoch == self._epoch:
                self.counts["errors"] += 1
            raise
        finally:
            for task in (response, watcher):
                if not task.done():
                    task.cancel()
            await asyncio.gather(response, watcher, return_exceptions=True)
            self.active -= 1

    async def _respond(self, body, request_id, control, mode, scope, receive, send) -> None:
        if mode == "429":
            self.counts["rate_limited"] += 1
            await JSONResponse(
                {"error": {"message": "Local load test rate limit", "type": "rate_limit_error", "code": "rate_limit"}},
                status_code=429,
                headers={"Retry-After": str(control.retry_after)},
            )(scope, receive, send)
            return
        if mode == "stall":
            self.counts["stalled"] += 1
            await asyncio.sleep(control.stall_seconds)
        if mode == "disconnect":
            self.counts["disconnected"] += 1
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"1000000")]})
            await send({"type": "http.response.body", "body": b"{", "more_body": True})
            # Uvicorn closes the socket after an error in a started response.
            raise ConnectionResetError("Intentional local model disconnect")

        base = {"id": f"chatcmpl-{request_id}", "created": 1, "model": body.get("model", "local-load-model")}
        usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
        if not body.get("stream"):
            await asyncio.sleep(control.delay_ms / 1000)
            await JSONResponse(
                {
                    **base,
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": request_id}, "finish_reason": "stop"}
                    ],
                    "usage": usage,
                }
            )(scope, receive, send)
            return

        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache")],
            }
        )

        async def emit(choices, **extra):
            event = {**base, "object": "chat.completion.chunk", "choices": choices, **extra}
            data = b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"
            await send({"type": "http.response.body", "body": data, "more_body": True})
            self.counts["chunks_sent"] += 1
            self.counts["bytes_sent"] += len(data)

        await asyncio.sleep(control.first_token_ms / 1000)
        for index in range(control.stream_chunk_count):
            if index:
                await asyncio.sleep(control.chunk_interval_ms / 1000)
            # The first chunk always contains the complete request ID.
            content = request_id if index == 0 else " " * control.stream_chunk_bytes
            delta = {"content": content}
            if index == 0:
                delta["role"] = "assistant"
            await emit([{"index": 0, "delta": delta, "finish_reason": None}])
        await emit([{"index": 0, "delta": {}, "finish_reason": "stop"}])
        if (body.get("stream_options") or {}).get("include_usage"):
            await emit([], usage=usage)
        await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False})


model_app = ModelSimulator()


def create_model_app() -> ModelSimulator:
    return ModelSimulator()


def build_graph(extra_tools=()):
    """Call the real model factory and keep the service's checkpoint state."""
    _disable_external_configuration()
    local_model_url()
    if extra_tools:
        raise ValueError("The local load agent does not accept external tools")

    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph import END, START, MessagesState, StateGraph

    from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory

    async def reply(state: MessagesState, config: RunnableConfig):
        messages = state["messages"]
        human = [message for message in messages if isinstance(message, HumanMessage)]
        text = human[-1].content
        data = json.loads(text) if text.startswith("{") else {"request_id": text}
        request_id = data["request_id"]
        if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
            raise ValueError("Invalid load request ID")
        streaming = data.get("upstream_stream", True)
        if not isinstance(streaming, bool):
            raise ValueError("upstream_stream must be a boolean")
        model = CompletionModelFactory.create(
            "openai",
            "local-load-model",
            configurable_fields=(),
            config_prefix="",
            model_parameter_values=(),
            api_key=MODEL_API_KEY,
            base_url=local_model_url(),
            streaming=streaming,
            disable_streaming=not streaming,
            use_responses_api=False,
        )
        result = await model.ainvoke([HumanMessage(json.dumps({"request_id": request_id}))], config=config)
        if not isinstance(result.content, str) or result.content.rstrip(" ") != request_id:
            raise RuntimeError("The local model returned a different request ID")
        # Keep padding out of the checkpoint and the final API message.
        result.content = request_id
        result.response_metadata.update(load_turn_count=len(human), load_input_message_count=len(messages))
        return {"messages": [result]}

    graph = StateGraph(MessagesState)
    graph.add_node("reply", reply)
    graph.add_edge(START, "reply")
    graph.add_edge("reply", END)
    return graph.compile()


def __getattr__(name: str):
    if name != "load_agent":
        raise AttributeError(name)
    _disable_external_configuration()
    from langgraph_agent_toolkit.agents.agent import Agent

    agent = Agent(
        name="load-agent",
        description="Call a local model simulator through the service HTTP pool.",
        graph=build_graph(),
        graph_factory=build_graph,
    )
    globals()[name] = agent
    return agent


class WorkerMetrics:
    """Expose test metrics outside the production admission middleware."""

    def __init__(self, app, metrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] == "/load/metrics":
            if scope["method"] != "GET" or not _is_loopback((scope.get("client") or ("", 0))[0]):
                await JSONResponse({"detail": "Local GET access is required"}, status_code=403)(scope, receive, send)
                return
            await JSONResponse(self.metrics())(scope, receive, send)
            return
        if scope["type"] == "http" and scope["path"].startswith("/load-agent/"):

            async def worker_header(message):
                if message["type"] == "http.response.start":
                    headers = [
                        (key, value) for key, value in message.get("headers", []) if key.lower() != b"x-load-worker"
                    ]
                    headers.append((b"x-load-worker", str(os.getpid()).encode("ascii")))
                    message = {**message, "headers": headers}
                await send(message)

            await self.app(scope, receive, worker_header)
            return
        await self.app(scope, receive, send)


def create_service_app():
    """Run the package service and add local worker metrics."""
    _disable_external_configuration()
    local_model_url()
    from langgraph_agent_toolkit.service.handler import create_app

    app = create_app()
    original_lifespan = app.router.lifespan_context
    lag = {"last_ms": 0.0, "max_ms": 0.0, "samples": 0}

    async def sample_lag() -> None:
        interval = 0.02
        while True:
            expected = time.monotonic() + interval
            await asyncio.sleep(interval)
            measured = max(0.0, time.monotonic() - expected) * 1000
            lag.update(last_ms=measured, max_ms=max(lag["max_ms"], measured), samples=lag["samples"] + 1)

    @asynccontextmanager
    async def lifespan(application):
        async with original_lifespan(application):
            ticker = asyncio.create_task(sample_lag(), name="load-event-loop-lag")
            try:
                yield
            finally:
                ticker.cancel()
                with suppress(asyncio.CancelledError):
                    await ticker

    app.router.lifespan_context = lifespan

    def metrics() -> dict[str, Any]:
        admission = getattr(app.state, "request_admission", None)
        manager = getattr(app.state, "llm_transport_manager", None)
        pools = list(manager._pools.values()) if manager is not None else []
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        snapshot = {
            "pid": os.getpid(),
            "active": admission.active if admission is not None else 0,
            "waiting": admission.waiting if admission is not None else 0,
            "stalled_cleanups": getattr(app.state, "stalled_request_cleanups", 0),
            "transport_pools": len(pools),
            "open_sync_clients": sum(not sync.is_closed for sync, _ in pools),
            "open_async_clients": sum(not asynchronous.is_closed for _, asynchronous in pools),
            "asyncio_tasks": len(asyncio.all_tasks()),
            "event_loop_lag_last_ms": lag["last_ms"],
            "event_loop_lag_max_ms": lag["max_ms"],
            "event_loop_lag_samples": lag["samples"],
            "rss_peak_bytes": peak_rss if sys.platform == "darwin" else peak_rss * 1024,
            "ready": bool(getattr(app.state, "ready", False)),
        }
        # Each sample reports the largest delay since the preceding read.
        lag.update(max_ms=0.0, samples=0)
        return snapshot

    # Add this last so it remains outside the service admission middleware.
    app.add_middleware(WorkerMetrics, metrics=metrics)
    return app
