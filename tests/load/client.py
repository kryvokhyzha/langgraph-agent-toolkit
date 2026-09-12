"""Measure bounded HTTP load without importing pytest or the agent package."""

import asyncio
import json
import math
import random
import time
from collections import Counter
from contextlib import asynccontextmanager, suppress
from typing import Any, Literal
from uuid import UUID

import httpx
from pydantic import BaseModel, Field, field_validator


class LoadPhase(BaseModel):
    """Define one finite load phase. Rate mode drops arrivals when the client is full."""

    route: Literal["/load-agent/invoke", "/load-agent/stream", "/load-agent/stream/jsonl"]
    concurrency: int = Field(gt=0)
    duration: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    prefix: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.-]+$")
    rate: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_inflight: int | None = Field(default=None, gt=0)
    max_requests: int = Field(default=5000, gt=0)
    shared_thread: bool = False
    upstream_stream: bool = True
    disconnect_after_first: bool = False
    request_timeout: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    client_pool: Literal["slots", "shared"] = "slots"
    client_keepalive_expiry: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @field_validator("route", mode="before")
    @classmethod
    def normalize_route(cls, value: str) -> str:
        return value if value.startswith("/load-agent/") else "/load-agent/" + value.lstrip("/")


class _Samples:
    """Keep exact counts and a bounded reservoir for percentile estimates."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0
        self.values: list[float] = []
        self.random = random.Random(0)

    def add(self, seconds: float) -> None:
        value = max(0.0, seconds * 1000)
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
        if len(self.values) < 100_000:
            self.values.append(value)
        else:
            index = self.random.randrange(self.count)
            if index < len(self.values):
                self.values[index] = value

    def report(self) -> dict[str, Any]:
        ordered = sorted(self.values)

        def percentile(fraction: float) -> float | None:
            if not ordered:
                return None
            position = (len(ordered) - 1) * fraction
            left = math.floor(position)
            right = math.ceil(position)
            return round(ordered[left] + (ordered[right] - ordered[left]) * (position - left), 3)

        return {
            "count": self.count,
            "sample_count": len(ordered),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": round(self.maximum, 3) if self.count else None,
            "mean": round(self.total / self.count, 3) if self.count else None,
        }


class _ProtocolError(Exception):
    pass


class _StreamError(Exception):
    pass


class _ClientPool:
    """Loan one persistent HTTP client to each active request slot."""

    def __init__(self, base_url: str, secret: str, capacity: int, phase: LoadPhase):
        self.base_url = base_url
        self.secret = secret
        self.capacity = capacity
        self.phase = phase
        self.clients: list[httpx.AsyncClient] = []
        self.available = asyncio.Queue()
        self.leased = 0
        self.peak_leased = 0
        self.wait = _Samples()

    async def __aenter__(self):
        try:
            # Load the certificate bundle once for all slot clients.
            context = httpx.create_ssl_context(trust_env=False)
            slots = self.capacity if self.phase.client_pool == "slots" else 1
            connections = 1 if self.phase.client_pool == "slots" else self.capacity
            for _ in range(slots):
                client = httpx.AsyncClient(
                    base_url=self.base_url.rstrip("/"),
                    headers={"Authorization": f"Bearer {self.secret}"},
                    limits=httpx.Limits(
                        max_connections=connections,
                        max_keepalive_connections=connections,
                        keepalive_expiry=self.phase.client_keepalive_expiry,
                    ),
                    timeout=httpx.Timeout(self.phase.request_timeout),
                    verify=context,
                    trust_env=False,
                    follow_redirects=False,
                )
                self.clients.append(client)
                self.available.put_nowait(client)
        except BaseException:
            await self.aclose()
            raise
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    @asynccontextmanager
    async def loan(self):
        began = time.perf_counter()
        client = await self.available.get() if self.phase.client_pool == "slots" else self.clients[0]
        self.wait.add(time.perf_counter() - began)
        self.leased += 1
        self.peak_leased = max(self.peak_leased, self.leased)
        try:
            yield client
        finally:
            self.leased -= 1
            if self.phase.client_pool == "slots":
                self.available.put_nowait(client)

    async def aclose(self) -> None:
        async def close_all():
            errors = await asyncio.gather(*(client.aclose() for client in self.clients), return_exceptions=True)
            if any(isinstance(error, BaseException) for error in errors):
                raise RuntimeError("One or more load clients did not close")

        cleanup = asyncio.create_task(close_all(), name="load-client-pool-close")
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        cleanup.result()
        if cancelled:
            raise asyncio.CancelledError

    def report(self) -> dict:
        return {
            "mode": self.phase.client_pool,
            "owned_clients": len(self.clients),
            "closed_clients": sum(client.is_closed for client in self.clients),
            "peak_leased": self.peak_leased,
            "active_leases": self.leased,
            "loan_wait_ms": self.wait.report(),
        }


class _TransportSamples:
    """Keep transport timing totals without request or response data."""

    def __init__(self):
        self.events = Counter()
        self.failures = Counter()
        self.exception_stages = Counter()
        self.requests_without_trace = 0
        self.exception_examples: list[dict] = []
        self.stages: dict[str, _Samples] = {}
        self.timings = {
            name: _Samples()
            for name in (
                "request_to_first_transport_ms",
                "request_to_send_headers_ms",
                "send_headers_to_response_headers_ms",
                "request_to_response_headers_ms",
            )
        }

    def report(self) -> dict:
        return {
            **{name: samples.report() for name, samples in self.timings.items()},
            "stage_duration_ms": {name: samples.report() for name, samples in sorted(self.stages.items())},
            "event_counts": dict(sorted(self.events.items())),
            "failure_stage_counts": dict(sorted(self.failures.items())),
            "exception_stage_counts": dict(sorted(self.exception_stages.items())),
            "requests_without_trace": self.requests_without_trace,
        }


class _RequestTrace:
    """Observe public HTTPX trace events for one request."""

    _stages = frozenset(
        {
            "connect_tcp",
            "start_tls",
            "send_request_headers",
            "send_request_body",
            "receive_response_headers",
            "receive_response_body",
            "response_closed",
        }
    )

    def __init__(self, samples: _TransportSamples, began: float):
        self.samples = samples
        self.began = began
        self.first_event = None
        self.send_headers = None
        self.response_headers = None
        self.stage = "before_transport"
        self.failed_stage = None
        self.connect_started = False
        self.connect_complete = False
        self.started: dict[str, float] = {}

    async def __call__(self, event: str, _info: dict) -> None:
        # Trace information can contain credentials. Read only the event name.
        parts = event.split(".")
        if len(parts) != 3:
            return
        _, stage, action = parts
        if stage not in self._stages or action not in {"started", "complete", "failed"}:
            return
        now = time.perf_counter()
        self.samples.events[event] += 1
        if self.first_event is None:
            self.first_event = now
        if action == "started":
            self.started[stage] = now
            self.stage = stage
            if stage == "send_request_headers" and self.send_headers is None:
                self.send_headers = now
            if stage == "connect_tcp":
                self.connect_started = True
        else:
            if stage in self.started:
                samples = self.samples.stages.get(stage)
                if samples is None:
                    samples = self.samples.stages[stage] = _Samples()
                samples.add(now - self.started.pop(stage))
            if action == "failed" and self.failed_stage is None:
                self.failed_stage = stage
            if stage == "receive_response_headers" and action == "complete" and self.response_headers is None:
                self.response_headers = now
            if stage == "connect_tcp" and action == "complete":
                self.connect_complete = True

    def failure(self, error: Exception) -> None:
        stage = self.failed_stage or self.stage
        self.samples.failures[stage] += 1
        self.samples.exception_stages[f"{type(error).__name__}:{stage}"] += 1
        if len(self.samples.exception_examples) < 5:
            chain, visited = [], set()
            current = error
            while current is not None and id(current) not in visited and len(chain) < 5:
                visited.add(id(current))
                error_type = type(current)
                code = getattr(current, "errno", None)
                chain.append(
                    {
                        "class": f"{error_type.__module__}.{error_type.__qualname__}"[:160],
                        "errno": code if type(code) is int else None,
                    }
                )
                current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
            now = time.perf_counter()

            def elapsed_from(value):
                return None if value is None else round(max(0, now - value) * 1000, 3)

            self.samples.exception_examples.append(
                {
                    "chain": chain,
                    "stage": stage,
                    "transport_started": self.first_event is not None,
                    "connect_started": self.connect_started,
                    "connect_complete": self.connect_complete,
                    "send_headers_started": self.send_headers is not None,
                    "response_headers_received": self.response_headers is not None,
                    "elapsed_ms": elapsed_from(self.began),
                    "since_first_transport_ms": elapsed_from(self.first_event),
                    "since_send_headers_ms": elapsed_from(self.send_headers),
                    "since_response_headers_ms": elapsed_from(self.response_headers),
                }
            )

    def finish(self) -> None:
        if self.first_event is None:
            self.samples.requests_without_trace += 1
        else:
            self.samples.timings["request_to_first_transport_ms"].add(self.first_event - self.began)
        if self.send_headers is not None:
            self.samples.timings["request_to_send_headers_ms"].add(self.send_headers - self.began)
        if self.response_headers is not None:
            self.samples.timings["request_to_response_headers_ms"].add(self.response_headers - self.began)
            if self.send_headers is not None:
                self.samples.timings["send_headers_to_response_headers_ms"].add(
                    self.response_headers - self.send_headers
                )


def _message_matches(message: Any, request_id: str, thread_id: str) -> tuple[bool, bool]:
    if not isinstance(message, dict):
        raise _ProtocolError("The response message is not an object")
    identity_matches = message.get("type") == "ai" and message.get("content") == request_id
    try:
        UUID(message.get("run_id", ""))
        metadata_matches = message.get("thread_id") == thread_id
    except (ValueError, AttributeError, TypeError):
        metadata_matches = False
    return identity_matches, metadata_matches


async def measure_phase(
    base_url: str,
    secret: str,
    route: str,
    concurrency: int,
    duration: float = 10.0,
    *,
    prefix: str,
    rate: float | None = None,
    max_inflight: int | None = None,
    max_requests: int = 5000,
    shared_thread: bool = False,
    upstream_stream: bool = True,
    disconnect_after_first: bool = False,
    request_timeout: float = 10.0,
    client_pool: Literal["slots", "shared"] = "slots",
    client_keepalive_expiry: float = 1.0,
) -> dict[str, Any]:
    """Measure one phase with bounded HTTP clients and no retries.

    Successful latency excludes HTTP rejections, stream errors, and incorrect IDs.
    Rate mode also measures latency from each scheduled arrival. It counts arrivals
    that the client cannot send. Percentiles use a reservoir above 100,000 samples.
    Elapsed time includes the final request drain. Send time excludes that drain.
    Transport trace timing separates pre-send work from response-header waits.
    Pre-send work includes pool waiting, connection setup, and request encoding.
    Slot mode loans a separate client to each active request. Shared mode keeps
    one client for comparison with HTTPX connection-pool contention.
    """
    phase = LoadPhase(
        route=route,
        concurrency=concurrency,
        duration=duration,
        prefix=prefix,
        rate=rate,
        max_inflight=max_inflight,
        max_requests=max_requests,
        shared_thread=shared_thread,
        upstream_stream=upstream_stream,
        disconnect_after_first=disconnect_after_first,
        request_timeout=request_timeout,
        client_pool=client_pool,
        client_keepalive_expiry=client_keepalive_expiry,
    )
    if phase.disconnect_after_first and phase.route.endswith("/invoke"):
        raise ValueError("Intentional disconnects require a streaming route")
    capacity = phase.max_inflight or phase.concurrency
    if phase.rate is None:
        capacity = phase.concurrency
    counts: Counter = Counter()
    statuses: Counter = Counter()
    exceptions: Counter = Counter()
    dropped: Counter = Counter()
    latency = _Samples()
    arrival_latency = _Samples()
    first_data = _Samples()
    disconnected_first_data = _Samples()
    scheduler_lag = _Samples()
    driver_loop_lag = _Samples()
    transport_samples = _TransportSamples()
    successful_request_ids: list[str] = []
    successful_workers = Counter()
    active: set[asyncio.Task] = set()
    sequence = 0

    async with _ClientPool(base_url, secret, capacity, phase) as clients:
        started = time.perf_counter()
        send_duration = min(phase.duration, phase.max_requests / phase.rate) if phase.rate else phase.duration
        deadline = started + send_duration

        async def request(index: int, scheduled: float) -> None:
            request_id = f"{phase.prefix}-{index}"
            thread_id = f"{phase.prefix}-shared" if phase.shared_thread else request_id
            payload = {
                "input": {
                    "message": request_id
                    if phase.upstream_stream
                    else json.dumps({"request_id": request_id, "upstream_stream": False})
                },
                "thread_id": thread_id,
                "user_id": "load-user",
                "stream_tokens": True,
            }
            began = time.perf_counter()
            trace = _RequestTrace(transport_samples, began)
            scheduler_lag.add(began - scheduled)
            counts["attempted_requests"] += 1
            counts["inflight"] += 1
            counts["peak_inflight"] = max(counts["peak_inflight"], counts["inflight"])
            response = None
            first_at = None
            try:
                async with asyncio.timeout(phase.request_timeout), clients.loan() as client:
                    async with client.stream(
                        "POST", phase.route, json=payload, extensions={"trace": trace}
                    ) as response:
                        statuses[str(response.status_code)] += 1
                        if response.status_code != 200:
                            await response.aread()
                            return
                        if phase.route.endswith("/invoke"):
                            await response.aread()
                            message = response.json()
                        else:
                            message = None
                            ended = False
                            is_sse = phase.route.endswith("/stream")
                            async for line in response.aiter_lines():
                                if not line.strip() or (is_sse and not line.startswith("data:")):
                                    continue
                                data = line[5:].strip() if is_sse else line
                                if is_sse and data == "[DONE]":
                                    ended = True
                                    break
                                event = json.loads(data)
                                if not isinstance(event, dict):
                                    raise _ProtocolError("A stream event is not an object")
                                kind = event.get("type")
                                if kind == "error":
                                    raise _StreamError
                                if kind not in {"token", "message"}:
                                    raise _ProtocolError("The stream event type is invalid")
                                content = event.get("content")
                                if kind == "token" and not isinstance(content, str):
                                    raise _ProtocolError("A token is not text")
                                if kind == "message" and not isinstance(content, dict):
                                    raise _ProtocolError("A message is not an object")
                                if first_at is None:
                                    first_at = time.perf_counter()
                                    if phase.disconnect_after_first:
                                        matches = (
                                            content == request_id
                                            if kind == "token"
                                            else content.get("content") == request_id
                                        )
                                        counts["request_id_checks"] += 1
                                        counts["request_id_mismatches"] += not matches
                                        if kind == "message":
                                            _, metadata_matches = _message_matches(content, request_id, thread_id)
                                            counts["metadata_checks"] += 1
                                            counts["metadata_mismatches"] += not metadata_matches
                                        counts["intentional_disconnects"] += 1
                                        disconnected_first_data.add(first_at - began)
                                        return
                                if kind == "message" and content.get("type") == "ai":
                                    message = content
                            if is_sse and not ended:
                                raise _ProtocolError("The SSE completion frame is missing")
                        identity_matches, metadata_matches = _message_matches(message, request_id, thread_id)
                        counts["request_id_checks"] += 1
                        counts["metadata_checks"] += 1
                        counts["request_id_mismatches"] += not identity_matches
                        counts["metadata_mismatches"] += not metadata_matches
                        if not identity_matches or not metadata_matches:
                            return
                    finished = time.perf_counter()
                    counts["successful_requests"] += 1
                    successful_request_ids.append(request_id)
                    worker = response.headers.get("x-load-worker", "")
                    if len(worker) <= 10 and worker.isascii() and worker.isdecimal() and 0 < int(worker) < 2**31:
                        successful_workers[str(int(worker))] += 1
                    latency.add(finished - began)
                    arrival_latency.add(finished - scheduled)
                    if first_at is not None:
                        first_data.add(first_at - began)
            except _StreamError as exc:
                counts["stream_errors"] += 1
                trace.failure(exc)
            except (_ProtocolError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                counts["protocol_errors"] += 1
                trace.failure(exc)
            except Exception as exc:
                exceptions[type(exc).__name__] += 1
                trace.failure(exc)
            finally:
                trace.finish()
                if response is not None:
                    counts["received_body_bytes"] += response.num_bytes_downloaded
                counts["completed_requests"] += 1
                counts["inflight"] -= 1

        async def worker() -> None:
            nonlocal sequence
            while time.perf_counter() < deadline and sequence < phase.max_requests:
                index = sequence
                sequence += 1
                await request(index, time.perf_counter())

        async def sample_driver_loop() -> None:
            while True:
                expected = time.perf_counter() + 0.02
                await asyncio.sleep(0.02)
                driver_loop_lag.add(time.perf_counter() - expected)

        loop_sampler = asyncio.create_task(sample_driver_loop(), name="load-client-event-loop-lag")
        try:
            if phase.rate is None:
                active.update(asyncio.create_task(worker()) for _ in range(min(capacity, phase.max_requests)))
                await asyncio.gather(*active)
                sent_until = min(time.perf_counter(), deadline)
                scheduled_arrivals = sequence
            else:
                scheduled_arrivals = min(math.ceil(phase.duration * phase.rate), phase.max_requests)
                while sequence < scheduled_arrivals:
                    due = started + sequence / phase.rate
                    await asyncio.sleep(max(0.0, due - time.perf_counter()))
                    now = time.perf_counter()
                    if now >= deadline:
                        dropped["scheduler_deadline"] += scheduled_arrivals - sequence
                        for index in range(sequence, scheduled_arrivals):
                            scheduler_lag.add(now - (started + index / phase.rate))
                        break
                    due_until = min(scheduled_arrivals, math.floor((now - started) * phase.rate) + 1)
                    admitted = min(due_until - sequence, capacity - len(active))
                    for index in range(sequence, sequence + admitted):
                        task = asyncio.create_task(request(index, started + index / phase.rate))
                        active.add(task)
                        task.add_done_callback(active.discard)
                    dropped["inflight_limit"] += due_until - sequence - admitted
                    for index in range(sequence + admitted, due_until):
                        scheduler_lag.add(now - (started + index / phase.rate))
                    sequence = due_until
                sent_until = deadline
                await asyncio.sleep(max(0.0, sent_until - time.perf_counter()))
                if active:
                    await asyncio.gather(*active)
        finally:
            try:
                for task in active:
                    if not task.done():
                        task.cancel()
                if active:
                    await asyncio.gather(*active, return_exceptions=True)
            finally:
                loop_sampler.cancel()
                with suppress(asyncio.CancelledError):
                    await loop_sampler

    elapsed = time.perf_counter() - started
    send_window = max(sent_until - started, 0.000001)
    return {
        "mode": "fixed_concurrency" if phase.rate is None else "constant_arrival_rate",
        "config": phase.model_dump(mode="json"),
        "prefix": phase.prefix,
        "shared_thread_id": f"{phase.prefix}-shared" if phase.shared_thread else None,
        "successful_request_ids": successful_request_ids,
        "successful_worker_counts": dict(sorted(successful_workers.items())),
        **{
            key: counts[key]
            for key in (
                "attempted_requests",
                "completed_requests",
                "successful_requests",
                "intentional_disconnects",
                "stream_errors",
                "protocol_errors",
                "request_id_checks",
                "request_id_mismatches",
                "metadata_checks",
                "metadata_mismatches",
                "peak_inflight",
                "received_body_bytes",
            )
        },
        "status_counts": dict(sorted(statuses.items())),
        "expected_rejections": {code: statuses[code] for code in ("409", "503")},
        "unexpected_http_errors": {
            code: count for code, count in statuses.items() if code not in {"200", "409", "503"}
        },
        "exceptions": dict(sorted(exceptions.items())),
        "exception_examples": transport_samples.exception_examples,
        "scheduled_arrivals": scheduled_arrivals,
        "dropped_arrivals": sum(dropped.values()),
        "dropped_arrivals_by_reason": dict(dropped),
        "elapsed_seconds": round(elapsed, 6),
        "send_window_seconds": round(send_window, 6),
        "attempted_rps": round(counts["attempted_requests"] / send_window, 3),
        "successful_goodput_rps": round(counts["successful_requests"] / elapsed, 3),
        "successful_latency_ms": latency.report(),
        "successful_latency_from_arrival_ms": arrival_latency.report(),
        "successful_first_data_ms": first_data.report(),
        "intentional_disconnect_first_data_ms": disconnected_first_data.report(),
        "scheduler_lag_ms": scheduler_lag.report(),
        "driver_event_loop_lag_ms": driver_loop_lag.report(),
        "client_transport": transport_samples.report(),
        "client_pool": clients.report(),
    }
