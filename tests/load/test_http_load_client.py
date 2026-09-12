"""Check load results with deterministic HTTP responses and streamed failures."""

import asyncio
import errno
import importlib.util
import json
import sys
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import httpcore
import httpx
import pytest


_spec = importlib.util.spec_from_file_location("load_http_client", Path(__file__).with_name("client.py"))
load_client = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = load_client
_spec.loader.exec_module(load_client)


@pytest.fixture
def use_transport(monkeypatch):
    original_client = httpx.AsyncClient

    def install(handler):
        monkeypatch.setattr(
            load_client.httpx,
            "AsyncClient",
            lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
        )

    return install


def reply(payload):
    return {
        "type": "ai",
        "content": payload["input"]["message"],
        "run_id": str(uuid4()),
        "thread_id": payload["thread_id"],
    }


def encode(event, sse):
    text = json.dumps(event)
    return (f"data: {text}\n\n" if sse else text + "\n").encode()


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, generate):
        self.generate = generate
        self.closed = False

    async def __aiter__(self):
        async for chunk in self.generate():
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("shared_thread", [False, True])
async def test_success_percentiles_exclude_rejections_bad_ids_and_connection_errors(use_transport, shared_thread):
    async def respond(request):
        payload = json.loads(request.content)
        assert payload["user_id"] == "load-user"
        assert payload["stream_tokens"] is True
        assert payload["thread_id"] == ("identity-shared" if shared_thread else payload["input"]["message"])
        assert request.headers["authorization"] == "Bearer local-test-secret"
        index = int(payload["input"]["message"].rsplit("-", 1)[1])
        if index in (1, 2):
            return httpx.Response(503 if index == 1 else 409)
        if index == 5:
            raise httpx.ConnectError("Synthetic connection failure", request=request)
        message = reply(payload)
        if index == 0:
            await asyncio.sleep(0.02)
        elif index == 3:
            message["content"] = "another-request"
        elif index == 4:
            message["thread_id"] = "another-thread"
        return httpx.Response(200, json=message)

    use_transport(respond)
    result = await load_client.measure_phase(
        "http://127.0.0.1",
        "local-test-secret",
        "invoke",
        2,
        1,
        prefix="identity",
        max_requests=6,
        shared_thread=shared_thread,
    )

    assert result["attempted_requests"] == result["completed_requests"] == 6
    assert result["status_counts"] == {"200": 3, "409": 1, "503": 1}
    assert result["expected_rejections"] == {"409": 1, "503": 1}
    assert result["exceptions"] == {"ConnectError": 1}
    assert result["request_id_checks"] == 3
    assert result["request_id_mismatches"] == result["metadata_mismatches"] == 1
    assert result["successful_requests"] == result["successful_latency_ms"]["count"] == 1
    assert result["successful_latency_ms"]["p50"] >= 15
    assert result["successful_latency_ms"]["p50"] == result["successful_latency_ms"]["p99"]
    assert result["successful_goodput_rps"] > 0
    assert result["successful_request_ids"] == ["identity-0"]
    assert result["shared_thread_id"] == ("identity-shared" if shared_thread else None)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("sse", [True, False], ids=["sse", "jsonl"])
async def test_stream_error_and_truncated_success_do_not_count_as_goodput(use_transport, sse):
    async def respond(request):
        payload = json.loads(request.content)
        index = int(payload["input"]["message"].rsplit("-", 1)[1])

        async def generate():
            if sse:
                yield b": keepalive\n\n"
            await asyncio.sleep(0.002)
            token = encode({"type": "token", "content": payload["input"]["message"]}, sse)
            yield token[:7]
            yield token[7:]
            await asyncio.sleep(0.005)
            if index == 1:
                yield encode({"type": "error", "content": "Synthetic database failure"}, sse)
            elif index == 0:
                yield encode({"type": "message", "content": reply(payload)}, sse)
                if sse:
                    yield b"data: [DONE]\n\n"

        return httpx.Response(200, stream=ByteStream(generate), headers={"x-load-worker": str(1000 + index)})

    use_transport(respond)
    result = await load_client.measure_phase(
        "http://127.0.0.1",
        "local-test-secret",
        "stream" if sse else "stream/jsonl",
        3,
        prefix="stream",
        max_requests=3,
    )

    assert result["status_counts"] == {"200": 3}
    assert result["stream_errors"] == result["protocol_errors"] == 1
    assert result["successful_requests"] == result["successful_first_data_ms"]["count"] == 1
    assert result["successful_first_data_ms"]["p50"] < result["successful_latency_ms"]["p50"]
    assert result["request_id_checks"] == 1
    assert result["received_body_bytes"] > 0
    assert result["successful_worker_counts"] == {"1000": 1}


@pytest.mark.parametrize("sse", [True, False], ids=["sse", "jsonl"])
async def test_intentional_disconnect_checks_first_token_and_closes_without_reading_more(use_transport, sse):
    streams = []
    unwanted_reads = []

    async def respond(request):
        payload = json.loads(request.content)

        async def generate():
            yield encode({"type": "token", "content": payload["input"]["message"]}, sse)
            unwanted_reads.append(True)
            await asyncio.Event().wait()

        stream = ByteStream(generate)
        streams.append(stream)
        return httpx.Response(200, stream=stream, headers={"x-load-worker": "1234"})

    use_transport(respond)
    result = await load_client.measure_phase(
        "http://127.0.0.1",
        "local-test-secret",
        "stream" if sse else "stream/jsonl",
        2,
        prefix="disconnect",
        max_requests=2,
        disconnect_after_first=True,
    )

    assert result["intentional_disconnects"] == result["request_id_checks"] == 2
    assert result["request_id_mismatches"] == 0
    assert result["successful_requests"] == 0
    assert result["successful_latency_ms"]["p99"] is None
    assert result["intentional_disconnect_first_data_ms"]["count"] == 2
    assert not unwanted_reads
    assert len(streams) == 2 and all(stream.closed for stream in streams)
    assert result["successful_worker_counts"] == {}


async def test_only_correct_completed_responses_count_toward_worker_warmup(use_transport):
    async def respond(request):
        payload = json.loads(request.content)
        index = int(payload["input"]["message"].rsplit("-", 1)[1])
        message = reply(payload)
        if index == 1:
            message["content"] = "incorrect-request"
        headers = {"x-load-worker": str(2000 + index)}
        if index == 3:
            headers["x-load-worker"] = "invalid-pid"
        return httpx.Response(503 if index == 2 else 200, json=message, headers=headers)

    use_transport(respond)
    result = await load_client.measure_phase(
        "http://127.0.0.1", "local-test-secret", "invoke", 1, prefix="worker", max_requests=5
    )
    assert result["successful_requests"] == 3
    assert result["request_id_mismatches"] == 1
    assert result["status_counts"] == {"200": 4, "503": 1}
    assert result["successful_worker_counts"] == {"2000": 1, "2004": 1}


async def test_constant_arrivals_report_unsent_work_and_bound_client_concurrency(use_transport):
    async def respond(request):
        await asyncio.sleep(0.09)
        return httpx.Response(200, json=reply(json.loads(request.content)))

    use_transport(respond)
    result = await load_client.measure_phase(
        "http://127.0.0.1",
        "local-test-secret",
        "invoke",
        1,
        0.06,
        prefix="rate",
        rate=500,
        max_inflight=2,
        max_requests=100,
    )

    assert result["scheduled_arrivals"] == 30
    assert result["attempted_requests"] == result["successful_requests"] == 2
    assert result["peak_inflight"] == 2
    assert result["dropped_arrivals"] == 28
    assert result["dropped_arrivals_by_reason"]["inflight_limit"] > 0
    assert set(result["dropped_arrivals_by_reason"]) <= {"inflight_limit", "scheduler_deadline"}
    assert result["scheduler_lag_ms"]["count"] == 30
    assert result["send_window_seconds"] == 0.06
    assert result["elapsed_seconds"] > result["send_window_seconds"]
    assert result["successful_latency_from_arrival_ms"]["mean"] >= result["successful_latency_ms"]["mean"]


async def test_stream_keepalives_cannot_extend_the_total_request_timeout(use_transport):
    streams = []

    async def respond(request):
        async def generate():
            while True:
                yield b": keepalive\n\n"
                await asyncio.sleep(0.005)

        stream = ByteStream(generate)
        streams.append(stream)
        return httpx.Response(200, stream=stream)

    use_transport(respond)
    result = await load_client.measure_phase(
        "http://127.0.0.1",
        "local-test-secret",
        "stream",
        1,
        prefix="timeout",
        max_requests=1,
        request_timeout=0.03,
    )

    assert result["status_counts"] == {"200": 1}
    assert result["exceptions"] == {"TimeoutError": 1}
    assert result["successful_requests"] == 0
    assert streams[0].closed


@pytest.mark.parametrize("upstream_stream", [True, False])
@pytest.mark.parametrize("shared_thread", [True, False])
async def test_upstream_stream_option_keeps_request_and_thread_identity(use_transport, upstream_stream, shared_thread):
    observed = []

    async def respond(request):
        payload = json.loads(request.content)
        expected_id = f"upstream-{len(observed)}"
        if upstream_stream:
            assert payload["input"]["message"] == expected_id
        else:
            assert json.loads(payload["input"]["message"]) == {
                "request_id": expected_id,
                "upstream_stream": False,
            }
        assert payload["thread_id"] == ("upstream-shared" if shared_thread else expected_id)
        observed.append(payload)
        return httpx.Response(200, json={**reply(payload), "content": expected_id})

    use_transport(respond)
    result = await load_client.measure_phase(
        "http://127.0.0.1",
        "local-test-secret",
        "invoke",
        1,
        prefix="upstream",
        max_requests=2,
        upstream_stream=upstream_stream,
        shared_thread=shared_thread,
    )

    assert len(observed) == result["successful_requests"] == 2
    assert result["config"]["upstream_stream"] is upstream_stream
    assert result["request_id_mismatches"] == result["metadata_mismatches"] == 0
    assert result["successful_request_ids"] == ["upstream-0", "upstream-1"]


async def test_transport_trace_separates_pre_send_and_response_wait(use_transport):
    async def respond(request):
        trace = request.extensions["trace"]
        private_info = {"authorization": "Bearer private-value-must-not-appear"}
        await asyncio.sleep(0.01)
        await trace("connection.connect_tcp.started", private_info)
        await asyncio.sleep(0.01)
        await trace("connection.connect_tcp.complete", private_info)
        await trace("http11.send_request_headers.started", private_info)
        await trace("http11.send_request_headers.complete", private_info)
        await trace("http11.receive_response_headers.started", private_info)
        await asyncio.sleep(0.04)
        await trace("http11.receive_response_headers.complete", private_info)
        return httpx.Response(200, json=reply(json.loads(request.content)))

    use_transport(respond)
    result = await load_client.measure_phase(
        "http://127.0.0.1", "local-test-secret", "invoke", 1, prefix="traced", max_requests=1
    )
    trace = result["client_transport"]
    assert result["successful_requests"] == 1
    assert trace["request_to_first_transport_ms"]["mean"] >= 5
    assert trace["request_to_send_headers_ms"]["mean"] >= 15
    assert trace["send_headers_to_response_headers_ms"]["mean"] >= 30
    assert trace["request_to_response_headers_ms"]["mean"] >= 45
    assert trace["stage_duration_ms"]["connect_tcp"]["mean"] >= 5
    assert trace["stage_duration_ms"]["receive_response_headers"]["mean"] >= 30
    assert trace["event_counts"]["http11.send_request_headers.started"] == 1
    assert trace["failure_stage_counts"] == {}
    assert trace["requests_without_trace"] == 0
    assert result["driver_event_loop_lag_ms"]["count"] >= 1
    assert "private-value-must-not-appear" not in json.dumps(result)


@pytest.mark.parametrize("during_headers", [True, False])
async def test_timeout_trace_reports_the_active_transport_stage(use_transport, during_headers):
    async def respond(request):
        trace = request.extensions["trace"]
        if during_headers:
            await trace("http11.send_request_headers.started", {})
            await trace("http11.send_request_headers.complete", {})
            await trace("http11.receive_response_headers.started", {})
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            if during_headers:
                await trace("http11.receive_response_headers.failed", {"exception": exc})
                await trace("http11.response_closed.started", {})
                await trace("http11.response_closed.complete", {})
            raise

    use_transport(respond)
    result = await load_client.measure_phase(
        "http://127.0.0.1",
        "local-test-secret",
        "invoke",
        1,
        prefix="trace-timeout",
        max_requests=1,
        request_timeout=0.03,
    )
    stage = "receive_response_headers" if during_headers else "before_transport"
    assert result["exceptions"] == {"TimeoutError": 1}
    assert result["client_transport"]["failure_stage_counts"] == {stage: 1}
    assert result["client_transport"]["exception_stage_counts"] == {f"TimeoutError:{stage}": 1}
    assert result["client_transport"]["requests_without_trace"] == (0 if during_headers else 1)
    assert result["client_transport"]["request_to_send_headers_ms"]["count"] == (1 if during_headers else 0)
    assert not any(task.get_name() == "load-client-event-loop-lag" for task in asyncio.all_tasks())


@pytest.mark.parametrize("mode", ["slots", "shared"])
@pytest.mark.parametrize("expiry", [None, 0.25])
async def test_bounded_clients_share_tls_context_and_loan_distinct_slots(monkeypatch, mode, expiry):
    original = httpx.AsyncClient
    clients, settings, observed = [], [], []
    active = {}
    peak = {}
    both_started = asyncio.Event()

    def create(**kwargs):
        index = len(clients)
        settings.append(kwargs)

        async def respond(request):
            active[index] = active.get(index, 0) + 1
            peak[index] = max(peak.get(index, 0), active[index])
            observed.append(index)
            if sum(active.values()) == 2:
                both_started.set()
            try:
                await asyncio.wait_for(both_started.wait(), 1)
                await asyncio.sleep(0.005)
                return httpx.Response(200, json=reply(json.loads(request.content)))
            finally:
                active[index] -= 1

        client = original(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(load_client.httpx, "AsyncClient", create)
    result = await load_client.measure_phase(
        "http://127.0.0.1",
        "local-test-secret",
        "invoke",
        2,
        prefix="loans",
        max_requests=4,
        client_pool=mode,
        **({"client_keepalive_expiry": expiry} if expiry is not None else {}),
    )
    expected = 2 if mode == "slots" else 1
    assert result["successful_requests"] == len(observed) == 4
    assert len(clients) == len(set(observed)) == expected
    assert all(client.is_closed for client in clients)
    assert len({id(value["verify"]) for value in settings}) == 1
    assert all(value["limits"].max_connections == (1 if mode == "slots" else 2) for value in settings)
    assert all(value["limits"].keepalive_expiry == (expiry or 1.0) for value in settings)
    assert result["config"]["client_keepalive_expiry"] == (expiry or 1.0)
    assert all(value == (1 if mode == "slots" else 2) for value in peak.values())
    assert result["client_pool"]["owned_clients"] == result["client_pool"]["closed_clients"] == expected
    assert result["client_pool"]["peak_leased"] == 2
    assert result["client_pool"]["active_leases"] == 0


async def test_client_cleanup_finishes_after_repeated_cancellation(monkeypatch):
    original = httpx.AsyncClient
    clients = []
    all_started, close_started, allow_close = asyncio.Event(), asyncio.Event(), asyncio.Event()
    requests = 0

    async def respond(request):
        nonlocal requests
        requests += 1
        if requests == 3:
            all_started.set()
        await asyncio.Event().wait()

    class SlowCloseClient(original):
        async def aclose(self):
            close_started.set()
            await allow_close.wait()
            await super().aclose()

    def create(**kwargs):
        client = SlowCloseClient(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(load_client.httpx, "AsyncClient", create)
    task = asyncio.create_task(
        load_client.measure_phase("http://127.0.0.1", "local-test-secret", "invoke", 3, prefix="cancel", max_requests=3)
    )
    try:
        await asyncio.wait_for(all_started.wait(), 1)
        task.cancel()
        await asyncio.wait_for(close_started.wait(), 1)
        task.cancel()
    finally:
        allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(clients) == 3
    assert all(client.is_closed for client in clients)
    assert not any(task.get_name().startswith("load-client-") for task in asyncio.all_tasks())


async def test_slot_clients_reuse_real_tcp_connections_without_parallel_loans():
    peers = set()
    server_tasks = set()
    active = peak = 0

    async def serve(reader, writer):
        nonlocal active, peak
        task = asyncio.current_task()
        server_tasks.add(task)
        peers.add(writer.get_extra_info("peername"))
        try:
            while True:
                header = await reader.readuntil(b"\r\n\r\n")
                fields = dict(line.split(b":", 1) for line in header.split(b"\r\n")[1:] if line)
                payload = json.loads(await reader.readexactly(int(fields[b"Content-Length"])))
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.sleep(0.01)
                    body = json.dumps(reply(payload)).encode()
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                        + str(len(body)).encode()
                        + b"\r\n\r\n"
                        + body
                    )
                    await writer.drain()
                finally:
                    active -= 1
        except (asyncio.IncompleteReadError, OSError):
            pass
        finally:
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()
            server_tasks.discard(task)

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        result = await load_client.measure_phase(
            f"http://127.0.0.1:{port}", "local-test-secret", "invoke", 2, prefix="real-slots", max_requests=6
        )
    finally:
        server.close()
        await server.wait_closed()
        async with asyncio.timeout(1):
            await asyncio.gather(*server_tasks, return_exceptions=True)
    assert result["successful_requests"] == 6
    assert peak == 2 and active == 0
    assert len(peers) == 2
    assert result["client_pool"]["owned_clients"] == result["client_pool"]["closed_clients"] == 2
    assert result["client_pool"]["active_leases"] == 0
    assert result["client_transport"]["event_counts"]["connection.connect_tcp.started"] == 2
    assert result["client_transport"]["event_counts"]["http11.send_request_headers.started"] == 6


@pytest.mark.parametrize("expiry", [0, -1, float("inf"), float("nan")])
async def test_keepalive_expiry_must_be_positive_and_finite(expiry):
    with pytest.raises(ValueError, match="client_keepalive_expiry"):
        await load_client.measure_phase(
            "http://127.0.0.1", "local-test-secret", "invoke", 1, prefix="expiry", client_keepalive_expiry=expiry
        )


async def test_exception_examples_are_bounded_and_exclude_private_values(use_transport):
    private = "private-message-URL-header-must-not-appear"

    async def respond(request):
        trace = request.extensions["trace"]
        await trace("http11.send_request_headers.started", {"authorization": private})
        await trace("http11.send_request_headers.complete", {})
        await trace("http11.receive_response_headers.started", {})
        try:
            try:
                raise ConnectionResetError(errno.ECONNRESET, private)
            except ConnectionResetError as exc:
                raise httpcore.ReadError(private) from exc
        except httpcore.ReadError as exc:
            await trace("http11.receive_response_headers.failed", {"exception": exc})
            await trace("http11.response_closed.started", {})
            await trace("http11.response_closed.complete", {})
            raise httpx.ReadError(private, request=request) from exc

    use_transport(respond)
    result = await load_client.measure_phase(
        "http://127.0.0.1", "local-test-secret", "invoke", 2, prefix="read-errors", max_requests=7
    )
    assert result["exceptions"] == {"ReadError": 7}
    assert result["successful_requests"] == 0
    assert len(result["exception_examples"]) == 5
    for example in result["exception_examples"]:
        assert example["chain"] == [
            {"class": "httpx.ReadError", "errno": None},
            {"class": "httpcore.ReadError", "errno": None},
            {"class": "builtins.ConnectionResetError", "errno": errno.ECONNRESET},
        ]
        assert example["stage"] == "receive_response_headers"
        assert example["transport_started"] is True
        assert example["connect_started"] is False
        assert example["connect_complete"] is False
        assert example["send_headers_started"] is True
        assert example["response_headers_received"] is False
        assert example["elapsed_ms"] >= example["since_send_headers_ms"] >= 0
        assert example["since_response_headers_ms"] is None
    assert private not in json.dumps(result)
    assert "local-test-secret" not in json.dumps(result)


def test_exception_example_chain_stops_on_cycles_and_depth_limit():
    samples = load_client._TransportSamples()
    trace = load_client._RequestTrace(samples, load_client.time.perf_counter())
    first = RuntimeError("not-recorded")
    second = ValueError("not-recorded")
    first.__cause__ = second
    second.__cause__ = first
    trace.failure(first)
    assert len(samples.exception_examples[0]["chain"]) == 2
    current = first
    for _ in range(10):
        replacement = RuntimeError("not-recorded")
        replacement.__cause__ = current
        current = replacement
    trace.failure(current)
    assert len(samples.exception_examples[1]["chain"]) == 5
    assert samples.exception_examples[1]["stage"] == "before_transport"
