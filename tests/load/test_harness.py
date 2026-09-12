"""Reject load reports that lack correct results or fault evidence."""

import asyncio
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest


@pytest.fixture(scope="module")
def load_harness():
    path = Path(__file__).resolve().parents[2] / "scripts" / "load_test.py"
    spec = importlib.util.spec_from_file_location("lat_load_harness_under_test", path)
    module = importlib.util.module_from_spec(spec)
    original_path = sys.path[:]
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_path
        sys.modules.pop(spec.name, None)


@pytest.fixture(scope="module")
def validate_report(load_harness):
    return load_harness.validate_report


def phase(name="invoke-c8", **changes):
    value = {
        "name": name,
        "configuration": {"name": "local-test", "queue_waiters": 2, "workers": 1},
        "successful_requests": 40,
        "status_counts": {"200": 40},
        "recovery": {"recovered": True, "model": {"rate_limited": 1, "stalled": 1, "disconnected": 1}},
        "samples": [{"time": 102.0, "workers": [{"pid": 812, "active": 0, "waiting": 0}]}],
    }
    value.update(changes)
    return value


def slow_reader_phase():
    return phase(
        "slow-reader",
        requested=2,
        statuses=[200, 200],
        errors=[],
        owned_count=2,
        closed_count=2,
        started_monotonic=100.0,
        results=[
            {"headers_seconds": 0.1, "held_seconds": 3.0},
            {"headers_seconds": 0.2, "held_seconds": 3.0},
        ],
        samples=[
            {
                "time": 102.0,
                "workers": [{"pid": 812, "active": 0, "waiting": 0}],
                "model": {"requests": 2, "active": 0},
            }
        ],
    )


def problems(validate_report, value):
    failures = validate_report({"phases": [value]})
    return {problem for failure in failures for problem in failure["problems"]}


def test_accept_correct_results_and_observed_faults(validate_report):
    values = [
        phase(),
        phase("same-thread", history={"ok": True}),
        phase("worker-kill", replacement={"new_pids": [813]}, exceptions={"ReadError": 1}),
        phase("provider-429", unexpected_http_errors={"429": 2}),
        phase("provider-stall", unexpected_http_errors={"504": 2}),
        phase("provider-disconnect"),
        phase("db-drop", fault={"result": [{"pid": 812, "terminated": True}]}),
        phase("disconnect", intentional_disconnects=2),
        slow_reader_phase(),
    ]
    assert validate_report({"phases": values}) == []


@pytest.mark.parametrize("field", ["request_id_mismatches", "metadata_mismatches", "protocol_errors", "stream_errors"])
def test_http_200_does_not_hide_incorrect_results(validate_report, field):
    assert field in problems(validate_report, phase(**{field: 1}))


def test_forty_successes_require_correct_stored_history(validate_report):
    assert "Shared conversation lost or changed a turn" in problems(
        validate_report, phase("same-thread", history={"ok": False})
    )


def test_worker_failure_requires_a_replacement(validate_report):
    assert "No replacement worker was observed" in problems(
        validate_report, phase("worker-kill", replacement={"new_pids": []})
    )


@pytest.mark.parametrize(
    "name,counter", [("429", "rate_limited"), ("stall", "stalled"), ("disconnect", "disconnected")]
)
def test_provider_fault_requires_observed_model_failure(validate_report, name, counter):
    value = phase("provider-" + name)
    value["recovery"]["model"][counter] = 0
    assert "Provider fault was not observed" in problems(validate_report, value)


@pytest.mark.parametrize("name", ["invoke-c8", "provider-429", "worker-kill"])
def test_http_500_is_unexpected_even_during_faults(validate_report, name):
    value = phase(name, unexpected_http_errors={"500": 1}, replacement={"new_pids": [813]})
    assert "Unexpected HTTP errors" in problems(validate_report, value)


@pytest.mark.parametrize("active,waiting", [(9, 0), (0, 3)])
def test_admission_limits_apply_to_each_worker(validate_report, active, waiting):
    value = phase(samples=[{"workers": [{"pid": 812, "active": active, "waiting": waiting}]}])
    assert "Admission limit exceeded" in problems(validate_report, value)


def test_database_fault_requires_a_terminated_session(validate_report):
    value = phase("db-drop", fault={"result": [{"pid": 812, "terminated": False}]})
    assert "No database session was terminated" in problems(validate_report, value)


def test_slow_reader_requires_drain_while_sockets_remain_open(validate_report):
    value = slow_reader_phase()
    value["samples"][0]["time"] = 104.0
    assert "No evidence of cleanup before slow readers closed" in problems(validate_report, value)
    assert value["drained_before_readers_closed"] is False


def test_slow_reader_accepts_drain_during_the_common_hold_interval(validate_report):
    value = slow_reader_phase()
    assert problems(validate_report, value) == set()
    assert value["drained_before_readers_closed"] is True


@pytest.mark.parametrize("results", [[], [{"headers_seconds": 0.1}], [{"headers_seconds": 0.1, "held_seconds": 0}]])
def test_slow_reader_cannot_pass_without_hold_evidence(validate_report, results):
    value = slow_reader_phase()
    value["results"] = results
    assert problems(validate_report, value)


@pytest.mark.parametrize("workers", [[], [{"pid": 812, "active": 0, "waiting": 0}]])
def test_slow_reader_requires_samples_from_every_worker(validate_report, workers):
    value = slow_reader_phase()
    value["configuration"]["workers"] = 2
    value["samples"][0]["workers"] = workers
    assert problems(validate_report, value)


def test_empty_report_has_no_successful_verdict(validate_report):
    assert validate_report({"phases": []}) == [{"problems": ["No measured phases selected"]}]


def disposable_database(load_harness):
    return load_harness.LoadDatabase(
        dsn="host=127.0.0.1 port=18516 dbname=lat_load user=lat_load",
        schema="lat_load_admin_test",
        application_prefix="lat-load-admin-test",
    )


async def test_schema_connection_pins_destination_and_bounds_waits(load_harness, monkeypatch):
    from langgraph_agent_toolkit.core.memory.pool import CancellableAsyncConnection

    monkeypatch.setenv("LAT_LOAD_TEST_DATABASE", "yes")
    monkeypatch.setenv("PGHOSTADDR", "192.0.2.8")
    connection = AsyncMock()
    connection.__aenter__.return_value = connection
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(CancellableAsyncConnection, "connect", connect)
    database = disposable_database(load_harness)

    async with load_harness.schema_connection(database) as selected:
        assert selected is connection

    connect.assert_awaited_once()
    assert connect.call_args.args == (database.dsn,)
    options = connect.call_args.kwargs
    assert options["hostaddr"] == "127.0.0.1"
    assert 0 < options["connect_timeout"] <= 3
    assert options["autocommit"] is True
    assert options["application_name"] == "lat-load-admin-test-admin"
    assert "-c lock_timeout=2000" in options["options"]
    assert "-c statement_timeout=3000" in options["options"]
    connection.__aexit__.assert_awaited_once()


async def test_schema_connection_requires_opt_in_before_connect(load_harness, monkeypatch):
    from langgraph_agent_toolkit.core.memory.pool import CancellableAsyncConnection

    monkeypatch.delenv("LAT_LOAD_TEST_DATABASE", raising=False)
    connect = AsyncMock()
    monkeypatch.setattr(CancellableAsyncConnection, "connect", connect)
    with pytest.raises(ValueError, match="LAT_LOAD_TEST_DATABASE=yes"):
        async with load_harness.schema_connection(disposable_database(load_harness)):
            pytest.fail("Schema access requires explicit test authorization")
    connect.assert_not_awaited()


@pytest.mark.parametrize("health_path", ["live", "ready"])
async def test_settle_retries_transient_health_timeout(load_harness, monkeypatch, health_path):
    counts = Counter()
    worker = {"pid": 812, "active": 0, "waiting": 0, "stalled_cleanups": 0}

    def respond(request):
        path = request.url.path
        counts[path] += 1
        if path == "/load/metrics":
            return httpx.Response(200, json=worker)
        if path == "/metrics":
            return httpx.Response(200, json={"active": 0})
        if path == f"/health/{health_path}" and counts[path] == 1:
            raise httpx.ReadTimeout("Transient test health timeout", request=request)
        return httpx.Response(200)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        load_harness.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    async with asyncio.timeout(2):
        result = await load_harness.settle(
            SimpleNamespace(url="http://127.0.0.1:8101"),
            SimpleNamespace(url="http://127.0.0.1:8102"),
            workers=1,
            timeout=1,
        )
    assert result["recovered"] is True
    assert result["seconds"] < 1
    assert result["health"] == {"live": 200, "ready": 200}
    assert counts[f"/health/{health_path}"] == 2
    assert counts["/load/metrics"] == counts["/metrics"] == 2


def warm_result(worker_counts, *, successes=None):
    successful = sum(worker_counts.values()) if successes is None else successes
    return {
        "attempted_requests": 32,
        "successful_requests": successful,
        "successful_worker_counts": worker_counts,
        "exceptions": {},
        "successful_latency_ms": {"count": successful, "p95": 100},
    }


def warm_baseline(*, recovered=True, pids=(812, 813), pools=(1, 1)):
    return {
        "recovered": recovered,
        "workers": [{"pid": pid, "transport_pools": pool} for pid, pool in zip(pids, pools, strict=True)],
    }


@pytest.mark.parametrize(
    "result,baseline",
    [
        (warm_result({}), warm_baseline()),
        (warm_result({"812": 32}), warm_baseline()),
        (warm_result({"812": 16, "811": 16}), warm_baseline()),
        (warm_result({"812": 16, "813": 16}), warm_baseline(recovered=False)),
        (warm_result({"812": 16, "813": 16}), warm_baseline(pools=(1, 0))),
    ],
    ids=["open-pools-without-replies", "all-replies-from-one-pid", "old-pid", "work-not-drained", "missing-pool"],
)
async def test_warmup_requires_current_worker_replies_and_stops_after_five_attempts(
    load_harness, monkeypatch, result, baseline
):
    measure = AsyncMock(return_value=result)
    settle = AsyncMock(return_value=baseline)
    monkeypatch.setattr(load_harness, "measure_phase", measure)
    monkeypatch.setattr(load_harness, "settle", settle)
    config = load_harness.Configuration(name="warm-test", backend="sqlite", workers=2)
    with pytest.raises(RuntimeError, match="Not all workers completed a correct agent reply"):
        await load_harness.warm_workers(
            SimpleNamespace(url="http://127.0.0.1:8101"),
            SimpleNamespace(url="http://127.0.0.1:8102"),
            config,
            "slots",
        )
    assert measure.await_count == settle.await_count == 5
    for call in measure.call_args_list:
        assert call.args[2:] == ("invoke", 8, 3)
        assert call.kwargs["max_requests"] == 32
        assert call.kwargs["request_timeout"] == 20
        assert call.kwargs["client_pool"] == "slots"


@pytest.mark.parametrize("timeout_profile,request_timeout", [("normal", 20), ("aggressive", 10)])
async def test_warmup_accepts_second_attempt_only_after_both_current_pids_reply(
    load_harness, monkeypatch, timeout_profile, request_timeout
):
    first = warm_result({"812": 32})
    second = warm_result({"812": 16, "813": 16})
    measure = AsyncMock(side_effect=[first, second])
    settle = AsyncMock(return_value=warm_baseline())
    monkeypatch.setattr(load_harness, "measure_phase", measure)
    monkeypatch.setattr(load_harness, "settle", settle)
    config = load_harness.Configuration(name="warm-test", backend="sqlite", workers=2, timeout_profile=timeout_profile)
    result = await load_harness.warm_workers(
        SimpleNamespace(url="http://127.0.0.1:8101"),
        SimpleNamespace(url="http://127.0.0.1:8102"),
        config,
        "slots",
    )
    assert measure.await_count == settle.await_count == 2
    assert result["attempts"] == [first, second]
    assert result["workers"] == warm_baseline()["workers"]
    assert result["configuration"] == config.model_dump()
    assert result["seconds"] >= 0
    assert len({call.kwargs["prefix"] for call in measure.call_args_list}) == 2
    assert all(call.kwargs["request_timeout"] == request_timeout for call in measure.call_args_list)


async def test_replacement_worker_cannot_use_a_dead_workers_successes(load_harness, monkeypatch):
    before = warm_result({"812": 16, "813": 16})
    after = warm_result({"812": 16, "814": 16})
    measure = AsyncMock(side_effect=[before, after])
    settle = AsyncMock(return_value=warm_baseline(pids=(812, 814)))
    monkeypatch.setattr(load_harness, "measure_phase", measure)
    monkeypatch.setattr(load_harness, "settle", settle)
    config = load_harness.Configuration(name="replacement-test", backend="sqlite", workers=2)
    result = await load_harness.warm_workers(
        SimpleNamespace(url="http://127.0.0.1:8101"),
        SimpleNamespace(url="http://127.0.0.1:8102"),
        config,
        "slots",
    )
    assert measure.await_count == 2
    assert result["attempts"][-1]["successful_worker_counts"] == {"812": 16, "814": 16}
    assert {worker["pid"] for worker in result["workers"]} == {812, 814}


async def test_warmup_cancellation_stops_without_retry(load_harness, monkeypatch):
    measure = AsyncMock(side_effect=asyncio.CancelledError)
    settle = AsyncMock()
    monkeypatch.setattr(load_harness, "measure_phase", measure)
    monkeypatch.setattr(load_harness, "settle", settle)
    config = load_harness.Configuration(name="cancel-test", backend="sqlite", workers=2)
    with pytest.raises(asyncio.CancelledError):
        await load_harness.warm_workers(
            SimpleNamespace(url="http://127.0.0.1:8101"),
            SimpleNamespace(url="http://127.0.0.1:8102"),
            config,
            "slots",
        )
    measure.assert_awaited_once()
    settle.assert_not_awaited()
