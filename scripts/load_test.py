"""Run repeatable pressure tests against disposable local toolkit processes."""

import asyncio
import json
import math
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal
from uuid import uuid4

import fire
import httpx
import rootutils
from pydantic import BaseModel, Field


ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=False)
sys.path.insert(0, str(ROOT / "tests" / "load"))

from client import measure_phase  # noqa: E402
from db_pressure import (  # noqa: E402
    LoadDatabase,
    gather_db_metrics,
    hold_checkpoint_table_lock,
    terminate_toolkit_sessions,
    verify_http_history,
)


NORMAL_CONTROL = {
    "mode": "none",
    "remaining": -1,
    "delay_ms": 100,
    "first_token_ms": 50,
    "chunk_interval_ms": 10,
    "stream_chunk_count": 5,
    "stream_chunk_bytes": 0,
    "stall_seconds": 30,
    "retry_after": 0.05,
}

TIMEOUT_PROFILES = {
    "normal": {
        "REQUEST_TIMEOUT": "15",
        "THREAD_QUEUE_TIMEOUT": "10",
        "POSTGRES_POOL_TIMEOUT": "10",
        "POSTGRES_HEALTH_CHECK_TIMEOUT": "5",
        "POSTGRES_CONNECT_TIMEOUT": "10",
        "THREAD_LOCK_HEARTBEAT_TIMEOUT": "5",
    },
    "aggressive": {
        "REQUEST_TIMEOUT": "5",
        "THREAD_QUEUE_TIMEOUT": "4",
        "POSTGRES_POOL_TIMEOUT": "2",
        "POSTGRES_HEALTH_CHECK_TIMEOUT": "0.5",
        "POSTGRES_CONNECT_TIMEOUT": "2",
        "THREAD_LOCK_HEARTBEAT_TIMEOUT": "0.5",
    },
}


class Configuration(BaseModel):
    name: str
    backend: Literal["sqlite", "postgres"]
    workers: int = Field(default=1, ge=1, le=4)
    transport: Literal["httpx", "aiohttp"] = "httpx"
    queue_waiters: int = Field(default=0, ge=0)
    timeout_profile: Literal["normal", "aggressive"] = "normal"


class LocalProcess:
    """Own a local Uvicorn process tree and its log."""

    def __init__(self, target: str, directory: Path, env: dict[str, str], workers: int = 1):
        directory.mkdir(parents=True, exist_ok=True)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.log_path = directory / "process.log"
        self.log = self.log_path.open("wb")
        retained = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"}
        environment = {name: value for name, value in os.environ.items() if name in retained}
        environment.update(
            PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "tests" / "load")]),
            PYTHONUNBUFFERED="1",
            PYTHON_DOTENV_DISABLED="1",
            LANGSMITH_TRACING="false",
            LANGCHAIN_TRACING_V2="false",
            LOG_LEVEL="WARNING",
            COLORIZE="false",
            **env,
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                target,
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--workers",
                str(workers),
                "--timeout-worker-healthcheck",
                "10",
                "--timeout-graceful-shutdown",
                "10",
                "--log-level",
                "info",
                "--no-access-log",
            ],
            cwd=directory,
            env=environment,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    async def ready(self, path: str = "/health/ready") -> None:
        async with httpx.AsyncClient(timeout=1, trust_env=False) as client:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Process startup failed; see {self.log_path}")
                try:
                    if (await client.get(self.url + path)).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.1)
        raise TimeoutError(f"Process startup timed out; see {self.log_path}")

    async def close(self) -> None:
        # The process group can outlive a failed supervisor.
        with suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGTERM)
        if self.process.poll() is None:
            try:
                await asyncio.to_thread(self.process.wait, timeout=12)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                await asyncio.to_thread(self.process.wait, timeout=5)
        # Give multiprocessing's resource tracker time to unlink named resources.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.killpg(self.process.pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.05)
        else:
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
        self.log.close()


async def worker_metrics(client: httpx.AsyncClient, url: str, workers: int) -> list[dict]:
    found = {}
    for _ in range(workers * 4):
        response = await client.get(url + "/load/metrics")
        response.raise_for_status()
        value = response.json()
        previous = found.get(value["pid"])
        if previous is not None:
            value["event_loop_lag_max_ms"] = max(value["event_loop_lag_max_ms"], previous["event_loop_lag_max_ms"])
        found[value["pid"]] = value
        if len(found) >= workers:
            break
    return list(found.values())


async def sample(service: LocalProcess, model: LocalProcess, workers: int, stop: asyncio.Event, samples: list) -> None:
    from resources import process_metrics

    async with httpx.AsyncClient(
        timeout=2, trust_env=False, limits=httpx.Limits(max_keepalive_connections=0)
    ) as client:
        while not stop.is_set():
            try:
                metrics = await worker_metrics(client, service.url, workers)
                model_metrics = (await client.get(model.url + "/metrics")).json()
                health = {}
                for path in ("live", "ready"):
                    started = time.monotonic()
                    try:
                        response = await client.get(service.url + "/health/" + path)
                        health[path] = {
                            "status": response.status_code,
                            "latency_ms": (time.monotonic() - started) * 1000,
                        }
                    except httpx.HTTPError as exc:
                        health[path] = {"error": type(exc).__name__}
                samples.append(
                    {
                        "time": time.monotonic(),
                        "workers": metrics,
                        "model": model_metrics,
                        "processes": await process_metrics([value["pid"] for value in metrics]),
                        "driver_and_model_processes": await process_metrics([os.getpid(), model.process.pid]),
                        "health": health,
                    }
                )
            except (httpx.HTTPError, ValueError) as exc:
                samples.append({"time": time.monotonic(), "sample_error": type(exc).__name__})
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), 1)


async def settle(service: LocalProcess, model: LocalProcess, workers: int, timeout: float = 10) -> dict:
    start = time.monotonic()
    metrics, upstream = [], {}
    async with httpx.AsyncClient(
        timeout=2, trust_env=False, limits=httpx.Limits(max_keepalive_connections=0)
    ) as client:
        while time.monotonic() - start < timeout:
            try:
                async with asyncio.timeout(max(0.001, timeout - (time.monotonic() - start))):
                    metrics = await worker_metrics(client, service.url, workers)
                    upstream = (await client.get(model.url + "/metrics")).json()
                    health = {
                        path: (await client.get(service.url + "/health/" + path)).status_code
                        for path in ("live", "ready")
                    }
            except (httpx.HTTPError, TimeoutError):
                await asyncio.sleep(0.1)
                continue
            if (
                len(metrics) == workers
                and all(
                    all(value.get(key, 0) == 0 for key in ("active", "waiting", "stalled_cleanups"))
                    for value in metrics
                )
                and upstream["active"] == 0
                and all(status == 200 for status in health.values())
            ):
                return {
                    "recovered": True,
                    "seconds": time.monotonic() - start,
                    "workers": metrics,
                    "model": upstream,
                    "health": health,
                }
            await asyncio.sleep(0.1)
        return {"recovered": False, "seconds": time.monotonic() - start, "workers": metrics, "model": upstream}


@asynccontextmanager
async def schema_connection(database: LoadDatabase):
    """Bound schema administration to the validated local database."""
    from psycopg.conninfo import conninfo_to_dict

    from langgraph_agent_toolkit.core.memory.pool import CancellableAsyncConnection

    database.validate()
    values = conninfo_to_dict(database.dsn)
    async with asyncio.timeout(6):
        async with await CancellableAsyncConnection.connect(
            database.dsn,
            hostaddr=values.get("hostaddr") or values["host"],
            connect_timeout=3,
            autocommit=True,
            application_name=database.application_prefix + "-admin",
            options="-c lock_timeout=2000 -c statement_timeout=3000",
        ) as connection:
            yield connection


@asynccontextmanager
async def service_process(config: Configuration, directory: Path, model_url: str, dsn: str | None):
    database = None
    env = {
        "ENV_MODE": "production",
        "AUTH_MODE": "trusted",
        "AUTH_SECRET": "local-load-test-token",
        "AUTH_USERS": "{}",
        "USE_FAKE_MODEL": "false",
        "OBSERVABILITY_BACKEND": "empty",
        "AGENT_PATHS": '["support:load_agent"]',
        "DEFAULT_AGENT": "load-agent",
        "MEMORY_BACKEND": config.backend,
        "SQLITE_DB_PATH": str(directory / "checkpoints.sqlite"),
        "LAT_LOAD_MODEL_URL": model_url + "/v1",
        "LLM_HTTP_ASYNC_TRANSPORT": config.transport,
        "LLM_HTTP_MAX_RETRIES": "1",
        "LLM_HTTP_CONNECT_TIMEOUT": "1",
        "LLM_HTTP_READ_TIMEOUT": "2",
        "LLM_HTTP_POOL_TIMEOUT": "1",
        "REQUEST_MAX_CONCURRENT": "8",
        "REQUEST_QUEUE_MAX_WAITERS": str(config.queue_waiters),
        "REQUEST_QUEUE_TIMEOUT": "0.25",
        "REQUEST_CLEANUP_TIMEOUT": "2",
        "RESPONSE_SEND_TIMEOUT": "1",
        "THREAD_QUEUE_MAX_WAITERS": "32",
        "THREAD_LOCK_HEARTBEAT": "0.2",
    }
    if config.backend == "postgres":
        from psycopg import sql
        from psycopg.conninfo import conninfo_to_dict

        values = conninfo_to_dict(dsn)
        suffix = uuid4().hex[:12]
        database = LoadDatabase(dsn=dsn, schema="lat_load_" + suffix, application_prefix="lat-load-" + suffix)
        database.validate()
        async with schema_connection(database) as connection:
            await connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(database.schema)))
        env.update(
            POSTGRES_HOST=values["host"],
            POSTGRES_PORT=values.get("port", "5432"),
            POSTGRES_USER=values["user"],
            POSTGRES_PASSWORD=values.get("password", "local-test-unused"),
            POSTGRES_DB=values["dbname"],
            POSTGRES_SCHEMA=database.schema,
            POSTGRES_APPLICATION_NAME=database.application_prefix,
            POSTGRES_RECONNECT_TIMEOUT="3",
            POSTGRES_LOCK_TIMEOUT="500",
            POSTGRES_STATEMENT_TIMEOUT="3000",
            POSTGRES_MIN_SIZE="1",
        )
    env.update(TIMEOUT_PROFILES[config.timeout_profile])
    service = None
    try:
        service = LocalProcess("support:create_service_app", directory, env, workers=config.workers)
        await service.ready()
        yield service, database
    finally:
        if service is not None:
            await service.close()
        if database is not None:
            async with schema_connection(database) as connection:
                await connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(database.schema)))


def save_report(output: Path, report: dict) -> None:
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


async def configure_model(client: httpx.AsyncClient, model: LocalProcess, values: dict) -> None:
    response = await client.post(model.url + "/control", json=values)
    response.raise_for_status()
    actual = response.json()["control"]
    if any(actual.get(key) != value for key, value in values.items() if key != "reset_metrics"):
        raise RuntimeError("The model simulator did not apply the requested control")


async def warm_workers(
    service: LocalProcess,
    model: LocalProcess,
    config: Configuration,
    client_pool: str,
    client_keepalive_expiry: float = 1.0,
) -> dict:
    """Require a complete correct agent reply from every worker before measurement."""
    started = time.monotonic()
    attempts = []
    for _ in range(5):
        result = await measure_phase(
            service.url,
            "local-load-test-token",
            "invoke",
            4 * config.workers,
            3,
            prefix=f"warmup-{config.name}-{uuid4().hex[:6]}",
            max_requests=32,
            client_pool=client_pool,
            client_keepalive_expiry=client_keepalive_expiry,
            request_timeout=20 if config.timeout_profile == "normal" else 10,
        )
        attempts.append(
            {
                key: result[key]
                for key in (
                    "attempted_requests",
                    "successful_requests",
                    "successful_worker_counts",
                    "exceptions",
                    "successful_latency_ms",
                )
            }
        )
        baseline = await settle(service, model, config.workers)
        pids = {str(worker["pid"]) for worker in baseline["workers"]}
        if (
            baseline["recovered"]
            and result["successful_requests"] >= 4 * config.workers
            and pids <= set(result["successful_worker_counts"])
            and all(worker["transport_pools"] for worker in baseline["workers"])
        ):
            return {
                "configuration": config.model_dump(),
                "seconds": time.monotonic() - started,
                "workers": baseline["workers"],
                "attempts": attempts,
            }
    raise RuntimeError("Not all workers completed a correct agent reply during warm-up")


def validate_report(report: dict) -> list[dict]:
    """Reject incorrect responses, unbounded work, and faults that did not run."""
    failures = []
    for phase in report["phases"]:
        problems = []
        name = phase["name"]
        config = phase["configuration"]
        if not phase["recovery"]["recovered"]:
            problems.append("Requests did not drain")
        for key in ("request_id_mismatches", "metadata_mismatches", "protocol_errors", "stream_errors"):
            if phase.get(key):
                problems.append(key)
        allowed = {429, 504} if name.startswith("provider-") else ({504} if name == "worker-kill" else set())
        if any(int(status) not in allowed for status in phase.get("unexpected_http_errors", {})):
            problems.append("Unexpected HTTP errors")
        if name != "worker-kill" and phase.get("exceptions"):
            problems.append("Unexpected client transport errors")
        pool = phase.get("client_pool")
        if pool and (pool["owned_clients"] != pool["closed_clients"] or pool["active_leases"]):
            problems.append("Load client connections did not close")
        if name not in {"slow-reader", "disconnect"} and not phase.get("successful_requests"):
            problems.append("No successful requests")
        for sample in phase["samples"]:
            for worker in sample.get("workers", []):
                if worker["active"] > 8 or worker["waiting"] > config["queue_waiters"]:
                    problems.append("Admission limit exceeded")
        if name == "same-thread":
            if phase["successful_requests"] != 40:
                problems.append("Some shared-conversation requests did not succeed")
            if not phase.get("history", {}).get("ok"):
                problems.append("Shared conversation lost or changed a turn")
        if name == "disconnect" and not phase.get("intentional_disconnects"):
            problems.append("No client disconnected after a token")
        if name.startswith("provider-"):
            counter = {
                "provider-429": "rate_limited",
                "provider-stall": "stalled",
                "provider-disconnect": "disconnected",
            }[name]
            if not phase["recovery"]["model"][counter]:
                problems.append("Provider fault was not observed")
        if name == "db-drop" and not any(item["terminated"] for item in phase["fault"]["result"]):
            problems.append("No database session was terminated")
        if name == "worker-kill" and not phase.get("replacement", {}).get("new_pids"):
            problems.append("No replacement worker was observed")
        if name == "slow-reader":
            if (
                phase["statuses"] != [200] * phase["requested"]
                or phase["errors"]
                or phase["owned_count"] != phase["closed_count"]
            ):
                problems.append("Slow reader setup or socket cleanup failed")
            # Both sockets must still be held when the server releases capacity.
            if phase["results"] and all(value.get("held_seconds") for value in phase["results"]):
                start = phase["started_monotonic"]
                lower = start + max(value["headers_seconds"] for value in phase["results"]) + 1
                upper = start + min(value["headers_seconds"] + value["held_seconds"] for value in phase["results"])
                drained = any(
                    lower < sample["time"] < upper
                    and sample.get("model", {}).get("requests", 0) >= phase["requested"]
                    and sample["model"]["active"] == 0
                    and len(sample.get("workers", [])) == config["workers"]
                    and all(worker["active"] == 0 for worker in sample.get("workers", []))
                    for sample in phase["samples"]
                )
                phase["drained_before_readers_closed"] = drained
                if not drained:
                    problems.append("No evidence of cleanup before slow readers closed")
            else:
                problems.append("No evidence that slow readers held their sockets open")
        database = phase.get("database_after", {})
        if database.get("advisory_locks", {}).get("granted", 0):
            problems.append("Conversation locks remained after drain")
        if any(session["state"] == "idle in transaction" for session in database.get("sessions", [])):
            problems.append("A database transaction remained open after drain")
        if (
            database
            and database["database_counters"]["deadlocks"]
            > phase.get("database_before", database)["database_counters"]["deadlocks"]
        ):
            problems.append("PostgreSQL detected a deadlock")
        if problems:
            failures.append({"configuration": config["name"], "phase": name, "problems": sorted(set(problems))})
    if not report["phases"]:
        failures.append({"problems": ["No measured phases selected"]})
    return failures


async def execute(
    output: Path,
    duration: float,
    soak_seconds: float,
    dsn: str | None,
    quick: bool,
    configuration_names: str | None = None,
    phase_names: str | None = None,
    client_pool: Literal["slots", "shared"] = "slots",
    timeout_profile: Literal["normal", "aggressive"] = "normal",
    client_keepalive_expiry: float = 1.0,
) -> dict:
    from resources import descriptor_counts, process_metrics

    output.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "versions": {},
        "model": (
            "local OpenAI protocol simulation; nonstream delay 100 ms; "
            "stream first token 50 ms plus four 10 ms gaps; no paid calls"
        ),
        "api_settings": {
            "active_per_worker": 8,
            "request_timeout_seconds": float(TIMEOUT_PROFILES[timeout_profile]["REQUEST_TIMEOUT"]),
            "send_timeout_seconds": 1,
            "worker_heartbeat_seconds": 10,
        },
        "load_client_pool": client_pool,
        "client_keepalive_expiry_seconds": client_keepalive_expiry,
        "timeout_profile": timeout_profile,
        "timeout_settings": TIMEOUT_PROFILES[timeout_profile],
        "warmups": [],
        "phases": [],
    }
    for name in (
        "openai",
        "httpx",
        "httpx2",
        "aiohttp",
        "langchain-openai",
        "langchain-core",
        "langgraph",
        "uvicorn",
        "psycopg",
        "psycopg-pool",
    ):
        with suppress(PackageNotFoundError):
            report["versions"][name] = version(name)
    model = LocalProcess("support:create_model_app", output / "model", {})
    try:
        await model.ready("/metrics")
        configurations = [Configuration(name="sqlite-httpx-1", backend="sqlite")]
        if dsn:
            configurations.extend(
                [
                    Configuration(name="postgres-httpx-1", backend="postgres"),
                    Configuration(name="postgres-aiohttp-1", backend="postgres", transport="aiohttp"),
                    Configuration(name="postgres-httpx-2", backend="postgres", workers=2),
                    Configuration(name="postgres-queue-1", backend="postgres", queue_waiters=8),
                ]
            )
        elif not quick:
            configurations.append(Configuration(name="sqlite-aiohttp-1", backend="sqlite", transport="aiohttp"))
        if configuration_names:
            requested = set(configuration_names.split(","))
            unknown = requested - {config.name for config in configurations}
            if unknown:
                raise ValueError(f"Unknown or unavailable configurations: {sorted(unknown)}")
            configurations = [config for config in configurations if config.name in requested]
        selected_phases = set(phase_names.split(",")) if phase_names else None
        async with httpx.AsyncClient(timeout=3, trust_env=False) as control:
            for config in configurations:
                config.timeout_profile = timeout_profile
                async with service_process(config, output / config.name, model.url, dsn) as (service, database):
                    await configure_model(control, model, NORMAL_CONTROL)
                    report["warmups"].append(
                        await warm_workers(service, model, config, client_pool, client_keepalive_expiry)
                    )

                    async def phase(
                        name: str,
                        route: str,
                        concurrency: int,
                        *,
                        seconds=duration,
                        fault=None,
                        operation=None,
                        model_control=None,
                        **kwargs,
                    ):
                        if selected_phases is not None and name not in selected_phases:
                            return None
                        baseline = await settle(service, model, config.workers)
                        if not baseline["recovered"]:
                            raise RuntimeError("Previous work did not drain before the next phase")
                        await configure_model(
                            control, model, {**NORMAL_CONTROL, **(model_control or {}), "reset_metrics": True}
                        )
                        pids = [worker["pid"] for worker in baseline["workers"]]
                        resources_before = {
                            "processes": await process_metrics(pids),
                            "file_descriptors": await descriptor_counts(pids),
                        }
                        database_before = await gather_db_metrics(database) if database is not None else None
                        stop = asyncio.Event()
                        samples = []
                        sampler = asyncio.create_task(sample(service, model, config.workers, stop, samples))
                        injection = asyncio.create_task(fault()) if fault is not None else None
                        started_monotonic = time.monotonic()
                        try:
                            result = (
                                await operation()
                                if operation is not None
                                else await measure_phase(
                                    service.url,
                                    "local-load-test-token",
                                    route,
                                    concurrency,
                                    seconds,
                                    prefix=f"{config.name}-{name}-{uuid4().hex[:6]}",
                                    client_pool=client_pool,
                                    client_keepalive_expiry=client_keepalive_expiry,
                                    request_timeout=20 if timeout_profile == "normal" else 10,
                                    **kwargs,
                                )
                            )
                            if injection is not None:
                                result["fault"] = await injection
                        finally:
                            stop.set()
                            await sampler
                            if injection is not None and not injection.done():
                                injection.cancel()
                                with suppress(asyncio.CancelledError):
                                    await injection
                        await configure_model(control, model, NORMAL_CONTROL)
                        result.update(
                            name=name,
                            configuration=config.model_dump(),
                            samples=samples,
                            recovery=await settle(service, model, config.workers),
                            baseline=baseline,
                            resources_before=resources_before,
                            started_monotonic=started_monotonic,
                        )
                        pids = [worker["pid"] for worker in result["recovery"]["workers"]]
                        result["resources_after"] = {
                            "processes": await process_metrics(pids),
                            "file_descriptors": await descriptor_counts(pids),
                        }
                        if database is not None:
                            result["database_before"] = database_before
                            result["database_after"] = await gather_db_metrics(database)
                        report["phases"].append(result)
                        save_report(output, report)
                        print(
                            json.dumps(
                                {
                                    "phase": config.name + "/" + name,
                                    "status": result.get("status_counts", result.get("statuses")),
                                    "goodput": round(result.get("successful_goodput_rps", 0), 2),
                                    "latency": result.get("successful_latency_ms"),
                                    "recovered": result["recovery"]["recovered"],
                                }
                            ),
                            flush=True,
                        )
                        if not result["recovery"]["recovered"]:
                            raise RuntimeError(f"Work did not drain after {config.name}/{name}")
                        return result

                    for concurrency in [1, 8] if quick else [1, 8 * config.workers, 32 * config.workers]:
                        await phase(f"invoke-c{concurrency}", "invoke", concurrency)
                    if not quick:
                        await phase("invoke-nonstream", "invoke", 8 * config.workers, upstream_stream=False)
                    await phase("sse", "stream", 8 * config.workers)
                    await phase("jsonl", "stream/jsonl", 8 * config.workers)
                    if not quick:
                        await phase("arrival-128", "invoke", 128, rate=128, max_inflight=256)
                        await phase("disconnect", "stream", 16, disconnect_after_first=True, max_requests=128)
                        history = await phase(
                            "same-thread", "invoke", 4, seconds=15, shared_thread=True, max_requests=40
                        )
                        if history is not None:
                            async with httpx.AsyncClient(
                                base_url=service.url,
                                headers={"Authorization": "Bearer local-load-test-token"},
                                trust_env=False,
                            ) as history_client:
                                ids = history["successful_request_ids"]
                                history["history"] = await verify_http_history(
                                    history_client,
                                    agent_id="load-agent",
                                    thread_id=history["shared_thread_id"],
                                    user_id="load-user",
                                    expected_human_messages=ids,
                                    expected_ai_messages=ids,
                                    expected_pairs={value: value for value in ids},
                                )
                            save_report(output, report)

                    if config.name in {"sqlite-httpx-1", "postgres-aiohttp-1"} and not quick:
                        for failure in ("429", "stall", "disconnect"):

                            async def provider_fault(failure=failure):
                                await asyncio.sleep(0.75)
                                await configure_model(
                                    control, model, {"mode": failure, "remaining": -1, "stall_seconds": 10}
                                )
                                await asyncio.sleep(2.5)
                                await configure_model(control, model, NORMAL_CONTROL)
                                return {"type": "provider-" + failure, "injected_seconds": 2.5}

                            await phase(
                                "provider-" + failure,
                                "invoke",
                                32,
                                seconds=max(duration, 8),
                                rate=32,
                                fault=provider_fault,
                            )

                        from slow_reader import measure_slow_readers

                        await phase(
                            "slow-reader",
                            "stream/jsonl",
                            2,
                            model_control={
                                "first_token_ms": 0,
                                "chunk_interval_ms": 0,
                                "stream_chunk_count": 512,
                                "stream_chunk_bytes": 16384,
                            },
                            operation=lambda: measure_slow_readers(service.url, "local-load-test-token"),
                        )

                    if config.name == "postgres-httpx-1":
                        acquired, release = asyncio.Event(), asyncio.Event()

                        async def lock_fault():
                            holder = asyncio.create_task(hold_checkpoint_table_lock(database, release, acquired))
                            try:
                                await asyncio.wait_for(acquired.wait(), 3)
                                await asyncio.sleep(2)
                            finally:
                                release.set()
                                await holder
                            return {"type": "checkpoint table lock", "held_seconds": 2}

                        async def connection_fault():
                            await asyncio.sleep(1)
                            return {
                                "type": "connection termination",
                                "result": await terminate_toolkit_sessions(database, roles=("locks", "saver")),
                            }

                        if not quick:
                            await phase("db-lock", "invoke", 32, rate=64, fault=lock_fault)
                            await phase("db-drop", "invoke", 32, rate=64, fault=connection_fault)

                    if config.name == "postgres-httpx-2" and not quick:

                        async def worker_fault():
                            async with httpx.AsyncClient(
                                timeout=3, trust_env=False, limits=httpx.Limits(max_keepalive_connections=0)
                            ) as client:
                                before = await worker_metrics(client, service.url, 2)
                            pid = min(value["pid"] for value in before)
                            parent = await asyncio.to_thread(
                                subprocess.run,
                                ["ps", "-p", str(pid), "-o", "ppid="],
                                capture_output=True,
                                check=True,
                                timeout=2,
                            )
                            if int(parent.stdout.strip()) != service.process.pid:
                                raise RuntimeError("The selected worker is not owned by this test")
                            await asyncio.sleep(1)
                            os.kill(pid, signal.SIGKILL)
                            return {
                                "type": "worker SIGKILL",
                                "killed_pid": pid,
                                "killed_monotonic": time.monotonic(),
                                "before_pids": [value["pid"] for value in before],
                            }

                        replacement = await phase("worker-kill", "invoke", 64, seconds=30, rate=64, fault=worker_fault)
                        if replacement is not None:
                            replacement["replacement"] = {
                                "new_pids": sorted(
                                    {value["pid"] for value in replacement["recovery"]["workers"]}
                                    - set(replacement["fault"]["before_pids"])
                                ),
                            }
                            first_seen = next(
                                (
                                    sample["time"]
                                    for sample in replacement["samples"]
                                    if any(
                                        worker["pid"] in replacement["replacement"]["new_pids"]
                                        for worker in sample.get("workers", [])
                                    )
                                ),
                                None,
                            )
                            replacement["replacement"]["observed_after_kill_seconds"] = (
                                first_seen - replacement["fault"]["killed_monotonic"]
                                if first_seen is not None
                                else None
                            )
                            verification = await warm_workers(
                                service, model, config, client_pool, client_keepalive_expiry
                            )
                            report["warmups"].append(verification)
                            replacement["replacement"]["agent_verification"] = verification
                            save_report(output, report)
                        await phase(
                            "soak",
                            "stream/jsonl",
                            128,
                            seconds=soak_seconds,
                            rate=80,
                            max_inflight=256,
                            max_requests=20000,
                        )
        report["validation_failures"] = validate_report(report)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_report(output, report)
        if report["validation_failures"]:
            raise RuntimeError(f"Load test checks failed; see {output / 'results.json'}")
        return report
    finally:
        await model.close()


def run(
    output: str = "data/load-test",
    duration: float = 10,
    soak_seconds: float = 60,
    postgres_dsn: str | None = None,
    quick: bool = False,
    configuration_names: str | None = None,
    phase_names: str | None = None,
    client_pool: Literal["slots", "shared"] = "slots",
    timeout_profile: Literal["normal", "aggressive"] = "normal",
    client_keepalive_expiry: float = 1.0,
) -> None:
    """Measure local API pressure. The optional PostgreSQL DSN must be disposable."""
    if not 1 <= duration <= 60 or not 1 <= soak_seconds <= 600:
        raise ValueError("Use duration between 1 and 60 seconds and soak_seconds between 1 and 600 seconds")
    if not math.isfinite(client_keepalive_expiry) or client_keepalive_expiry <= 0:
        raise ValueError("Use a positive finite client_keepalive_expiry in seconds")
    if timeout_profile not in TIMEOUT_PROFILES:
        raise ValueError("Use timeout_profile=normal or timeout_profile=aggressive")
    asyncio.run(
        execute(
            Path(output).resolve(),
            duration,
            soak_seconds,
            postgres_dsn,
            quick,
            configuration_names,
            phase_names,
            client_pool,
            timeout_profile,
            client_keepalive_expiry,
        )
    )


if __name__ == "__main__":
    fire.Fire(run)
