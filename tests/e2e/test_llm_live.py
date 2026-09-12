"""Check real OpenAI tools, usage, streams, and cancellation through the API.

Use --run-e2e --run-llm. Each transport can spend at most six model calls.
Collection does not read credentials or start a provider client.
"""

import asyncio
import json
import os
import re
import sys
import time
from uuid import UUID, uuid4

import httpx
import pytest
from conftest import ApiProcess


pytestmark = [pytest.mark.e2e, pytest.mark.llm]


class LiveHTTPFailure(AssertionError):
    """Keep safe HTTP failure fields without request or response bodies."""

    def __init__(self, operation, status, error_code):
        self.summary = {"operation": operation, "http_status": status, "error_code": error_code}
        super().__init__(f"Live test HTTP operation {operation} failed with status {status}")


async def require_success(response, operation):
    if response.status_code == 200:
        return
    await response.aread()
    error_code = None
    try:
        body = response.json()
        candidate = body.get("error_code") if isinstance(body, dict) else None
        if isinstance(candidate, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", candidate):
            error_code = candidate
    except ValueError:
        pass
    raise LiveHTTPFailure(operation, response.status_code, error_code)


@pytest.fixture
def live_api_service(tmp_path, monkeypatch, request):
    """Pass explicit credentials only to the isolated child process."""
    if not request.config.getoption("--run-llm") or not request.config.getoption("--run-e2e"):
        pytest.skip("Use --run-e2e --run-llm to run live model tests.")
    key = os.environ.get("LAT_TEST_OPENAI_API_KEY", "")
    model = os.environ.get("LAT_TEST_OPENAI_MODEL", "")
    if not key or not model:
        pytest.fail("Set LAT_TEST_OPENAI_API_KEY and LAT_TEST_OPENAI_MODEL explicitly.")
    parameters = json.loads(os.environ.get("LAT_TEST_OPENAI_MODEL_KWARGS", "{}"))
    if not isinstance(parameters, dict) or set(parameters) - {"reasoning_effort", "temperature"}:
        pytest.fail("LAT_TEST_OPENAI_MODEL_KWARGS can contain only reasoning_effort and temperature.")
    transport = request.param
    service = ApiProcess(tmp_path)
    service.env.update(
        AGENT_PATHS=json.dumps(["live_support_agent:live_agent"]),
        DEFAULT_AGENT="live-agent",
        USE_FAKE_MODEL="false",
        MODEL_CONFIGS="{}",
        OPENAI_API_KEY=key,
        OPENAI_BASE_URL="https://api.openai.com/v1",
        OPENAI_API_BASE="https://api.openai.com/v1",
        OPENAI_MODEL_NAME=model,
        LAT_TEST_OPENAI_MODEL=model,
        LAT_TEST_OPENAI_MODEL_KWARGS=json.dumps(parameters),
        LAT_TEST_LLM_CHILD="yes",
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
        LANGFUSE_TRACING_ENABLED="false",
        LLM_HTTP_ASYNC_TRANSPORT=transport,
        LLM_HTTP_MAX_RETRIES="0",
        LLM_HTTP_READ_TIMEOUT="60",
        REQUEST_TIMEOUT="90",
        REQUEST_CLEANUP_TIMEOUT="10",
        RESPONSE_SEND_TIMEOUT="10",
    )
    bootstrap_module = sys.modules[ApiProcess.__module__]
    bootstrap = bootstrap_module.BOOTSTRAP.replace(
        "import runpy",
        "from langgraph_agent_toolkit.service import factory\n"
        "from live_support_agent import create_app\n"
        "factory.create_app = create_app\n"
        "import runpy",
    )
    monkeypatch.setattr(bootstrap_module, "BOOTSTRAP", bootstrap)
    try:
        service.start()
        yield service, transport, model
    finally:
        service.stop()


def checked_reply(message, thread_id):
    assert message["type"] == "ai"
    assert isinstance(message["content"], str) and message["content"].strip()
    assert message["thread_id"] == thread_id
    assert str(UUID(message["run_id"])) == message["run_id"]
    usage = message["usage_metadata"]
    assert all(
        type(usage[name]) is int and usage[name] > 0 for name in ("input_tokens", "output_tokens", "total_tokens")
    )
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    return {"thread_id": thread_id, "run_id": message["run_id"], "usage_metadata": usage}


async def metrics(http):
    response = await http.get("/live/metrics")
    await require_success(response, "metrics")
    result = response.json()
    assert all(type(value) is int and value >= 0 for value in result.values())
    assert result["attempted"] <= 6
    return result


async def drained(http, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = await metrics(http)
        if all(result[key] == 0 for key in ("active", "admission_active", "admission_waiting", "stalled_cleanups")):
            return result
        await asyncio.sleep(0.05)
    pytest.fail("The live request did not release model and conversation resources before the deadline.")


def payload(message, thread_id):
    return {"input": {"message": message}, "thread_id": thread_id, "user_id": "live-user", "stream_tokens": True}


async def read_stream(http, protocol, thread_id, message, *, disconnect=False):
    began = time.monotonic()
    route = "stream" if protocol == "sse" else "stream/jsonl"
    final = None
    first_token = None
    done = 0
    async with http.stream("POST", f"/live-agent/{route}", json=payload(message, thread_id)) as response:
        await require_success(response, protocol)
        expected_type = "text/event-stream" if protocol == "sse" else "application/jsonl"
        assert response.headers["content-type"].startswith(expected_type)
        async for line in response.aiter_lines():
            if not line or line.startswith(":"):
                continue
            if protocol == "sse":
                assert line.startswith("data:")
                line = line.removeprefix("data:").strip()
            if line == "[DONE]":
                assert protocol == "sse"
                done += 1
                continue
            assert done == 0
            event = json.loads(line)
            assert event["type"] in {"token", "message"}
            if event["type"] == "token" and event["content"]:
                if first_token is None:
                    first_token = time.monotonic() - began
                if disconnect:
                    return {
                        "thread_id": thread_id,
                        "first_token_seconds": first_token,
                        "closed_after_first_token": True,
                    }
            if event["type"] == "message" and event["content"]["type"] == "ai":
                final = event["content"]
    assert not disconnect, "The stream ended before the test could close it after a token."
    assert done == (1 if protocol == "sse" else 0)
    assert first_token is not None
    return {
        **checked_reply(final, thread_id),
        "protocol": protocol,
        "first_token_seconds": first_token,
        "elapsed_seconds": time.monotonic() - began,
    }


@pytest.mark.parametrize("live_api_service", ["httpx", "aiohttp"], indirect=True)
@pytest.mark.asyncio
async def test_live_model_tools_streams_and_disconnect_recovery(live_api_service):
    service, transport, model = live_api_service
    report = {"transport": transport, "model": model, "call_limit": 6, "steps": [], "passed": False}
    prefix = f"live-{transport}-{uuid4().hex[:8]}"
    async with httpx.AsyncClient(
        base_url=service.url,
        headers=service.headers,
        timeout=95,
        trust_env=False,
        limits=httpx.Limits(max_connections=4, max_keepalive_connections=4, keepalive_expiry=1),
    ) as http:
        try:
            assert (await metrics(http))["attempted"] == 0
            tool_thread = prefix + "-tools"
            began = time.monotonic()
            response = await http.post(
                "/live-agent/invoke", json=payload("live-tools: Use add to calculate 2 plus 5.", tool_thread)
            )
            await require_success(response, "invoke-tools")
            tool_reply = checked_reply(response.json(), tool_thread)
            report["steps"].append({"phase": "tools", **tool_reply, "elapsed_seconds": time.monotonic() - began})
            history = await http.get("/live-agent/history", params={"thread_id": tool_thread, "user_id": "live-user"})
            await require_success(history, "history")
            messages = history.json()["messages"]
            assert [message["type"] for message in messages] == ["human", "ai", "tool", "ai"]
            requested = messages[1]["tool_calls"]
            assert len(requested) == 1
            assert requested[0]["name"] == "add"
            assert requested[0]["args"] == {"a": 2, "b": 5}
            assert requested[0]["id"] == messages[2]["tool_call_id"]
            assert messages[2]["content"] == "7"
            assert messages[-1]["content"] == response.json()["content"]
            assert messages[-1]["usage_metadata"] == tool_reply["usage_metadata"]
            assert messages[1]["usage_metadata"]["total_tokens"] > 0
            report["tool_usage_metadata"] = messages[1]["usage_metadata"]
            assert (await drained(http))["attempted"] == 2

            replies = await asyncio.gather(
                read_stream(http, "sse", prefix + "-sse", "Write one short sentence about a river."),
                read_stream(http, "jsonl", prefix + "-jsonl", "Write one short sentence about a mountain."),
            )
            assert len({reply["run_id"] for reply in replies}) == 2
            report["steps"].extend({"phase": "concurrent-stream", **reply} for reply in replies)
            after_streams = await drained(http)
            assert after_streams["attempted"] == 4
            assert after_streams["peak_active"] >= 2

            cancel_thread = prefix + "-cancel"
            cancelled_before = after_streams["cancelled"]
            early = await read_stream(
                http,
                "sse",
                cancel_thread,
                "Write a numbered list of 100 different flowers, one flower per line. Start immediately.",
                disconnect=True,
            )
            after_close = await drained(http)
            assert after_close["attempted"] == 5
            early["model_cancellation_observed"] = after_close["cancelled"] > cancelled_before
            if not early["model_cancellation_observed"]:
                early["limitation"] = "The provider call completed before cancellation reached the model."
            report["steps"].append({"phase": "early-disconnect", **early})
            for path in ("live", "ready"):
                await require_success(await http.get("/health/" + path), "health-" + path)

            began = time.monotonic()
            recovered = await http.post(
                "/live-agent/invoke",
                json=payload("Now write one short greeting. Do not continue the list.", cancel_thread),
            )
            await require_success(recovered, "invoke-recovery")
            report["steps"].append(
                {
                    "phase": "same-thread-recovery",
                    **checked_reply(recovered.json(), cancel_thread),
                    "elapsed_seconds": time.monotonic() - began,
                }
            )
            final_metrics = await drained(http)
            assert final_metrics["attempted"] == 6
            assert final_metrics["failed"] == 0
            assert final_metrics["completed"] + final_metrics["cancelled"] == 6
            assert final_metrics["tools_completed"] == 1
            assert final_metrics["transport_pools"] == 1
            report["passed"] = True
        except LiveHTTPFailure as error:
            report["http_failure"] = error.summary
            raise
        finally:
            try:
                report["metrics"] = await metrics(http)
            except LiveHTTPFailure as error:
                report["metrics_http_failure"] = error.summary
            except Exception as error:
                report["metrics_error_type"] = type(error).__name__
            (service.directory / "live-result.json").write_text(json.dumps(report, indent=2) + "\n")
