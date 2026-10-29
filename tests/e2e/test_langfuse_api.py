"""Read real Langfuse records after toolkit API worker shutdown and restart.

Use only a dedicated LAT_TEST_LANGFUSE_* project. These tests leave records
with a lat-test- prefix in that project. They do not call a model service.
"""

import ast
import json
import os
import time
from importlib.metadata import version
from uuid import uuid4

import httpx
import pytest
from api_process import ApiProcess
from packaging.version import Version
from pydantic import SecretStr

from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.schema import ChatMessage


pytestmark = pytest.mark.langfuse


def decode(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
    return value


def read_trace(reader, trace_id, server_major):
    """Read stored observations and scores through the server's public APIs."""
    if server_major < 4:
        response = reader.get(f"api/public/traces/{trace_id}")
        if response.status_code == 404:
            return None, [], []
        response.raise_for_status()
        trace = response.json()
        root = next((item for item in trace["observations"] if item["name"] == "lat-process-agent"), trace)
        root = {**root, "userId": trace["userId"], "sessionId": trace["sessionId"]}
        return root, trace["observations"], trace["scores"]
    observations = {}
    cursors = set()
    cursor = None
    while True:
        params = {"traceId": trace_id, "fields": "core,basic,io,prompt", "limit": 100}
        if cursor:
            params["cursor"] = cursor
        response = reader.get("api/public/v2/observations", params=params)
        response.raise_for_status()
        page = response.json()
        for item in page["data"]:
            previous = observations.setdefault(item["id"], item)
            assert previous == item, "One observation ID has conflicting stored rows"
        cursor = page.get("meta", {}).get("cursor")
        if not cursor:
            break
        assert cursor not in cursors, "The observation API repeated a cursor"
        cursors.add(cursor)
    rows = list(observations.values())
    root = next((item for item in rows if item["name"] == "lat-process-agent"), None)
    response = reader.get("api/public/v3/scores", params={"traceId": trace_id, "name": "lat-test-correctness"})
    response.raise_for_status()
    return root, rows, response.json()["data"]


def check_trace(
    root,
    observations,
    scores,
    *,
    trace_id,
    user,
    thread,
    request,
    answer,
    prompt_name,
    prompt_version,
    mode,
    sdk,
    server_major,
):
    """Require complete graph, model, tool, identity, and prompt records."""
    assert root is not None, "The application root is missing"
    if sdk >= 3:
        assert root.get("endTime"), "The application root is incomplete"
    assert root["userId"] == user and root["sessionId"] == thread
    assert request in json.dumps(decode(root.get("input"))), "The root input is not stored"
    assert answer in json.dumps(decode(root.get("output"))), "The root output is not stored"
    by_id = {item["id"]: item for item in observations}
    assert all(item["traceId"] == trace_id for item in observations)
    if server_major >= 4:
        assert all(item.get("userId") == user and item.get("sessionId") == thread for item in observations)
    generations = [item for item in observations if item["type"].upper() == "GENERATION"]
    assert len(generations) == (2 if mode == "native" else 5), "Model observations are missing"
    for item in generations:
        assert item.get("input") and item.get("output") and item.get("endTime"), "A model has incomplete I/O"
        parents = set()
        parent = item.get("parentObservationId")
        while parent:
            assert parent not in parents, "Observation parents contain a cycle"
            parents.add(parent)
            assert parent in by_id, "A stored model parent is missing"
            if parent == root["id"]:
                break  # The application root can have an external parent.
            parent = by_id[parent].get("parentObservationId")
        if sdk >= 3:
            assert root["id"] in parents, "A model is outside the application root"
    addition = [item for item in observations if item["name"] == "add"]
    assert len(addition) == 1 and addition[0].get("endTime"), "The arithmetic tool did not complete exactly once"
    assert decode(addition[0].get("input")) == {"a": 2, "b": 5}
    if mode == "deepagents":
        assert len([item for item in observations if item["name"] == "multiply"]) == 1
        assert len([item for item in observations if item["name"] == "task"]) == 1
    if sdk >= 3:
        assert any(
            item.get("promptName") == prompt_name and item.get("promptVersion") == prompt_version
            for item in generations
        ), "The stored model does not link to the selected prompt version"
    feedback = [item for item in scores if item["name"] == "lat-test-correctness"]
    assert len(feedback) == 1 and feedback[0]["value"] == 1, "Feedback is missing or duplicated"


def await_trace(reader, trace_id, config, expected):
    deadline = time.monotonic() + config["TIMEOUT"]
    last_error = "No records"
    while time.monotonic() < deadline:
        try:
            check_trace(*read_trace(reader, trace_id, config["VERSION"].major), trace_id=trace_id, **expected)
            return
        except AssertionError as exc:
            last_error = str(exc)
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    pytest.fail(f"Langfuse did not store complete API process records: {last_error}")


@pytest.mark.parametrize("mode", ["native", "deepagents"])
def test_api_workers_flush_linked_traces_and_feedback_across_restart(tmp_path, mode):
    names = ("BASE_URL", "PUBLIC_KEY", "SECRET_KEY", "SERVER_VERSION")
    missing = [f"LAT_TEST_LANGFUSE_{name}" for name in names if not os.environ.get(f"LAT_TEST_LANGFUSE_{name}")]
    if missing:
        pytest.fail("Set the Langfuse test project configuration: " + ", ".join(missing))
    config = {name: os.environ[f"LAT_TEST_LANGFUSE_{name}"] for name in names}
    for key in ("PUBLIC_KEY", "SECRET_KEY"):
        config[key] = SecretStr(config[key])
    config["VERSION"] = Version(config["SERVER_VERSION"])
    config["TIMEOUT"] = float(os.environ.get("LAT_TEST_LANGFUSE_TIMEOUT", "90"))
    assert 0 < config["TIMEOUT"] <= 1800
    sdk = Version(version("langfuse")).major
    flush_interval = 3600 if sdk >= 5 else type(settings).model_fields["LANGFUSE_FLUSH_INTERVAL"].default
    if sdk >= 5 and config["VERSION"] < Version("4.59.0"):
        pytest.fail("Langfuse SDK v5 requires server 4.59.0 or later")
    if config["VERSION"].major == 4 and sdk == 2:
        pytest.fail("Langfuse server v4 events_only mode does not accept SDK v2 traces")
    if config["VERSION"].major == 3 and config["VERSION"] < Version("3.63.0") and sdk >= 3:
        pytest.fail("Langfuse server versions before 3.63.0 require SDK v2")
    assert config["VERSION"].major in (3, 4), "This process journey requires a Langfuse v3 or v4 test server"
    if mode == "deepagents":
        pytest.importorskip("deepagents")
    identifier = "lat-test-" + uuid4().hex
    user, thread, prompt_name = identifier + "-user", identifier + "-thread", identifier + "-prompt"
    api = ApiProcess(tmp_path)
    api.env.update(
        AGENT_PATHS=json.dumps(["langfuse_support_agent:langfuse_agent"]),
        DEFAULT_AGENT="lat-process-agent",
        OBSERVABILITY_BACKEND="langfuse",
        LANGFUSE_HOST=config["BASE_URL"],
        LANGFUSE_PUBLIC_KEY=config["PUBLIC_KEY"].get_secret_value(),
        LANGFUSE_SECRET_KEY=config["SECRET_KEY"].get_secret_value(),
        LANGFUSE_FLUSH_AT="100",
        LANGFUSE_FLUSH_INTERVAL=str(flush_interval),
        LANGFUSE_MEDIA_UPLOAD_ENABLED="false",
        LAT_TEST_LANGFUSE_PROMPT_NAME=prompt_name,
        LAT_TEST_LANGFUSE_BUILDER=mode,
    )
    runs = []
    with httpx.Client(
        base_url=config["BASE_URL"].rstrip("/") + "/",
        auth=(config["PUBLIC_KEY"].get_secret_value(), config["SECRET_KEY"].get_secret_value()),
        timeout=10,
        trust_env=False,
    ) as reader:
        health = reader.get("api/public/health")
        health.raise_for_status()
        assert Version(health.json()["version"]) == config["VERSION"]
        try:
            for index, protocol in enumerate(("invoke", "stream_jsonl"), start=1):
                created = reader.post(
                    "api/public/v2/prompts",
                    json={
                        "name": prompt_name,
                        "type": "text",
                        "prompt": f"Prompt revision {index}. {{{{request}}}}",
                        "labels": ["production"],
                    },
                )
                created.raise_for_status()
                prompt_version = created.json()["version"]
                api.start()
                with AgentClient(api.url, auth_secret=api.secret, timeout=30) as client:
                    if runs:
                        history = client.get_history(thread, user_id=user)
                        assert any(message.content == runs[0][1]["answer"] for message in history.messages)
                    request = f"Calculate revision {index}."
                    if protocol == "invoke":
                        message = client.invoke({"message": request}, thread_id=thread, user_id=user)
                    else:
                        streamed = list(client.stream_jsonl({"message": request}, thread_id=thread, user_id=user))
                        message = next(
                            item for item in reversed(streamed) if isinstance(item, ChatMessage) and item.type == "ai"
                        )
                    answer = "The sum is 7." if mode == "native" else "The checked result is 21."
                    assert message.content == answer
                    trace_id = message.run_id.replace("-", "") if sdk >= 3 else message.run_id
                    expected = dict(
                        user=user,
                        thread=thread,
                        request=request,
                        answer=answer,
                        prompt_name=prompt_name,
                        prompt_version=prompt_version,
                        mode=mode,
                        sdk=sdk,
                        server_major=config["VERSION"].major,
                    )
                    runs.append((trace_id, expected))
                    client.create_feedback(message.run_id, "lat-test-correctness", 1, user_id=user)
                # SDK v5 wakes buffered scores. Earlier SDKs wait for their interval.
                api.stop()
                await_trace(reader, trace_id, config, expected)
            for trace_id, expected in runs:
                await_trace(reader, trace_id, config, expected)
            api.start()
            with AgentClient(api.url, auth_secret=api.secret, timeout=15) as client:
                saved = client.get_history(thread, user_id=user)
                assert saved.pending_interrupts == []
                assert sum(message.content == runs[0][1]["answer"] for message in saved.messages) == 2
        finally:
            api.stop()
