"""Read real Phoenix records after API and observability server restarts.

Use only the managed disposable LAT_TEST_PHOENIX_* server. These tests use
synthetic prompts and local tools. They do not call a model service.
"""

import json
import os
import signal
import time
from uuid import uuid4

import httpx
import pytest
from api_process import ApiProcess

from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.schema import ChatMessage
from scripts.ci.phoenix_stack import _command, restart


pytestmark = pytest.mark.phoenix_live


@pytest.fixture
def phoenix_server():
    names = ("BASE_URL", "SERVER_VERSION", "STACK_DIRECTORY")
    missing = [f"LAT_TEST_PHOENIX_{name}" for name in names if not os.environ.get(f"LAT_TEST_PHOENIX_{name}")]
    if missing:
        pytest.fail("Set the managed Phoenix test server configuration: " + ", ".join(missing))
    config = {name: os.environ[f"LAT_TEST_PHOENIX_{name}"] for name in names}
    _, managed, _ = _command(config["STACK_DIRECTORY"])
    assert config["BASE_URL"] == managed["LAT_TEST_PHOENIX_BASE_URL"], "The server does not match its managed stack"
    assert config["SERVER_VERSION"] == managed["LAT_TEST_PHOENIX_SERVER_VERSION"]
    config["TIMEOUT"] = float(os.environ.get("LAT_TEST_PHOENIX_TIMEOUT", "30"))
    assert 0 < config["TIMEOUT"] <= 180
    return config


def make_api(directory, config, project, prompt_name, mode):
    api = ApiProcess(directory)
    api.env.update(
        AGENT_PATHS=json.dumps(["phoenix_support_agent:phoenix_agent"]),
        DEFAULT_AGENT="lat-process-agent",
        OBSERVABILITY_BACKEND="phoenix",
        PHOENIX_ENDPOINT=config["BASE_URL"],
        PHOENIX_PROJECT_NAME=project,
        LAT_TEST_PHOENIX_PROMPT_NAME=prompt_name,
        LAT_TEST_PHOENIX_BUILDER=mode,
    )
    return api


def create_prompt(reader, name, index):
    response = reader.post(
        "v1/prompts",
        json={
            "prompt": {"name": name},
            "version": {
                "model_provider": "OPENAI",
                "model_name": "lat-test-calculator",
                "template": {
                    "type": "chat",
                    "messages": [{"role": "user", "content": f"Prompt revision {index}. {{request}}"}],
                },
                "template_type": "CHAT",
                "template_format": "F_STRING",
                "invocation_parameters": {"type": "openai", "openai": {}},
            },
        },
    )
    response.raise_for_status()
    return response.json()["data"]["id"]


def read_trace(reader, project, trace_id):
    """Read stored spans and native trace annotations through independent HTTP calls."""
    response = reader.get(f"v1/projects/{project}/spans", params={"trace_id": trace_id, "limit": 1000})
    if response.status_code == 404:
        return [], []
    response.raise_for_status()
    page = response.json()
    assert not page.get("next_cursor"), "The test trace exceeds the bounded span query"
    response = reader.get(f"v1/projects/{project}/trace_annotations", params={"trace_ids": trace_id, "limit": 1000})
    response.raise_for_status()
    annotations = response.json()
    assert not annotations.get("next_cursor"), "The test trace exceeds the bounded annotation query"
    return page["data"], annotations["data"]


def check_trace(spans, annotations, *, trace_id, user, thread, request, answer, prompt_version, mode, feedback=True):
    """Require complete model, tool, identity, prompt, and parent records."""
    roots = [item for item in spans if item["name"] == "lat-process-agent"]
    assert len(roots) == 1, "The application root is missing or duplicated"
    root = roots[0]
    attributes = root["attributes"]
    assert attributes["user.id"] == user
    assert attributes["session.id"] == thread
    assert request in attributes["input.value"], "The root input is missing"
    assert answer in attributes["output.value"], "The root output is missing"
    assert root["parent_id"] is None, "The application root has an unexpected parent"
    by_id = {item["context"]["span_id"]: item for item in spans}
    assert len(by_id) == len(spans), "Stored spans contain duplicate IDs"
    assert all(item["context"]["trace_id"] == trace_id for item in spans)
    assert all(item["end_time"] and item["status_code"] in ("OK", "UNSET") for item in spans), "A span did not complete"
    for item in spans:
        attributes = item["attributes"]
        assert attributes["user.id"] == user and attributes["session.id"] == thread
        parents = set()
        parent = item["parent_id"]
        while parent:
            assert parent not in parents, "Span parents contain a cycle"
            parents.add(parent)
            assert parent in by_id, "A stored span parent is missing"
            parent = by_id[parent]["parent_id"]
        if item is not root:
            assert root["context"]["span_id"] in parents, "A span is outside the application root"
    generations = [item for item in spans if item["span_kind"] == "LLM"]
    assert len(generations) == (2 if mode == "native" else 5), "Model spans are missing"
    for item in generations:
        attributes = item["attributes"]
        assert item["status_code"] == "OK", "A model did not complete successfully"
        assert attributes.get("input.value") and attributes.get("output.value"), "A model has incomplete I/O"
        assert attributes["llm.prompt_template.version"] == prompt_version, "A model lost its prompt version"
    addition = [item for item in spans if item["name"] == "add" and item["span_kind"] == "TOOL"]
    assert len(addition) == 1, "The arithmetic tool did not run exactly once"
    assert json.loads(addition[0]["attributes"]["input.value"]) == {"a": 2, "b": 5}
    if mode == "deepagents":
        assert len([item for item in spans if item["name"] == "multiply" and item["span_kind"] == "TOOL"]) == 1
        assert len([item for item in spans if item["name"] == "task" and item["span_kind"] == "TOOL"]) == 1
    if feedback:
        scores = [item for item in annotations if item["name"] == "lat-test-correctness"]
        assert len(scores) == 1, "Native feedback is missing or duplicated"
        assert scores[0]["trace_id"] == trace_id and scores[0]["annotator_kind"] == "HUMAN"
        assert scores[0]["result"]["score"] == 1
        assert scores[0]["result"]["explanation"] == "The arithmetic result is correct."
    else:
        assert annotations == [], "The run has unexpected annotations"


def await_trace(reader, project, trace_id, expected, timeout):
    deadline = time.monotonic() + timeout
    last_error = "No records"
    while time.monotonic() < deadline:
        try:
            records = read_trace(reader, project, trace_id)
            check_trace(*records, trace_id=trace_id, **expected)
            return records
        except AssertionError as exc:
            last_error = str(exc)
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    pytest.fail(f"Phoenix did not store complete API process records: {last_error}")


@pytest.mark.parametrize("mode", ["native", "deepagents"])
def test_api_and_phoenix_restarts_preserve_linked_traces_and_feedback(tmp_path, mode, phoenix_server):
    config = phoenix_server
    if mode == "deepagents":
        pytest.importorskip("deepagents")
    identifier = "lat-test-" + uuid4().hex
    user, thread, prompt_name = identifier + "-user", identifier + "-thread", identifier + "-prompt"
    project = identifier + "-project"
    api = make_api(tmp_path, config, project, prompt_name, mode)
    runs = []
    snapshots = {}
    with httpx.Client(base_url=config["BASE_URL"].rstrip("/") + "/", timeout=10, trust_env=False) as reader:
        health = reader.get("healthz")
        health.raise_for_status()
        assert health.headers["x-phoenix-server-version"] == config["SERVER_VERSION"]
        try:
            for index, protocol in enumerate(("invoke", "stream_jsonl"), start=1):
                prompt_version = create_prompt(reader, prompt_name, index)
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
                    trace_id = message.run_id.replace("-", "")
                    expected = dict(
                        user=user,
                        thread=thread,
                        request=request,
                        answer=answer,
                        prompt_version=prompt_version,
                        mode=mode,
                    )
                    runs.append((trace_id, expected))
                    client.create_feedback(
                        message.run_id,
                        "lat-test-correctness",
                        1,
                        kwargs={"comment": "The arithmetic result is correct."},
                        user_id=user,
                    )
                api.stop()
                snapshots[trace_id] = await_trace(reader, project, trace_id, expected, config["TIMEOUT"])
            api.start()
            with AgentClient(api.url, auth_secret=api.secret, timeout=15) as client:
                saved = client.get_history(thread, user_id=user)
                assert saved.pending_interrupts == []
                assert sum(message.content == runs[0][1]["answer"] for message in saved.messages) == 2
            api.stop()
            restart(config["STACK_DIRECTORY"])
            for trace_id, expected in runs:
                stored = await_trace(reader, project, trace_id, expected, config["TIMEOUT"])
                assert stored == snapshots[trace_id], "The server restart changed stored spans or annotations"
            (tmp_path / "phoenix-records.json").write_text(
                json.dumps({"project": project, "traces": snapshots}, indent=2) + "\n"
            )
            versions = reader.get(f"v1/prompts/{prompt_name}/versions")
            versions.raise_for_status()
            assert {item["id"] for item in versions.json()["data"]} == {run[1]["prompt_version"] for run in runs}
        finally:
            api.stop()


def test_shutdown_drains_a_buffered_custom_provider(tmp_path, phoenix_server):
    config = phoenix_server
    identifier = "lat-test-" + uuid4().hex
    project, prompt_name = identifier + "-project", identifier + "-prompt"
    user, thread = identifier + "-user", identifier + "-thread"
    api = make_api(tmp_path, config, project, prompt_name, "native")
    api.env["OTEL_BSP_SCHEDULE_DELAY"] = "3600000"
    with httpx.Client(base_url=config["BASE_URL"].rstrip("/") + "/", timeout=10, trust_env=False) as reader:
        prompt_version = create_prompt(reader, prompt_name, 1)
        try:
            api.start()
            with AgentClient(api.url, auth_secret=api.secret, timeout=30) as client:
                message = client.invoke({"message": "Calculate the buffered request."}, thread_id=thread, user_id=user)
            assert message.content == "The sum is 7."
            trace_id = message.run_id.replace("-", "")
            assert read_trace(reader, project, trace_id) == ([], []), "The background interval did not retain the spans"
            api.stop()
            assert api.process.returncode in (0, -signal.SIGTERM), "The API did not stop gracefully"
            records = await_trace(
                reader,
                project,
                trace_id,
                dict(
                    user=user,
                    thread=thread,
                    request="Calculate the buffered request.",
                    answer="The sum is 7.",
                    prompt_version=prompt_version,
                    mode="native",
                    feedback=False,
                ),
                config["TIMEOUT"],
            )
            (tmp_path / "phoenix-records.json").write_text(
                json.dumps({"project": project, "traces": {trace_id: records}}, indent=2) + "\n"
            )
        finally:
            api.stop()
