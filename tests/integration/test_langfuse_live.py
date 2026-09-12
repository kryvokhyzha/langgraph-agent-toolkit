"""Check ingestion and reads in an explicitly configured Langfuse test project.

Set only LAT_TEST_LANGFUSE_* variables for these tests.
The test leaves records with the lat-test- prefix in that project.
"""

import ast
import asyncio
import json
import os
import time
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from packaging.version import Version
from pydantic import SecretStr

from langgraph_agent_toolkit.core.observability import langfuse as adapter


pytestmark = pytest.mark.langfuse


class ScriptedModel(FakeMessagesListChatModel):
    """Run model callbacks without a model service or tokenizer download."""

    def bind_tools(self, tools, **kwargs):
        return self

    def get_num_tokens(self, text):
        return len(text.split())

    def get_num_tokens_from_messages(self, messages, tools=None):
        return sum(self.get_num_tokens(str(message.content)) for message in messages)


def tool_call(name, **arguments):
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": uuid4().hex}])


def build_live_graph(builder):
    """Use the package builders and real local arithmetic tools."""
    responses = [tool_call("add", a=2, b=5)]
    if builder == "native":
        from langgraph_agent_toolkit.agents.blueprints.create_agent._shared import build_tool_graph

        answer = "The sum is 7."
        graph = build_tool_graph(ScriptedModel(responses=[*responses, AIMessage(answer)]))
    else:
        from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph
        from langgraph_agent_toolkit.agents.components.tools import multiply

        answer = "The checked result is 21."
        specialist = ScriptedModel(responses=[tool_call("multiply", a=7, b=3), AIMessage("Verified result: 21.")])
        responses.extend(
            [
                tool_call("task", subagent_type="verifier", description="Multiply 7 by 3 and verify the result."),
                AIMessage(answer),
            ]
        )
        graph = build_graph(
            model=ScriptedModel(responses=responses),
            subagents=[
                {
                    "name": "verifier",
                    "description": "Check the calculation.",
                    "model": specialist,
                    "tools": [multiply],
                }
            ],
        )
    return graph, answer


@pytest.fixture(scope="module")
def live_langfuse(request):
    """Keep one SDK client open for all runs in the same test project."""
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    names = ("BASE_URL", "PUBLIC_KEY", "SECRET_KEY", "SERVER_VERSION")
    missing = [f"LAT_TEST_LANGFUSE_{name}" for name in names if not os.environ.get(f"LAT_TEST_LANGFUSE_{name}")]
    if missing:
        pytest.fail("Set the Langfuse test project configuration: " + ", ".join(missing))
    config = {name: os.environ[f"LAT_TEST_LANGFUSE_{name}"] for name in names}
    server_version = Version(config["SERVER_VERSION"])
    server_major = server_version.major
    if server_major not in (2, 3, 4):
        pytest.fail("LAT_TEST_LANGFUSE_SERVER_VERSION must select server v2, v3, or v4")
    if server_major == 2 and adapter._SDK_MAJOR != 2:
        pytest.fail("Langfuse server v2 requires SDK v2")
    if server_major == 3 and server_version < Version("3.63.0") and adapter._SDK_MAJOR != 2:
        pytest.fail("Langfuse server versions before 3.63.0 require SDK v2")
    if server_major == 4 and adapter._SDK_MAJOR == 2:
        pytest.fail("Langfuse server v4 events_only mode does not accept SDK v2 traces")
    timeout = float(os.environ.get("LAT_TEST_LANGFUSE_TIMEOUT", "90"))
    if not 0 < timeout <= 1800:
        pytest.fail("LAT_TEST_LANGFUSE_TIMEOUT must be greater than 0 and at most 1800 seconds")
    for name in tuple(os.environ):
        if name.startswith(("LANGFUSE_", "OTEL_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", config["PUBLIC_KEY"])
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", config["SECRET_KEY"])
    monkeypatch.setenv("LANGFUSE_HOST", config["BASE_URL"])
    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")
    monkeypatch.setattr(
        adapter,
        "settings",
        adapter.settings.model_copy(
            update={
                "LANGFUSE_PUBLIC_KEY": SecretStr(config["PUBLIC_KEY"]),
                "LANGFUSE_SECRET_KEY": SecretStr(config["SECRET_KEY"]),
                "LANGFUSE_HOST": config["BASE_URL"],
            }
        ),
    )
    with httpx.Client(
        base_url=config["BASE_URL"].rstrip("/") + "/",
        auth=(config["PUBLIC_KEY"], config["SECRET_KEY"]),
        timeout=10,
        trust_env=False,
    ) as reader:
        health = reader.get("api/public/health")
        health.raise_for_status()
        actual_version = Version(health.json()["version"])
        assert actual_version == server_version, "The server version differs from LAT_TEST_LANGFUSE_SERVER_VERSION"
        kwargs = {
            "public_key": config["PUBLIC_KEY"],
            "secret_key": config["SECRET_KEY"],
            "host": config["BASE_URL"],
            "flush_at": 1,
            "flush_interval": 0.01,
            "timeout": 10,
        }
        if adapter._IS_NEW_LANGFUSE:
            from opentelemetry.sdk.trace import TracerProvider

            kwargs["tracer_provider"] = TracerProvider()
        client = adapter.Langfuse(**kwargs)
        monkeypatch.setattr(adapter, "_get_langfuse_client", lambda: client)
        observation = adapter.LangfuseObservability()
        prompts = []
        try:
            yield SimpleNamespace(
                observation=observation,
                client=client,
                reader=reader,
                server_major=server_major,
                timeout=timeout,
                prompts=prompts,
            )
        finally:
            try:
                if adapter._SDK_MAJOR >= 3:
                    for name in prompts:
                        observation.delete_prompt(name)
            finally:
                client.shutdown()


def deduplicate_observations(observations):
    """Keep one row per observation ID. Reject conflicting data."""
    by_id = {}
    for item in observations:
        previous = by_id.setdefault(item["id"], item)
        assert previous == item, "The observation API returned conflicting rows for one ID"
    return list(by_id.values())


def read_records(live, trace_id, deadline):
    def get(path, **kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("The ingestion polling deadline expired")
        return live.reader.get(path, timeout=min(10, remaining), **kwargs)

    if live.server_major < 4:
        response = get(f"api/public/traces/{trace_id}")
        if response.status_code == 404:
            return None, [], []
        response.raise_for_status()
        trace = response.json()
        return trace, trace["observations"], trace["scores"]
    observations = []
    cursor = None
    while True:
        params = {"traceId": trace_id, "fields": "core,basic,io", "limit": 100}
        if cursor:
            params["cursor"] = cursor
        response = get("api/public/v2/observations", params=params)
        response.raise_for_status()
        page = response.json()
        for item in page["data"]:
            for field in ("input", "output"):
                item[field] = decoded(item.get(field))
        observations.extend(page["data"])
        next_cursor = page.get("meta", {}).get("cursor")
        if not next_cursor:
            break
        assert next_cursor != cursor, "The observation API repeated its pagination cursor"
        cursor = next_cursor
    observations = deduplicate_observations(observations)
    root = next((item for item in observations if item["name"] == "lat-test-execution"), None)
    response = get("api/public/v3/scores", params={"traceId": trace_id, "name": "lat-test-correctness", "limit": 100})
    response.raise_for_status()
    return root, observations, response.json()["data"]


def decoded(value):
    """Read JSON values and the legacy tool-input representation."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
    return value


def contains(value, text):
    return text in json.dumps(decoded(value), ensure_ascii=False)


def ancestors(item, by_id):
    """Check parent links up to the application root observation."""
    parents = []
    parent = item.get("parentObservationId")
    while parent:
        assert parent in by_id, "A stored observation has a missing parent"
        assert parent not in parents and parent != item["id"], "Stored observations contain a parent cycle"
        parents.append(parent)
        if by_id[parent]["name"] == "lat-test-execution":
            break
        parent = by_id[parent].get("parentObservationId")
    return parents


def assert_graph_records(root, observations, scores, *, server_major, builder, identifier, trace_id, prompt, answer):
    """Require stored graph results and a connected model and tool tree."""
    assert root is not None, "The root record is not available"
    assert root.get("input") == prompt, "The root input is not stored"
    assert root.get("output") == answer, "The root output is not stored"
    assert root["userId"] == identifier
    assert root["sessionId"] == identifier
    if server_major == 4:
        for item in observations:
            assert item.get("userId") == identifier, f"The {item['name']} observation has a different user"
            assert item.get("sessionId") == identifier, f"The {item['name']} observation has a different session"
    by_id = {item["id"]: item for item in observations}
    graphs = [item for item in observations if item["name"] == "lat-test-agent"]
    assert len(graphs) == 1, "The graph observation is not available"
    graph = graphs[0]
    assert graph.get("endTime") and contains(graph.get("output"), answer), "The graph result is not stored"
    assert contains(graph.get("input"), prompt), "The graph input is not stored"
    if adapter._IS_NEW_LANGFUSE:
        roots = [item for item in observations if item["name"] == "lat-test-execution"]
        assert len(roots) == 1 and roots[0].get("endTime"), "The root observation is not complete"
        assert roots[0]["id"] in ancestors(graph, by_id), "The graph is not below the root observation"

    generations = [item for item in observations if item.get("type", "").upper() == "GENERATION"]
    assert len(generations) == (2 if builder == "native" else 5), "Model observations are missing"
    for item in [graph, *generations]:
        assert item["traceId"] == trace_id
        assert item.get("endTime") and item.get("input") and item.get("output"), "A model or graph has incomplete I/O"
    assert any(contains(item["input"], prompt) for item in generations), "The model input is not stored"
    assert any(contains(item["output"], answer) for item in generations), "The final model answer is not stored"
    assert all(graph["id"] in ancestors(item, by_id) for item in generations)

    def stored_tool(name, arguments, expected_output):
        matches = [item for item in observations if item["name"] == name]
        assert len(matches) == 1, f"The {name} tool observation is not available"
        item = matches[0]
        assert item["traceId"] == trace_id
        assert item.get("endTime"), f"The {name} tool is not complete"
        assert decoded(item.get("input")) == arguments, f"The {name} tool input differs"
        assert graph["id"] in ancestors(item, by_id), f"The {name} tool is not below the graph"
        result = decoded(item.get("output"))
        if name == "task" and isinstance(result, dict):
            messages = result.get("update", {}).get("messages", [])
            assert len(messages) == 1 and messages[0]["type"] == "tool", "The task result message is not stored"
            result = messages[0]
        if isinstance(result, dict):
            result = result.get("content")
        assert result is not None, f"The {name} tool output is not stored"
        if isinstance(expected_output, int):
            assert float(result) == expected_output, f"The {name} tool result differs"
        else:
            assert result == expected_output, f"The {name} tool result differs"
        return item

    addition = stored_tool("add", {"a": 2, "b": 5}, 7)
    if builder == "deepagents":
        delegated = stored_tool(
            "task",
            {"subagent_type": "verifier", "description": "Multiply 7 by 3 and verify the result."},
            "Verified result: 21.",
        )
        multiplication = stored_tool("multiply", {"a": 7, "b": 3}, 21)
        assert delegated["id"] in ancestors(multiplication, by_id), "The subagent tool is outside its task"
        assert delegated["id"] not in ancestors(addition, by_id), "The parent tool is inside the subagent task"
        child_generations = [item for item in generations if delegated["id"] in ancestors(item, by_id)]
        assert len(child_generations) == 2, "The subagent model observations are outside their task"
        assert any(contains(item["output"], "Verified result: 21.") for item in child_generations)

    feedback = [item for item in scores if item["name"] == "lat-test-correctness"]
    assert len(feedback) == 1 and feedback[0]["value"] == 1, "The feedback is not stored"


@pytest.mark.asyncio
@pytest.mark.parametrize("builder", ["native", "deepagents"])
async def test_prompt_versions_and_completed_trace_are_readable(live_langfuse, builder):
    """Read prompt versions, graph I/O, model and tool links, and feedback."""
    live = live_langfuse
    obs = live.observation
    identifier = f"lat-test-{uuid4()}"
    obs.push_prompt(identifier, "Add {{a}} and {{b}}.", force_create_new_version=False)
    live.prompts.append(identifier)
    first, first_remote = obs.pull_prompt(
        identifier, return_with_prompt_object=True, template_format="jinja2", cache_ttl_seconds=0
    )
    assert first.invoke({"a": 2, "b": 5}).to_messages()[0].content == "Add 2 and 5."
    obs.push_prompt(identifier, "Calculate {{a}} plus {{b}}.", force_create_new_version=False)
    second, second_remote = obs.pull_prompt(
        identifier, return_with_prompt_object=True, template_format="jinja2", cache_ttl_seconds=0
    )
    assert second_remote.version == first_remote.version + 1
    messages = second.invoke({"a": 2, "b": 5}).to_messages()
    assert messages[0].content == "Calculate 2 plus 5."
    graph, answer = build_live_graph(builder)

    run_id = str(uuid4())
    callback = obs.get_callback_handler(run_id=run_id, user_id=identifier, session_id=identifier)
    with obs.trace_context(
        run_id, agent_name="lat-test-execution", user_id=identifier, session_id=identifier, input=messages[0].content
    ) as trace:
        result = await graph.ainvoke(
            {"messages": messages},
            config={"callbacks": [callback], "run_name": "lat-test-agent", "recursion_limit": 30},
        )
        output = result["messages"][-1].content
        assert output == answer
        assert float(next(item for item in result["messages"] if isinstance(item, ToolMessage)).content) == 7
        obs.update_trace(trace, output=output)
    obs.record_feedback(run_id, "lat-test-correctness", 1, comment=identifier)
    live.client.flush()
    trace_id = run_id.replace("-", "") if adapter._IS_NEW_LANGFUSE else run_id
    deadline = time.monotonic() + live.timeout
    last_failure = "The server has not returned graph records"
    while True:
        try:
            root, observations, scores = read_records(live, trace_id, deadline)
            assert_graph_records(
                root,
                observations,
                scores,
                server_major=live.server_major,
                builder=builder,
                identifier=identifier,
                trace_id=trace_id,
                prompt=messages[0].content,
                answer=answer,
            )
            return
        except (AssertionError, TimeoutError, httpx.TimeoutException) as exc:
            last_failure = str(exc)
        if time.monotonic() >= deadline:
            pytest.fail(
                f"Langfuse trace {trace_id} did not expose complete graph records "
                f"within {live.timeout:g} seconds: {last_failure}"
            )
        await asyncio.sleep(min(1, deadline - time.monotonic()))
