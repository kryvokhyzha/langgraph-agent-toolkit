"""Compare actual API tool selection with deterministic model responses."""

import importlib.util
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest
from api_process import ApiProcess

from langgraph_agent_toolkit.client import AgentClient, AgentClientError
from langgraph_agent_toolkit.schema import ChatMessage


def events(path, case=None):
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return records if case is None else [item for item in records if item["case"] == case]


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def client_for(api, agent):
    return AgentClient(api.url, agent=agent, auth_secret=api.secret, timeout=30)


def invoke_case(api, agent, case, *, secondary=False, protocol="invoke", bad_selector=False):
    instruction = f"workflow {'secondary' if secondary else 'primary'} case={case}"
    if bad_selector:
        instruction += " bad-selector"
    with client_for(api, agent) as client:
        started = time.perf_counter()
        if protocol == "invoke":
            result = client.invoke({"message": instruction}, thread_id=case, user_id="benchmark-user")
        else:
            method = client.stream if protocol == "sse" else client.stream_jsonl
            streamed = list(method({"message": instruction}, thread_id=case, user_id="benchmark-user"))
            assert "ToolSelectionResponse" not in str(streamed), "Internal selector calls reached the public stream"
            result = next(item for item in reversed(streamed) if isinstance(item, ChatMessage) and item.type == "ai")
        duration = time.perf_counter() - started
    assert result.content == f"Complete: verified-{case}"
    return duration


def check_case(records, *, mode, builder, secondary=False):
    main = [item for item in records if item["kind"] == "main"]
    selector = [item for item in records if item["kind"] == "selector"]
    assert len(main) == (3 if builder == "native" else 7)
    assert len(selector) == (0 if mode == "off" else len(main))
    assert len([item for item in records if item["kind"] == "effect"]) == 1
    required = {"lookup_secondary", "record_secondary"} if secondary else {"lookup_primary", "record_primary"}
    other_pair = {"lookup_primary", "record_primary", "lookup_secondary", "record_secondary"} - required
    for item in main:
        visible = set(item["tools"])
        assert required <= visible
        assert "add" in visible
        if mode == "llm":
            assert not any(name.startswith("unrelated_") for name in visible)
            assert not other_pair & visible
        if builder == "deepagents":
            assert {"task", "write_todos", "read_file", "write_file"} <= visible
    if builder == "deepagents":
        assert len([item for item in records if item["kind"] == "specialist_tool"]) == 1
    return main, selector


@pytest.mark.parametrize("builder", ["native", "deepagents"])
def test_tool_selection_benchmark_and_api_recovery(tmp_path, builder):
    if builder == "deepagents" and importlib.util.find_spec("deepagents") is None:
        pytest.skip("Install the deepagents extra for this API journey.")
    event_file = tmp_path / "model-events.jsonl"
    api = ApiProcess(tmp_path)
    paths = [f"tool_selection_support_agent:agent_{mode}_{size}" for size in (8, 32, 128) for mode in ("off", "llm")]
    paths.append("tool_selection_support_agent:agent_fallback")
    api.env.update(
        AGENT_PATHS=json.dumps(paths),
        DEFAULT_AGENT="selection-off-8",
        LAT_TEST_TOOL_SELECTION_EVENTS=str(event_file),
        LAT_TEST_TOOL_SELECTION_BUILDER=builder,
    )
    rows = []
    first = None
    try:
        api.start()
        for size in (8, 32, 128):
            for mode in ("off", "llm"):
                durations, main_calls, selector_calls, main_bytes, selector_bytes = [], [], [], [], []
                visible_counts, specialist_calls, total_calls, total_bytes = [], [], [], []
                for trial in range(3):
                    case = f"benchmark-{uuid4().hex}"
                    agent = f"selection-{mode}-{size}"
                    duration = invoke_case(
                        api, agent, case, secondary=trial == 1, protocol=("invoke", "sse", "jsonl")[trial]
                    )
                    records = events(event_file, case)
                    main, selector = check_case(records, mode=mode, builder=builder, secondary=trial == 1)
                    specialists = [item for item in records if item["kind"] == "specialist"]
                    durations.append(duration)
                    main_calls.append(len(main))
                    selector_calls.append(len(selector))
                    main_bytes.append(sum(item["schema_bytes"] for item in main))
                    selector_bytes.append(sum(item["schema_bytes"] for item in selector))
                    specialist_calls.append(len(specialists))
                    total_calls.append(len(main) + len(selector) + len(specialists))
                    total_bytes.append(sum(item["schema_bytes"] for item in [*main, *selector, *specialists]))
                    visible_counts.append(max(len(item["tools"]) for item in main))
                    if size == 32 and mode == "llm" and trial == 0:
                        first = (agent, case)
                rows.append(
                    {
                        "mode": mode,
                        "distractor_tools": size,
                        "requests": len(durations),
                        "successful_requests": len(durations),
                        "protocol_per_request": ["invoke", "sse", "jsonl"],
                        "latency_ms_per_request": [round(value * 1000, 3) for value in durations],
                        "p50_ms": round(statistics.median(durations) * 1000, 3),
                        "p95_ms": round(percentile(durations, 0.95) * 1000, 3),
                        "main_model_calls_per_request": main_calls,
                        "selector_model_calls_per_request": selector_calls,
                        "specialist_model_calls_per_request": specialist_calls,
                        "total_model_calls_per_request": total_calls,
                        "main_tool_schema_bytes_per_request": main_bytes,
                        "selector_tool_schema_bytes_per_request": selector_bytes,
                        "total_bound_schema_bytes_per_request": total_bytes,
                        "max_main_visible_tools_per_request": visible_counts,
                        "write_effects_per_request": [1] * len(durations),
                    }
                )
        for size in (8, 32, 128):
            baseline, selected = [row for row in rows if row["distractor_tools"] == size]
            assert max(selected["main_tool_schema_bytes_per_request"]) < min(
                baseline["main_tool_schema_bytes_per_request"]
            )
            assert max(selected["max_main_visible_tools_per_request"]) < min(
                baseline["max_main_visible_tools_per_request"]
            )

        concurrent = [(f"parallel-{uuid4().hex}", bool(index % 2)) for index in range(6)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [
                pool.submit(invoke_case, api, "selection-llm-32", case, secondary=secondary)
                for case, secondary in concurrent
            ]
            for future in futures:
                future.result(timeout=30)
        for case, secondary in concurrent:
            check_case(events(event_file, case), mode="llm", builder=builder, secondary=secondary)

        failed = f"selector-error-{uuid4().hex}"
        with client_for(api, "selection-llm-32") as client:
            with pytest.raises(AgentClientError):
                client.invoke({"message": f"workflow primary bad-selector case={failed}"}, thread_id=failed)
        failure_events = events(event_file, failed)
        assert len([item for item in failure_events if item["kind"] == "selector"]) == 2
        assert not any(item["kind"] in ("effect", "main") for item in failure_events)
        recovered = f"selector-recovered-{uuid4().hex}"
        invoke_case(api, "selection-llm-32", recovered)
        check_case(events(event_file, recovered), mode="llm", builder=builder)
        fallback = f"selector-fallback-{uuid4().hex}"
        invoke_case(api, "selection-llm-32-all", fallback, bad_selector=True)
        fallback_events = events(event_file, fallback)
        main_count = 3 if builder == "native" else 7
        assert len([item for item in fallback_events if item["kind"] == "selector"]) == main_count * 2
        assert len([item for item in fallback_events if item["kind"] == "effect"]) == 1

        api.restart()
        agent, case = first
        with client_for(api, agent) as client:
            history = client.get_history(case, user_id="benchmark-user")
            assert history.messages[-1].content == f"Complete: verified-{case}"
            assert any(message.tool_calls for message in history.messages if message.type == "ai")
            if builder == "deepagents":
                recalled = client.invoke(
                    {"message": f"read-note case=recall-{uuid4().hex}"}, thread_id=case, user_id="benchmark-user"
                )
                assert f"verified-{case}" in recalled.content
    finally:
        api.stop()
    report = {
        "builder": builder,
        "model_kind": "scripted",
        "transport": "real toolkit TCP API",
        "storage": "SQLite",
        "schema_measure": "UTF-8 bytes of serialized OpenAI-style tool definitions, not token counts",
        "latency_scope": "Local API, graph, SQLite, and scripted model overhead. No provider inference or network.",
        "latency_assertions": "No speedup or latency threshold is asserted.",
        "latency_samples": "Each variant has one invoke, one SSE, and one JSONL request. The summary mixes protocols.",
        "repetitions_per_variant": 3,
        "rows": rows,
        "concurrent_requests_verified": len(concurrent),
        "selector_failure_verified": True,
        "healthy_request_after_failure_verified": True,
        "fallback_write_count_verified": 1,
        "api_restart_verified": True,
    }
    (tmp_path / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    if directory := os.environ.get("LAT_TEST_TOOL_SELECTION_REPORT_DIR"):
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{builder}.json").write_text(json.dumps(report, indent=2) + "\n")
