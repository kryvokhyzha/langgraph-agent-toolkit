"""Check the live-test reader against the v4 API response contract."""

import time
from types import SimpleNamespace

import httpx
import pytest
import test_langfuse_live as live
from packaging.version import Version


def test_v4_reader_decodes_json_and_coalesces_identical_rows_across_pages():
    root = {
        "id": "root-id",
        "name": "lat-test-execution",
        "input": '"Calculate 2 plus 5."',
        "output": '"The sum is 7."',
    }
    generation = {"id": "model-id", "name": "model", "input": '{"a":2}', "output": '{"content":"7"}'}
    requests = []

    def respond(request):
        requests.append(request)
        assert request.url.params["traceId"] == "trace-id"
        if request.url.path.endswith("/scores"):
            return httpx.Response(200, json={"data": [{"name": "lat-test-correctness", "value": 1}]})
        assert "parseIoAsJson" not in request.url.params
        assert request.url.params["fields"] == "core,basic,io"
        if "cursor" not in request.url.params:
            return httpx.Response(200, json={"data": [generation], "meta": {"cursor": "next-page"}})
        assert request.url.params["cursor"] == "next-page"
        return httpx.Response(200, json={"data": [generation, root], "meta": {}})

    with httpx.Client(base_url="http://langfuse.invalid/", transport=httpx.MockTransport(respond)) as reader:
        result, observations, scores = live.read_records(
            SimpleNamespace(reader=reader, server_major=4), "trace-id", time.monotonic() + 5
        )

    assert result["input"] == "Calculate 2 plus 5."
    assert result["output"] == "The sum is 7."
    assert len(observations) == 2
    assert observations[0]["input"] == {"a": 2}
    assert observations[0]["output"] == {"content": "7"}
    assert scores == [{"name": "lat-test-correctness", "value": 1}]
    assert len(requests) == 3


@pytest.mark.parametrize("field", ["output", "parentObservationId", "endTime"])
def test_duplicate_ids_cannot_hide_conflicting_results_or_parent_links(field):
    original = {"id": "model-id", "output": "7", "parentObservationId": "graph-id", "endTime": "completed"}
    conflicting = {**original, field: "different"}
    with pytest.raises(AssertionError, match="conflicting rows"):
        live.deduplicate_observations([original, conflicting])


@pytest.mark.parametrize("server", ["3.194.0", "4.58.9"])
def test_sdk_v5_rejects_servers_without_the_modern_score_api(monkeypatch, server):
    monkeypatch.setattr(live.adapter, "_SDK_MAJOR", 5)
    with pytest.raises(pytest.fail.Exception, match="SDK v5 requires server 4.59.0"):
        live.validate_server_version(Version(server))


def test_sdk_v5_accepts_the_minimum_supported_server(monkeypatch):
    monkeypatch.setattr(live.adapter, "_SDK_MAJOR", 5)
    live.validate_server_version(Version("4.59.0"))
