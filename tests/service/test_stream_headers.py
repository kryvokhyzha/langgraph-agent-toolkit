"""Keep SSE responses uncached and unbuffered by supporting proxies."""

import pytest


@pytest.mark.parametrize("path", ["/stream", "/agent/stream"])
def test_sse_routes_send_proxy_headers(test_client, path):
    response = test_client.post(path, json={"input": {"message": "hello"}})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.text.endswith("data: [DONE]\n\n")


def test_jsonl_keeps_its_existing_header_contract(test_client):
    response = test_client.post("/stream/jsonl", json={"input": {"message": "hello"}})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/jsonl")
    assert "x-accel-buffering" not in response.headers
    assert "[DONE]" not in response.text
