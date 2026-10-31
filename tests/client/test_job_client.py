"""Check job requests through the actual HTTPX client interfaces."""

import json

import httpx
import pytest

from langgraph_agent_toolkit.client import AgentClient, AgentClientError
from langgraph_agent_toolkit.schema.execution import ExecutionRequest


JOB_ID = "7d638a81-7d4a-4e2d-b16f-633354efbe22"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_submit_inspect_cancel_keep_identity_and_operation_key(asynchronous):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            202 if request.url.path.endswith("/jobs") else 200,
            json={
                "job_id": JOB_ID,
                "agent_id": "processor",
                "thread_id": "work-item",
                "status": "queued",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
        )

    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as sync_http:
        async with httpx.AsyncClient(transport=transport) as async_http:
            async with AgentClient(
                "http://test",
                agent="processor",
                get_info=False,
                auth_secret="test-token",
                http_client=sync_http,
                async_http_client=async_http,
            ) as client:
                value = ExecutionRequest(thread_id="work-item", input={"number": 7}, user_id="owner")
                if asynchronous:
                    job = await client.asubmit_job(value, idempotency_key="operation-1")
                    await client.aget_job(job.job_id, user_id="owner")
                    await client.acancel_job(job.job_id, user_id="owner")
                else:
                    job = client.submit_job(value, idempotency_key="operation-1")
                    client.get_job(job.job_id, user_id="owner")
                    client.cancel_job(job.job_id, user_id="owner")
                assert job.status == "queued"
                assert requests[0].url.path == "/processor/jobs"
                assert requests[0].headers["idempotency-key"] == "operation-1"
                body = json.loads(requests[0].content)
                assert body == {"thread_id": "work-item", "input": {"number": 7}, "user_id": "owner"}
                assert requests[1].method == "GET"
                assert requests[1].url.path == f"/jobs/{JOB_ID}"
                assert requests[2].method == "POST"
                assert requests[2].url.path == f"/jobs/{JOB_ID}/cancel"
                assert all(request.url.params["user_id"] == "owner" for request in requests[1:])
                assert all(request.headers["authorization"] == "Bearer test-token" for request in requests)
            assert not sync_http.is_closed
            assert not async_http.is_closed


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_job_conflict_retains_http_status_and_does_not_retry(asynchronous):
    calls = []

    def conflict(request):
        calls.append(request)
        return httpx.Response(409, json={"detail": "Operation key already used"})

    transport = httpx.MockTransport(conflict)
    with httpx.Client(transport=transport) as sync_http:
        async with httpx.AsyncClient(transport=transport) as async_http:
            async with AgentClient(
                "http://test", get_info=False, http_client=sync_http, async_http_client=async_http
            ) as client:
                with pytest.raises(AgentClientError) as error:
                    value = ExecutionRequest(thread_id="work-item", input={"number": 8})
                    if asynchronous:
                        await client.asubmit_job(value, idempotency_key="operation-1")
                    else:
                        client.submit_job(value, idempotency_key="operation-1")
                assert error.value.status_code == 409
                assert len(calls) == 1
                assert calls[0].url.path == "/jobs"
