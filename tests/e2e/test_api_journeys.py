"""Exercise user journeys through a real API, client, and SQLite database."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import httpx
import pytest

from langgraph_agent_toolkit.client import AgentClient, AgentClientError
from langgraph_agent_toolkit.schema import ChatMessage


def client_for(service):
    return AgentClient(service.url, agent="journey-agent", auth_secret=service.secret, timeout=10, stream_timeout=10)


def history_contents(client, thread_id, user_id="user-1"):
    return [message.content for message in client.get_history(thread_id, user_id=user_id).messages]


def test_shared_client_auth_keeps_user_memory_and_thread_history_separate(api_service):
    with httpx.Client(base_url=api_service.url, timeout=5, trust_env=False) as http:
        assert http.get("/info").status_code == 401
        denied = http.post(
            "/invoke", headers={"Authorization": "Bearer wrong-credential"}, json={"input": {"message": "denied"}}
        )
        assert denied.status_code == 401
        assert denied.headers["www-authenticate"] == "Bearer"

    with client_for(api_service) as client:
        assert client.info.default_agent == "journey-agent"
        first = client.invoke({"message": "remember:tea"}, user_id="user-1")
        assert str(UUID(first.thread_id)) == first.thread_id
        assert first.content == "turn=1; preference=tea; message=remember:tea"
        second = client.invoke({"message": "recall"}, thread_id="second-thread", user_id="user-1")
        assert second.content == "turn=1; preference=tea; message=recall"
        other_user = client.invoke({"message": "recall"}, thread_id="second-thread", user_id="user-2")
        assert other_user.content == "turn=1; preference=none; message=recall"
        assert history_contents(client, first.thread_id) == ["remember:tea", first.content]
        assert history_contents(client, "second-thread") == ["recall", second.content]
        assert history_contents(client, "second-thread", "user-2") == ["recall", other_user.content]

        client.clear_history(first.thread_id, user_id="user-1")
        assert history_contents(client, first.thread_id) == []
        assert history_contents(client, "second-thread") == ["recall", second.content]
        after_clear = client.invoke({"message": "recall"}, thread_id=first.thread_id, user_id="user-1")
        assert after_clear.content == "turn=1; preference=tea; message=recall"


@pytest.mark.parametrize("protocol", ["sse", "jsonl"])
def test_stream_protocol_finishes_and_reports_errors_to_the_client(api_service, protocol):
    endpoint = "stream" if protocol == "sse" else "stream/jsonl"
    with client_for(api_service) as client:
        stream = client.stream if protocol == "sse" else client.stream_jsonl
        events = list(stream({"message": "hello"}, thread_id="stream-thread", user_id="user-1"))
        messages = [event for event in events if isinstance(event, ChatMessage)]
        tokens = [event for event in events if isinstance(event, str)]
        assert messages[-1].content == "turn=1; preference=none; message=hello"
        assert "".join(tokens) == messages[-1].content
        assert messages[-1].thread_id == "stream-thread"
        assert str(UUID(messages[-1].run_id)) == messages[-1].run_id
        assert history_contents(client, "stream-thread") == ["hello", messages[-1].content]

        with pytest.raises(AgentClientError, match="Internal server error") as error:
            list(stream({"message": "fail"}, thread_id="failed-client", user_id="user-1"))
        assert "E2E internal failure detail" not in str(error.value)
        recovered = list(stream({"message": "recovered"}, thread_id="failed-client", user_id="user-1"))
        assert recovered[-1].content.endswith("message=recovered")

    with httpx.Client(base_url=api_service.url, headers=api_service.headers, timeout=10, trust_env=False) as http:
        for text in ("complete", "fail"):
            response = http.post(
                f"/journey-agent/{endpoint}",
                json={"input": {"message": text}, "thread_id": f"wire-{text}", "user_id": "user-1"},
            )
            assert response.status_code == 200
            lines = [line for line in response.text.splitlines() if line]
            if protocol == "sse":
                assert response.headers["content-type"].startswith("text/event-stream")
                assert lines[-1] == "data: [DONE]"
                assert lines.count("data: [DONE]") == 1
                chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
            else:
                assert response.headers["content-type"].startswith("application/jsonl")
                assert "[DONE]" not in response.text
                chunks = [json.loads(line) for line in lines]
            if text == "complete":
                assert chunks[-1]["type"] == "message"
                assert chunks[-1]["content"]["content"] == "turn=1; preference=none; message=complete"
                assert not any(chunk["type"] == "error" for chunk in chunks)
            else:
                assert chunks == [{"type": "error", "content": "Internal server error"}]
        assert http.get("/health/ready").status_code == 200


def test_concurrent_turns_and_pending_interrupt_survive_process_replacement(api_service):
    def send(index):
        with client_for(api_service) as client:
            return client.invoke({"message": f"slow:{index}"}, thread_id="shared-thread", user_id="user-1")

    with ThreadPoolExecutor(max_workers=4) as workers:
        responses = list(workers.map(send, range(4)))
    assert sorted(int(response.content.split(";", 1)[0].removeprefix("turn=")) for response in responses) == [
        1,
        2,
        3,
        4,
    ]
    with client_for(api_service) as client:
        before = history_contents(client, "shared-thread")
        assert len(before) == 8
        assert set(before[::2]) == {f"slow:{index}" for index in range(4)}
        paused = client.invoke({"message": "pause"}, thread_id="pending-thread", user_id="user-1")
        assert paused.content == "Choose the next action."
        assert paused.custom_data["interrupts"][0]["id"]

    api_service.restart()

    with client_for(api_service) as client:
        assert history_contents(client, "shared-thread") == before
        continued = client.invoke({"message": "after-restart"}, thread_id="shared-thread", user_id="user-1")
        assert continued.content == "turn=5; preference=none; message=after-restart"
        resumed = client.invoke({"message": "continue"}, thread_id="pending-thread", user_id="user-1")
        assert resumed.content == "turn=1; preference=none; message=resumed:continue"
        assert resumed.custom_data == {}
        assert history_contents(client, "pending-thread") == ["pause", resumed.content]

    api_service.stop()
    assert api_service.database.is_file()
    with sqlite3.connect(f"file:{api_service.database}?mode=ro", uri=True) as database:
        assert database.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
