"""Recover typed questions after a real API process exits."""

import json

import httpx
import pytest
from api_process import ApiProcess

from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.schema import ChatMessage


@pytest.mark.parametrize("protocol", ["invoke", "stream", "stream_jsonl"])
def test_typed_questions_survive_process_crash_and_resume_without_repeating_writes(tmp_path, protocol):
    events = tmp_path / "typed-events.jsonl"
    api = ApiProcess(tmp_path)
    api.env.update(
        AGENT_PATHS=json.dumps(["typed_support_agent:typed_agent"]),
        DEFAULT_AGENT="typed-journey",
        LAT_TEST_TYPED_EVENTS=str(events),
        CHECKPOINT_DURABILITY="sync",
    )

    def connect():
        return AgentClient(api.url, auth_secret=api.secret, timeout=15)

    def by_id(records):
        return {item["id"]: item for item in records}

    try:
        api.start()
        with connect() as client:
            if protocol == "invoke":
                paused = client.invoke({"message": "Begin."}, thread_id="typed", user_id="user-one")
            else:
                messages = list(getattr(client, protocol)({"message": "Begin."}, thread_id="typed", user_id="user-one"))
                paused = next(
                    message
                    for message in messages
                    if isinstance(message, ChatMessage) and message.custom_data.get("interrupts")
                )
        pending = paused.custom_data["interrupts"]
        assert len(pending) == 2
        assert all(item["response_schema"]["properties"]["quantity"]["minimum"] == 1 for item in pending)
        assert [json.loads(line) for line in events.read_text().splitlines()] == [{"event": "committed"}]

        api.restart()

        with connect() as client:
            assert by_id(client.get_history("typed", user_id="user-one").pending_interrupts) == by_id(pending)
            assert client.get_history("typed", user_id="user-two").pending_interrupts == []
            assert client.get_history("another", user_id="user-one").pending_interrupts == []
        before = events.read_text()
        with httpx.Client(timeout=15, trust_env=False) as http:
            rejected = http.post(
                f"{api.url}/invoke",
                headers=api.headers,
                json={
                    "thread_id": "typed",
                    "user_id": "user-one",
                    "input": {"resume": {item["id"]: {"quantity": 0} for item in pending}},
                },
            )
            assert rejected.status_code == 400, rejected.text
        assert events.read_text() == before

        api.restart()

        with connect() as client:
            recovered = client.get_history("typed", user_id="user-one").pending_interrupts
            assert by_id(recovered) == by_id(pending)
            quantities = {"first": 2, "second": 4}
            client.invoke(
                {"resume": {item["id"]: {"quantity": quantities[item["value"]["label"]]} for item in recovered}},
                thread_id="typed",
                user_id="user-one",
            )
            saved = client.get_history("typed", user_id="user-one")
            assert saved.pending_interrupts == []
            contents = [message.content for message in saved.messages]
            assert contents.count("The initial write is complete.") == 1
            assert "Quantity for first: 2." in contents
            assert "Quantity for second: 4." in contents

        api.restart()
        with connect() as client:
            assert client.get_history("typed", user_id="user-one").pending_interrupts == []
        effects = [json.loads(line) for line in events.read_text().splitlines()]
        assert effects.count({"event": "committed"}) == 1
        assert sorted((item["label"], item["quantity"]) for item in effects if item["event"] == "answered") == [
            ("first", 2),
            ("second", 4),
        ]
    finally:
        api.stop()
