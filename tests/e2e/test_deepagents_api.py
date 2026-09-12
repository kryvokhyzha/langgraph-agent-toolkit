"""Keep virtual files private and durable across a real API process crash."""

import json

import pytest

from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.schema import ChatMessage


pytest.importorskip("deepagents")


@pytest.mark.parametrize(
    "api_service",
    [{"AGENT_PATHS": json.dumps(["deep_support_agent:deep_agent"]), "DEFAULT_AGENT": "deep-agent"}],
    indirect=True,
)
def test_deepagents_files_survive_process_replacement_and_remain_private(api_service):
    def connect():
        return AgentClient(api_service.url, agent="deep-agent", auth_secret=api_service.secret, timeout=15)

    with connect() as client:
        saved = client.invoke({"message": "write:Private report for user one."}, thread_id="report", user_id="user-1")
        assert saved.content == "Saved."

    api_service.restart()

    with connect() as client:
        events = list(client.stream_jsonl({"message": "read"}, thread_id="report", user_id="user-1"))
        replies = [event for event in events if isinstance(event, ChatMessage) and event.type == "ai"]
        assert "Private report for user one." in replies[-1].content
        history = client.get_history("report", user_id="user-1").messages
        assert history[0].content == "write:Private report for user one."

        for thread, user in (("another-thread", "user-1"), ("report", "user-2")):
            missing = client.invoke({"message": "read"}, thread_id=thread, user_id=user)
            assert "not found" in missing.content.lower()
            assert "Private report" not in missing.content

        client.clear_history("report", user_id="user-1")
        missing = client.invoke({"message": "read"}, thread_id="report", user_id="user-1")
        assert "not found" in missing.content.lower()
