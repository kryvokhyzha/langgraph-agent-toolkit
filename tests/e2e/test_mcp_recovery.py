"""Recover parallel MCP requests after the API process loses its response."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from api_process import ApiProcess

from langgraph_agent_toolkit.client import AgentClient


pytest.importorskip("fastmcp")


def test_parallel_mcp_questions_survive_process_replacement_without_replaying_history(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    event_file = tmp_path / "mcp-events.jsonl"
    mcp_log = tmp_path / "mcp.log"
    api = ApiProcess(tmp_path)
    api.env.update(
        AGENT_PATHS=json.dumps(["mcp_support_agent:mcp_agent"]),
        DEFAULT_AGENT="mcp-journey",
        MCP_SERVERS=json.dumps({"journey": {"url": f"http://127.0.0.1:{port}/mcp"}}),
        CHECKPOINT_DURABILITY="sync",
    )
    with mcp_log.open("wb") as log:
        server = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("mcp_support_server.py")), str(port), str(event_file)],
            env=api.env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        try:
            deadline = time.monotonic() + 15
            while True:
                if server.poll() is not None:
                    pytest.fail(f"The MCP server exited before startup.\n{mcp_log.read_text()}")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        pytest.fail(f"The MCP server did not start.\n{mcp_log.read_text()}")
                    time.sleep(0.05)
            api.start()
            with httpx.Client(timeout=15, trust_env=False) as http:
                paused = http.post(
                    f"{api.url}/invoke",
                    headers=api.headers,
                    json={"thread_id": "parallel", "input": {"message": "Begin."}},
                )
                assert paused.status_code == 200, paused.text
                # Ignore the response. The next worker must recover the question IDs.
                before_read = event_file.read_text()
                api.restart()
                with AgentClient(api.url, auth_secret=api.secret, timeout=15) as client:
                    first = client.get_history("parallel", offset=99, limit=1)
                    assert first.messages == []
                    assert len(first.pending_interrupts) == 2
                    assert event_file.read_text() == before_read
                    assert client.get_history("another").pending_interrupts == []
                    assert client.get_history("parallel", user_id="another").pending_interrupts == []
                    resumed = client.invoke(
                        {
                            "resume": {
                                item["id"]: {
                                    "responses": {
                                        "details": {
                                            "action": "accept",
                                            "content": {"answer": item["id"]},
                                        }
                                    }
                                }
                                for item in first.pending_interrupts
                            }
                        },
                        thread_id="parallel",
                    )
                    second = resumed.custom_data["interrupts"]
                    assert len(second) == 2
                    assert {item["id"] for item in second} == {item["id"] for item in first.pending_interrupts}
                api.restart()
                before_read = event_file.read_text()
                with AgentClient(api.url, auth_secret=api.secret, timeout=15) as client:
                    recovered = client.get_history("parallel")
                    assert recovered.pending_interrupts == second
                    assert event_file.read_text() == before_read
                    complete = client.invoke(
                        {
                            "resume": {
                                item["id"]: {
                                    "responses": {
                                        "details": {
                                            "action": "accept",
                                            "content": {"confirmation": "yes"},
                                        }
                                    }
                                }
                                for item in recovered.pending_interrupts
                            }
                        },
                        thread_id="parallel",
                    )
                    assert complete.content == "Complete."
                    history = client.get_history("parallel")
                    assert history.pending_interrupts == []
                    tools = [message for message in history.messages if message.type == "tool"]
                    assert len(tools) == 2
                    assert all(message.tool_status == "success" for message in tools)
                    assert {message.tool_artifact["structured_content"]["label"] for message in tools} == {
                        "first",
                        "second",
                    }
                    writes = [
                        json.loads(line)
                        for line in event_file.read_text().splitlines()
                        if json.loads(line)["event"] == "write"
                    ]
                    assert len(writes) == 2
                    assert {item["answer"] for item in writes} == {item["id"] for item in first.pending_interrupts}
                    before_clear = event_file.read_text()
                    client.clear_history("parallel")
                    assert client.get_history("parallel").pending_interrupts == []
                    assert event_file.read_text() == before_clear
        finally:
            api.stop()
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
