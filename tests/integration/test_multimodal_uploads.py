"""Verify document uploads through HTTP, checkpoints, and OpenAI serialization."""

import json
import sys
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.graph import END, MessagesState, StateGraph
from pydantic import SecretStr

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core.memory.types import MemoryBackends
from langgraph_agent_toolkit.core.models.chat_openai import ChatOpenAIPatched
from langgraph_agent_toolkit.core.observability.types import ObservabilityBackend
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import constants
from langgraph_agent_toolkit.helper.types import EnvironmentMode
from langgraph_agent_toolkit.service.handler import create_app
from langgraph_agent_toolkit.ui.utils.message import build_chat_message


HEADERS = {"Authorization": "Bearer document-test-credential"}


async def test_document_uploads_survive_restart_and_reach_responses_with_filenames(monkeypatch, tmp_path):
    """Preserve document bytes and names through API history and model requests."""
    requests = []

    def provider(request):
        assert request.method == "POST"
        assert request.url.path == "/v1/responses"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_documents",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-4.1",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_documents",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Documents received.", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "total_tokens": 13,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    transport = httpx.MockTransport(provider)
    with httpx.Client(transport=transport) as sync_http:
        async with httpx.AsyncClient(transport=transport) as async_http:
            model = ChatOpenAIPatched(
                model="gpt-4.1",
                api_key="offline-document-test",
                base_url="https://provider.invalid/v1",
                use_responses_api=True,
                http_client=sync_http,
                http_async_client=async_http,
                max_retries=0,
            )

            async def reply(state):
                response = await model.ainvoke(state["messages"])
                assert isinstance(response, AIMessage)
                return {"messages": [response]}

            def build_graph():
                builder = StateGraph(MessagesState)
                builder.add_node("reply", reply)
                builder.set_entry_point("reply")
                builder.add_edge("reply", END)
                return builder.compile()

            module = ModuleType("lat_document_test_" + uuid4().hex)
            module.agent = Agent("documents", "Read uploaded documents.", build_graph())
            monkeypatch.setitem(sys.modules, module.__name__, module)
            for key, value in {
                "ENV_MODE": EnvironmentMode.PRODUCTION,
                "AUTH_MODE": "trusted",
                "AUTH_SECRET": SecretStr("document-test-credential"),
                "AUTH_USERS": {},
                "MEMORY_BACKEND": MemoryBackends.SQLITE,
                "SQLITE_DB_PATH": str(tmp_path / "documents.sqlite"),
                "OBSERVABILITY_BACKEND": ObservabilityBackend.EMPTY,
                "AGENT_PATHS": [f"{module.__name__}:agent"],
                "DEFAULT_AGENT": "documents",
                "MCP_SERVERS": {},
                "MCP_AGENT_SERVERS": {},
                "MULTIMODAL_MAX_ATTACHMENTS": 6,
            }.items():
                monkeypatch.setattr(settings, key, value)
            monkeypatch.setattr(constants, "_runtime_default_agent", "documents")
            uploads = [
                SimpleNamespace(name=f"report.{extension}", type=None, getvalue=lambda: b"Document test bytes.")
                for extension in ("docx", "pptx", "xlsx", "csv", "json", "md")
            ]
            message = build_chat_message("Read these documents.", uploads)
            with TestClient(create_app()) as client:
                response = client.post(
                    "/documents/invoke",
                    headers=HEADERS,
                    json={"input": {"message": message}, "thread_id": "document-thread"},
                )
                assert response.status_code == 200, response.text

            module.agent.graph = build_graph()
            with TestClient(create_app()) as client:
                history = client.get("/documents/history", headers=HEADERS, params={"thread_id": "document-thread"})
                assert history.status_code == 200, history.text
                assert history.json()["messages"][0]["content"] == message
                follow_up = client.post(
                    "/documents/invoke",
                    headers=HEADERS,
                    json={"input": {"message": "Read them again."}, "thread_id": "document-thread"},
                )
                assert follow_up.status_code == 200, follow_up.text

    expected_files = [
        {
            "type": "input_file",
            "filename": block["extras"]["filename"],
            "file_data": f"data:{block['mime_type']};base64,{block['base64']}",
        }
        for block in message
        if block["type"] == "file"
    ]
    assert len(requests) == 2
    for request in requests:
        files = [
            block
            for item in request["input"]
            if isinstance(item.get("content"), list)
            for block in item["content"]
            if block["type"] == "input_file"
        ]
        assert files == expected_files
