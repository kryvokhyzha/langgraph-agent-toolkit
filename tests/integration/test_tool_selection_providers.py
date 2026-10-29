"""Verify deferred tools through native provider SDK serialization."""

import json

import httpx
import pytest
from langchain_core.tools import tool

from langgraph_agent_toolkit.agents.blueprints.create_agent._shared import build_tool_graph
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory
from langgraph_agent_toolkit.core.tool_selection import ToolSelectionConfig


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("use_async", [False, True])
async def test_provider_search_preserves_eager_tools_and_serializes_deferred_schemas(provider, use_async, monkeypatch):
    """Send real SDK requests without contacting either provider."""
    requests = []
    provider_http = httpx if provider == "openai" else pytest.importorskip("httpx2")

    @tool
    def lookup_record(record_id: str) -> str:
        """Read one synthetic record by its ID."""
        return record_id

    def respond(request):
        requests.append(request)
        if provider == "openai":
            return provider_http.Response(
                200,
                json={
                    "id": "resp_tool_search",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-5.5",
                    "output": [
                        {
                            "id": "msg_complete",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "Complete.", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
                },
            )
        return provider_http.Response(
            200,
            json={
                "id": "msg_tool_search",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "Complete."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        )

    transport = provider_http.MockTransport(respond)
    with provider_http.Client(transport=transport) as sync_client:
        async with provider_http.AsyncClient(transport=transport) as async_client:
            if provider == "openai":
                pytest.importorskip("langchain_openai")
                model = CompletionModelFactory.create(
                    "openai",
                    "gpt-5.5",
                    configurable_fields=(),
                    config_prefix="",
                    model_parameter_values=(),
                    api_key="synthetic-sdk-test",
                    base_url="https://provider.invalid/v1",
                    http_client=sync_client,
                    http_async_client=async_client,
                    use_responses_api=True,
                    max_retries=0,
                )
            else:
                module = pytest.importorskip("langchain_anthropic.chat_models")
                monkeypatch.setattr(module, "_get_default_httpx_client", lambda **kwargs: sync_client)
                monkeypatch.setattr(module, "_get_default_async_httpx_client", lambda **kwargs: async_client)
                model = CompletionModelFactory.create(
                    "anthropic",
                    "claude-sonnet-4-6",
                    configurable_fields=(),
                    config_prefix="",
                    model_parameter_values=(),
                    api_key="synthetic-sdk-test",
                    base_url="https://provider.invalid",
                    max_retries=0,
                )
            graph = build_tool_graph(
                model,
                [lookup_record],
                tool_selection=ToolSelectionConfig(mode="provider", always_include=["add"]),
            )
            state = {"messages": [{"role": "user", "content": "No record lookup is required."}]}
            result = await graph.ainvoke(state) if use_async else graph.invoke(state)
            assert result["messages"][-1].text == "Complete."

    assert len(requests) == 1
    request = requests[0]
    payload = json.loads(request.content)
    expected_type = "tool_search" if provider == "openai" else "tool_search_tool_bm25_20251119"
    assert len([item for item in payload["tools"] if item.get("type") == expected_type]) == 1
    by_name = {item["name"]: item for item in payload["tools"] if "name" in item}
    assert not by_name["add"].get("defer_loading", False)
    assert by_name["lookup_record"]["defer_loading"] is True
    assert by_name["lookup_record"].get("parameters", by_name["lookup_record"].get("input_schema"))
    assert request.url.path == ("/v1/responses" if provider == "openai" else "/v1/messages")
    assert not (lookup_record.extras or {}).get("defer_loading", False)
