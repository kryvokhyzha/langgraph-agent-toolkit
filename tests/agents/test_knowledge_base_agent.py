"""Check knowledge-base queries and sources with the real Bedrock retriever."""

from copy import deepcopy
from importlib import import_module
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


@pytest.fixture
def kb_module():
    return import_module("langgraph_agent_toolkit.agents.blueprints.knowledge_base_agent.agent")


@pytest.fixture
def bedrock_client(monkeypatch, kb_module):
    client = Mock()
    client.retrieve.return_value = {"retrievalResults": []}
    retriever = AmazonKnowledgeBasesRetriever.model_construct(knowledge_base_id="synthetic-kb", client=client)
    monkeypatch.setattr(kb_module, "get_kb_retriever", lambda: retriever)
    return client


@pytest.mark.parametrize(
    ("content", "expected_query"),
    [
        ("  product guide  ", "product guide"),
        (
            [
                {"type": "text", "text": "product"},
                {"type": "image", "base64": "SYNTHETIC_MEDIA", "mime_type": "image/png"},
                {"type": "file", "url": "https://example.com/attachment.pdf", "text": "ignored media text"},
                {"type": "text", "text": "guide"},
            ],
            "product\nguide",
        ),
        (["product", {"type": "text", "text": "guide"}], "product\nguide"),
    ],
)
async def test_retrieval_uses_latest_text_without_changing_messages(kb_module, bedrock_client, content, expected_query):
    messages = [HumanMessage("previous question"), HumanMessage(content=content)]
    original = deepcopy(messages)

    await kb_module.retrieve_documents({"messages": messages}, {})

    assert bedrock_client.retrieve.call_args.kwargs["retrievalQuery"] == {"text": expected_query}
    assert messages == original


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [HumanMessage(" \n ")],
        [HumanMessage([{"type": "image", "base64": "SYNTHETIC_MEDIA", "mime_type": "image/png"}])],
    ],
)
async def test_missing_query_skips_retriever_and_clears_previous_context(monkeypatch, kb_module, messages):
    factory = Mock(side_effect=AssertionError("An empty query must not create a retriever."))
    monkeypatch.setattr(kb_module, "get_kb_retriever", factory)
    state = {"messages": messages, "retrieved_documents": [{"content": "old document"}], "kb_documents": "old document"}

    update = await kb_module.retrieve_documents(state, {})
    state.update(update)
    state.update(await kb_module.prepare_augmented_prompt(state, {}))

    factory.assert_not_called()
    assert state["retrieved_documents"] == []
    assert state["kb_documents"] == ""


@pytest.mark.parametrize(
    ("location", "source_metadata", "expected_source"),
    [
        ({"type": "S3", "s3Location": {"uri": "s3://synthetic-docs/guide.txt"}}, {}, "s3://synthetic-docs/guide.txt"),
        ({"type": "WEB", "webLocation": {"url": "https://example.com/guide"}}, {}, "https://example.com/guide"),
        ({"type": "CUSTOM", "customDocumentLocation": {"id": "guide-id"}}, {}, "guide-id"),
        ({}, {"x-amz-bedrock-kb-source-uri": "s3://synthetic-docs/fallback.txt"}, "s3://synthetic-docs/fallback.txt"),
        ({}, {"source": "custom source"}, "custom source"),
    ],
)
async def test_bedrock_sources_and_nested_metadata_reach_prompt(
    kb_module, bedrock_client, location, source_metadata, expected_source
):
    bedrock_client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"type": "TEXT", "text": "Synthetic product guide."},
                "location": location,
                "metadata": {"id": "custom-id", "title": "Synthetic title", **source_metadata},
                "score": 0.9,
            }
        ]
    }

    retrieved = await kb_module.retrieve_documents({"messages": [HumanMessage("product guide")]}, {})
    prompt = await kb_module.prepare_augmented_prompt(retrieved, {})

    assert retrieved["retrieved_documents"] == [
        {
            "id": "custom-id",
            "source": expected_source,
            "title": "Synthetic title",
            "content": "Synthetic product guide.",
            "relevance_score": 0.9,
        }
    ]
    assert f"Source: {expected_source}\nTitle: Synthetic title" in prompt["kb_documents"]
    assert "Synthetic product guide." in prompt["kb_documents"]


async def test_custom_metadata_keeps_precedence_and_remains_unchanged(monkeypatch, kb_module):
    document = Document(
        page_content="Synthetic content.",
        metadata={
            "id": "existing-id",
            "source": "existing source",
            "title": "Existing title",
            "location": {"type": "WEB", "webLocation": {"url": "https://example.com/guide"}},
            "source_metadata": {"id": "nested-id", "source": "nested source", "title": "Nested title"},
        },
    )
    original = deepcopy(document)
    monkeypatch.setattr(kb_module, "get_kb_retriever", lambda: Mock(ainvoke=AsyncMock(return_value=[document])))

    result = await kb_module.retrieve_documents({"messages": [HumanMessage("product guide")]}, {})

    summary = result["retrieved_documents"][0]
    assert (summary["id"], summary["source"], summary["title"]) == ("existing-id", "existing source", "Existing title")
    assert document == original
