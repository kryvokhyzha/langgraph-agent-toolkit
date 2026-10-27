"""Verify media recovery and active-model checks in real Deep Agents graphs."""

import base64
import importlib
from uuid import uuid4

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.exceptions import ModelInvalidRequestError
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field


deepagents = pytest.importorskip("deepagents")
backends = importlib.import_module("deepagents.backends")
blueprint = importlib.import_module("langgraph_agent_toolkit.agents.blueprints.deep_agent.agent")


class RecordingModel(FakeMessagesListChatModel):
    """Record model requests and reject selected requests."""

    seen: list = Field(default_factory=list)
    reject_calls: set[int] = Field(default_factory=set)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        if len(self.seen) in self.reject_calls:
            raise ModelInvalidRequestError("The provider rejected the new file.")
        return super()._generate(messages, *args, **kwargs)

    def get_num_tokens(self, text):
        return len(text.split())

    def get_num_tokens_from_messages(self, messages, tools=None):
        return sum(self.get_num_tokens(str(message.content)) for message in messages)


class SwitchModelMiddleware(AgentMiddleware):
    """Select a different model before the final content check."""

    def __init__(self, model):
        self.model = model

    async def awrap_model_call(self, request, handler):
        return await handler(request.override(model=self.model))


def media_blocks(messages):
    return [
        block
        for message in messages
        if isinstance(message, (HumanMessage, ToolMessage))
        for block in message.content_blocks
        if block["type"] == "image"
    ]


@pytest.mark.parametrize("reject_twice", [False, True])
@pytest.mark.parametrize("offload", [False, True])
async def test_new_binary_read_retries_model_once_without_repeating_external_write(tmp_path, reject_twice, offload):
    """Retry a rejected file once and keep completed tool effects unchanged."""
    writes = []

    @tool
    async def save_record(value: str) -> str:
        """Save one external record."""
        writes.append(value)
        return "Saved."

    backend = backends.FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    uploaded = await backend.aupload_files([("/image.png", b"Synthetic image bytes.")])
    assert uploaded[0].error is None
    model = RecordingModel(
        profile={"image_inputs": True, "image_tool_message": True},
        reject_calls={2, 3} if reject_twice else {2},
        responses=[
            AIMessage(
                "",
                tool_calls=[
                    {"name": "save_record", "args": {"value": "saved once"}, "id": uuid4().hex},
                    {"name": "read_file", "args": {"file_path": "/image.png"}, "id": uuid4().hex},
                ],
            ),
            AIMessage("The record is saved."),
        ],
    )
    graph = blueprint.build_graph(
        extra_tools=[save_record], model=model, backend=backend, offload_binary_content=offload
    )
    if reject_twice:
        with pytest.raises(ModelInvalidRequestError, match="provider rejected"):
            await graph.ainvoke({"messages": [HumanMessage("Save the record and read the image.")]})
    else:
        result = await graph.ainvoke({"messages": [HumanMessage("Save the record and read the image.")]})
        assert result["messages"][-1].content == "The record is saved."
        results = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        assert [message.name for message in results].count("save_record") == 1
        assert [message.name for message in results].count("read_file") == 1

    assert writes == ["saved once"]
    assert len(model.seen) == 3
    assert len(media_blocks(model.seen[1])) == 1
    assert media_blocks(model.seen[2]) == []
    assert any(
        isinstance(message, ToolMessage) and message.name == "read_file" and "Unsupported content" in message.content
        for message in model.seen[2]
    )


@pytest.mark.parametrize("selected_accepts_images", [False, True])
async def test_model_switch_checks_selected_profile_and_keeps_checkpoint_media(selected_accepts_images):
    """Filter the model view with the selected profile and preserve stored media."""
    initial = RecordingModel(
        profile={"image_inputs": not selected_accepts_images}, responses=[AIMessage("Unused model.")]
    )
    selected = RecordingModel(
        profile={"image_inputs": selected_accepts_images}, responses=[AIMessage("The selected model answered.")]
    )
    content = [
        {"type": "text", "text": "Read this image."},
        {
            "type": "image",
            "base64": base64.b64encode(b"Synthetic image bytes.").decode("ascii"),
            "mime_type": "image/png",
        },
    ]
    graph = deepagents.create_deep_agent(
        model=initial,
        middleware=[SwitchModelMiddleware(selected)],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "selected-model-media"}}
    result = await graph.ainvoke({"messages": [HumanMessage(content)]}, config, durability="sync")

    assert result["messages"][-1].content == "The selected model answered."
    assert initial.seen == []
    assert len(selected.seen) == 1
    assert len(media_blocks(selected.seen[0])) == int(selected_accepts_images)
    if not selected_accepts_images:
        assert "does not support image content" in str(selected.seen[0])
    snapshot = await graph.aget_state(config)
    original = next(message for message in snapshot.values["messages"] if isinstance(message, HumanMessage))
    assert original.content == content
