from typing import Any, Dict, List, Literal, NotRequired

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import TypedDict

from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.constants import (
    DEFAULT_MODEL_PARAMETER_VALUES,
    get_default_agent,
)


class AgentInfo(BaseModel):
    """Information about an available agent."""

    key: str = Field(
        description="Agent key.",
        examples=["langgraph-supervisor-agent"],
    )
    description: str = Field(
        description="Description of the agent.",
        examples=["A research assistant."],
    )


class ServiceMetadata(BaseModel):
    """Service metadata, including available agents and models."""

    agents: list[AgentInfo] = Field(
        description="List of available agents.",
    )
    default_agent: str = Field(
        description="Default agent used when none is specified.",
        examples=[get_default_agent()],
    )


class UserComplexInput(BaseModel):
    """User input for an agent with dynamic fields."""

    message: str | list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "User input to the agent: either plain text, or a list of LangChain content blocks for "
            "multimodal input. Each block is {'type': 'text'|'image'|'file'|'audio'|'video', ...} with "
            "'text', a 'url', or 'base64'+'mime_type'. LangChain translates blocks to the provider's "
            "native format, so the chosen model must support the modality."
        ),
        examples=[
            "What is the weather in Tokyo?",
            [
                {"type": "text", "text": "Describe this image."},
                {"type": "image", "url": "https://example.com/image.jpg"},
            ],
            [
                {"type": "text", "text": "Summarize this document."},
                {"type": "file", "base64": "<base64-bytes>", "mime_type": "application/pdf"},
            ],
        ],
    )

    model_config = {
        "extra": "allow"  # allow unknown fields
    }

    @field_validator("message")
    @classmethod
    def _validate_content_blocks(cls, value: "str | list[dict[str, Any]] | None"):
        """Validate basic multimodal content block requirements.

        LangChain does detailed validation later.
        """
        if not isinstance(value, list):
            return value
        allowed = {"text", "image", "file", "audio", "video"}
        # Accept each recognized content source.
        # LangChain validates alternative forms, such as `file_id`, `id`, and `source_type`.
        # Reject only blocks without a content reference, such as {"type": "image"}.
        content_keys = {"url", "base64", "data", "file_id", "id", "source_type", "source", "path"}
        media_count = 0
        for i, block in enumerate(value):
            if not isinstance(block, dict) or "type" not in block:
                raise ValueError(f"content block {i} must be a dict with a 'type' field")
            btype = block["type"]
            if btype not in allowed:
                raise ValueError(f"content block {i} has unsupported type {btype!r}; expected one of {sorted(allowed)}")
            if btype == "text":
                if not isinstance(block.get("text"), str):
                    raise ValueError(f"content block {i} of type 'text' must include a string 'text' field")
            else:
                media_count += 1
                if block.get("base64") and not isinstance(block.get("mime_type"), str):
                    raise ValueError(
                        f"content block {i} of type {btype!r} must include a string 'mime_type' when using 'base64'"
                    )
                if not any(block.get(k) for k in content_keys):
                    raise ValueError(
                        f"content block {i} of type {btype!r} must include a content source "
                        "(e.g. a 'url', or 'base64' + 'mime_type')"
                    )
        max_attachments = settings.MULTIMODAL_MAX_ATTACHMENTS
        if max_attachments is not None and media_count > max_attachments:
            raise ValueError(
                f"too many attachments: {media_count} (max {max_attachments}); "
                f"adjust MULTIMODAL_MAX_ATTACHMENTS to change the limit"
            )
        return value


class UserInput(BaseModel):
    """User input for an agent."""

    input: UserComplexInput = Field(
        description="Structured input from the user, including a message and optional dynamic fields.",
        examples=[
            {
                "message": "What is the weather in Tokyo?",
            }
        ],
    )
    model_name: str | None = Field(
        title="Model",
        description="LLM Model Name to use for the agent.",
        default=None,
        examples=["gpt-3.5-turbo", "gpt-4o"],
    )
    model_provider: str | None = Field(
        title="Model Provider",
        description="LLM Model Provider to use for the agent.",
        default=None,
        examples=["openai", "anthropic"],
    )
    model_config_key: str | None = Field(
        title="Model Configuration Key",
        description="Key for predefined model configuration in MODEL_CONFIGS.",
        default=None,
        examples=["gpt4o", "gemini"],
    )
    thread_id: str | None = Field(
        description="Thread ID for one conversation and its short-term checkpoint state.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: str | None = Field(
        description="Stable user ID for observability and long-term memory across threads when the agent has a store.",
        default=None,
        examples=["521c0a60-ea75-43fa-a793-a4cf11e013ae"],
    )
    agent_config: dict[str, Any] = Field(
        description="Additional configuration to pass through to the agent",
        default={},
        examples=[
            {
                "checkpointer_params": {"k": 6},
                **DEFAULT_MODEL_PARAMETER_VALUES,
            },
        ],
    )
    recursion_limit: int | None = Field(
        description="Recursion limit for the agent.",
        default=None,
        examples=[settings.DEFAULT_RECURSION_LIMIT],
    )


class StreamInput(UserInput):
    """User input for streaming an agent response."""

    stream_tokens: bool = Field(
        description="Whether to stream LLM tokens to the client.",
        default=True,
    )


class ToolCall(TypedDict):
    """Tool call request."""

    name: str
    """Tool name."""
    args: dict[str, Any]
    """Tool call arguments."""
    id: str | None
    """Tool call identifier."""
    type: NotRequired[Literal["tool_call"]]


class UsageMetadata(TypedDict):
    """Provider token counts in the LangChain usage format."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_token_details: NotRequired[dict[str, int]]
    output_token_details: NotRequired[dict[str, int]]


class ChatMessage(BaseModel):
    """Chat message."""

    type: Literal["human", "ai", "tool", "custom"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool", "custom"],
    )
    content: str | Dict[str, Any] | List[str | Dict[str, Any]] = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: list[ToolCall] = Field(
        description="Tool calls in the message.",
        default=[],
    )
    tool_call_id: str | None = Field(
        description="Tool call that this message is responding to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: str | None = Field(
        description="Run ID of the message.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    feedback_token: str | None = Field(
        default=None,
        description="Server proof for feedback on this run. Send it with feedback from a token user.",
        max_length=128,
    )
    thread_id: str | None = Field(
        default=None, description="Public thread ID. Use this ID to continue the conversation."
    )
    response_metadata: dict[str, Any] = Field(
        description="Response metadata. For example: response headers, logprobs, token counts.",
        default={},
    )
    usage_metadata: UsageMetadata | None = Field(
        default=None, description="Provider token counts. None means that counts are unavailable."
    )
    custom_data: dict[str, Any] = Field(
        description="Custom message data.",
        default={},
    )

    def pretty_repr(self) -> str:
        """Get a readable message representation."""
        base_title = self.type.title() + " Message"
        padded = " " + base_title + " "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        print(self.pretty_repr())  # noqa: T201


class StreamChunk(BaseModel):
    """One JSON Lines (NDJSON) chunk from an agent stream.

    The ``/stream/jsonl`` endpoint emits one `StreamChunk` per line.
    `type="token"` contains an incremental token string.
    `type="message"` contains a complete `ChatMessage`.
    `type="error"` contains an error description string.
    """

    type: Literal["token", "message", "error"] = Field(description="The kind of chunk.")
    content: str | ChatMessage = Field(description="Token text, a full ChatMessage, or an error string.")


class ErrorResponse(BaseModel):
    """Standard error response from service exception handlers."""

    detail: str = Field(description="Human-readable error message.")
    error_code: str | None = Field(default=None, description="Stable machine-readable error code, when present.")


class Feedback(BaseModel):
    """Feedback for the configured observability platform."""

    run_id: str = Field(
        description="Run ID to record feedback for.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    feedback_token: str | None = Field(
        default=None,
        description="Server proof from the response. Required for feedback from a token user.",
        max_length=128,
    )
    key: str = Field(
        description="Feedback key.",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="Feedback score.",
        examples=[0.8],
    )
    user_id: str | None = Field(
        description="User ID to associate with the feedback.",
        default=None,
        examples=["521c0a60-ea75-43fa-a793-a4cf11e013ae"],
    )
    kwargs: dict[str, Any] = Field(
        description="Additional feedback kwargs, passed to LangSmith.",
        default={},
        examples=[{"comment": "In-line human feedback"}],
    )


class FeedbackResponse(BaseModel):
    """Response after feedback is recorded."""

    status: Literal["success"] = "success"
    run_id: str = Field(
        description="Run ID for which feedback was recorded.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    message: str = Field(
        description="Descriptive message about the feedback operation.",
        default="Feedback recorded successfully.",
    )


class MessageInput(BaseModel):
    """Input for a chat history message."""

    type: Literal["human", "ai", "tool", "custom"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool", "custom"],
    )
    content: str | list[str | dict[str, Any]] = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    custom_data: dict[str, Any] | None = Field(default=None, description="Payload for a custom message.")
    tool_call_id: str | None = Field(default=None, description="ID of the tool call for a tool response.")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Tool calls in an AI message.")
    usage_metadata: UsageMetadata | None = Field(default=None, description="Provider token counts for an AI message.")
    response_metadata: dict[str, Any] = Field(default_factory=dict, description="Provider response metadata.")

    @model_validator(mode="after")
    def validate_tool_message(self) -> "MessageInput":
        """Require the call ID for tool responses."""
        if self.type == "custom" and self.custom_data is None:
            raise ValueError("A custom message must include custom_data.")
        if self.type == "tool" and not self.tool_call_id:
            raise ValueError("A tool message must include tool_call_id.")
        if self.tool_calls and self.type != "ai":
            raise ValueError("Only AI messages can include tool_calls.")
        if self.usage_metadata is not None and self.type != "ai":
            raise ValueError("Only AI messages can include usage_metadata.")
        return self


class AddMessagesInput(BaseModel):
    """Input for adding chat history messages."""

    thread_id: str | None = Field(
        description="Thread ID for one conversation and its short-term checkpoint state.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: str | None = Field(
        description="Owner of this thread. This operation does not read or change long-term memory.",
        default=None,
        examples=["521c0a60-ea75-43fa-a793-a4cf11e013ae"],
    )
    messages: list[MessageInput] = Field(
        description="List of messages to add to the chat history.",
        examples=[
            [
                {
                    "type": "human",
                    "content": "Hello, how are you?",
                },
                {
                    "type": "ai",
                    "content": "I'm doing well, thank you! How can I assist you today?",
                },
            ]
        ],
    )


class AddMessagesResponse(BaseModel):
    """Response after chat history messages are added."""

    status: Literal["success"] = "success"
    thread_id: str | None = Field(
        description="Thread ID for which the message was added.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: str | None = Field(
        description="User ID associated with the message.",
        default=None,
        examples=["521c0a60-ea75-43fa-a793-a4cf11e013ae"],
    )
    message: str = Field(
        description="Descriptive message about the operation.",
        default="Messages added successfully.",
    )


class ClearHistoryInput(BaseModel):
    """Input for clearing one thread's checkpoints without changing long-term memory."""

    thread_id: str | None = Field(
        description="Thread ID for one conversation and its short-term checkpoint state.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: str | None = Field(
        description="Owner of this thread. This operation does not read or change long-term memory.",
        default=None,
        examples=["521c0a60-ea75-43fa-a793-a4cf11e013ae"],
    )


class ClearHistoryResponse(BaseModel):
    """Response after chat history messages are cleared."""

    status: Literal["success"] = "success"
    thread_id: str | None = Field(
        description="Thread ID for which the messages were cleared.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: str | None = Field(
        description="User ID associated with the operation.",
        default=None,
        examples=["521c0a60-ea75-43fa-a793-a4cf11e013ae"],
    )
    message: str = Field(
        description="Descriptive message about the operation.",
        default="Messages cleared successfully.",
    )


class ChatHistoryInput(BaseModel):
    """Input for retrieving chat history."""

    thread_id: str | None = Field(
        description="Thread ID for one conversation and its short-term checkpoint state.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: str | None = Field(
        description="Owner of this thread. This operation does not read or change long-term memory.",
        default=None,
        examples=["521c0a60-ea75-43fa-a793-a4cf11e013ae"],
    )
    offset: int = Field(default=0, ge=0, description="Number of messages to skip.")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of messages to return.")


class ChatHistory(BaseModel):
    messages: list[ChatMessage]
    next_offset: int | None = None
    total: int | None = None


class HealthCheck(BaseModel):
    """Response model for a health check."""

    content: str = Field(
        ...,
        description="Health status of the service.",
        examples=["healthy"],
    )
    version: str = Field(
        ...,
        description="Version of the service.",
        examples=["1.0.0"],
    )


class LivenessResponse(BaseModel):
    """Response model for a liveness probe."""

    status: Literal["alive", "unhealthy"] = Field(
        description="Liveness status of the service.",
        examples=["alive"],
    )
    version: str = Field(
        description="Version of the service.",
        examples=["1.0.0"],
    )


class ReadinessResponse(BaseModel):
    """Response model for a readiness probe."""

    status: Literal["ready", "not_ready"] = Field(
        description="Readiness status of the service.",
        examples=["ready"],
    )
    version: str = Field(
        description="Version of the service.",
        examples=["1.0.0"],
    )
    initialized_agents: List[str] = Field(
        default=[],
        description="List of successfully initialized agent IDs.",
        examples=[["react_agent", "chatbot_agent"]],
    )
    message: str = Field(
        default="",
        description="Additional information about readiness status.",
        examples=["All agents initialized successfully"],
    )


class StartupResponse(BaseModel):
    """Response model for a startup probe."""

    status: Literal["started", "starting"] = Field(
        description="Startup status of the service.",
        examples=["started"],
    )
    version: str = Field(
        description="Version of the service.",
        examples=["1.0.0"],
    )
    message: str = Field(
        default="",
        description="Additional information about startup status.",
        examples=["Application startup complete"],
    )


class DatabaseHealthResponse(BaseModel):
    """Response model for a database health check."""

    status: Literal["healthy", "exhausted", "no_pool", "error"] = Field(
        description="Database connection pool status.",
        examples=["healthy"],
    )
    message: str | None = Field(
        default=None,
        description="Additional information about the database status.",
    )
    pool_size: int | None = Field(
        default=None,
        description="Total size of the connection pool.",
    )
    pool_available: int | None = Field(
        default=None,
        description="Number of available connections in the pool.",
    )
    requests_waiting: int | None = Field(
        default=None,
        description="Number of requests waiting for a connection.",
    )
    connections_num: int | None = Field(
        default=None,
        description="Current number of connections.",
    )
