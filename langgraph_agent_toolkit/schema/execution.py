"""Transport models for explicit graph execution outside the chat contract."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


def validate_agent_config(agent_config: dict[str, Any]) -> None:
    """Keep identity, checkpoint, and operation fields under runtime control."""
    reserved = {
        "thread_id",
        "user_id",
        "checkpoint_id",
        "checkpoint_ns",
        "lat_user_id",
        "lat_agent_id",
        "lat_thread_id",
        "lat_operation_id",
        "lat_operation_fingerprint",
        "lat_execution_run_id",
        "lat_execution_triggers",
    }
    if reserved.intersection(agent_config) or any(key.startswith("__") for key in agent_config):
        raise ValueError("agent_config cannot override identity or checkpoint fields")


class ExecutionRequest(BaseModel):
    """Submit new input, answer interrupts, or recover unfinished execution."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    operation: Literal["input", "resume", "recover"] = "input"
    thread_id: str = Field(min_length=1, max_length=256)
    user_id: str | None = Field(default=None, min_length=1, max_length=256)
    input: dict[str, JsonValue] = Field(
        default_factory=dict, description="Native graph input. Chat graphs usually accept a messages list."
    )
    resume: dict[str, JsonValue] | None = None
    expected_checkpoint_id: str | None = Field(default=None, min_length=1, max_length=256)
    model_name: str | None = None
    model_provider: str | None = None
    model_config_key: str | None = None
    agent_config: dict[str, JsonValue] = Field(default_factory=dict)
    recursion_limit: int | None = Field(default=None, ge=1)
    timeout_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)

    @field_validator("agent_config")
    @classmethod
    def reject_reserved_config(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        validate_agent_config(value)
        return value

    @field_validator("thread_id", "user_id", "expected_checkpoint_id")
    @classmethod
    def reject_nul(cls, value: str | None) -> str | None:
        if value is not None and "\x00" in value:
            raise ValueError("Execution identifiers must not contain NUL characters")
        return value

    @model_validator(mode="after")
    def validate_operation(self):
        if self.operation in {"resume", "recover"} and self.expected_checkpoint_id is None:
            raise ValueError("resume and recover require expected_checkpoint_id")
        if self.operation == "resume":
            if not self.resume:
                raise ValueError("resume requires answers keyed by interrupt ID")
        elif self.resume is not None:
            raise ValueError("Only the resume operation accepts interrupt answers")
        if self.operation != "input" and self.input:
            raise ValueError("Only the input operation accepts new graph input")
        return self


class ExecutionResult(BaseModel):
    """Return graph output and recovery references without a chat envelope."""

    status: Literal["completed", "interrupted"]
    output: JsonValue = None
    run_id: str
    thread_id: str
    checkpoint_id: str | None = None
    interrupts: list[dict[str, Any]] = Field(default_factory=list)


class ExecutionSnapshot(BaseModel):
    """Describe saved execution state without starting or resuming the graph."""

    status: Literal["empty", "completed", "interrupted", "unfinished"]
    output: JsonValue = None
    thread_id: str
    checkpoint_id: str | None = None
    operation_id: str | None = None
    run_id: str | None = None
    interrupts: list[dict[str, Any]] = Field(default_factory=list)
