"""Validate tool selection settings without agent runtime dependencies."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolSelectionConfig(BaseModel):
    """Select the middleware and keep tool selection disabled by default.

    `max_tools` limits selected tools. Required tools do not use this limit.
    Parsing fallback does not retry tool execution.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["off", "llm", "provider"] = "off"
    max_tools: int = Field(default=8, ge=1)
    always_include: list[str] = Field(default_factory=list)
    model_config_key: str | None = None
    max_retries: int = Field(default=1, ge=0, le=3)
    on_parsing_failure: Literal["error", "none", "all"] = "error"

    @field_validator("always_include")
    @classmethod
    def validate_tool_names(cls, value: list[str]) -> list[str]:
        """Reject empty, repeated, or padded tool names."""
        if any(not name or name != name.strip() for name in value):
            raise ValueError("Tool names must be nonempty and must not contain surrounding whitespace")
        if len(set(value)) != len(value):
            raise ValueError("always_include must not contain duplicate tool names")
        return value

    @field_validator("model_config_key")
    @classmethod
    def validate_model_key(cls, value: str | None) -> str | None:
        """Reject an empty model configuration key."""
        if value is not None and (not value or value != value.strip()):
            raise ValueError("model_config_key must be nonempty and must not contain surrounding whitespace")
        return value
