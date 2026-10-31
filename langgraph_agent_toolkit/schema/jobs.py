"""Public job records without queue backend imports."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from langgraph_agent_toolkit.schema.execution import ExecutionResult


class BackgroundJob(BaseModel):
    """A durable operation and its latest execution result."""

    job_id: str
    agent_id: str
    thread_id: str
    status: Literal["queued", "running", "cancelling", "cancelled", "completed", "interrupted", "failed"]
    created_at: datetime
    updated_at: datetime
    attempts: int = 0
    result: ExecutionResult | None = None
    error: str | None = None
