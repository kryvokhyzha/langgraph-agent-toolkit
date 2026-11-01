"""Submit and inspect optional durable jobs without waiting for execution."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Annotated
from uuid import UUID

import psycopg
from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from psycopg_pool import PoolTimeout

from langgraph_agent_toolkit.core.background import BackgroundUnavailable, JobConflict, PostgresJobQueue
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.constants import get_default_agent
from langgraph_agent_toolkit.schema.execution import ExecutionRequest
from langgraph_agent_toolkit.schema.jobs import BackgroundJob
from langgraph_agent_toolkit.service.auth import authenticated_user_id
from langgraph_agent_toolkit.service.utils import get_agent


jobs_router = APIRouter(tags=["jobs"])


def _queue(request: Request) -> PostgresJobQueue:
    queue = getattr(request.app.state, "background_jobs", None)
    if queue is None or not getattr(request.app.state, "ready", False):
        raise HTTPException(503, "Background jobs are unavailable")
    return queue


@contextmanager
def _job_errors() -> Generator[None, None, None]:
    try:
        yield
    except JobConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except (BackgroundUnavailable, psycopg.Error, PoolTimeout) as exc:
        raise HTTPException(503, "Background job storage is unavailable", headers={"Retry-After": "1"}) from exc


@jobs_router.post("/jobs", status_code=202, response_model=BackgroundJob)
@jobs_router.post("/{agent_id}/jobs", status_code=202, response_model=BackgroundJob, operation_id="submit_agent_job")
async def submit_job(
    value: ExecutionRequest,
    request: Request,
    response: Response,
    idempotency_key: Annotated[str, Header(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")],
    agent_id: str | None = None,
) -> BackgroundJob:
    """Persist a job and return its status before the agent runs."""
    agent_id = agent_id or get_default_agent()
    get_agent(request, agent_id)
    owner = authenticated_user_id(request, value.user_id)
    if value.recursion_limit is not None and value.recursion_limit > settings.DEFAULT_RECURSION_LIMIT:
        raise HTTPException(422, "recursion_limit exceeds the service limit")
    timeout = (
        value.timeout_seconds if "timeout_seconds" in value.model_fields_set else settings.BACKGROUND_EXECUTION_TIMEOUT
    )
    if timeout > settings.BACKGROUND_EXECUTION_TIMEOUT:
        raise HTTPException(422, "timeout_seconds exceeds BACKGROUND_EXECUTION_TIMEOUT")
    value = value.model_copy(update={"user_id": owner, "timeout_seconds": timeout})
    with _job_errors():
        job = await _queue(request).submit(agent_id, owner, value, idempotency_key)
    location = request.url_for("get_job", job_id=job.job_id)
    if owner != request.state.principal.user_id:
        location = location.include_query_params(user_id=owner)
    response.headers["Location"] = str(location)
    return job


@jobs_router.get("/jobs/{job_id}", response_model=BackgroundJob)
async def get_job(
    job_id: UUID, request: Request, user_id: Annotated[str | None, Query(min_length=1, max_length=256)] = None
) -> BackgroundJob:
    """Read one job within the authenticated owner's scope."""
    owner = authenticated_user_id(request, user_id)
    with _job_errors():
        job = await _queue(request).get(str(job_id), owner)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@jobs_router.post("/jobs/{job_id}/cancel", response_model=BackgroundJob)
async def cancel_job(
    job_id: UUID, request: Request, user_id: Annotated[str | None, Query(min_length=1, max_length=256)] = None
) -> BackgroundJob:
    """Request cancellation. Completed external actions are not reversed."""
    owner = authenticated_user_id(request, user_id)
    with _job_errors():
        job = await _queue(request).cancel(str(job_id), owner)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job
