import asyncio
from contextlib import aclosing

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from langchain_core.messages import AIMessage
from langchain_core.messages import ChatMessage as LangchainChatMessage
from langchain_core.runnables import RunnableConfig

from langgraph_agent_toolkit import __version__
from langgraph_agent_toolkit.agents.agent_executor import add_graph_history, get_graph_history
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.constants import get_default_agent
from langgraph_agent_toolkit.helper.utils import langchain_to_chat_message
from langgraph_agent_toolkit.schema import (
    AddMessagesInput,
    AddMessagesResponse,
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    ClearHistoryInput,
    ClearHistoryResponse,
    DatabaseHealthResponse,
    ErrorResponse,
    Feedback,
    FeedbackResponse,
    HealthCheck,
    LivenessResponse,
    ReadinessResponse,
    ServiceMetadata,
    StartupResponse,
    StreamChunk,
    StreamInput,
    UserInput,
)
from langgraph_agent_toolkit.service.auth import conversation_identity, execution_input
from langgraph_agent_toolkit.service.feedback import authorize_feedback, sign_feedback_message
from langgraph_agent_toolkit.service.responses import ClosingStreamingResponse as StreamingResponse
from langgraph_agent_toolkit.service.utils import (
    _sse_response_example,
    get_agent,
    get_agent_executor,
    get_all_agent_info,
    jsonl_message_generator,
    message_generator,
)


# Create routers for private and public endpoints.
private_router = APIRouter()
public_router = APIRouter()

# Authenticated endpoints share these OpenAPI error responses.
COMMON_ERROR_RESPONSES = {
    status.HTTP_403_FORBIDDEN: {"model": ErrorResponse, "description": "User identity does not match token"},
    status.HTTP_409_CONFLICT: {"model": ErrorResponse, "description": "Conversation queue is full or wait expired"},
    status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse, "description": "Request body exceeds byte limit"},
    status.HTTP_504_GATEWAY_TIMEOUT: {"model": ErrorResponse, "description": "Request time limit expired"},
    status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse, "description": "Missing or invalid bearer token"},
    status.HTTP_404_NOT_FOUND: {"model": ErrorResponse, "description": "Agent or resource not found"},
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse, "description": "Request validation error"},
    status.HTTP_429_TOO_MANY_REQUESTS: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse, "description": "Internal server error"},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "Service unavailable"},
}


@private_router.get(
    "/info",
    status_code=status.HTTP_200_OK,
    tags=["info"],
    summary="Get information about available agents",
    description="Returns metadata about the service including available agents and default agent.",
)
async def info(request: Request) -> ServiceMetadata:
    return ServiceMetadata(
        agents=get_all_agent_info(request),
        default_agent=get_default_agent(),
    )


@private_router.post(
    "/{agent_id}/invoke",
    operation_id="invoke_with_agent_id",
    status_code=status.HTTP_200_OK,
    tags=["agent"],
    summary="Invoke a specific agent to get a response",
    description="Invoke a specified agent with user input to retrieve a final response.",
)
@private_router.post(
    "/invoke",
    status_code=status.HTTP_200_OK,
    tags=["agent"],
    summary="Invoke an agent to get a response",
    description="Invoke an agent with user input to retrieve a final response.",
)
async def invoke(user_input: UserInput, agent_id: str = None, request: Request = None) -> ChatMessage:
    """Invoke an agent and return its final response.

    Use the default agent when `agent_id` is not provided.
    Use `thread_id` to persist a multi-turn conversation.
    Messages include the `run_id` keyword argument for feedback recording.
    """
    executor = get_agent_executor(request)

    if agent_id is None:
        agent_id = get_default_agent()

    get_agent(request, agent_id)
    public_id, user_input = execution_input(request, agent_id, user_input)
    response = await executor.invoke(
        agent_id=agent_id,
        input=user_input.input,
        thread_id=user_input.thread_id,
        user_id=user_input.user_id,
        model_name=user_input.model_name,
        model_provider=user_input.model_provider,
        model_config_key=user_input.model_config_key,
        agent_config=user_input.agent_config,
        recursion_limit=user_input.recursion_limit,
    )
    response.thread_id = public_id
    return sign_feedback_message(request, agent_id, response)


@private_router.post(
    "/{agent_id}/stream",
    operation_id="stream_with_agent_id",
    status_code=status.HTTP_200_OK,
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    tags=["agent"],
    summary="Stream a specific agent's response",
    description="Stream a specified agent's response to a user input, including intermediate messages and tokens.",
)
@private_router.post(
    "/stream",
    status_code=status.HTTP_200_OK,
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    tags=["agent"],
    summary="Stream an agent's response",
    description="Stream an agent's response to a user input, including intermediate messages and tokens.",
)
async def stream(user_input: StreamInput, agent_id: str | None = None, request: Request = None) -> StreamingResponse:
    """Stream agent responses, including messages and tokens.

    Use the default agent when `agent_id` is not provided.
    Use `thread_id` to persist a multi-turn conversation.
    Messages include the `run_id` keyword argument for feedback recording.
    Set `stream_tokens=false` to exclude token output.
    """
    if agent_id is None:
        agent_id = get_default_agent()

    get_agent(request, agent_id)
    public_id, user_input = execution_input(request, agent_id, user_input)
    return StreamingResponse(
        message_generator(user_input, request, agent_id, public_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@private_router.post(
    "/{agent_id}/stream/jsonl",
    operation_id="stream_jsonl_with_agent_id",
    status_code=status.HTTP_200_OK,
    tags=["agent"],
    summary="Stream a specific agent's response as JSON Lines (NDJSON)",
    description=(
        "JSON Lines (application/jsonl) alternative to the SSE `/{agent_id}/stream` endpoint: one "
        "typed StreamChunk per line (type=token|message|error). Useful for non-browser clients."
    ),
    responses={
        status.HTTP_200_OK: {
            "model": StreamChunk,
            "description": "A JSON Lines (application/jsonl) stream — one StreamChunk object per line.",
        }
    },
)
@private_router.post(
    "/stream/jsonl",
    status_code=status.HTTP_200_OK,
    tags=["agent"],
    summary="Stream an agent's response as JSON Lines (NDJSON)",
    description=(
        "JSON Lines (application/jsonl) alternative to the SSE `/stream` endpoint: one typed "
        "StreamChunk per line (type=token|message|error). The SSE `/stream` endpoint is unchanged."
    ),
    responses={
        status.HTTP_200_OK: {
            "model": StreamChunk,
            "description": "A JSON Lines (application/jsonl) stream — one StreamChunk object per line.",
        }
    },
)
async def stream_jsonl(
    user_input: StreamInput, agent_id: str | None = None, request: Request = None
) -> StreamingResponse:
    """Stream agent responses as JSON Lines with one `StreamChunk` per line.

    This typed, OpenAPI-documented endpoint is an alternative to SSE `/stream`.
    It emits token chunks when `stream_tokens` is true and full message chunks.
    A failure emits a final error chunk.
    The response body end terminates the stream.
    """
    if agent_id is None:
        agent_id = get_default_agent()

    get_agent(request, agent_id)
    public_id, user_input = execution_input(request, agent_id, user_input)

    async def encoded():
        async with aclosing(jsonl_message_generator(user_input, request, agent_id, public_id)) as chunks:
            async for chunk in chunks:
                yield chunk.model_dump_json() + "\n"

    return StreamingResponse(encoded(), media_type="application/jsonl")


@private_router.post(
    "/feedback",
    status_code=status.HTTP_201_CREATED,
    tags=["feedback"],
    summary="Record feedback",
    description="Record feedback for a run to the configured observability platform.",
)
@private_router.post(
    "/{agent_id}/feedback",
    operation_id="feedback_with_agent_id",
    status_code=status.HTTP_201_CREATED,
    tags=["feedback"],
    summary="Record feedback for a specific agent",
    description="Record feedback for a run to the configured observability platform for a specific agent.",
)
async def feedback(feedback: Feedback, agent_id: str | None = None, request: Request = None) -> FeedbackResponse:
    """Record feedback for a run on the configured observability platform.

    The agent configuration selects the observability platform.
    """
    try:
        if agent_id is None:
            agent_id = get_default_agent()

        agent = get_agent(request, agent_id)
        owner = authorize_feedback(request, agent_id, feedback)
        await request.app.state.blocking_executor.run(
            agent.observability.record_feedback,
            run_id=feedback.run_id,
            key=feedback.key,
            score=feedback.score,
            user_id=owner,
            **feedback.kwargs,
        )

        return FeedbackResponse(
            run_id=feedback.run_id,
            message=f"Feedback '{feedback.key}' recorded successfully for run {feedback.run_id}.",
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        # Let the global exception handler process other exceptions.
        raise


@private_router.get(
    "/history",
    status_code=status.HTTP_200_OK,
    tags=["history"],
    summary="Get chat history",
    description="Get short-term chat history for one thread. user_id identifies its owner.",
)
@private_router.get(
    "/{agent_id}/history",
    operation_id="history_with_agent_id",
    status_code=status.HTTP_200_OK,
    tags=["history"],
    summary="Get chat history for a specific agent",
    description="Get short-term chat history for one agent and thread. user_id identifies its owner.",
)
async def history(
    input: ChatHistoryInput = Depends(),
    agent_id: str | None = None,
    request: Request = None,
) -> ChatHistory:
    """Get short-term chat history for one thread."""
    agent_id = agent_id or get_default_agent()
    _, owner, key = conversation_identity(request, agent_id, input.thread_id, input.user_id)
    executor = get_agent_executor(request)
    agent = get_agent(request, agent_id)
    async with executor.concurrency.lock(key):
        messages = await get_graph_history(
            agent.graph, RunnableConfig(configurable={"thread_id": key, "user_id": owner})
        )
        limit = min(input.limit, settings.HISTORY_MAX_PAGE_SIZE)
        page = messages[input.offset : input.offset + limit]
        next_offset = input.offset + len(page)
        return ChatHistory(
            messages=[langchain_to_chat_message(m) for m in page],
            total=len(messages),
            next_offset=next_offset if next_offset < len(messages) else None,
        )


@private_router.delete(
    "/history/clear",
    status_code=status.HTTP_200_OK,
    tags=["history"],
    summary="Clear chat history",
    description="Delete checkpoints for one thread. Keep long-term memory.",
)
@private_router.delete(
    "/{agent_id}/history/clear",
    operation_id="clear_history_with_agent_id",
    status_code=status.HTTP_200_OK,
    tags=["history"],
    summary="Clear chat history for a specific agent",
    description="Delete checkpoints for one agent and thread. Keep long-term memory.",
)
async def clear_history(
    input: ClearHistoryInput,
    agent_id: str | None = None,
    request: Request = None,
) -> ClearHistoryResponse:
    """Delete one thread's checkpoints. Keep long-term memory."""
    agent_id = agent_id or get_default_agent()
    public_id, _, key = conversation_identity(request, agent_id, input.thread_id, input.user_id)
    executor = get_agent_executor(request)
    agent = get_agent(request, agent_id)
    async with executor.concurrency.lock(key):
        if not agent.graph.checkpointer:
            raise HTTPException(503, "This agent has no checkpointer")
        await agent.graph.checkpointer.adelete_thread(key)
    return ClearHistoryResponse(
        status="success",
        thread_id=public_id,
        user_id=input.user_id,
        message="Deleted all checkpoints for this conversation.",
    )


@private_router.post(
    "/history/add_messages",
    status_code=status.HTTP_201_CREATED,
    tags=["history"],
    summary="Add messages to chat history",
    description="Add messages to one short-term conversation. user_id identifies its owner.",
)
@private_router.post(
    "/{agent_id}/history/add_messages",
    operation_id="add_messages_with_agent_id",
    status_code=status.HTTP_201_CREATED,
    tags=["history"],
    summary="Add messages to chat history for a specific agent",
    description="Add messages to one agent and thread. user_id identifies its owner.",
)
async def add_messages(
    input: AddMessagesInput,
    agent_id: str | None = None,
    request: Request = None,
) -> AddMessagesResponse:
    """Add messages to one thread's short-term history."""
    agent_id = agent_id or get_default_agent()
    public_id, owner, key = conversation_identity(request, agent_id, input.thread_id, input.user_id)
    executor = get_agent_executor(request)
    agent = get_agent(request, agent_id)
    async with executor.concurrency.lock(key):
        messages = []
        for message in input.messages:
            if message.type == "custom":
                messages.append(LangchainChatMessage(role="custom", content=[message.custom_data]))
            elif message.type == "ai":
                messages.append(
                    AIMessage(
                        content=message.content,
                        tool_calls=message.tool_calls,
                        usage_metadata=message.usage_metadata,
                        response_metadata=message.response_metadata,
                    )
                )
            else:
                messages.append(message.model_dump(exclude_none=True, exclude={"custom_data"}))
        await add_graph_history(
            agent.graph,
            config=RunnableConfig(configurable={"thread_id": key, "user_id": owner}),
            messages=messages,
        )
    return AddMessagesResponse(
        status="success",
        thread_id=public_id,
        user_id=input.user_id,
        message=f"Added {len(input.messages)} messages to chat history.",
    )


@public_router.get(
    "/",
    tags=["public"],
    summary="API Home",
    description="Redirects to the API documentation.",
)
async def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@public_router.get(
    "/health",
    tags=["healthcheck"],
    summary="Health Check",
    description="Perform a health check to verify the service is running correctly.",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
async def health_check() -> HealthCheck:
    """Health check endpoint."""
    return HealthCheck(
        content="healthy",
        version=__version__,
    )


@public_router.get(
    "/health/live",
    tags=["healthcheck"],
    summary="Liveness Probe",
    description="Check whether the worker responds and request cleanup is progressing. "
    "Returns 503 for stalled cleanup. Configure the supervisor to restart on failure.",
    response_description="Return HTTP Status Code 200 (OK) if alive",
    status_code=status.HTTP_200_OK,
    response_model=LivenessResponse,
    responses={503: {"description": "Request cleanup is stalled", "model": LivenessResponse}},
)
async def liveness_probe(request: Request):
    """Return the Kubernetes liveness probe response.

    This probe confirms that the process responds.
    Kubernetes restarts the process when the probe fails.
    """
    if getattr(request.app.state, "stalled_request_cleanups", 0):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503, content=LivenessResponse(status="unhealthy", version=__version__).model_dump()
        )
    return LivenessResponse(
        status="alive",
        version=__version__,
    )


@public_router.get(
    "/health/ready",
    tags=["healthcheck"],
    summary="Readiness Probe",
    description="Kubernetes readiness probe - checks if service is ready to accept traffic. "
    "Returns 200 only after all agents are initialized. Traffic won't be routed until ready.",
    response_description="Return HTTP Status Code 200 (OK) if ready, 503 if not ready",
    status_code=status.HTTP_200_OK,
    response_model=ReadinessResponse,
    responses={
        503: {
            "description": "Service not ready",
            "model": ReadinessResponse,
        }
    },
)
async def readiness_probe(request: Request):
    """Return the Kubernetes readiness probe response.

    This probe checks that the service initializes agents successfully.
    Kubernetes routes traffic to the pod only after this returns 200.
    """
    from fastapi.responses import JSONResponse

    is_ready = getattr(request.app.state, "ready", False) and not getattr(
        request.app.state, "stalled_request_cleanups", 0
    )
    initialized_agents = getattr(request.app.state, "initialized_agents", [])

    if is_ready:
        try:
            async with asyncio.timeout(2):
                for name in ("db_pool", "lock_pool"):
                    pool = getattr(request.app.state, name, None)
                    if pool is not None:
                        async with pool.connection(timeout=1) as connection:
                            await connection.execute("SELECT 1")
                sqlite = getattr(request.app.state, "sqlite_connection", None)
                if sqlite is not None:
                    async with sqlite.execute("SELECT 1") as cursor:
                        await cursor.fetchone()
        except Exception:
            is_ready = False

    if is_ready and initialized_agents:
        return ReadinessResponse(
            status="ready",
            version=__version__,
            initialized_agents=initialized_agents,
            message=f"All {len(initialized_agents)} agents initialized successfully",
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ReadinessResponse(
                status="not_ready",
                version=__version__,
                initialized_agents=initialized_agents,
                message="Agents not yet initialized" if not initialized_agents else "Service not ready",
            ).model_dump(),
        )


@public_router.get(
    "/health/startup",
    tags=["healthcheck"],
    summary="Startup Probe",
    description="Kubernetes startup probe - checks if application has finished starting. "
    "Use with initialDelaySeconds to give agents time to initialize before checking readiness.",
    response_description="Return HTTP Status Code 200 (OK) if started, 503 if still starting",
    status_code=status.HTTP_200_OK,
    response_model=StartupResponse,
    responses={
        503: {
            "description": "Still starting up",
            "model": StartupResponse,
        }
    },
)
async def startup_probe(request: Request):
    """Return the Kubernetes startup probe response.

    This probe checks whether the application finished initialization.
    It supports slow-starting containers.
    Kubernetes checks it before liveness and readiness probes.
    """
    from fastapi.responses import JSONResponse

    startup_complete = getattr(request.app.state, "startup_complete", False)

    if startup_complete:
        return StartupResponse(
            status="started",
            version=__version__,
            message="Application startup complete",
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=StartupResponse(
                status="starting",
                version=__version__,
                message="Application still initializing",
            ).model_dump(),
        )


@public_router.get(
    "/health/db",
    tags=["healthcheck"],
    summary="Database Pool Health",
    description="Get database connection pool statistics for monitoring and debugging.",
    response_description="Return database pool statistics",
    status_code=status.HTTP_200_OK,
    response_model=DatabaseHealthResponse,
)
async def db_health_check(request: Request) -> DatabaseHealthResponse:
    """Database pool health check endpoint."""
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        return DatabaseHealthResponse(
            status="no_pool",
            message="No database pool configured or memory backend not using PostgreSQL",
        )

    if not hasattr(pool, "get_stats"):
        # The SQLite saver exposes a raw connection, not a pool with statistics.
        return DatabaseHealthResponse(
            status="no_pool",
            message="Connection pool statistics are only available for the PostgreSQL backend",
        )

    try:
        stats = pool.get_stats()
        return DatabaseHealthResponse(
            status="healthy" if stats.get("pool_available", 0) > 0 else "exhausted",
            pool_size=stats.get("pool_size", 0),
            pool_available=stats.get("pool_available", 0),
            requests_waiting=stats.get("requests_waiting", 0),
            connections_num=stats.get("connections_num", 0),
        )
    except Exception:
        return DatabaseHealthResponse(
            status="error",
            message="Could not read database pool statistics",
        )
