from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse, StreamingResponse
from langchain_core.messages import AnyMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from langgraph_agent_toolkit import __version__
from langgraph_agent_toolkit.agents.agent import Agent
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
    Feedback,
    FeedbackResponse,
    HealthCheck,
    LivenessResponse,
    ReadinessResponse,
    ServiceMetadata,
    StartupResponse,
    StreamInput,
    UserInput,
)
from langgraph_agent_toolkit.service.utils import (
    _sse_response_example,
    _validate_thread_or_user_id,
    get_agent,
    get_agent_executor,
    get_all_agent_info,
    message_generator,
)


# Create separate routers for private and public endpoints
private_router = APIRouter()
public_router = APIRouter(tags=["public"])


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
    """Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    executor = get_agent_executor(request)

    if agent_id is None:
        agent_id = get_default_agent()

    try:
        return await executor.invoke(
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
    except Exception:
        # Let the global exception handler deal with all exceptions
        raise


@private_router.post(
    "/{agent_id}/stream",
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
    """Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    if agent_id is None:
        agent_id = get_default_agent()

    return StreamingResponse(
        message_generator(user_input, request, agent_id),
        media_type="text/event-stream",
    )


@private_router.post(
    "/feedback",
    status_code=status.HTTP_201_CREATED,
    tags=["feedback"],
    summary="Record feedback",
    description="Record feedback for a run to the configured observability platform.",
)
@private_router.post(
    "/{agent_id}/feedback",
    status_code=status.HTTP_201_CREATED,
    tags=["feedback"],
    summary="Record feedback for a specific agent",
    description="Record feedback for a run to the configured observability platform for a specific agent.",
)
async def feedback(feedback: Feedback, agent_id: str | None = None, request: Request = None) -> FeedbackResponse:
    """Record feedback for a run to the configured observability platform.

    This routes the feedback to the appropriate platform based on the agent's configuration.
    """
    try:
        if agent_id is None:
            agent_id = get_default_agent()

        agent = get_agent(request, agent_id)
        agent.observability.record_feedback(
            run_id=feedback.run_id,
            key=feedback.key,
            score=feedback.score,
            user_id=feedback.user_id,
            **feedback.kwargs,
        )

        return FeedbackResponse(
            run_id=feedback.run_id,
            message=f"Feedback '{feedback.key}' recorded successfully for run {feedback.run_id}.",
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        # Let the global exception handler deal with all other exceptions
        raise


@private_router.get(
    "/history",
    status_code=status.HTTP_200_OK,
    tags=["chat"],
    summary="Get chat history",
    description="Get chat history for a thread or user.",
)
@private_router.get(
    "/{agent_id}/history",
    status_code=status.HTTP_200_OK,
    tags=["chat"],
    summary="Get chat history for a specific agent",
    description="Get chat history for a thread or user with a specific agent.",
)
async def history(
    input: ChatHistoryInput = Depends(),
    agent_id: str | None = None,
    request: Request = None,
) -> ChatHistory:
    """Get chat history."""
    _validate_thread_or_user_id(input.thread_id, input.user_id)

    if agent_id is None:
        agent_id = get_default_agent()

    agent: Agent = get_agent(request, agent_id)
    try:
        state_snapshot = await agent.graph.aget_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                    "user_id": input.user_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        # Let the global exception handler deal with all other exceptions
        raise


@private_router.delete(
    "/history/clear",
    status_code=status.HTTP_200_OK,
    tags=["chat"],
    summary="Clear chat history",
    description="Clear chat history for a thread or user.",
)
@private_router.delete(
    "/{agent_id}/history/clear",
    status_code=status.HTTP_200_OK,
    tags=["chat"],
    summary="Clear chat history for a specific agent",
    description="Clear chat history for a thread or user with a specific agent.",
)
async def clear_history(
    input: ClearHistoryInput,
    agent_id: str | None = None,
    request: Request = None,
) -> ClearHistoryResponse:
    """Clear chat history."""
    _validate_thread_or_user_id(input.thread_id, input.user_id)

    if agent_id is None:
        agent_id = get_default_agent()

    agent: Agent = get_agent(request, agent_id)
    try:
        state_snapshot = await agent.graph.aget_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                    "user_id": input.user_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]

        await agent.graph.aupdate_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                    "user_id": input.user_id,
                }
            ),
            values={"messages": [RemoveMessage(id=m.id) for m in messages]},
        )

        return ClearHistoryResponse(
            status="success",
            thread_id=input.thread_id,
            user_id=input.user_id,
            message=f"Cleared {len(messages)} messages from chat history.",
        )
    except Exception:
        # Let the global exception handler deal with all exceptions
        raise


@private_router.post(
    "/history/add_messages",
    status_code=status.HTTP_201_CREATED,
    tags=["chat"],
    summary="Add messages to chat history",
    description="Add messages to the end of chat history for a thread or user.",
)
@private_router.post(
    "/{agent_id}/history/add_messages",
    status_code=status.HTTP_201_CREATED,
    tags=["chat"],
    summary="Add messages to chat history for a specific agent",
    description="Add messages to the end of chat history for a thread or user with a specific agent.",
)
async def add_messages(
    input: AddMessagesInput,
    agent_id: str | None = None,
    request: Request = None,
) -> AddMessagesResponse:
    """Add messages to the end of chat history."""
    _validate_thread_or_user_id(input.thread_id, input.user_id)

    if agent_id is None:
        agent_id = get_default_agent()

    agent: Agent = get_agent(request, agent_id)
    try:
        await agent.graph.aupdate_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                    "user_id": input.user_id,
                }
            ),
            values={"messages": [{"type": m.type, "content": m.content} for m in input.messages]},
        )

        return AddMessagesResponse(
            status="success",
            thread_id=input.thread_id,
            user_id=input.user_id,
            message=f"Added {len(input.messages)} messages to chat history.",
        )
    except Exception:
        # Let the global exception handler deal with all exceptions
        raise


@public_router.get(
    "/",
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
    description="Kubernetes liveness probe - checks if the process is alive. "
    "Returns 200 if process is running, use to trigger container restart on failure.",
    response_description="Return HTTP Status Code 200 (OK) if alive",
    status_code=status.HTTP_200_OK,
    response_model=LivenessResponse,
)
async def liveness_probe() -> LivenessResponse:
    """Liveness probe for Kubernetes.

    This probe indicates if the process is alive and should be restarted if it fails.
    It performs minimal checks - just confirms the process can respond.
    """
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
    """Readiness probe for Kubernetes.

    This probe indicates if the service is ready to accept traffic.
    It checks that agents have been initialized successfully.
    Kubernetes will not route traffic to the pod until this returns 200.
    """
    from fastapi.responses import JSONResponse

    is_ready = getattr(request.app.state, "ready", False)
    initialized_agents = getattr(request.app.state, "initialized_agents", [])

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
    """Startup probe for Kubernetes.

    This probe indicates if the application has finished its initialization.
    It's designed for slow-starting containers and allows more time than liveness probe.
    The startup probe is checked before liveness/readiness probes are activated.
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

    try:
        stats = pool.get_stats()
        return DatabaseHealthResponse(
            status="healthy" if stats.get("pool_available", 0) > 0 else "exhausted",
            pool_size=stats.get("pool_size", 0),
            pool_available=stats.get("pool_available", 0),
            requests_waiting=stats.get("requests_waiting", 0),
            connections_num=stats.get("connections_num", 0),
        )
    except Exception as e:
        return DatabaseHealthResponse(
            status="error",
            message=str(e),
        )
