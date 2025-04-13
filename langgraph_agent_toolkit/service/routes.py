from fastapi import APIRouter, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from fastapi.responses import RedirectResponse
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.helper.constants import DEFAULT_AGENT
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.helper.logging import logger
from langgraph_agent_toolkit.schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
    HealthCheck,
)
from langgraph_agent_toolkit.helper.utils import langchain_to_chat_message
from langgraph_agent_toolkit.service.utils import (
    message_generator,
    _sse_response_example,
    get_agent,
    get_agent_executor,
    get_all_agent_info,
)

# Create separate routers for private and public endpoints
private_router = APIRouter()
public_router = APIRouter()


@private_router.get("/info", tags=["info"])
async def info(request: Request) -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(request),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


@private_router.post("/{agent_id}/invoke", tags=["agent"])
@private_router.post("/invoke", tags=["agent"])
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT, request: Request = None) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    executor = get_agent_executor(request)

    try:
        return await executor.invoke(
            agent_id=agent_id,
            message=user_input.message,
            thread_id=user_input.thread_id,
            model=user_input.model,
            agent_config=user_input.agent_config,
            recursion_limit=user_input.recursion_limit,
        )
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error")


@private_router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    tags=["agent"],
)
@private_router.post(
    "/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    tags=["agent"],
)
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT, request: Request = None) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input, request, agent_id),
        media_type="text/event-stream",
    )


@private_router.post("/feedback", status_code=status.HTTP_201_CREATED, tags=["feedback"])
async def feedback(feedback: Feedback, agent_id: str = DEFAULT_AGENT, request: Request = None) -> FeedbackResponse:
    """
    Record feedback for a run to the configured observability platform.

    This routes the feedback to the appropriate platform based on the agent's configuration.
    """
    try:
        agent = get_agent(request, agent_id)
        agent.observability.record_feedback(
            run_id=feedback.run_id,
            key=feedback.key,
            score=feedback.score,
            **feedback.kwargs,
        )

        return FeedbackResponse(
            run_id=feedback.run_id,
            message=f"Feedback '{feedback.key}' recorded successfully for run {feedback.run_id}.",
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"An exception occurred while recording feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error recording feedback"
        )


@private_router.post("/history", tags=["history"])
def history(input: ChatHistoryInput, request: Request = None) -> ChatHistory:
    """
    Get chat history.
    """
    agent: Agent = get_agent(request, DEFAULT_AGENT)
    try:
        agent_graph: CompiledStateGraph = agent.graph
        state_snapshot = agent_graph.get_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error")


@public_router.get("/", tags=["public"])
async def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@public_router.get(
    "/health",
    tags=["public", "healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
async def health_check() -> HealthCheck:
    """Health check endpoint."""
    return HealthCheck(status="OK")
