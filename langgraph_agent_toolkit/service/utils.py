import logging
import warnings
from contextlib import aclosing
from typing import Annotated, Any, AsyncGenerator

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.exceptions import ModelAuthenticationError

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.core.memory.concurrency import ConversationBusyError
from langgraph_agent_toolkit.helper.logging import InterceptHandler, logger
from langgraph_agent_toolkit.helper.types import EnvironmentMode
from langgraph_agent_toolkit.schema import ChatMessage, StreamChunk, StreamInput
from langgraph_agent_toolkit.service.auth import authenticate
from langgraph_agent_toolkit.service.feedback import sign_feedback_message


def _safe_stream_error(exc: Exception) -> str:
    """Return client text for a stream failure.

    Provider authentication failures always use fixed text.
    For other failures, non-production clients receive the error detail.
    Production clients receive a general message for other failures.
    """
    if isinstance(exc, ModelAuthenticationError):
        return "The model provider credentials were rejected"
    if isinstance(exc, ConversationBusyError):
        return str(exc)
    if settings.ENV_MODE != EnvironmentMode.PRODUCTION:
        return f"Internal server error: {exc}"
    return "Internal server error"


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Provide a configured bearer token.", auto_error=False)),
    ],
    request: Request,
) -> None:
    request.state.principal = authenticate(http_auth.credentials if http_auth else None)


def get_agent_executor(request: Request) -> AgentExecutor:
    """Get the `AgentExecutor` initialized during lifespan."""
    app = request.app
    if not hasattr(app.state, "agent_executor") or getattr(app.state, "ready", True) is False:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent executor not initialized. Service might be starting up.",
        )
    return app.state.agent_executor


def get_agent(request: Request, agent_id: str) -> Agent:
    """Get an agent by ID from the initialized `AgentExecutor`."""
    executor = get_agent_executor(request)
    try:
        return executor.get_agent(agent_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found",
        )


def get_all_agent_info(request: Request):
    """Get all agent information from the initialized `AgentExecutor`."""
    executor = get_agent_executor(request)
    return executor.get_all_agent_info()


async def jsonl_message_generator(
    stream_input: StreamInput,
    request: Request,
    agent_id: str,
    public_thread_id: str | None = None,
) -> AsyncGenerator[StreamChunk, None]:
    """Yield typed events and close execution when the consumer disconnects."""
    executor = get_agent_executor(request)
    try:
        stream = executor.stream(
            agent_id=agent_id,
            input=stream_input.input,
            thread_id=stream_input.thread_id,
            user_id=stream_input.user_id,
            model_name=stream_input.model_name,
            model_provider=stream_input.model_provider,
            model_config_key=stream_input.model_config_key,
            stream_tokens=stream_input.stream_tokens,
            agent_config=stream_input.agent_config,
            recursion_limit=stream_input.recursion_limit,
        )
        async with aclosing(stream):
            async for message in stream:
                if isinstance(message, str):
                    yield StreamChunk(type="token", content=message)
                elif isinstance(message, ChatMessage):
                    message.thread_id = public_thread_id
                    yield StreamChunk(type="message", content=sign_feedback_message(request, agent_id, message))
    except Exception as exc:
        if isinstance(exc, ModelAuthenticationError):
            logger.warning("The model provider credentials were rejected")
        else:
            logger.opt(exception=True).error("Agent stream failed")
        yield StreamChunk(type="error", content=_safe_stream_error(exc))


async def message_generator(
    stream_input: StreamInput,
    request: Request,
    agent_id: str,
    public_thread_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Encode typed events as SSE frames."""
    async with aclosing(jsonl_message_generator(stream_input, request, agent_id, public_thread_id)) as stream:
        async for chunk in stream:
            yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


def _sse_response_example() -> dict[int, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": (
                        "data: {'type': 'token', 'content': 'Hello'}\n\n"
                        "data: {'type': 'token', 'content': ' World'}\n\n"
                        "data: [DONE]\n\n"
                    ),
                    "schema": {"type": "string"},
                }
            },
        }
    }


def setup_logging():
    """Configure application logging to use loguru."""
    # Configure logging once and redirect standard-library logging to loguru.
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Configure the root logger with this handler.
    root_logger = logging.getLogger()
    root_logger.handlers = [InterceptHandler()]
    root_logger.setLevel(logging.NOTSET)

    # Configure uvicorn and related loggers.
    for logger_name in [
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "uvicorn.asgi",
        "watchfiles",
        "watchfiles.main",
    ]:
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers = [InterceptHandler()]
        uvicorn_logger.setLevel(logging.INFO)
        uvicorn_logger.propagate = False

    # Reduce selected logger output in production.
    if not settings.is_dev():
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("fastapi").setLevel(logging.WARNING)

    # Suppress LangChain beta warnings.
    warnings.filterwarnings("ignore", category=LangChainBetaWarning)

    return logger
