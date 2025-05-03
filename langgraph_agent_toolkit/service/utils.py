import json
import logging
import secrets
import warnings
from typing import Annotated, Any, AsyncGenerator, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.helper.logging import InterceptHandler, logger
from langgraph_agent_toolkit.schema import ChatMessage, StreamInput


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return

    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or not secrets.compare_digest(http_auth.credentials, auth_secret):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


def get_agent_executor(request: Request) -> AgentExecutor:
    """Get the AgentExecutor instance that was initialized in lifespan."""
    app = request.app
    if not hasattr(app.state, "agent_executor"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent executor not initialized. Service might be starting up.",
        )
    return app.state.agent_executor


def get_agent(request: Request, agent_id: str) -> Agent:
    """Get an agent by its ID from the initialized AgentExecutor."""
    executor = get_agent_executor(request)
    try:
        return executor.get_agent(agent_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found",
        )


def get_all_agent_info(request: Request):
    """Get information about all available agents from the initialized AgentExecutor."""
    executor = get_agent_executor(request)
    return executor.get_all_agent_info()


def _validate_thread_or_user_id(thread_id: Optional[str], user_id: Optional[str]) -> None:
    """Validate that either thread_id or user_id is provided."""
    if thread_id is None and user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either thread_id or user_id must be provided.",
        )


async def message_generator(
    stream_input: StreamInput,
    request: Request,
    agent_id: str,
) -> AsyncGenerator[str, None]:
    """Generate messages from an agent."""
    executor = get_agent_executor(request)

    try:
        async for message in executor.stream(
            agent_id=agent_id,
            message=stream_input.message,
            thread_id=stream_input.thread_id,
            user_id=stream_input.user_id,
            model_name=stream_input.model_name,
            model_provider=stream_input.model_provider,
            model_config_key=stream_input.model_config_key,
            stream_tokens=stream_input.stream_tokens,
            agent_config=stream_input.agent_config,
            recursion_limit=stream_input.recursion_limit,
        ):
            if isinstance(message, str):
                # Token output
                yield f"data: {json.dumps({'type': 'token', 'content': message})}\n\n"
            elif isinstance(message, ChatMessage):
                # Complete message
                yield f"data: {json.dumps({'type': 'message', 'content': message.model_dump()})}\n\n"

    except Exception as e:
        logger.error(f"Error in message_generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'Internal server error: {e}'})}\n\n"
    finally:
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
    # Setup logging once - redirect standard library logging to loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    # Reduce noise from uvicorn logs in production
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").handlers = [InterceptHandler()]

    # Additional loggers that should be quieter in production
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)

    # Suppress LangChain beta warnings
    warnings.filterwarnings("ignore", category=LangChainBetaWarning)

    return logger
