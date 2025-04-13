import json
from typing import Annotated, Any, AsyncGenerator

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.agent_executor import AgentExecutor
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.helper.constants import DEFAULT_AGENT
from langgraph_agent_toolkit.helper.logging import logger
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
    if not http_auth or http_auth.credentials != auth_secret:
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


async def message_generator(
    user_input: StreamInput, request: Request, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    executor = get_agent_executor(request)

    try:
        async for item in executor.stream(
            agent_id=agent_id,
            message=user_input.message,
            thread_id=user_input.thread_id,
            model=user_input.model,
            stream_tokens=user_input.stream_tokens,
            agent_config=user_input.agent_config,
            recursion_limit=user_input.recursion_limit,
        ):
            if isinstance(item, str):
                # Token output
                yield f"data: {json.dumps({'type': 'token', 'content': item})}\n\n"
            elif isinstance(item, ChatMessage):
                # Complete message
                yield f"data: {json.dumps({'type': 'message', 'content': item.model_dump()})}\n\n"

        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in message_generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
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
