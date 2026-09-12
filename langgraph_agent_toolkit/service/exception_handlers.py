import sys
import traceback

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from psycopg import OperationalError
from psycopg_pool import PoolTimeout, TooManyRequests

from langgraph_agent_toolkit.core.memory.concurrency import (
    ConversationBusyError,
    ConversationLockLostError,
    NestedConversationError,
)
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.exceptions import (
    AgentToolkitError,
    AuthenticationError,
    AuthorizationError,
    FeedbackError,
    InputValidationError,
    ModelConfigurationError,
    ModelNotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    ToolExecutionError,
    ToolNotFoundError,
    UnsupportedMessageTypeError,
    ValidationError,
)
from langgraph_agent_toolkit.helper.logging import logger
from langgraph_agent_toolkit.helper.types import EnvironmentMode
from langgraph_agent_toolkit.service.admission import ServiceBusyError


def _expose_error_detail() -> bool:
    """Return whether clients can receive internal error details.

    Return `True` outside production.
    Production clients receive a general message.
    The service logs full errors on the server.
    Read `settings.ENV_MODE` for each request.
    """
    return settings.ENV_MODE != EnvironmentMode.PRODUCTION


def register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers on the FastAPI app."""

    @app.exception_handler(ModelAuthenticationError)
    async def model_authentication_handler(request: Request, exc: ModelAuthenticationError) -> JSONResponse:
        logger.warning("The model provider credentials were rejected")
        return JSONResponse(
            status_code=503,
            content={
                "detail": "The model provider credentials were rejected",
                "error_code": "model_authentication_failed",
            },
        )

    @app.exception_handler(ModelRateLimitError)
    async def model_rate_limit_handler(request: Request, exc: ModelRateLimitError) -> JSONResponse:
        logger.warning("The model provider rate limit was exceeded")
        return JSONResponse(
            status_code=429,
            content={"detail": "The model provider rate limit was exceeded", "error_code": "model_rate_limit"},
        )

    @app.exception_handler(ModelConnectionError)
    async def model_connection_handler(request: Request, exc: ModelConnectionError) -> JSONResponse:
        logger.warning("The model provider is unavailable")
        return JSONResponse(
            status_code=503,
            content={"detail": "The model provider is unavailable", "error_code": "model_unavailable"},
        )

    @app.exception_handler(ModelTimeoutError)
    async def model_timeout_handler(request: Request, exc: ModelTimeoutError) -> JSONResponse:
        logger.warning("The model provider request timed out")
        return JSONResponse(
            status_code=504,
            content={"detail": "The model provider request timed out", "error_code": "model_timeout"},
        )

    @app.exception_handler(ServiceBusyError)
    async def service_busy_handler(request: Request, exc: ServiceBusyError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": "Worker request capacity is full", "error_code": "service_busy"},
            headers={"Retry-After": "1"},
        )

    @app.exception_handler(OperationalError)
    @app.exception_handler(PoolTimeout)
    @app.exception_handler(TooManyRequests)
    async def database_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.opt(exception=True).error("Database operation failed")
        return JSONResponse(status_code=503, content={"detail": "Database temporarily unavailable"})

    @app.exception_handler(NestedConversationError)
    async def nested_conversation_handler(request: Request, exc: NestedConversationError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc), "error_code": "nested_conversation"})

    @app.exception_handler(ConversationBusyError)
    async def conversation_busy_handler(request: Request, exc: ConversationBusyError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)}, headers={"Retry-After": "1"})

    @app.exception_handler(ConversationLockLostError)
    async def conversation_lock_handler(request: Request, exc: ConversationLockLostError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "The database conversation lock was lost"})

    @app.exception_handler(TimeoutError)
    async def request_timeout_handler(request: Request, exc: TimeoutError) -> JSONResponse:
        return JSONResponse(status_code=504, content={"detail": "The request time limit expired"})

    @app.exception_handler(AuthenticationError)
    async def authentication_error_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
        """Handle authentication errors."""
        logger.warning(f"Authentication error: {exc}")
        content = {"detail": str(exc)}
        if exc.error_code:
            content["error_code"] = exc.error_code
        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content=content)

    @app.exception_handler(AuthorizationError)
    async def authorization_error_handler(request: Request, exc: AuthorizationError) -> JSONResponse:
        """Handle authorization errors."""
        logger.warning(f"Authorization error: {exc}")
        content = {"detail": str(exc)}
        if exc.error_code:
            content["error_code"] = exc.error_code
        return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content=content)

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
        """Handle validation errors."""
        logger.warning(f"Validation error: {exc}")
        content = {"detail": str(exc)}
        if exc.error_code:
            content["error_code"] = exc.error_code
        if exc.details:
            content["details"] = exc.details
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=content)

    @app.exception_handler(InputValidationError)
    async def input_validation_error_handler(request: Request, exc: InputValidationError) -> JSONResponse:
        """Handle input validation errors."""
        logger.warning(f"Input validation error: {exc}")
        content = {"detail": str(exc)}
        if exc.error_code:
            content["error_code"] = exc.error_code
        if exc.details:
            content["details"] = exc.details
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content=content)

    @app.exception_handler(UnsupportedMessageTypeError)
    async def unsupported_message_type_handler(request: Request, exc: UnsupportedMessageTypeError) -> JSONResponse:
        """Handle unsupported message type errors."""
        logger.warning(f"Unsupported message type: {exc}")
        content = {"detail": str(exc), "message_type": exc.message_type, "supported_types": exc.supported_types}
        if exc.error_code:
            content["error_code"] = exc.error_code
        if _expose_error_detail():
            content["traceback"] = traceback.format_exc()
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content=content)

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        """Handle model not found errors."""
        logger.warning(f"Model not found: {exc}")
        content = {"detail": str(exc), "model_name": exc.model_name}
        if exc.error_code:
            content["error_code"] = exc.error_code
        if exc.provider:
            content["provider"] = exc.provider
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=content)

    @app.exception_handler(ModelConfigurationError)
    async def model_configuration_error_handler(request: Request, exc: ModelConfigurationError) -> JSONResponse:
        """Handle model configuration errors."""
        logger.warning(f"Model configuration error: {exc}")
        content = {"detail": str(exc)}
        if exc.error_code:
            content["error_code"] = exc.error_code
        if exc.details:
            content["details"] = exc.details
        if _expose_error_detail():
            content["traceback"] = traceback.format_exc()
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=content)

    @app.exception_handler(ToolNotFoundError)
    async def tool_not_found_handler(request: Request, exc: ToolNotFoundError) -> JSONResponse:
        """Handle tool not found errors."""
        logger.warning(f"Tool not found: {exc}")
        content = {"detail": str(exc), "tool_name": exc.tool_name, "available_tools": exc.available_tools}
        if exc.error_code:
            content["error_code"] = exc.error_code
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=content)

    @app.exception_handler(ToolExecutionError)
    async def tool_execution_error_handler(request: Request, exc: ToolExecutionError) -> JSONResponse:
        """Handle tool execution errors."""
        logger.error(f"Tool execution error: {exc}")
        content = {
            "detail": str(exc) if _expose_error_detail() else "Tool execution failed",
            "tool_name": exc.tool_name,
        }
        if exc.error_code:
            content["error_code"] = exc.error_code
        if _expose_error_detail():
            content["traceback"] = traceback.format_exc()
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=content)

    @app.exception_handler(RateLimitError)
    async def rate_limit_error_handler(request: Request, exc: RateLimitError) -> JSONResponse:
        """Handle rate limit errors."""
        logger.warning(f"Rate limit exceeded: {exc}")
        content = {"detail": str(exc), "resource": exc.resource, "limit": exc.limit}
        if exc.error_code:
            content["error_code"] = exc.error_code
        if exc.reset_time:
            content["reset_time"] = exc.reset_time
        return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS, content=content)

    @app.exception_handler(ServiceUnavailableError)
    async def service_unavailable_handler(request: Request, exc: ServiceUnavailableError) -> JSONResponse:
        """Handle service unavailable errors."""
        logger.error(f"Service unavailable: {exc}")
        content = {"detail": str(exc), "service": exc.service}
        if exc.error_code:
            content["error_code"] = exc.error_code
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=content)

    @app.exception_handler(FeedbackError)
    async def feedback_error_handler(request: Request, exc: FeedbackError) -> JSONResponse:
        """Handle feedback operation errors."""
        logger.error(f"Feedback error: {exc}")
        content = {"detail": str(exc), "run_id": exc.run_id, "operation": exc.operation}
        if exc.error_code:
            content["error_code"] = exc.error_code
        if exc.reason:
            content["reason"] = exc.reason
        if _expose_error_detail():
            content["traceback"] = traceback.format_exc()
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=content)

    @app.exception_handler(AgentToolkitError)
    async def agent_toolkit_error_handler(request: Request, exc: AgentToolkitError) -> JSONResponse:
        """Handle `AgentToolkitError` and its subclasses."""
        logger.error(f"Agent toolkit error: {exc}")
        content = {"detail": str(exc), "error_type": exc.__class__.__name__}
        if exc.error_code:
            content["error_code"] = exc.error_code
        if exc.details:
            content["details"] = exc.details
        if _expose_error_detail():
            content["traceback"] = traceback.format_exc()
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=content)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle an `HTTPException` and log it."""
        logger.warning(f"HTTPException: {exc.detail} (status {exc.status_code})")

        content = {"detail": exc.detail}

        # Include response headers when provided.
        if exc.headers:
            return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)

        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle an unexpected `ValueError`.

        Client validation must raise `InputValidationError` or `ValidationError`.
        The service logs the full error.
        Only non-production clients receive the error message.
        """
        logger.opt(exception=sys.exc_info()).error(f"ValueError: {exc}")

        if _expose_error_detail():
            content = {"detail": str(exc), "traceback": traceback.format_exc()}
        else:
            content = {"detail": "Invalid request"}

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=content,
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle an unexpected exception.

        The service logs the exception type, message, and traceback.
        Non-production clients receive the message and type.
        Production clients receive a general message.
        """
        error_detail = f"{exc.__class__.__name__}: {exc}"
        logger.opt(exception=sys.exc_info()).error(f"Agent error: {error_detail}")

        if _expose_error_detail():
            content = {"detail": str(exc), "error_type": exc.__class__.__name__, "traceback": traceback.format_exc()}
        else:
            content = {"detail": "Internal server error"}

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=content,
        )
