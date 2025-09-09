import os
import sys
import traceback

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

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


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers to the FastAPI app using decorators."""
    env_mode = EnvironmentMode(os.environ.get("ENV_MODE", EnvironmentMode.PRODUCTION))
    include_traceback = env_mode != EnvironmentMode.PRODUCTION

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
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=content)

    @app.exception_handler(UnsupportedMessageTypeError)
    async def unsupported_message_type_handler(request: Request, exc: UnsupportedMessageTypeError) -> JSONResponse:
        """Handle unsupported message type errors."""
        logger.error(f"Unsupported message type: {exc}")
        content = {"detail": str(exc), "message_type": exc.message_type, "supported_types": exc.supported_types}
        if exc.error_code:
            content["error_code"] = exc.error_code
        if include_traceback:
            content["traceback"] = traceback.format_exc()
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=content)

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        """Handle model not found errors."""
        logger.error(f"Model not found: {exc}")
        content = {"detail": str(exc), "model_name": exc.model_name}
        if exc.error_code:
            content["error_code"] = exc.error_code
        if exc.provider:
            content["provider"] = exc.provider
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=content)

    @app.exception_handler(ModelConfigurationError)
    async def model_configuration_error_handler(request: Request, exc: ModelConfigurationError) -> JSONResponse:
        """Handle model configuration errors."""
        logger.error(f"Model configuration error: {exc}")
        content = {"detail": str(exc)}
        if exc.error_code:
            content["error_code"] = exc.error_code
        if exc.details:
            content["details"] = exc.details
        if include_traceback:
            content["traceback"] = traceback.format_exc()
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=content)

    @app.exception_handler(ToolNotFoundError)
    async def tool_not_found_handler(request: Request, exc: ToolNotFoundError) -> JSONResponse:
        """Handle tool not found errors."""
        logger.error(f"Tool not found: {exc}")
        content = {"detail": str(exc), "tool_name": exc.tool_name, "available_tools": exc.available_tools}
        if exc.error_code:
            content["error_code"] = exc.error_code
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=content)

    @app.exception_handler(ToolExecutionError)
    async def tool_execution_error_handler(request: Request, exc: ToolExecutionError) -> JSONResponse:
        """Handle tool execution errors."""
        logger.error(f"Tool execution error: {exc}")
        content = {"detail": str(exc), "tool_name": exc.tool_name}
        if exc.error_code:
            content["error_code"] = exc.error_code
        if include_traceback:
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
        if include_traceback:
            content["traceback"] = traceback.format_exc()
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=content)

    @app.exception_handler(AgentToolkitError)
    async def agent_toolkit_error_handler(request: Request, exc: AgentToolkitError) -> JSONResponse:
        """Handle base AgentToolkitError and any custom errors that inherit from it."""
        logger.error(f"Agent toolkit error: {exc}")
        content = {"detail": str(exc), "error_type": exc.__class__.__name__}
        if exc.error_code:
            content["error_code"] = exc.error_code
        if exc.details:
            content["details"] = exc.details
        if include_traceback:
            content["traceback"] = traceback.format_exc()
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=content)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTPException with appropriate logging."""
        logger.warning(f"HTTPException: {exc.detail} (status {exc.status_code})")

        content = {"detail": exc.detail}

        # Include headers if present
        if exc.headers:
            return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)

        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle ValueError exceptions."""
        logger.opt(exception=sys.exc_info()).error(f"ValueError: {exc}")

        content = {"detail": str(exc)}
        if include_traceback:
            content["traceback"] = traceback.format_exc()

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=content,
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle all other unexpected exceptions."""
        # For all other exceptions, preserve the original error details
        error_detail = f"{exc.__class__.__name__}: {exc}"
        logger.opt(exception=sys.exc_info()).error(f"Agent error: {error_detail}")

        # Use the original exception message instead of generic "Unexpected error"
        content = {"detail": str(exc), "error_type": exc.__class__.__name__}
        if include_traceback:
            content["traceback"] = traceback.format_exc()

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=content,
        )
