class AgentToolkitError(Exception):
    """Base class for LangGraph Agent Toolkit errors."""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        """Initialize the exception.

        Args:
            message: Error message for a user.
            error_code: Optional error code for a program.
            details: Optional error context.

        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class AgentError(AgentToolkitError):
    """Base class for agent errors."""

    pass


class AgentConfigurationError(AgentError):
    """Raised when an agent configuration has an error."""

    pass


class AgentExecutionError(AgentError):
    """Raised when an agent cannot run."""

    pass


class AgentTimeoutError(AgentError):
    """Raised when an agent operation times out."""

    pass


class MessageError(AgentToolkitError):
    """Base class for message errors."""

    pass


class UnsupportedMessageTypeError(MessageError):
    """Raised for an unsupported message type."""

    def __init__(self, message_type: str, supported_types: list = None):
        """Initialize the error with message type information.

        Args:
            message_type: Unsupported message type.
            supported_types: Supported message types.

        """
        supported = f" Supported types: {supported_types}" if supported_types else ""
        message = f"Unsupported message type: {message_type}.{supported}"
        super().__init__(message, error_code="UNSUPPORTED_MESSAGE_TYPE")
        self.message_type = message_type
        self.supported_types = supported_types or []


class MessageConversionError(MessageError):
    """Raised when message conversion fails."""

    pass


class ValidationError(AgentToolkitError):
    """Base class for validation errors."""

    pass


class InputValidationError(ValidationError):
    """Raised when input validation fails."""

    pass


class ConfigurationValidationError(ValidationError):
    """Raised when configuration validation fails."""

    pass


class ModelError(AgentToolkitError):
    """Base class for model errors."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""

    def __init__(self, model_name: str, provider: str = None):
        """Initialize the error with model information.

        Args:
            model_name: Name of the missing model.
            provider: Optional model provider.

        """
        provider_info = f" from provider '{provider}'" if provider else ""
        message = f"Model '{model_name}'{provider_info} not found"
        super().__init__(message, error_code="MODEL_NOT_FOUND")
        self.model_name = model_name
        self.provider = provider


class ModelConfigurationError(ModelError):
    """Raised when a model configuration has an error."""

    pass


class ToolError(AgentToolkitError):
    """Base class for tool errors."""

    pass


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found."""

    def __init__(self, tool_name: str, available_tools: list = None):
        """Initialize the error with tool information.

        Args:
            tool_name: Name of the missing tool.
            available_tools: Available tools.

        """
        available = f" Available tools: {available_tools}" if available_tools else ""
        message = f"Tool '{tool_name}' not found.{available}"
        super().__init__(message, error_code="TOOL_NOT_FOUND")
        self.tool_name = tool_name
        self.available_tools = available_tools or []


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""

    def __init__(self, tool_name: str, original_error: Exception = None):
        """Initialize the error with tool execution information.

        Args:
            tool_name: Name of the failed tool.
            original_error: Exception that caused the failure.

        """
        message = f"Tool '{tool_name}' execution failed"
        if original_error:
            message += f": {str(original_error)}"
        super().__init__(message, error_code="TOOL_EXECUTION_FAILED")
        self.tool_name = tool_name
        self.original_error = original_error


class MemoryError(AgentToolkitError):
    """Base class for memory errors."""

    pass


class MemoryNotFoundError(MemoryError):
    """Raised when the requested memory or thread is missing."""

    def __init__(self, identifier: str, identifier_type: str = "thread"):
        """Initialize the error with memory identifier information.

        Args:
            identifier: Missing identifier.
            identifier_type: Identifier type, such as `thread` or `user`.

        """
        message = f"{identifier_type.capitalize()} '{identifier}' not found"
        super().__init__(message, error_code="MEMORY_NOT_FOUND")
        self.identifier = identifier
        self.identifier_type = identifier_type


class MemoryOperationError(MemoryError):
    """Raised when memory operations fail."""

    pass


class ObservabilityError(AgentToolkitError):
    """Base class for observability errors."""

    pass


class FeedbackError(ObservabilityError):
    """Raised when feedback operations fail."""

    def __init__(self, run_id: str, operation: str, reason: str = None):
        """Initialize the error with feedback operation information.

        Args:
            run_id: Run ID for the feedback.
            operation: Failed feedback operation.
            reason: Optional failure reason.

        """
        message = f"Feedback {operation} failed for run {run_id}"
        if reason:
            message += f": {reason}"
        super().__init__(message, error_code="FEEDBACK_FAILED")
        self.run_id = run_id
        self.operation = operation
        self.reason = reason


class AuthenticationError(AgentToolkitError):
    """Raised when authentication fails."""

    pass


class AuthorizationError(AgentToolkitError):
    """Raised when authorization fails."""

    pass


class RateLimitError(AgentToolkitError):
    """Raised when rate limits are exceeded."""

    def __init__(self, resource: str, limit: int, reset_time: float = None):
        """Initialize the error with rate-limit information.

        Args:
            resource: Resource that reached the rate limit.
            limit: Exceeded rate limit.
            reset_time: Optional limit reset time.

        """
        message = f"Rate limit exceeded for {resource} (limit: {limit})"
        if reset_time:
            message += f". Resets at {reset_time}"
        super().__init__(message, error_code="RATE_LIMIT_EXCEEDED")
        self.resource = resource
        self.limit = limit
        self.reset_time = reset_time


class NetworkError(AgentToolkitError):
    """Base class for network errors."""

    pass


class ServiceUnavailableError(NetworkError):
    """Raised when an external service is unavailable."""

    def __init__(self, service: str, details: str = None):
        """Initialize the error with service information.

        Args:
            service: Name of the unavailable service.
            details: Optional unavailability details.

        """
        message = f"Service '{service}' is unavailable"
        if details:
            message += f": {details}"
        super().__init__(message, error_code="SERVICE_UNAVAILABLE")
        self.service = service
