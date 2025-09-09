class AgentToolkitError(Exception):
    """Base exception for all LangGraph Agent Toolkit errors.

    All custom exceptions should inherit from this base class.
    """

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        """Initialize the base exception.

        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error context

        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class AgentError(AgentToolkitError):
    """Base exception for agent-related errors."""

    pass


class AgentConfigurationError(AgentError):
    """Raised when there's an error in agent configuration."""

    pass


class AgentExecutionError(AgentError):
    """Raised when an agent fails to execute properly."""

    pass


class AgentTimeoutError(AgentError):
    """Raised when an agent operation times out."""

    pass


class MessageError(AgentToolkitError):
    """Base exception for message-related errors."""

    pass


class UnsupportedMessageTypeError(MessageError):
    """Raised when encountering an unsupported message type.

    This replaces the string-based check for "Unsupported message type".
    """

    def __init__(self, message_type: str, supported_types: list = None):
        """Initialize with specific message type information.

        Args:
            message_type: The unsupported message type
            supported_types: List of supported message types

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
    """Base exception for validation errors."""

    pass


class InputValidationError(ValidationError):
    """Raised when input validation fails."""

    pass


class ConfigurationValidationError(ValidationError):
    """Raised when configuration validation fails."""

    pass


class ModelError(AgentToolkitError):
    """Base exception for model-related errors."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""

    def __init__(self, model_name: str, provider: str = None):
        """Initialize with model information.

        Args:
            model_name: Name of the model that wasn't found
            provider: Optional model provider

        """
        provider_info = f" from provider '{provider}'" if provider else ""
        message = f"Model '{model_name}'{provider_info} not found"
        super().__init__(message, error_code="MODEL_NOT_FOUND")
        self.model_name = model_name
        self.provider = provider


class ModelConfigurationError(ModelError):
    """Raised when there's an error in model configuration."""

    pass


class ToolError(AgentToolkitError):
    """Base exception for tool-related errors."""

    pass


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found."""

    def __init__(self, tool_name: str, available_tools: list = None):
        """Initialize with tool information.

        Args:
            tool_name: Name of the tool that wasn't found
            available_tools: List of available tools

        """
        available = f" Available tools: {available_tools}" if available_tools else ""
        message = f"Tool '{tool_name}' not found.{available}"
        super().__init__(message, error_code="TOOL_NOT_FOUND")
        self.tool_name = tool_name
        self.available_tools = available_tools or []


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""

    def __init__(self, tool_name: str, original_error: Exception = None):
        """Initialize with tool execution information.

        Args:
            tool_name: Name of the tool that failed
            original_error: The original exception that caused the failure

        """
        message = f"Tool '{tool_name}' execution failed"
        if original_error:
            message += f": {str(original_error)}"
        super().__init__(message, error_code="TOOL_EXECUTION_FAILED")
        self.tool_name = tool_name
        self.original_error = original_error


class MemoryError(AgentToolkitError):
    """Base exception for memory-related errors."""

    pass


class MemoryNotFoundError(MemoryError):
    """Raised when requested memory/thread is not found."""

    def __init__(self, identifier: str, identifier_type: str = "thread"):
        """Initialize with memory identifier information.

        Args:
            identifier: The identifier that wasn't found
            identifier_type: Type of identifier (thread, user, etc.)

        """
        message = f"{identifier_type.capitalize()} '{identifier}' not found"
        super().__init__(message, error_code="MEMORY_NOT_FOUND")
        self.identifier = identifier
        self.identifier_type = identifier_type


class MemoryOperationError(MemoryError):
    """Raised when memory operations fail."""

    pass


class ObservabilityError(AgentToolkitError):
    """Base exception for observability-related errors."""

    pass


class FeedbackError(ObservabilityError):
    """Raised when feedback operations fail."""

    def __init__(self, run_id: str, operation: str, reason: str = None):
        """Initialize with feedback operation information.

        Args:
            run_id: The run ID associated with the feedback
            operation: The feedback operation that failed
            reason: Optional reason for the failure

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
        """Initialize with rate limit information.

        Args:
            resource: The resource that hit the rate limit
            limit: The rate limit that was exceeded
            reset_time: Optional time when the limit resets

        """
        message = f"Rate limit exceeded for {resource} (limit: {limit})"
        if reset_time:
            message += f". Resets at {reset_time}"
        super().__init__(message, error_code="RATE_LIMIT_EXCEEDED")
        self.resource = resource
        self.limit = limit
        self.reset_time = reset_time


class NetworkError(AgentToolkitError):
    """Base exception for network-related errors."""

    pass


class ServiceUnavailableError(NetworkError):
    """Raised when an external service is unavailable."""

    def __init__(self, service: str, details: str = None):
        """Initialize with service information.

        Args:
            service: Name of the unavailable service
            details: Optional details about the unavailability

        """
        message = f"Service '{service}' is unavailable"
        if details:
            message += f": {details}"
        super().__init__(message, error_code="SERVICE_UNAVAILABLE")
        self.service = service
