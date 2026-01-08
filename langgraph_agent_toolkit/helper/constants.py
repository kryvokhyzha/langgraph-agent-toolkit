import os


# Runtime override (set by agent_executor when validating loaded agents)
# This takes precedence over settings to allow dynamic adjustment at startup
_runtime_default_agent: str | None = None


def get_default_agent() -> str:
    """Get the current default agent name."""
    from langgraph_agent_toolkit.core.settings import settings

    if _runtime_default_agent is not None:
        return _runtime_default_agent

    return settings.DEFAULT_AGENT


def set_default_agent(agent_name: str) -> str:
    """Set the runtime default agent name."""
    global _runtime_default_agent
    _runtime_default_agent = agent_name
    return _runtime_default_agent


DEFAULT_CONFIG_PREFIX = os.getenv("DEFAULT_CONFIG_PREFIX", "agent")
DEFAULT_CONFIGURABLE_FIELDS = ("temperature", "max_tokens", "top_p", "streaming")
DEFAULT_MODEL_PARAMETER_VALUES = dict(
    temperature=0.0,
    max_tokens=2048,
    top_p=0.95,
    streaming=True,
)
