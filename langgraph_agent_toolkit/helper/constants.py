# This is the initial default agent name, but it may be overridden at runtime
DEFAULT_AGENT = "react-agent"
_CURRENT_DEFAULT_AGENT = DEFAULT_AGENT


def get_default_agent():
    return _CURRENT_DEFAULT_AGENT


def set_default_agent(agent_name):
    global _CURRENT_DEFAULT_AGENT
    _CURRENT_DEFAULT_AGENT = agent_name
    return _CURRENT_DEFAULT_AGENT


DEFAULT_MAX_MESSAGE_HISTORY_LENGTH = 6 + 1  # N messages + 1 system message
DEFAULT_RECURSION_LIMIT = 25
DEFAULT_CONFIG_PREFIX = "agent"
DEFAULT_CONFIGURABLE_FIELDS = ("temperature", "max_tokens", "top_p", "streaming")
DEFAULT_MODEL_PARAMETER_VALUES = dict(
    temperature=0.0,
    max_tokens=1024,
    top_p=0.7,
    streaming=True,
)
DEFAULT_CACHE_TTL_SECOND = 60 * 10  # 10 minutes

DEFAULT_STREAMLIT_USER_ID = "streamlit-user"
