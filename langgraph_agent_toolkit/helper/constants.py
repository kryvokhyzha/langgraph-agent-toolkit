DEFAULT_AGENT = "react-agent"
DEFAULT_MAX_MESSAGE_HISTORY_LENGTH = 6 + 1  # N messages + 1 system message
DEFAULT_OPENAI_COMPATIBLE_MODEL_PARAMS = dict(
    temperature=0.0,
    max_tokens=1024,
    top_p=0.7,
    streaming=True,
)
