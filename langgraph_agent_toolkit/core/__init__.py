from langgraph_agent_toolkit.core.settings import settings


# Lazy import to avoid requiring openai at import time
def __getattr__(name: str):
    if name == "CompletionModelFactory":
        from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory

        return CompletionModelFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["settings", "CompletionModelFactory"]
