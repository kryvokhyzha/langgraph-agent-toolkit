try:
    from importlib.metadata import version

    __version__ = version("langgraph_agent_toolkit")
except Exception:
    # Fallback for local development
    __version__ = "0.0.0.dev0"
