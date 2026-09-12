"""Check validated Langfuse settings without network access."""

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from langgraph_agent_toolkit.core.observability import langfuse as adapter


def test_validated_credentials_are_passed_to_sdk(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    configured = adapter.settings.model_copy(
        update={
            "LANGFUSE_PUBLIC_KEY": SecretStr("pk-from-settings"),
            "LANGFUSE_SECRET_KEY": SecretStr("sk-from-settings"),
            "LANGFUSE_HOST": "https://langfuse.invalid",
            "LANGFUSE_FLUSH_AT": 20,
            "LANGFUSE_TIMEOUT": 3,
        }
    )
    monkeypatch.setattr(adapter, "settings", configured)
    constructor = MagicMock()
    monkeypatch.setattr(adapter, "Langfuse", constructor)
    obs = adapter.LangfuseObservability()
    assert obs.validate_environment()
    obs._get_client()
    assert constructor.call_args.kwargs["public_key"] == "pk-from-settings"
    assert constructor.call_args.kwargs["secret_key"] == "sk-from-settings"
    assert constructor.call_args.kwargs["host"] == "https://langfuse.invalid"
    assert constructor.call_args.kwargs["flush_at"] == 20
    assert constructor.call_args.kwargs["timeout"] == 3
    obs._get_client()
    constructor.assert_called_once()


@pytest.mark.skipif(not adapter._IS_NEW_LANGFUSE, reason="Modern SDK callback project selection")
def test_callback_keeps_the_initialized_project_when_configuration_changes(monkeypatch):
    configured = adapter.settings.model_copy(
        update={
            "LANGFUSE_PUBLIC_KEY": SecretStr("pk-first"),
            "LANGFUSE_SECRET_KEY": SecretStr("sk-first"),
        }
    )
    monkeypatch.setattr(adapter, "settings", configured)
    monkeypatch.setattr(adapter, "_get_langfuse_client", MagicMock())
    monkeypatch.setattr(adapter, "CallbackHandler", lambda public_key: public_key)
    obs = adapter.LangfuseObservability()
    assert obs.get_callback_handler() == "pk-first"
    monkeypatch.setattr(
        adapter,
        "settings",
        configured.model_copy(
            update={
                "LANGFUSE_PUBLIC_KEY": SecretStr("pk-second"),
                "LANGFUSE_SECRET_KEY": SecretStr("sk-second"),
            }
        ),
    )
    assert obs.get_callback_handler() == "pk-first"
