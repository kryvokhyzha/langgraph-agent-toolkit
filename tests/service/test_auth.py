from unittest.mock import patch

from pydantic import SecretStr

from langgraph_agent_toolkit.schema import ChatMessage
from langgraph_agent_toolkit.service import utils as service_utils


# NOTE: verify_bearer reads `settings.AUTH_SECRET` from the singleton bound in
# `service.utils`, so we patch that exact object's attribute. Patching
# `core.settings.settings` (as the old tests did) does NOT affect the reference
# verify_bearer captured at import time, which is why those tests never actually gated.


def test_no_auth_secret(mock_agent_executor, test_client):
    """When AUTH_SECRET is unset, requests are allowed with or without a token."""
    mock_agent_executor.invoke.return_value = ChatMessage(type="ai", content="ok")

    with patch.object(service_utils.settings, "AUTH_SECRET", None):
        with patch("langgraph_agent_toolkit.service.routes.get_agent_executor", return_value=mock_agent_executor):
            resp = test_client.post(
                "/invoke",
                json={"input": {"message": "test"}},
                headers={"Authorization": "Bearer any-token"},
            )
            assert resp.status_code == 200

            # ...and with no Authorization header at all
            resp = test_client.post("/invoke", json={"input": {"message": "test"}})
            assert resp.status_code == 200


def test_auth_secret_correct(mock_agent_executor, test_client):
    """When AUTH_SECRET is set, the matching bearer token is accepted."""
    mock_agent_executor.invoke.return_value = ChatMessage(type="ai", content="ok")

    with patch.object(service_utils.settings, "AUTH_SECRET", SecretStr("test-secret")):
        with patch("langgraph_agent_toolkit.service.routes.get_agent_executor", return_value=mock_agent_executor):
            resp = test_client.post(
                "/invoke",
                json={"input": {"message": "test"}},
                headers={"Authorization": "Bearer test-secret"},
            )
            assert resp.status_code == 200


def test_auth_secret_incorrect(mock_agent_executor, test_client):
    """When AUTH_SECRET is set, a wrong or missing bearer token is rejected with 401."""
    with patch.object(service_utils.settings, "AUTH_SECRET", SecretStr("test-secret")):
        with patch("langgraph_agent_toolkit.service.routes.get_agent_executor", return_value=mock_agent_executor):
            # Wrong token -> 401
            resp = test_client.post(
                "/invoke",
                json={"input": {"message": "test"}},
                headers={"Authorization": "Bearer wrong-secret"},
            )
            assert resp.status_code == 401

            # Missing Authorization header -> 401
            resp = test_client.post("/invoke", json={"input": {"message": "test"}})
            assert resp.status_code == 401
