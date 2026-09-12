"""Test conversation navigation through the Streamlit page."""

from types import SimpleNamespace

import httpx
import pytest
from streamlit import config
from streamlit.runtime.context import ContextProxy
from streamlit.testing.v1 import AppTest

from langgraph_agent_toolkit.client import AgentClient, AgentClientError
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.schema import ChatHistory, ChatMessage
from langgraph_agent_toolkit.ui import main_page


@pytest.fixture
def page(monkeypatch):
    """Run the real page with a deterministic service client."""

    def respond(request):
        assert request.method == "GET"
        selected = request.url.path.split("/")[1]
        assert request.url.path == f"/{selected}/history"
        thread_id = request.url.params["thread_id"]
        user_id = request.url.params.get("user_id")
        offset = int(request.url.params["offset"])
        client.history_calls.append((selected, thread_id, user_id))
        client.history_page_calls.append((selected, thread_id, user_id, offset))
        history = client.history_pages.get((thread_id, offset))
        if isinstance(history, httpx.Response):
            return history
        if history is None:
            history = ChatHistory(
                messages=[ChatMessage(type="ai", content=f"Saved reply from {selected}: {thread_id}")]
            )
        return httpx.Response(200, json=history.model_dump(mode="json"))

    http_client = httpx.Client(transport=httpx.MockTransport(respond))
    client = AgentClient(base_url="http://service.test", agent="first-agent", get_info=False, http_client=http_client)
    client.info = SimpleNamespace(
        default_agent="first-agent",
        agents=[SimpleNamespace(key="first-agent"), SimpleNamespace(key="second-agent")],
    )
    client.history_calls = []
    client.history_page_calls = []
    client.history_pages = {}
    monkeypatch.setattr(main_page, "AgentClient", lambda **kwargs: client)
    monkeypatch.setattr(settings, "AUTH_MODE", "trusted")
    monkeypatch.setattr(settings, "DEFAULT_STREAMLIT_USER_ID", "configured-user")
    old_toolbar = config.get_option("client.toolbarMode")
    config.set_option("client.toolbarMode", "minimal")
    app = AppTest.from_string(
        "import asyncio\nfrom langgraph_agent_toolkit.ui.main_page import main_page\nasyncio.run(main_page())"
    )
    try:
        yield app, client
    finally:
        http_client.close()
        config.set_option("client.toolbarMode", old_toolbar)


def test_resume_selects_url_agent_before_loading_history(page):
    app, client = page
    app.query_params.update(agent="second-agent", thread_id="saved-thread", user_id="untrusted-user")
    app.run()

    assert not app.exception
    assert client.history_calls == [("second-agent", "saved-thread", "configured-user")]
    assert app.selectbox[0].value == "second-agent"
    assert app.chat_message[0].markdown[0].value == "Saved reply from second-agent: saved-thread"


def test_new_chat_updates_url_and_agent_change_starts_a_new_conversation(page):
    app, client = page
    app.query_params.update(agent="first-agent", thread_id="saved-thread")
    app.run()
    app.button[0].click().run()

    assert not app.exception
    first_new_thread = app.session_state.thread_id
    assert first_new_thread != "saved-thread"
    assert app.query_params["thread_id"] == [first_new_thread]
    assert "Saved reply" not in app.chat_message[0].markdown[0].value

    app.selectbox[0].select("second-agent").run()
    assert not app.exception
    assert app.query_params["agent"] == ["second-agent"]
    assert app.session_state.thread_id != first_new_thread
    assert app.query_params["thread_id"] == [app.session_state.thread_id]
    assert client.agent == "second-agent"
    assert len(client.history_calls) == 1


def test_url_navigation_reloads_history_in_an_existing_session(page):
    app, client = page
    app.query_params.update(agent="first-agent", thread_id="first-thread")
    app.run()
    app.query_params.update(agent="second-agent", thread_id="second-thread")
    app.run()

    assert not app.exception
    assert app.selectbox[0].value == "second-agent"
    assert app.chat_message[0].markdown[0].value == "Saved reply from second-agent: second-thread"
    assert len(client.history_calls) == 2


def test_unknown_url_agent_stops_before_fetching_history(page):
    app, client = page
    app.query_params.update(agent="missing-agent", thread_id="saved-thread")
    app.run()

    assert not app.exception
    assert "not available" in app.error[0].value
    assert not client.history_calls
    assert not app.chat_input


def test_token_mode_does_not_take_user_identity_from_url(page, monkeypatch):
    app, client = page
    monkeypatch.setattr(settings, "AUTH_MODE", "token")
    app.query_params.update(agent="second-agent", thread_id="saved-thread", user_id="another-user")
    app.run()

    assert not app.exception
    assert client.history_calls == [("second-agent", "saved-thread", None)]


def test_history_failure_does_not_display_or_continue_another_conversation(page):
    app, client = page
    app.query_params.update(thread_id="first-thread")
    app.run()

    def failed_history(**kwargs):
        raise AgentClientError("Service unavailable")

    client.get_history = failed_history
    app.query_params["thread_id"] = "unavailable-thread"
    app.run()

    assert not app.exception
    assert "Could not load" in app.error[0].value
    assert not app.chat_message
    assert not app.chat_input


def test_resume_loads_all_history_pages_in_order(page):
    app, client = page
    messages = [ChatMessage(type="ai", content=f"Saved reply {index}") for index in range(205)]
    for offset in (0, 100, 200):
        client.history_pages[("long-thread", offset)] = ChatHistory(
            messages=messages[offset : offset + 100],
            total=len(messages),
            next_offset=offset + 100 if offset < 200 else None,
        )
    app.query_params.update(agent="second-agent", thread_id="long-thread")
    app.run()

    assert not app.exception
    assert not app.error
    assert client.history_page_calls == [
        ("second-agent", "long-thread", "configured-user", offset) for offset in (0, 100, 200)
    ]
    assert app.session_state.messages == messages
    assert [element.value for element in app.chat_message[0].markdown] == [message.content for message in messages]
    assert app.chat_input


def test_later_history_page_failure_hides_partial_and_previous_conversations(page):
    app, client = page
    app.query_params.update(thread_id="previous-thread")
    app.run()
    previous_messages = list(app.session_state.messages)
    previous_conversation = app.session_state.conversation
    client.history_pages[("unavailable-thread", 0)] = ChatHistory(
        messages=[ChatMessage(type="ai", content="Partial reply")], total=2, next_offset=1
    )
    client.history_pages[("unavailable-thread", 1)] = httpx.Response(503, json={"detail": "Service unavailable"})
    app.query_params["thread_id"] = "unavailable-thread"
    app.run()

    assert not app.exception
    assert "Could not load" in app.error[0].value
    assert client.history_page_calls[-2:] == [
        ("first-agent", "unavailable-thread", "configured-user", offset) for offset in (0, 1)
    ]
    assert app.session_state.messages == previous_messages
    assert app.session_state.conversation == previous_conversation
    assert not app.chat_message
    assert not app.chat_input


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:8501/client/chat", "https://chat.example/client/chat"])
def test_share_uses_current_public_url_and_encodes_conversation(page, monkeypatch, base_url):
    app, _ = page
    monkeypatch.setattr(ContextProxy, "url", property(lambda self: base_url))
    app.query_params.update(agent="second-agent", thread_id="thread & one", user_id="do-not-share")
    app.run()
    next(button for button in app.button if "Share/resume" in button.label).click().run()

    assert not app.exception
    text = "\n".join(element.value for element in app.markdown)
    assert f"{base_url}?agent=second-agent&thread_id=thread+%26+one" in text
    assert "do-not-share" not in text


def test_share_reports_missing_url_without_using_private_session_state(page, monkeypatch):
    app, _ = page
    monkeypatch.setattr(ContextProxy, "url", property(lambda self: None))
    app.run()
    next(button for button in app.button if "Share/resume" in button.label).click().run()

    assert not app.exception
    assert any("Could not determine the app URL" in error.value for error in app.error)


def test_feedback_sends_the_proof_from_the_generated_message(page, monkeypatch):
    app, client = page
    monkeypatch.setattr(settings, "AUTH_MODE", "token")
    feedback_calls = []

    async def stream(**kwargs):
        yield ChatMessage(type="ai", content="New answer", run_id="generated-run", feedback_token="server-proof")

    async def record_feedback(**kwargs):
        feedback_calls.append(kwargs)

    client.astream = stream
    client.acreate_feedback = record_feedback
    app.run()
    app.chat_input[0].set_value("A new question").run()
    assert not app.exception
    app.feedback[0].set_value(4).run()
    assert not app.exception
    assert feedback_calls == [
        {
            "run_id": "generated-run",
            "key": "human-feedback-stars",
            "score": 1.0,
            "kwargs": {"comment": "In-line human feedback"},
            "user_id": None,
            "feedback_token": "server-proof",
        }
    ]
    app.run()
    assert len(feedback_calls) == 1
