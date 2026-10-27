"""Test MCP questions through the real Streamlit page."""

from types import SimpleNamespace

import pytest
from streamlit import config
from streamlit.testing.v1 import AppTest

from langgraph_agent_toolkit.client import AgentClientError
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.schema import ChatMessage
from langgraph_agent_toolkit.ui import main_page


def question(identifier="one", *, schema=None, mode="form", key="details"):
    request = {"key": key, "message": f"Question {identifier}", "mode": mode}
    if mode == "url":
        request["url"] = "https://example.com/authorize"
    else:
        request["requested_schema"] = schema or {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
    return {"id": identifier, "value": {"type": "mcp_elicitation", "tool_name": "remote", "requests": [request]}}


def paused(*questions):
    return ChatMessage(type="ai", content="Please answer.", custom_data={"interrupts": list(questions)})


@pytest.fixture
def page(monkeypatch):
    """Use the normal page with an isolated, deterministic client."""
    client = SimpleNamespace(
        agent="first-agent",
        info=SimpleNamespace(default_agent="first-agent", agents=[SimpleNamespace(key="first-agent")]),
        pending=[question()],
        calls=[],
        result=ChatMessage(type="ai", content="Complete."),
        history_calls=[],
    )

    def history(**kwargs):
        client.history_calls.append(kwargs)
        return SimpleNamespace(messages=[], next_offset=None, pending_interrupts=client.pending)

    async def invoke(**kwargs):
        client.calls.append(("invoke", kwargs))
        if isinstance(client.result, Exception):
            raise client.result
        return client.result

    async def stream(protocol, **kwargs):
        client.calls.append((protocol, kwargs))
        if isinstance(client.result, Exception):
            raise client.result
        yield client.result

    async def close():
        pass

    client.get_history = history
    client.ainvoke = invoke
    client.astream = lambda **kwargs: stream("sse", **kwargs)
    client.astream_jsonl = lambda **kwargs: stream("jsonl", **kwargs)
    client.aclose = close
    monkeypatch.setattr(main_page, "AgentClient", lambda **kwargs: client)
    monkeypatch.setattr(settings, "AUTH_MODE", "trusted")
    monkeypatch.setattr(settings, "DEFAULT_STREAMLIT_USER_ID", "test-user")
    old_toolbar = config.get_option("client.toolbarMode")
    config.set_option("client.toolbarMode", "minimal")
    app = AppTest.from_string(
        "import asyncio\nfrom langgraph_agent_toolkit.ui.main_page import main_page\nasyncio.run(main_page())"
    )
    app.query_params.update(thread_id="saved-thread")
    try:
        yield app, client
    finally:
        config.set_option("client.toolbarMode", old_toolbar)


def submit(app):
    next(button for button in app.button if button.label == "Send answers").click().run()
    assert not app.exception


@pytest.mark.parametrize("protocol", ["invoke", "sse", "jsonl"])
def test_history_questions_submit_through_each_transport(page, protocol):
    app, client = page
    app.run()
    assert not app.exception
    assert app.chat_input[0].disabled
    if protocol == "invoke":
        next(toggle for toggle in app.toggle if toggle.label == "Stream results").set_value(False).run()
    elif protocol == "jsonl":
        app.radio[0].set_value("JSON Lines").run()
    app.text_input[0].set_value("chosen value")
    submit(app)

    assert client.calls == [
        (
            protocol,
            {
                "input": {"responses": {"details": {"action": "accept", "content": {"answer": "chosen value"}}}},
                "thread_id": "saved-thread",
                "user_id": "test-user",
            },
        )
    ]
    assert not app.chat_input[0].disabled
    assert app.session_state.pending_interrupts == []
    assert not app.text_input


@pytest.mark.parametrize("action", ["decline", "cancel"])
def test_decline_and_cancel_do_not_require_form_values(page, action):
    app, client = page
    client.pending = [
        question(schema={"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]})
    ]
    app.run()
    next(widget for widget in app.selectbox if widget.label == "Action").select(action)
    submit(app)
    assert client.calls[0][1]["input"] == {"responses": {"details": {"action": action}}}


def test_parallel_questions_route_answers_by_interrupt_id(page):
    app, client = page
    client.pending = [question("one"), question("two")]
    app.run()
    app.text_input[0].set_value("first")
    app.text_input[1].set_value("second")
    submit(app)
    assert client.calls[0][1]["input"] == {
        "resume": {
            "one": {"responses": {"details": {"action": "accept", "content": {"answer": "first"}}}},
            "two": {"responses": {"details": {"action": "accept", "content": {"answer": "second"}}}},
        }
    }


def test_url_question_has_explicit_accept_without_form_content(page):
    app, client = page
    client.pending = [question(mode="url")]
    app.run()
    assert not app.exception
    assert "Complete the requested page" in app.caption[0].value
    assert not app.text_input
    submit(app)
    assert client.calls[0][1]["input"] == {"responses": {"details": {"action": "accept"}}}


@pytest.mark.parametrize("url", ["http://[", "javascript:alert(1)"])
def test_invalid_or_non_web_url_is_plain_text_and_can_be_declined(page, url):
    app, client = page
    client.pending = [question(mode="url")]
    client.pending[0]["value"]["requests"][0]["url"] = url
    app.run()
    assert not app.exception
    assert app.code[0].value == url
    next(widget for widget in app.selectbox if widget.label == "Action").select("decline")
    submit(app)
    assert client.calls[0][1]["input"] == {"responses": {"details": {"action": "decline"}}}


@pytest.mark.parametrize("protocol", ["invoke", "sse", "jsonl"])
def test_live_question_and_next_round_replace_form_state(page, protocol):
    app, client = page
    client.pending = []
    client.result = paused(question())
    app.run()
    if protocol == "invoke":
        next(toggle for toggle in app.toggle if toggle.label == "Stream results").set_value(False).run()
    elif protocol == "jsonl":
        app.radio[0].set_value("JSON Lines").run()
    app.chat_input[0].set_value("Begin").run()
    assert not app.exception
    assert app.chat_input[0].disabled
    app.text_input[0].set_value("first answer")
    client.result = paused(question(key="second"))
    submit(app)
    assert app.text_input[0].value == ""
    assert app.session_state.pending_interrupts[0]["value"]["requests"][0]["key"] == "second"
    app.text_input[0].set_value("second answer")
    client.result = ChatMessage(type="ai", content="Complete.")
    submit(app)
    assert client.calls[-1][1]["input"] == {
        "responses": {"second": {"action": "accept", "content": {"answer": "second answer"}}}
    }
    assert not app.chat_input[0].disabled


def test_repeated_question_in_a_later_round_clears_old_answers(page):
    app, client = page
    app.run()
    client.result = paused(question())
    app.text_input[0].set_value("previous answer")
    submit(app)
    assert app.text_input[0].value == ""
    assert len(client.calls) == 1


def test_one_interrupt_can_request_several_answers_and_include_optional_values(page):
    app, client = page
    client.pending[0]["value"]["requests"].append(question(key="extra")["value"]["requests"][0])
    client.pending[0]["value"]["requests"][0]["requested_schema"]["properties"]["action"] = {"type": "string"}
    app.run()
    next(widget for widget in app.checkbox if widget.label == "Include action").check()
    app.text_input[0].set_value("first")
    app.text_input[1].set_value("an optional field named action")
    app.text_input[2].set_value("second")
    submit(app)
    assert client.calls[0][1]["input"] == {
        "responses": {
            "details": {"action": "accept", "content": {"answer": "first", "action": "an optional field named action"}},
            "extra": {"action": "accept", "content": {"answer": "second"}},
        }
    }


def test_lost_resume_reply_reloads_current_checkpoint_questions(page):
    app, client = page
    app.run()
    client.result = AgentClientError("Reply lost")
    app.text_input[0].set_value("sent once")
    submit(app)
    assert "conversation" not in app.session_state
    assert len(client.calls) == 1
    client.pending = [question("next-round", key="followup")]
    app.run()
    assert not app.exception
    assert len(client.history_calls) == 2
    assert app.text_input[0].value == ""
    assert app.session_state.pending_interrupts == client.pending
    assert len(client.calls) == 1


def test_navigation_and_user_change_do_not_keep_old_questions_or_answers(page, monkeypatch):
    app, client = page
    app.run()
    app.text_input[0].set_value("private answer").run()
    monkeypatch.setattr(settings, "DEFAULT_STREAMLIT_USER_ID", "other-user")
    app.run()
    assert not app.exception
    assert app.text_input[0].value == ""
    client.pending = []
    app.query_params["thread_id"] = "other-thread"
    app.run()
    assert not app.exception
    assert not app.text_input
    assert not app.chat_input[0].disabled


def test_basic_field_types_optional_values_and_arrays(page):
    app, client = page
    client.pending = [
        question(
            schema={
                "type": "object",
                "properties": {
                    "choice": {"type": "string", "enum": ["a", "b"]},
                    "enabled": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "tags": {"type": "array", "items": {"type": "string", "enum": ["x", "y"]}},
                    "names": {"type": "array", "items": {"type": "string"}},
                    "note": {"type": "string"},
                },
                "required": ["choice", "enabled", "count", "ratio", "tags", "names"],
            }
        )
    ]
    app.run()
    next(widget for widget in app.selectbox if widget.label == "choice").select("b")
    next(widget for widget in app.checkbox if widget.label == "enabled").check()
    next(widget for widget in app.text_input if widget.label == "count").set_value("9007199254740993")
    next(widget for widget in app.text_input if widget.label == "ratio").set_value("0.25")
    app.multiselect[0].set_value(["y"])
    app.text_area[0].set_value('["first", "second"]')
    submit(app)
    assert client.calls[0][1]["input"]["responses"]["details"]["content"] == {
        "choice": "b",
        "enabled": True,
        "count": 9007199254740993,
        "ratio": 0.25,
        "tags": ["y"],
        "names": ["first", "second"],
    }


def test_unsupported_schema_uses_json_and_invalid_input_does_not_send(page):
    app, client = page
    client.pending = [
        question(
            schema={
                "type": "object",
                "properties": {"answer": {"type": ["string", "null"]}},
                "required": ["answer"],
            }
        )
    ]
    app.run()
    assert not app.exception
    assert app.text_area[0].label == "Answers as JSON"
    app.text_area[0].set_value('{"answer":')
    submit(app)
    assert not client.calls
    assert any("valid JSON" in error.value for error in app.error)
    app.text_area[0].set_value('{"answer": null}')
    submit(app)
    assert client.calls[0][1]["input"]["responses"]["details"]["content"] == {"answer": None}


@pytest.mark.parametrize("kind,raw", [("integer", "1.5"), ("number", "NaN"), ("array", '[1, "two"]')])
def test_invalid_required_values_do_not_resume_the_agent(page, kind, raw):
    app, client = page
    spec = {"type": kind}
    if kind == "array":
        spec["items"] = {"type": "string"}
    client.pending = [question(schema={"type": "object", "properties": {"value": spec}, "required": ["value"]})]
    app.run()
    widget = app.text_area[0] if kind == "array" else app.text_input[0]
    widget.set_value(raw)
    submit(app)
    assert not client.calls
    assert app.error


def test_mixed_interrupts_keep_the_other_resume_value(page):
    app, client = page
    client.pending.append({"id": "custom-question", "value": "Provide another answer."})
    app.run()
    app.text_input[0].set_value("MCP answer")
    app.text_area[0].set_value('{"message": "custom answer"}')
    submit(app)
    assert client.calls[0][1]["input"] == {
        "resume": {
            "one": {"responses": {"details": {"action": "accept", "content": {"answer": "MCP answer"}}}},
            "custom-question": {"message": "custom answer"},
        }
    }
