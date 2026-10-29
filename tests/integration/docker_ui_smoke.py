"""Run the installed Streamlit page against a disposable API container.

Set AGENT_URL and AUTH_SECRET for that API. Start the API with USE_FAKE_MODEL=true
and SQLite persistence. Mount this script in the app image and run it with Python.
This check uses real HTTP requests. It does not test browser rendering.
"""

import json
import os
from pathlib import Path
from uuid import uuid4


def check_page(app) -> None:
    """Stop when the page reports an error."""
    assert not app.exception, [item.message for item in app.exception]
    assert not app.error, [item.value for item in app.error]


def main() -> None:
    """Send messages with each protocol and restore their saved history."""
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    assert os.environ.get("AGENT_URL"), "Set AGENT_URL to the disposable API."
    assert os.environ.get("AUTH_SECRET"), "Set AUTH_SECRET to the disposable API token."

    from streamlit import config
    from streamlit.testing.v1 import AppTest

    import langgraph_agent_toolkit
    from langgraph_agent_toolkit.client import AgentClient
    from langgraph_agent_toolkit.core.settings import settings

    config.set_option("client.toolbarMode", "minimal")
    entrypoint = Path(langgraph_agent_toolkit.__file__).with_name("run_app.py")
    thread_id = f"docker-ui-{uuid4()}"
    agent = "chatbot-agent"
    user_id = settings.DEFAULT_STREAMLIT_USER_ID if settings.AUTH_MODE == "trusted" else None
    expected_reply = "This is a test response from the fake model."
    expected_messages = []
    protocols = ("SSE", "JSON Lines", "invoke")

    with AgentClient(base_url=os.environ["AGENT_URL"], agent=agent) as client:
        try:
            app = AppTest.from_file(str(entrypoint), default_timeout=30)
            app.query_params.update(agent=agent, thread_id=thread_id)
            app.run()
            check_page(app)
            assert app.session_state.messages == []
            for protocol in protocols:
                if protocol == "invoke":
                    next(item for item in app.toggle if item.label == "Stream results").set_value(False).run()
                else:
                    next(item for item in app.radio if item.label == "Streaming protocol").set_value(protocol).run()
                check_page(app)
                prompt = f"Check the container chat with {protocol}."
                app.chat_input[0].set_value(prompt).run()
                check_page(app)
                expected_messages.extend([prompt, expected_reply])
                assert [item.content for item in app.session_state.messages] == expected_messages
                assert expected_reply in [item.value for item in app.markdown]
                saved = client.get_history(thread_id, user_id=user_id)
                assert [item.content for item in saved.messages] == expected_messages

            restored = AppTest.from_file(str(entrypoint), default_timeout=30)
            restored.query_params.update(agent=agent, thread_id=thread_id)
            restored.run()
            check_page(restored)
            assert [item.content for item in restored.session_state.messages] == expected_messages
            assert [item.value for item in restored.markdown].count(expected_reply) == len(protocols)
            next(item for item in restored.button if "New Chat" in item.label).click().run()
            check_page(restored)
            assert restored.session_state.thread_id != thread_id
            assert all(item.content not in expected_messages for item in restored.session_state.messages)
        finally:
            client.clear_history(thread_id, user_id=user_id)
        assert not client.get_history(thread_id, user_id=user_id).messages

    print(
        json.dumps(
            {
                "ui_protocols": list(protocols),
                "saved_messages": len(expected_messages),
                "resume": True,
                "new_chat": True,
                "cleanup": True,
            }
        )
    )


if __name__ == "__main__":
    main()
