"""Interrupt agent built with native ``create_agent`` + ``HumanInTheLoopMiddleware`` (tool approval).

Before the agent runs a sensitive tool (here ``send_email``), ``HumanInTheLoopMiddleware`` pauses and
surfaces the pending call for human approval. The toolkit's service resumes it through the normal
interrupt flow — reply ``approve`` to run the tool, ``reject: <reason>`` to decline, or any other text
to send guidance back to the model (the executor maps the reply to a HITL decision; see
``agent_executor.build_resume_command``).

Contrast with ``blueprints/interrupt_agent`` — a hand-built ``StateGraph`` using raw ``interrupt()``.
This one gets human-in-the-loop purely by composing a middleware onto ``create_agent``.
"""

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.components.middlewares import (
    SanitizeHistoryMiddleware,
    TrimMessagesMiddleware,
)
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory
from langgraph_agent_toolkit.schema.models import ModelProvider


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email. This is a sensitive action that requires human approval before it runs."""
    return f"Email sent to {recipient} (subject: {subject!r})."


model = CompletionModelFactory.create(
    model_provider=ModelProvider.OPENAI,
    model_name=settings.OPENAI_MODEL_NAME,
    openai_api_base=settings.OPENAI_API_BASE_URL,
    openai_api_key=settings.OPENAI_API_KEY,
)

hitl_agent = Agent(
    name="hitl-agent",
    description="A create_agent assistant that requires human approval before sending email.",
    graph=create_agent(
        model=model,
        tools=[send_email],
        middleware=[
            # Pause for human approval before send_email runs. interrupt_on=True allows the full set
            # of decisions (approve / edit / reject / respond); the executor maps the user's reply.
            HumanInTheLoopMiddleware(
                interrupt_on={"send_email": True},
                description_prefix="The assistant wants to send an email and needs your approval",
            ),
            # Conversational hygiene: bound the model's view and repair any broken tool pairing.
            TrimMessagesMiddleware(),
            SanitizeHistoryMiddleware(),
        ],
        system_prompt=(
            "You are a helpful assistant that can send emails on the user's behalf using the "
            "send_email tool. Call the tool whenever the user asks to send an email."
        ),
        state_schema=AgentState,
        checkpointer=MemorySaver(),
    ),
)

__all__ = ["hitl_agent"]
