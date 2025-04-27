from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.agents.components.tools import add, multiply
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.core.models.factory import ModelFactory
from langgraph_agent_toolkit.schema.models import ModelProvider


model = ModelFactory.create(
    model_provider=ModelProvider.OPENAI,
    model_name=settings.OPENAI_MODEL_NAME,
    openai_api_base=settings.OPENAI_API_BASE_URL,
    openai_api_key=settings.OPENAI_API_KEY,
)

math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time.",
).with_config(tags=["skip_stream"])

research_agent = create_react_agent(
    model=model,
    tools=[DuckDuckGoSearchResults()],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math.",
).with_config(tags=["skip_stream"])

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    ),
    add_handoff_back_messages=False,
)

supervisor_agent = Agent(
    name="supervisor-agent",
    description="A langgraph supervisor agent",
    graph=workflow.compile(checkpointer=MemorySaver()),
)
