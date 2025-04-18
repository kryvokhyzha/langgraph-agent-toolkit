from dataclasses import dataclass

from langgraph.func import Pregel
from langgraph.graph.state import CompiledStateGraph

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability


@dataclass
class Agent:
    name: str
    description: str
    graph: CompiledStateGraph | Pregel
    # prompt_manager: None = None
    observability: BaseObservabilityPlatform = EmptyObservability()
