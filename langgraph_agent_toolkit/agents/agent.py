from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.func import Pregel

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability


@dataclass
class Agent:
    name: str
    description: str
    graph: CompiledStateGraph | Pregel
    observability: BaseObservabilityPlatform = EmptyObservability()
