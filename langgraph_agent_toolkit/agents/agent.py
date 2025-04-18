from dataclasses import dataclass

from langgraph.func import Pregel
from langgraph.graph.state import CompiledStateGraph

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform


@dataclass
class Agent:
    name: str
    description: str
    graph: CompiledStateGraph | Pregel
    observability: BaseObservabilityPlatform | None = None
