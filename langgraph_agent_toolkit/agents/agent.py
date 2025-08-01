from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rootutils
from langgraph.func import Pregel

from langgraph_agent_toolkit.core.observability.base import BaseObservabilityPlatform


@dataclass
class Agent:
    name: str
    description: str
    graph: Pregel
    observability: BaseObservabilityPlatform | None = None


def draw_agent_graph(agent: Agent, image_path: Optional[str | Path] = None, **kwargs):
    if image_path:
        image_path = Path(image_path)
    else:
        path_to_root = rootutils.find_root(search_from=__file__, indicator=".project-root")

        image_path = path_to_root / "docs" / "visualizations"
        image_path.mkdir(exist_ok=True, parents=True)

        image_path = image_path / f"{agent.name}.png"

    graph_image = agent.graph.get_graph(xray=1).draw_mermaid_png(**kwargs)
    with open(image_path, "wb") as f:
        f.write(graph_image)


__all__ = ["Agent", "draw_agent_graph"]
