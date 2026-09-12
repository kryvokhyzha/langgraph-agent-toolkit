import uuid
from typing import Any, Iterator, Optional, Sequence, Tuple

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.base import SerializerProtocol


class NoOpSaver(BaseCheckpointSaver):
    """A `BaseCheckpointSaver` that does not persist checkpoints."""

    def __init__(self, *, serde: Optional[SerializerProtocol] = None) -> None:
        super().__init__(serde=serde)

    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        """Return `None` because no checkpoint exists."""
        return None

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Return `None` because no checkpoint tuple exists."""
        return None

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Return an empty iterator because no checkpoints exist."""
        return iter([])

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Return a configuration with a generated checkpoint ID."""
        fake_checkpoint_id = str(uuid.uuid4())

        return {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "checkpoint_id": fake_checkpoint_id,
                "checkpoint_ns": "",
            },
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Do not persist writes."""
        pass

    async def aget(self, config: RunnableConfig) -> Optional[Checkpoint]:
        """Return `None` asynchronously."""
        return None

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Return `None` asynchronously."""
        return None

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Return an empty iterator asynchronously."""
        return
        yield

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Return a configuration without persisting a checkpoint."""
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Do not persist writes asynchronously."""
        pass
