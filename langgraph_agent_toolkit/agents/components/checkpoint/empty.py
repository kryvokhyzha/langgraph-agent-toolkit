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
    """A fake checkpointer that implements the BaseCheckpointSaver interface but doesn't actually persist anything.

    All operations are no-ops.
    This is useful for:
    - Testing without persistence
    - Running graphs without memory between executions
    - Situations where you want the graph structure but no state saving
    """

    def __init__(self, *, serde: Optional[SerializerProtocol] = None) -> None:
        super().__init__(serde=serde)

    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        """Return None - no checkpoint exists."""
        return None

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Return None - no checkpoint tuple exists."""
        return None

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Return an empty iterator - no checkpoints exist."""
        return iter([])

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Pretend to save but not actually persist anything.

        Return a config with a fake checkpoint_id to satisfy the interface.
        """
        # Generate a fake checkpoint ID to return
        fake_checkpoint_id = str(uuid.uuid4())

        # Return the config with the fake checkpoint_id
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
        """Do nothing - no writes are persisted."""
        pass

    # Async versions (required for async graph execution)
    async def aget(self, config: RunnableConfig) -> Optional[Checkpoint]:
        """Async version - always return None."""
        return None

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Async version - always return None."""
        return None

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Async version - always return an empty iterator."""
        return
        yield  # Makes this an async generator that yields nothing

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async version - pretend to save but not persist."""
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async version - do nothing."""
        pass
