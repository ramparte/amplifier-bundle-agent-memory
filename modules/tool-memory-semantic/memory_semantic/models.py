"""Data models for semantic memory."""

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class Memory(BaseModel):
    """Minimal memory model for V1.0.

    Design decisions:
    - id: Auto-generated UUID4 (Qdrant requires valid UUIDs)
    - agent_id: For namespace isolation
    - content: The actual memory text
    - timestamp: For recency ranking
    - embedding: Vector for semantic search
    - tags: Optional categorization
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: list[float]
    tags: list[str] = []

    def dict_for_storage(self) -> dict:
        """Return dict without embedding for Qdrant payload.

        Embedding is stored separately in Qdrant's vector field.
        Payload contains searchable metadata only.
        """
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }
