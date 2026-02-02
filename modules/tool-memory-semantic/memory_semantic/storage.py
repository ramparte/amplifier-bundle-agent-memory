"""Qdrant-based semantic storage with agent isolation."""

import os
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter

from .models import Memory
from .embeddings import EmbeddingGenerator


class MemoryStorage:
    """Semantic memory storage with agent namespace isolation.

    Design decisions:
    - Qdrant embedded mode (local file storage)
    - One database file per agent_id for isolation
    - Simple cosine similarity + 20% recency boost
    - Agent ID validation prevents path traversal
    """

    AGENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

    def __init__(self, agent_id: str, config: Optional[dict] = None):
        """Initialize storage for specific agent.

        Args:
            agent_id: Agent identity for namespace isolation
            config: Optional configuration:
                - storage_root: Base directory (default: ~/.amplifier/memory)
                - embedding_model: OpenAI model (default: text-embedding-3-small)
                - max_memories_per_agent: Memory limit (default: 10000)

        Raises:
            ValueError: If agent_id is invalid (path traversal attempt)
        """
        # Validate agent_id for security
        if not self.AGENT_ID_PATTERN.match(agent_id):
            raise ValueError(
                f"Invalid agent_id: {agent_id}. "
                "Only alphanumeric, underscore, and hyphen allowed."
            )

        self.agent_id = agent_id
        config = config or {}

        # Storage paths
        storage_root = config.get(
            "storage_root", os.path.expanduser("~/.amplifier/memory")
        )
        self.storage_path = Path(storage_root) / agent_id
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize Qdrant client (embedded mode)
        db_path = str(self.storage_path / "qdrant.db")
        self.client = QdrantClient(path=db_path)

        # Initialize embeddings
        embedding_model = config.get("embedding_model", "text-embedding-3-small")
        self.embeddings = EmbeddingGenerator(model=embedding_model)

        # Memory limits
        self.max_memories = config.get("max_memories_per_agent", 10000)

        # Collection name
        self.collection = f"memories_{agent_id}"

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection not in collection_names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.embeddings.dimensions, distance=Distance.COSINE
                ),
            )

    async def store(self, content: str, tags: Optional[list[str]] = None) -> str:
        """Store memory with embedding.

        Args:
            content: Memory text
            tags: Optional tags for categorization

        Returns:
            Memory ID

        Raises:
            ValueError: If memory limit exceeded
            Exception: If storage fails
        """
        # Check memory limit
        current_count = self.count()
        if current_count >= self.max_memories:
            raise ValueError(
                f"Memory limit reached: {current_count}/{self.max_memories}. "
                "Delete old memories before adding new ones."
            )

        # Generate embedding
        embedding = await self.embeddings.generate(content)

        # Create memory
        memory = Memory(
            agent_id=self.agent_id,
            content=content,
            embedding=embedding,
            tags=tags or [],
        )

        # Store to Qdrant
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=memory.id,
                    vector=memory.embedding,
                    payload=memory.dict_for_storage(),
                )
            ],
        )

        return memory.id

    async def search(
        self, query: str, limit: int = 5, since: Optional[datetime] = None
    ) -> list[Memory]:
        """Search memories semantically.

        Args:
            query: Natural language query
            limit: Max results to return
            since: Only return memories after this timestamp

        Returns:
            List of memories ranked by relevance
        """
        # Generate query embedding
        query_embedding = await self.embeddings.generate(query)

        # Build filter
        query_filter = None
        if since:
            query_filter = Filter(
                must=[{"key": "timestamp", "range": {"gte": since.isoformat()}}]
            )

        # Search (over-fetch for ranking)
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=limit * 2,
            query_filter=query_filter,
            with_payload=True,
        ).points

        # Apply simple ranking
        ranked = self._rank_results(results)

        return ranked[:limit]

    def _rank_results(self, results) -> list[Memory]:
        """Simple similarity + recency boost.

        Applies 20% boost to memories from last 7 days.
        """
        from datetime import timezone

        scored = []
        for result in results:
            # Reconstruct memory (embedding not needed for return)
            payload = result.payload
            memory = Memory(
                id=payload["id"],
                agent_id=payload["agent_id"],
                content=payload["content"],
                timestamp=datetime.fromisoformat(payload["timestamp"]),
                embedding=[],  # Don't return embedding to save tokens
                tags=payload.get("tags", []),
            )

            # Calculate score
            score = result.score  # Cosine similarity

            # Recency boost (20% for last 7 days)
            days_old = (datetime.now(timezone.utc) - memory.timestamp).days
            if days_old < 7:
                score *= 1.2

            scored.append((score, memory))

        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored]

    async def get(self, memory_id: str) -> Optional[Memory]:
        """Get memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory if found, None otherwise
        """
        results = self.client.retrieve(collection_name=self.collection, ids=[memory_id])

        if not results:
            return None

        result = results[0]
        payload = result.payload

        return Memory(
            id=payload["id"],
            agent_id=payload["agent_id"],
            content=payload["content"],
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            embedding=[],
            tags=payload.get("tags", []),
        )

    def count(self) -> int:
        """Count total memories for this agent.

        Returns:
            Number of memories stored
        """
        collection_info = self.client.get_collection(self.collection)
        return collection_info.points_count
