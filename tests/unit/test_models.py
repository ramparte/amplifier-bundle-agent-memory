"""Unit tests for Memory data model."""

from datetime import datetime
from memory_semantic.models import Memory


class TestMemoryModel:
    """Unit tests for Memory data model."""

    def test_memory_creation_with_defaults(self):
        """Memory can be created with minimal fields."""
        import uuid

        memory = Memory(
            agent_id="test-agent", content="Test memory content", embedding=[0.1, 0.2, 0.3]
        )

        assert memory.agent_id == "test-agent"
        assert memory.content == "Test memory content"
        assert memory.embedding == [0.1, 0.2, 0.3]
        # ID should be valid UUID4 format
        uuid.UUID(memory.id)  # Raises ValueError if invalid
        assert isinstance(memory.timestamp, datetime)
        assert memory.tags == []

    def test_memory_with_tags(self):
        """Memory can include tags."""
        memory = Memory(
            agent_id="test-agent",
            content="PostgreSQL decision",
            embedding=[0.1],
            tags=["database", "decisions"],
        )

        assert memory.tags == ["database", "decisions"]

    def test_memory_dict_for_storage(self):
        """dict_for_storage excludes embedding."""
        memory = Memory(
            agent_id="test-agent",
            content="Test",
            embedding=[0.1, 0.2],
            tags=["tag1"],
        )

        storage_dict = memory.dict_for_storage()

        assert "id" in storage_dict
        assert "agent_id" in storage_dict
        assert "content" in storage_dict
        assert "timestamp" in storage_dict
        assert "tags" in storage_dict
        assert "embedding" not in storage_dict
        assert isinstance(storage_dict["timestamp"], str)  # ISO format

    def test_memory_id_uniqueness(self):
        """Each memory gets unique ID."""
        m1 = Memory(agent_id="test", content="A", embedding=[0.1])
        m2 = Memory(agent_id="test", content="B", embedding=[0.2])

        assert m1.id != m2.id

    def test_memory_serialization(self):
        """Memory can serialize to/from dict."""
        original = Memory(
            agent_id="test-agent",
            content="Test content",
            embedding=[0.1, 0.2],
            tags=["tag1"],
        )

        # To dict
        data = original.model_dump()

        # From dict
        restored = Memory(**data)

        assert restored.id == original.id
        assert restored.agent_id == original.agent_id
        assert restored.content == original.content
        assert restored.embedding == original.embedding
        assert restored.tags == original.tags
