"""Unit tests for storage layer."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from memory_semantic.storage import MemoryStorage
from memory_semantic.models import Memory


@pytest.fixture
def temp_storage():
    """Create temporary storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"storage_root": tmpdir}
        storage = MemoryStorage("test-agent", config)
        yield storage


@pytest.mark.asyncio
class TestMemoryStorage:
    """Unit tests for storage layer."""

    def test_agent_id_validation_accepts_valid(self, temp_storage):
        """Valid agent IDs are accepted."""
        # Already created with "test-agent"
        assert temp_storage.agent_id == "test-agent"

    def test_agent_id_validation_rejects_path_traversal(self):
        """Path traversal attempts are rejected."""
        with pytest.raises(ValueError, match="Invalid agent_id"):
            MemoryStorage("../alice", {})

        with pytest.raises(ValueError, match="Invalid agent_id"):
            MemoryStorage("alice/../bob", {})

    def test_agent_id_validation_rejects_invalid_chars(self):
        """Invalid characters are rejected."""
        with pytest.raises(ValueError, match="Invalid agent_id"):
            MemoryStorage("alice/bob", {})

        with pytest.raises(ValueError, match="Invalid agent_id"):
            MemoryStorage("alice bob", {})

    def test_storage_path_isolation(self):
        """Each agent gets isolated storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"storage_root": tmpdir}

            alice_storage = MemoryStorage("alice", config)
            bob_storage = MemoryStorage("bob", config)

            # Different paths
            path1 = Path(tmpdir) / "alice"
            path2 = Path(tmpdir) / "bob"

            assert path1.exists()
            assert path2.exists()
            assert path1 != path2
            assert alice_storage.agent_id == "alice"
            assert bob_storage.agent_id == "bob"

    async def test_store_and_retrieve(self, temp_storage):
        """Can store and retrieve memory by ID."""
        memory_id = await temp_storage.store(content="Test memory", tags=["test"])

        # UUID4 format (36 chars with hyphens)
        assert len(memory_id) == 36
        assert memory_id.count("-") == 4

        # Retrieve
        memory = await temp_storage.get(memory_id)

        assert memory is not None
        assert memory.id == memory_id
        assert memory.content == "Test memory"
        assert memory.agent_id == "test-agent"
        assert "test" in memory.tags

    async def test_search_basic(self, temp_storage):
        """Basic semantic search works."""
        # Store memories
        await temp_storage.store("PostgreSQL is great for relational data", ["database"])
        await temp_storage.store("MongoDB is good for documents", ["database"])
        await temp_storage.store("Python is a programming language", ["language"])

        # Search
        results = await temp_storage.search("database systems", limit=2)

        assert len(results) <= 2
        assert all(isinstance(m, Memory) for m in results)
        # Should find database-related memories
        assert any("database" in m.tags for m in results)

    async def test_search_with_since_filter(self, temp_storage):
        """Search respects since timestamp filter."""
        import asyncio

        # Store old memory (simulate by waiting)
        await temp_storage.store("Old memory", [])

        # Wait
        await asyncio.sleep(0.5)

        # Store recent memory
        recent_id = await temp_storage.store("Recent memory", [])

        # Search with since filter (only recent)
        since = datetime.utcnow() - timedelta(seconds=0.3)
        results = await temp_storage.search("memory", since=since, limit=10)

        result_ids = [m.id for m in results]
        assert recent_id in result_ids

    async def test_count(self, temp_storage):
        """Count returns number of stored memories."""
        initial_count = temp_storage.count()

        await temp_storage.store("Memory 1", [])
        await temp_storage.store("Memory 2", [])

        assert temp_storage.count() == initial_count + 2
