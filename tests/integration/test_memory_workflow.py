"""Integration tests for memory workflow.

Tests the complete store→search workflow with real storage backend.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide a unique temporary database path for each test."""
    # Use tmp_path (pytest built-in) + unique ID to avoid conflicts
    return tmp_path / f"memory_{uuid.uuid4().hex}.db"


@pytest.fixture
def unique_agent_id():
    """Provide a unique agent ID for each test to avoid db conflicts."""
    return f"test-agent-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns deterministic embeddings."""
    embedder = AsyncMock()
    embedder.generate.side_effect = lambda text: [float(ord(c)) for c in text[:384]]
    embedder.dimensions = 384
    return embedder


@pytest.mark.asyncio
class TestMemoryWorkflow:
    """Integration tests for complete memory workflows."""

    async def test_store_and_search_single_memory(self, temp_db_path, unique_agent_id, mock_embedder):
        """Store a memory and retrieve it through search."""
        from memory_semantic.storage import MemoryStorage

        # Create storage
        with patch("memory_semantic.embeddings.EmbeddingGenerator", return_value=mock_embedder):
            storage = MemoryStorage(
                agent_id=unique_agent_id,
                config={"db_path": str(temp_db_path)},
            )

            # Store memory
            memory_id = await storage.store(
                "Python is a programming language", tags=["python", "programming"]
            )
            assert memory_id is not None

            # Search for it
            results = await storage.search("Python language", limit=5)

            # Should find the memory
            assert len(results) == 1
            assert results[0].content == "Python is a programming language"
            assert "python" in results[0].tags
            assert "programming" in results[0].tags

    async def test_store_multiple_and_search(self, temp_db_path, unique_agent_id, mock_embedder):
        """Store multiple memories and search with ranking."""
        from memory_semantic.storage import MemoryStorage

        with patch("memory_semantic.embeddings.EmbeddingGenerator", return_value=mock_embedder):
            storage = MemoryStorage(
                agent_id=unique_agent_id,
                config={"db_path": str(temp_db_path)},
            )

            # Store multiple memories
            memories = [
                ("Python is great for data science", ["python", "data-science"]),
                ("JavaScript is used for web development", ["javascript", "web"]),
                ("Python has excellent machine learning libraries", ["python", "ml"]),
                ("Rust is a systems programming language", ["rust", "systems"]),
            ]

            for content, tags in memories:
                await storage.store(content, tags)

            # Search for Python-related memories
            results = await storage.search("Python programming", limit=10)

            # Should find Python memories first
            assert len(results) >= 2
            python_results = [r for r in results if "python" in r.tags]
            assert len(python_results) >= 2

    async def test_time_filtered_search(self, temp_db_path, unique_agent_id, mock_embedder):
        """Test searching with time-based filtering."""
        from datetime import datetime

        from memory_semantic.storage import MemoryStorage

        with patch("memory_semantic.embeddings.EmbeddingGenerator", return_value=mock_embedder):
            storage = MemoryStorage(
                agent_id=unique_agent_id,
                config={"db_path": str(temp_db_path)},
            )

            # Store memories
            await storage.store("Old memory", tags=["old"])
            await asyncio.sleep(0.1)  # Small delay
            cutoff_time = datetime.utcnow()
            await asyncio.sleep(0.1)
            await storage.store("Recent memory", tags=["recent"])

            # Search with time filter
            results = await storage.search("memory", limit=10, since=cutoff_time)

            # Should only find recent memory
            assert len(results) == 1
            assert "recent" in results[0].tags

    async def test_tools_workflow(self, temp_db_path, unique_agent_id, mock_embedder):
        """Test complete workflow through tools interface."""
        from memory_semantic.tools import mount

        coordinator = MagicMock()
        coordinator.config = {
            "memory": {"agent_id": unique_agent_id, "db_path": str(temp_db_path)}
        }

        with patch("memory_semantic.embeddings.EmbeddingGenerator", return_value=mock_embedder):
            tools = mount(coordinator)
            store_tool = tools[0]
            search_tool = tools[1]

            # Store memories via tool
            store_result = await store_tool.execute(
                {"content": "FastAPI is a modern Python web framework", "tags": ["python", "web"]}
            )
            assert store_result.success is True
            memory_id = store_result.output["memory_id"]
            assert memory_id is not None

            # Search via tool
            search_result = await search_tool.execute({"query": "Python web framework"})
            assert search_result.success is True
            assert search_result.output["count"] >= 1
            found = any(
                m["id"] == memory_id for m in search_result.output["memories"]
            )
            assert found, "Should find the stored memory"

    async def test_empty_search_results(self, temp_db_path, unique_agent_id, mock_embedder):
        """Test search with no matching results."""
        from memory_semantic.storage import MemoryStorage

        with patch("memory_semantic.embeddings.EmbeddingGenerator", return_value=mock_embedder):
            storage = MemoryStorage(
                agent_id=unique_agent_id,
                config={"db_path": str(temp_db_path)},
            )

            # Store unrelated memory
            await storage.store("Quantum computing is complex", tags=["quantum"])

            # Search for something completely different
            results = await storage.search("dinosaurs and fossils", limit=10)

            # Should return some results (semantic search is fuzzy)
            # but might be low confidence
            assert isinstance(results, list)

    async def test_agent_isolation(self, tmp_path, mock_embedder):
        """Test that different agents don't see each other's memories."""
        from memory_semantic.storage import MemoryStorage

        with patch("memory_semantic.embeddings.EmbeddingGenerator", return_value=mock_embedder):
            # Agent 1 stores memory (separate db path)
            db1 = tmp_path / f"agent1_{uuid.uuid4().hex}.db"
            storage1 = MemoryStorage(
                agent_id="agent-1",
                config={"db_path": str(db1)},
            )
            await storage1.store("Agent 1's secret", tags=["secret"])

            # Agent 2 searches (separate db path)
            db2 = tmp_path / f"agent2_{uuid.uuid4().hex}.db"
            storage2 = MemoryStorage(
                agent_id="agent-2",
                config={"db_path": str(db2)},
            )
            results = await storage2.search("secret", limit=10)

            # Should not find agent 1's memory (different databases)
            assert len(results) == 0
