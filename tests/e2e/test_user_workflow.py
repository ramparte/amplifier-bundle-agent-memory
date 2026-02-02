"""End-to-end tests simulating real user workflows.

These tests simulate how users would actually interact with the memory system
through Amplifier sessions.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture
def mock_coordinator(tmp_path):
    """Mock Amplifier coordinator with memory configuration."""
    coordinator = MagicMock()
    agent_id = f"test-agent-{uuid.uuid4().hex[:8]}"
    coordinator.config = {
        "memory": {
            "agent_id": agent_id,
            "db_path": str(tmp_path / f"{agent_id}.db"),
        }
    }
    return coordinator


@pytest.fixture
def mock_embedder():
    """Mock embedder for testing."""
    embedder = AsyncMock()
    embedder.generate.side_effect = lambda text: [float(ord(c)) for c in text[:384]]
    embedder.dimensions = 384
    return embedder


@pytest.mark.asyncio
class TestUserWorkflow:
    """E2E tests for complete user workflows."""

    async def test_new_user_first_memory(self, mock_coordinator, mock_embedder):
        """Simulate a new user storing their first memory."""
        from memory_semantic.tools import mount

        with patch(
            "memory_semantic.embeddings.EmbeddingGenerator", return_value=mock_embedder
        ):
            # User mounts the tool
            tools = mount(mock_coordinator)
            assert len(tools) == 2
            store_tool, search_tool = tools

            # User stores their first memory
            result = await store_tool.execute(
                {
                    "content": "I prefer using TypeScript for web development",
                    "tags": ["preference", "typescript", "web"],
                }
            )

            # Should succeed
            assert result.success is True
            assert "memory_id" in result.output
            memory_id = result.output["memory_id"]

            # User searches for their preference
            search_result = await search_tool.execute(
                {"query": "TypeScript preference"}
            )

            # Should find their memory
            assert search_result.success is True
            assert search_result.output["count"] == 1
            assert search_result.output["memories"][0]["id"] == memory_id

    async def test_multi_session_continuity(self, mock_coordinator, mock_embedder):
        """Simulate user across multiple sessions with memory persistence.
        
        Note: Due to Qdrant's single-client locking, we simulate session separation
        by using the same tools instance but treating calls as separate sessions.
        In real usage, sessions would be temporally separated.
        """
        from memory_semantic.tools import mount

        with patch(
            "memory_semantic.embeddings.EmbeddingGenerator", return_value=mock_embedder
        ):
            # Session 1: User learns a new pattern
            tools = mount(mock_coordinator)
            store_tool, search_tool = tools

            await store_tool.execute(
                {
                    "content": "Repository pattern separates data access from business logic",
                    "tags": ["pattern", "architecture"],
                }
            )

            # Session 2: User recalls the pattern (same tools, simulating later session)
            result = await search_tool.execute({"query": "repository pattern"})

            # Should find memory from previous "session"
            assert result.success is True
            assert result.output["count"] == 1
            assert "Repository pattern" in result.output["memories"][0]["content"]

    async def test_building_knowledge_base(self, mock_coordinator, mock_embedder):
        """Simulate user building a knowledge base over time."""
        from memory_semantic.tools import mount

        with patch(
            "memory_semantic.embeddings.EmbeddingGenerator", return_value=mock_embedder
        ):
            tools = mount(mock_coordinator)
            store_tool, search_tool = tools

            # User stores multiple related memories
            memories = [
                {
                    "content": "FastAPI uses Pydantic for request validation",
                    "tags": ["fastapi", "python", "validation"],
                },
                {
                    "content": "FastAPI automatically generates OpenAPI documentation",
                    "tags": ["fastapi", "documentation"],
                },
                {
                    "content": "Use dependency injection in FastAPI for testability",
                    "tags": ["fastapi", "testing", "best-practice"],
                },
            ]

            for memory in memories:
                result = await store_tool.execute(memory)
                assert result.success is True

            # User searches for FastAPI knowledge
            result = await search_tool.execute({"query": "FastAPI best practices"})

            # Should find multiple relevant memories
            assert result.success is True
            assert result.output["count"] >= 2

    async def test_error_recovery(self, mock_coordinator, mock_embedder):
        """Simulate user handling errors gracefully."""
        from memory_semantic.tools import mount

        with patch(
            "memory_semantic.embeddings.EmbeddingGenerator", return_value=mock_embedder
        ):
            tools = mount(mock_coordinator)
            store_tool, search_tool = tools

            # User tries to store empty content (error)
            result = await store_tool.execute({"content": ""})
            assert result.success is False
            assert "empty" in result.error["message"].lower()

            # User corrects and tries again
            result = await store_tool.execute({"content": "Corrected memory content"})
            assert result.success is True

            # User tries to search with empty query (error)
            result = await search_tool.execute({"query": ""})
            assert result.success is False

            # User corrects and tries again
            result = await search_tool.execute({"query": "corrected"})
            assert result.success is True

    async def test_memory_evolution(self, mock_coordinator, mock_embedder):
        """Simulate user's understanding evolving over time."""
        from memory_semantic.tools import mount

        with patch(
            "memory_semantic.embeddings.EmbeddingGenerator", return_value=mock_embedder
        ):
            tools = mount(mock_coordinator)
            store_tool, search_tool = tools

            # User's initial understanding
            await store_tool.execute(
                {
                    "content": "Microservices are always better than monoliths",
                    "tags": ["architecture", "microservices"],
                }
            )

            # User's refined understanding
            await store_tool.execute(
                {
                    "content": "Microservices add complexity; use only when team is large enough",
                    "tags": ["architecture", "microservices", "tradeoffs"],
                }
            )

            # User searches for microservices knowledge
            result = await search_tool.execute({"query": "microservices architecture"})

            # Should find both memories showing evolution
            assert result.success is True
            assert result.output["count"] == 2

    async def test_context_specific_recall(self, mock_coordinator, mock_embedder):
        """Simulate user recalling memories for specific contexts."""
        from memory_semantic.tools import mount

        with patch(
            "memory_semantic.embeddings.EmbeddingGenerator", return_value=mock_embedder
        ):
            tools = mount(mock_coordinator)
            store_tool, search_tool = tools

            # User stores memories for different contexts
            await store_tool.execute(
                {
                    "content": "Use pytest fixtures for test setup",
                    "tags": ["testing", "pytest"],
                }
            )
            await store_tool.execute(
                {
                    "content": "Use React hooks for state management",
                    "tags": ["react", "frontend"],
                }
            )
            await store_tool.execute(
                {
                    "content": "Use pytest-asyncio for async tests",
                    "tags": ["testing", "pytest", "async"],
                }
            )

            # User searches in testing context
            result = await search_tool.execute({"query": "pytest testing setup"})

            # Should find testing-related memories
            assert result.success is True
            pytest_memories = [
                m for m in result.output["memories"] if "pytest" in m["tags"]
            ]
            assert len(pytest_memories) >= 2

    async def test_large_memory_content(self, mock_coordinator, mock_embedder):
        """Simulate user storing larger code snippets or documentation."""
        from memory_semantic.tools import mount

        with patch(
            "memory_semantic.embeddings.EmbeddingGenerator", return_value=mock_embedder
        ):
            tools = mount(mock_coordinator)
            store_tool, search_tool = tools

            # User stores a larger memory (code snippet)
            large_content = """
            async def handle_request(request: Request) -> Response:
                # Validate input
                data = await request.json()
                validator = RequestValidator()
                validated = validator.validate(data)
                
                # Process
                result = await processor.process(validated)
                
                # Return response
                return Response(result)
            """.strip()

            result = await store_tool.execute(
                {"content": large_content, "tags": ["code", "async", "validation"]}
            )

            assert result.success is True

            # User recalls the pattern
            search_result = await search_tool.execute(
                {"query": "async request handling"}
            )

            assert search_result.success is True
            assert search_result.output["count"] == 1

    async def test_memory_limits(self, mock_coordinator, mock_embedder):
        """Test that memory enforces reasonable limits."""
        from memory_semantic.tools import mount

        with patch(
            "memory_semantic.embeddings.EmbeddingGenerator", return_value=mock_embedder
        ):
            tools = mount(mock_coordinator)
            store_tool, search_tool = tools

            # Try to store content that's too large
            huge_content = "x" * 10001
            result = await store_tool.execute({"content": huge_content})

            # Should fail with clear error
            assert result.success is False
            assert "too long" in result.error["message"].lower()

            # Try to request too many search results
            await store_tool.execute({"content": "Valid memory"})
            result = await search_tool.execute({"query": "memory", "limit": 100})

            # Should succeed but cap at 20
            assert result.success is True
            assert result.output["count"] <= 20
