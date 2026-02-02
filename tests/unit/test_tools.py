"""Unit tests for memory tools."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestToolMount:
    """Tests for tool mounting and configuration."""

    def test_mount_requires_agent_id(self):
        """Mount fails without agent_id in config."""
        from memory_semantic.tools import mount
        
        coordinator = MagicMock()
        coordinator.config = {}

        with pytest.raises(ValueError, match="agent_id"):
            mount(coordinator)

    def test_mount_returns_tools(self):
        """Mount returns list of Tool objects."""
        from memory_semantic.tools import mount
        
        coordinator = MagicMock()
        coordinator.config = {"memory": {"agent_id": "test-agent"}}

        with patch("memory_semantic.storage.MemoryStorage"):
            tools = mount(coordinator)

            assert len(tools) == 2
            assert tools[0].name == "memory_store"
            assert tools[1].name == "memory_search"


@pytest.mark.asyncio
class TestMemoryStoreTool:
    """Tests for memory_store tool."""

    async def test_store_success(self):
        """Successful storage returns memory ID."""
        from memory_semantic.tools import mount
        
        coordinator = MagicMock()
        coordinator.config = {"memory": {"agent_id": "test-agent"}}

        mock_storage = AsyncMock()
        mock_storage.store.return_value = "12345678-1234-5678-1234-567812345678"

        with patch("memory_semantic.storage.MemoryStorage", return_value=mock_storage):
            tools = mount(coordinator)
            store_tool = tools[0]

            result = await store_tool.execute({"content": "Test memory", "tags": ["test"]})

            assert result.success is True
            assert result.output["memory_id"] == "12345678-1234-5678-1234-567812345678"
            mock_storage.store.assert_called_once_with("Test memory", ["test"])

    async def test_store_validates_empty_content(self):
        """Empty content is rejected."""
        from memory_semantic.tools import mount
        
        coordinator = MagicMock()
        coordinator.config = {"memory": {"agent_id": "test-agent"}}

        with patch("memory_semantic.storage.MemoryStorage"):
            tools = mount(coordinator)
            store_tool = tools[0]

            result = await store_tool.execute({"content": ""})

            assert result.success is False
            assert "empty" in result.error["message"].lower()

    async def test_store_validates_content_length(self):
        """Content over 10,000 chars is rejected."""
        from memory_semantic.tools import mount
        
        coordinator = MagicMock()
        coordinator.config = {"memory": {"agent_id": "test-agent"}}

        with patch("memory_semantic.storage.MemoryStorage"):
            tools = mount(coordinator)
            store_tool = tools[0]

            long_content = "x" * 10001
            result = await store_tool.execute({"content": long_content})

            assert result.success is False
            assert "too long" in result.error["message"].lower()

    async def test_store_handles_exceptions(self):
        """Storage exceptions are caught and returned as errors."""
        from memory_semantic.tools import mount
        
        coordinator = MagicMock()
        coordinator.config = {"memory": {"agent_id": "test-agent"}}

        mock_storage = AsyncMock()
        mock_storage.store.side_effect = Exception("Storage failed")

        with patch("memory_semantic.storage.MemoryStorage", return_value=mock_storage):
            tools = mount(coordinator)
            store_tool = tools[0]

            result = await store_tool.execute({"content": "Test"})

            assert result.success is False
            assert "Storage failed" in result.error["message"]


@pytest.mark.asyncio
class TestMemorySearchTool:
    """Tests for memory_search tool."""

    async def test_search_success(self):
        """Successful search returns memories."""
        from memory_semantic.tools import mount

        coordinator = MagicMock()
        coordinator.config = {"memory": {"agent_id": "test-agent"}}

        mock_memory = MagicMock()
        mock_memory.id = "12345678-1234-5678-1234-567812345678"
        mock_memory.content = "Test memory"
        mock_memory.timestamp = datetime(2026, 1, 31)
        mock_memory.tags = ["test"]

        mock_storage = AsyncMock()
        mock_storage.search.return_value = [mock_memory]

        with patch("memory_semantic.storage.MemoryStorage", return_value=mock_storage):
            tools = mount(coordinator)
            search_tool = tools[1]

            result = await search_tool.execute({"query": "test"})

            assert result.success is True
            assert len(result.output["memories"]) == 1
            assert (
                result.output["memories"][0]["id"]
                == "12345678-1234-5678-1234-567812345678"
            )
            assert "query_time_ms" in result.output

    async def test_search_validates_empty_query(self):
        """Empty query is rejected."""
        from memory_semantic.tools import mount

        coordinator = MagicMock()
        coordinator.config = {"memory": {"agent_id": "test-agent"}}

        with patch("memory_semantic.storage.MemoryStorage"):
            tools = mount(coordinator)
            search_tool = tools[1]

            result = await search_tool.execute({"query": ""})

            assert result.success is False
            assert "empty" in result.error["message"].lower()

    async def test_search_caps_limit(self):
        """Limit is capped at 20."""
        from memory_semantic.tools import mount

        coordinator = MagicMock()
        coordinator.config = {"memory": {"agent_id": "test-agent"}}

        mock_storage = AsyncMock()
        mock_storage.search.return_value = []

        with patch("memory_semantic.storage.MemoryStorage", return_value=mock_storage):
            tools = mount(coordinator)
            search_tool = tools[1]

            await search_tool.execute({"query": "test", "limit": 100})

            # Should cap at 20
            mock_storage.search.assert_called_once()
            call_args = mock_storage.search.call_args
            assert call_args[1]["limit"] == 20

    async def test_search_handles_invalid_since(self):
        """Invalid since timestamp returns error."""
        from memory_semantic.tools import mount

        coordinator = MagicMock()
        coordinator.config = {"memory": {"agent_id": "test-agent"}}

        with patch("memory_semantic.storage.MemoryStorage"):
            tools = mount(coordinator)
            search_tool = tools[1]

            result = await search_tool.execute({"query": "test", "since": "not-a-timestamp"})

            assert result.success is False
            assert "Invalid since timestamp" in result.error["message"]
