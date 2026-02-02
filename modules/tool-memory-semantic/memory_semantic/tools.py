"""Memory tools for Amplifier."""

import time
from datetime import datetime
from typing import Any

from amplifier_core import ToolResult


class MemoryStoreTool:
    """Tool for storing memories."""

    def __init__(self, storage):
        self.storage = storage
        self.name = "memory_store"
        self.description = "Store a memory with semantic embedding"
        self.input_schema = {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Memory content to store",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorization",
                },
            },
            "required": ["content"],
        }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Store memory."""
        try:
            content = input.get("content", "")
            tags = input.get("tags", [])

            # Validation - content
            if not content or not content.strip():
                return ToolResult(
                    success=False,
                    output=None,
                    error={"message": "Content cannot be empty"},
                )

            if len(content) > 10000:
                return ToolResult(
                    success=False,
                    output=None,
                    error={"message": "Content too long (max 10,000 chars)"},
                )

            # Validation - tags
            if not isinstance(tags, list):
                return ToolResult(
                    success=False,
                    output=None,
                    error={"message": "Tags must be an array"},
                )

            if len(tags) > 50:
                return ToolResult(
                    success=False,
                    output=None,
                    error={"message": "Too many tags (max 50)"},
                )

            for tag in tags:
                if not isinstance(tag, str):
                    return ToolResult(
                        success=False,
                        output=None,
                        error={"message": "All tags must be strings"},
                    )
                if len(tag) > 100:
                    return ToolResult(
                        success=False,
                        output=None,
                        error={"message": f"Tag too long: '{tag[:20]}...' (max 100 chars)"},
                    )
                if not tag.strip():
                    return ToolResult(
                        success=False,
                        output=None,
                        error={"message": "Tags cannot be empty strings"},
                    )

            # Store
            memory_id = await self.storage.store(content, tags)

            return ToolResult(
                success=True,
                output={
                    "memory_id": memory_id,
                    "message": f"Memory stored with ID: {memory_id}",
                },
            )

        except Exception as e:
            return ToolResult(success=False, output=None, error={"message": str(e)})


class MemorySearchTool:
    """Tool for searching memories."""

    def __init__(self, storage):
        self.storage = storage
        self.name = "memory_search"
        self.description = "Search memories semantically"
        self.input_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 5, max 20)",
                    "default": 5,
                },
                "since": {
                    "type": "string",
                    "description": "ISO timestamp - only return memories after this time",
                },
            },
            "required": ["query"],
        }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Search memories."""
        try:
            query = input.get("query", "")
            limit = min(input.get("limit", 5), 20)  # Cap at 20
            since_str = input.get("since")

            # Validation
            if not query or not query.strip():
                return ToolResult(
                    success=False,
                    output=None,
                    error={"message": "Query cannot be empty"},
                )

            # Parse since timestamp
            since = None
            if since_str:
                try:
                    since = datetime.fromisoformat(since_str.replace("Z", "+00:00"))
                except ValueError:
                    return ToolResult(
                        success=False,
                        output=None,
                        error={"message": "Invalid since timestamp (use ISO format)"},
                    )

            # Search
            start_time = time.time()
            memories = await self.storage.search(query, limit=limit, since=since)
            query_time_ms = int((time.time() - start_time) * 1000)

            # Format results
            results = [
                {
                    "id": mem.id,
                    "content": mem.content,
                    "timestamp": mem.timestamp.isoformat(),
                    "tags": mem.tags,
                }
                for mem in memories
            ]

            return ToolResult(
                success=True,
                output={
                    "memories": results,
                    "count": len(results),
                    "query_time_ms": query_time_ms,
                },
            )

        except Exception as e:
            return ToolResult(success=False, output=None, error={"message": str(e)})


def mount(coordinator, config: dict | None = None):
    """Mount memory tools.

    Agent ID resolution order:
    1. config["agent_id"] (explicit configuration)
    2. coordinator.session.agent_id (runtime detection)
    3. "default-agent" (fallback)

    Args:
        coordinator: Amplifier coordinator
        config: Optional configuration

    Returns:
        List of Tool instances
    """
    # Import here to avoid circular dependency
    from .storage import MemoryStorage

    # Get memory config
    memory_config = config or coordinator.config.get("memory", {})

    # Determine agent_id with fallback chain
    agent_id = memory_config.get("agent_id")

    if not agent_id:
        # Try to get from coordinator session
        session = getattr(coordinator, "session", None)
        if session:
            agent_id = getattr(session, "agent_id", None)

    if not agent_id:
        # Fallback to default
        agent_id = "default-agent"

    # Initialize storage
    storage = MemoryStorage(agent_id, memory_config)

    # Create tool instances
    store_tool = MemoryStoreTool(storage)
    search_tool = MemorySearchTool(storage)

    return [store_tool, search_tool]
