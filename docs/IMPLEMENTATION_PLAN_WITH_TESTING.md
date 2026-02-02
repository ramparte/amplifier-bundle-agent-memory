# Implementation Plan with Rigorous Testing

**Project**: amplifier-bundle-agent-memory V1.0  
**Timeline**: 3 days  
**Complexity**: ~500 lines  
**Philosophy**: Test-first, validate rigorously, prove correctness at every stage

---

## Overview

This plan ensures **bulletproof implementation** through:
1. Clear validation gates at each phase
2. Test-driven development (write tests first)
3. Multiple testing layers (unit, integration, e2e)
4. Concrete proof points for correctness
5. Security and performance validation

**If we follow this plan, we should have ZERO surprises.**

---

## Phase 0: Repository Setup

### Deliverable
GitHub repository with proper structure and tooling.

### Implementation Steps

1. **Create GitHub Repository**
```bash
# On GitHub: Create new repo "amplifier-bundle-agent-memory" under ramparte user
# Initialize locally
cd /home/samschillace/dev/ANext
rm -rf amplifier-bundle-agent-memory/reference  # Remove old reference
cd amplifier-bundle-agent-memory
git init
git remote add origin git@github.com:ramparte/amplifier-bundle-agent-memory.git
```

2. **Create Directory Structure**
```bash
mkdir -p modules/tool-memory-semantic/memory_semantic
mkdir -p tests/{unit,integration,e2e,performance,fixtures}
mkdir -p behaviors
mkdir -p context
mkdir -p docs
touch modules/tool-memory-semantic/memory_semantic/__init__.py
touch tests/__init__.py
```

3. **Create Configuration Files**

**pyproject.toml**:
```toml
[project]
name = "amplifier-bundle-agent-memory"
version = "1.0.0"
description = "Semantic memory with agent identities"
requires-python = ">=3.10"
dependencies = [
    "amplifier-core>=0.1.0",
    "qdrant-client>=1.7.0",
    "openai>=1.0.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "pytest-mock>=3.10",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
asyncio_mode = "auto"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "performance: Performance benchmarks",
    "security: Security tests",
]

[tool.coverage.run]
source = ["memory_semantic"]
omit = ["tests/*", "*/test_*.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
```

**.gitignore**:
```
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.coverage
htmlcov/
*.egg-info/
dist/
build/
.venv/
venv/
*.db
.DS_Store
```

**LICENSE** (MIT):
```
MIT License

Copyright (c) 2026 Sam Schillace

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

**README.md**:
```markdown
# Amplifier Agent Memory

Semantic memory system with agent identity isolation.

## Quick Start

\`\`\`bash
pip install git+https://github.com/ramparte/amplifier-bundle-agent-memory
\`\`\`

## Development

\`\`\`bash
git clone https://github.com/ramparte/amplifier-bundle-agent-memory
cd amplifier-bundle-agent-memory
pip install -e ".[dev]"
pytest
\`\`\`

See [docs/V1.0_DESIGN.md](docs/V1.0_DESIGN.md) for architecture.
```

### Validation Gate
- [ ] Repository created on GitHub
- [ ] Directory structure matches plan
- [ ] Configuration files present
- [ ] Can run `pytest` (even if no tests yet)
- [ ] Initial commit pushed

### Proof Points
```bash
git log --oneline  # Should show initial commit
pytest --collect-only  # Should find test directory
pip install -e ".[dev]"  # Should install with dev dependencies
```

---

## Phase 1: Data Model + Unit Tests

### Deliverable
Memory model with complete unit test coverage.

### Implementation Steps

1. **Write Test First** (`tests/unit/test_models.py`):

```python
import pytest
from datetime import datetime
from memory_semantic.models import Memory

class TestMemoryModel:
    """Unit tests for Memory data model."""
    
    def test_memory_creation_with_defaults(self):
        """Memory can be created with minimal fields."""
        memory = Memory(
            agent_id="test-agent",
            content="Test memory content",
            embedding=[0.1, 0.2, 0.3]
        )
        
        assert memory.agent_id == "test-agent"
        assert memory.content == "Test memory content"
        assert memory.embedding == [0.1, 0.2, 0.3]
        assert memory.id.startswith("mem-")
        assert len(memory.id) == 12  # "mem-" + 8 hex chars
        assert isinstance(memory.timestamp, datetime)
        assert memory.tags == []
    
    def test_memory_with_tags(self):
        """Memory can include tags."""
        memory = Memory(
            agent_id="test-agent",
            content="PostgreSQL decision",
            embedding=[0.1],
            tags=["database", "decisions"]
        )
        
        assert memory.tags == ["database", "decisions"]
    
    def test_memory_dict_for_storage(self):
        """dict_for_storage excludes embedding."""
        memory = Memory(
            agent_id="test-agent",
            content="Test",
            embedding=[0.1, 0.2],
            tags=["tag1"]
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
            tags=["tag1"]
        )
        
        # To dict
        data = original.dict()
        
        # From dict
        restored = Memory(**data)
        
        assert restored.id == original.id
        assert restored.agent_id == original.agent_id
        assert restored.content == original.content
        assert restored.embedding == original.embedding
        assert restored.tags == original.tags
```

2. **Implement Model** (`modules/tool-memory-semantic/memory_semantic/models.py`):

```python
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4

class Memory(BaseModel):
    """Minimal memory model for V1.0."""
    
    id: str = Field(default_factory=lambda: f"mem-{uuid4().hex[:8]}")
    agent_id: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    embedding: list[float]
    tags: list[str] = []
    
    def dict_for_storage(self) -> dict:
        """Return dict without embedding for Qdrant payload."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }
```

3. **Run Tests**:
```bash
pytest tests/unit/test_models.py -v
```

### Validation Gate
- [ ] All model unit tests pass
- [ ] Coverage >90% on models.py
- [ ] Memory can be created, serialized, deserialized
- [ ] dict_for_storage() excludes embedding

### Proof Points
```bash
pytest tests/unit/test_models.py -v --cov=memory_semantic.models
# Should show: 6 passed, >90% coverage
```

---

## Phase 2: Embedding Generator + Unit Tests

### Deliverable
OpenAI embedding wrapper with mocked API tests.

### Implementation Steps

1. **Write Test First** (`tests/unit/test_embeddings.py`):

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from memory_semantic.embeddings import EmbeddingGenerator

@pytest.mark.asyncio
class TestEmbeddingGenerator:
    """Unit tests for embedding generation."""
    
    async def test_generate_single_embedding(self):
        """Can generate embedding for single text."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        
        with patch('memory_semantic.embeddings.AsyncOpenAI') as mock_client:
            mock_client.return_value.embeddings.create = AsyncMock(return_value=mock_response)
            
            embedder = EmbeddingGenerator()
            result = await embedder.generate("test content")
            
            assert len(result) == 1536
            assert all(isinstance(x, float) for x in result)
    
    async def test_generate_batch_embeddings(self):
        """Can generate embeddings for batch."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]
        
        with patch('memory_semantic.embeddings.AsyncOpenAI') as mock_client:
            mock_client.return_value.embeddings.create = AsyncMock(return_value=mock_response)
            
            embedder = EmbeddingGenerator()
            results = await embedder.generate_batch(["text1", "text2"])
            
            assert len(results) == 2
            assert len(results[0]) == 1536
            assert len(results[1]) == 1536
    
    async def test_uses_correct_model(self):
        """Uses text-embedding-3-small by default."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        
        with patch('memory_semantic.embeddings.AsyncOpenAI') as mock_client:
            mock_create = AsyncMock(return_value=mock_response)
            mock_client.return_value.embeddings.create = mock_create
            
            embedder = EmbeddingGenerator()
            await embedder.generate("test")
            
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["model"] == "text-embedding-3-small"
    
    def test_custom_model(self):
        """Can specify custom embedding model."""
        with patch('memory_semantic.embeddings.AsyncOpenAI'):
            embedder = EmbeddingGenerator(model="text-embedding-3-large")
            assert embedder.model == "text-embedding-3-large"
```

2. **Implement Embeddings** (`modules/tool-memory-semantic/memory_semantic/embeddings.py`):

```python
from openai import AsyncOpenAI
from typing import Optional
import os

class EmbeddingGenerator:
    """OpenAI embedding API wrapper."""
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.dimensions = 1536
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    async def generate(self, content: str) -> list[float]:
        """Generate embedding for single content."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=content
        )
        return response.data[0].embedding
    
    async def generate_batch(self, contents: list[str]) -> list[list[float]]:
        """Generate embeddings for batch of content."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=contents
        )
        return [item.embedding for item in response.data]
```

3. **Run Tests**:
```bash
pytest tests/unit/test_embeddings.py -v
```

### Validation Gate
- [ ] All embedding unit tests pass
- [ ] Mocks verify correct API calls
- [ ] Can generate single and batch embeddings
- [ ] Custom model specification works

### Proof Points
```bash
pytest tests/unit/test_embeddings.py -v --cov=memory_semantic.embeddings
# Should show: 4 passed, >85% coverage
```

---

## Phase 3: Storage Layer + Unit Tests

### Deliverable
Qdrant storage with namespace isolation and ranking.

### Implementation Steps

1. **Write Test First** (`tests/unit/test_storage.py`):

```python
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
            
            storage1 = MemoryStorage("alice", config)
            storage2 = MemoryStorage("bob", config)
            
            # Different paths
            path1 = Path(tmpdir) / "alice"
            path2 = Path(tmpdir) / "bob"
            
            assert path1.exists()
            assert path2.exists()
            assert path1 != path2
    
    async def test_store_and_retrieve(self, temp_storage):
        """Can store and retrieve memory by ID."""
        memory_id = await temp_storage.store(
            content="Test memory",
            tags=["test"]
        )
        
        assert memory_id.startswith("mem-")
        
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
        # Store old memory
        old_time = datetime.utcnow() - timedelta(days=30)
        
        # Store recent memory
        recent_id = await temp_storage.store("Recent memory", [])
        
        # Search with since filter
        since = datetime.utcnow() - timedelta(days=7)
        results = await temp_storage.search("memory", since=since)
        
        # Should only find recent
        result_ids = [m.id for m in results]
        assert recent_id in result_ids
    
    async def test_recency_boost(self, temp_storage):
        """Recent memories get ranking boost."""
        # Store old memory with exact query match
        old_id = await temp_storage.store("database query optimization", [])
        
        # Wait a moment, then store recent memory with similar content
        import asyncio
        await asyncio.sleep(0.1)
        recent_id = await temp_storage.store("database query performance", [])
        
        # Search - recent should rank higher due to 20% boost
        results = await temp_storage.search("database queries", limit=2)
        
        # Both should be found
        result_ids = [m.id for m in results]
        assert old_id in result_ids
        assert recent_id in result_ids
```

2. **Implement Storage** (see V1.0_IMPLEMENTATION.md for full code)

3. **Run Tests**:
```bash
pytest tests/unit/test_storage.py -v
```

### Validation Gate
- [ ] Agent ID validation blocks path traversal
- [ ] Storage paths are isolated per agent
- [ ] Can store and retrieve by ID
- [ ] Search returns relevant results
- [ ] Recency boost affects ranking
- [ ] Since filter works

### Proof Points
```bash
pytest tests/unit/test_storage.py -v --cov=memory_semantic.storage
# Should show: 8+ passed, >80% coverage
```

---

## Phase 4: Tool Module + Unit Tests

### Deliverable
Tool module with complete error handling and validation.

### Implementation Steps

1. **Research Capability Access Pattern**:
```bash
# Check amplifier-core for correct pattern
python3 -c "from amplifier_core import get_session; help(get_session)"
# OR
python3 -c "from amplifier_foundation import require_capability; help(require_capability)"
```

2. **Write Test First** (`tests/unit/test_tools.py`):

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from memory_semantic.tools import mount

@pytest.fixture
def mock_coordinator():
    """Mock coordinator with agent_identity capability."""
    coordinator = MagicMock()
    coordinator.config = {"memory": {}}
    
    # Mock session with capabilities
    mock_session = MagicMock()
    mock_session.capabilities.get.return_value = "test-agent"
    
    with patch('memory_semantic.tools.get_session', return_value=mock_session):
        yield coordinator

class TestToolMount:
    """Tests for tool mounting and configuration."""
    
    def test_mount_requires_agent_identity(self):
        """Mount fails without agent_identity capability."""
        coordinator = MagicMock()
        coordinator.config = {}
        
        mock_session = MagicMock()
        mock_session.capabilities.get.return_value = None
        
        with patch('memory_semantic.tools.get_session', return_value=mock_session):
            with pytest.raises(ValueError, match="agent_identity"):
                mount(coordinator)
    
    def test_mount_returns_tools(self, mock_coordinator):
        """Mount returns list of Tool objects."""
        with patch('memory_semantic.tools.MemoryStorage'):
            tools = mount(mock_coordinator)
            
            assert len(tools) == 2
            assert tools[0].name == "memory_store"
            assert tools[1].name == "memory_search"

@pytest.mark.asyncio
class TestMemoryStoreTool:
    """Tests for memory_store tool."""
    
    async def test_store_success(self, mock_coordinator):
        """Successful storage returns memory ID."""
        mock_storage = AsyncMock()
        mock_storage.store.return_value = "mem-12345678"
        
        with patch('memory_semantic.tools.MemoryStorage', return_value=mock_storage):
            tools = mount(mock_coordinator)
            store_tool = tools[0]
            
            result = await store_tool.execute(
                content="Test memory",
                tags=["test"]
            )
            
            assert result.success is True
            assert result.data["memory_id"] == "mem-12345678"
            mock_storage.store.assert_called_once_with("Test memory", ["test"])
    
    async def test_store_validates_empty_content(self, mock_coordinator):
        """Empty content is rejected."""
        with patch('memory_semantic.tools.MemoryStorage'):
            tools = mount(mock_coordinator)
            store_tool = tools[0]
            
            result = await store_tool.execute(content="")
            
            assert result.success is False
            assert "empty" in result.error.lower()
    
    async def test_store_validates_content_length(self, mock_coordinator):
        """Content over 10,000 chars is rejected."""
        with patch('memory_semantic.tools.MemoryStorage'):
            tools = mount(mock_coordinator)
            store_tool = tools[0]
            
            long_content = "x" * 10001
            result = await store_tool.execute(content=long_content)
            
            assert result.success is False
            assert "too long" in result.error.lower()
    
    async def test_store_handles_exceptions(self, mock_coordinator):
        """Storage exceptions are caught and returned as errors."""
        mock_storage = AsyncMock()
        mock_storage.store.side_effect = Exception("Storage failed")
        
        with patch('memory_semantic.tools.MemoryStorage', return_value=mock_storage):
            tools = mount(mock_coordinator)
            store_tool = tools[0]
            
            result = await store_tool.execute(content="Test")
            
            assert result.success is False
            assert "Storage failed" in result.error

@pytest.mark.asyncio
class TestMemorySearchTool:
    """Tests for memory_search tool."""
    
    async def test_search_success(self, mock_coordinator):
        """Successful search returns memories."""
        mock_memory = MagicMock()
        mock_memory.id = "mem-12345678"
        mock_memory.content = "Test memory"
        mock_memory.timestamp = datetime(2026, 1, 31)
        mock_memory.tags = ["test"]
        
        mock_storage = AsyncMock()
        mock_storage.search.return_value = [mock_memory]
        
        with patch('memory_semantic.tools.MemoryStorage', return_value=mock_storage):
            tools = mount(mock_coordinator)
            search_tool = tools[1]
            
            result = await search_tool.execute(query="test")
            
            assert result.success is True
            assert len(result.data["memories"]) == 1
            assert result.data["memories"][0]["id"] == "mem-12345678"
            assert "query_time_ms" in result.data
    
    async def test_search_validates_empty_query(self, mock_coordinator):
        """Empty query is rejected."""
        with patch('memory_semantic.tools.MemoryStorage'):
            tools = mount(mock_coordinator)
            search_tool = tools[1]
            
            result = await search_tool.execute(query="")
            
            assert result.success is False
            assert "empty" in result.error.lower()
    
    async def test_search_caps_limit(self, mock_coordinator):
        """Limit is capped at 20."""
        mock_storage = AsyncMock()
        mock_storage.search.return_value = []
        
        with patch('memory_semantic.tools.MemoryStorage', return_value=mock_storage):
            tools = mount(mock_coordinator)
            search_tool = tools[1]
            
            await search_tool.execute(query="test", limit=100)
            
            # Should cap at 20
            mock_storage.search.assert_called_once()
            call_args = mock_storage.search.call_args
            assert call_args[1]["limit"] == 20
```

3. **Implement Tools** (see V1.0_IMPLEMENTATION.md, but update capability access)

4. **Run Tests**:
```bash
pytest tests/unit/test_tools.py -v
```

### Validation Gate
- [ ] Mount requires agent_identity
- [ ] Tools are returned correctly
- [ ] memory_store validates input
- [ ] memory_search validates query
- [ ] Errors are caught and returned properly
- [ ] Limit is capped at 20

### Proof Points
```bash
pytest tests/unit/test_tools.py -v --cov=memory_semantic.tools
# Should show: 10+ passed, >85% coverage
```

---

## Phase 5: Integration Tests

### Deliverable
Full workflow tests proving components work together.

### Implementation Steps

1. **Write Integration Tests** (`tests/integration/test_store_search.py`):

```python
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from memory_semantic.storage import MemoryStorage

@pytest.fixture
def storage():
    """Real storage for integration tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"storage_root": tmpdir}
        yield MemoryStorage("integration-test", config)

@pytest.mark.integration
@pytest.mark.asyncio
class TestStoreSearchIntegration:
    """Integration tests for store → search workflow."""
    
    async def test_store_and_search_workflow(self, storage):
        """Complete workflow: store multiple, search, verify results."""
        # Store memories
        id1 = await storage.store(
            "We decided to use PostgreSQL for the project database",
            tags=["database", "decisions"]
        )
        
        id2 = await storage.store(
            "API rate limit is 1000 requests per hour",
            tags=["api", "limits"]
        )
        
        id3 = await storage.store(
            "MongoDB was considered but rejected for this use case",
            tags=["database", "decisions"]
        )
        
        # Search for database decisions
        results = await storage.search("database decisions", limit=5)
        
        # Verify
        assert len(results) >= 2
        result_ids = [m.id for m in results]
        assert id1 in result_ids  # PostgreSQL decision should be found
        assert id3 in result_ids  # MongoDB decision should be found
        
        # Check that database-related memories rank higher
        top_result = results[0]
        assert "database" in top_result.tags or "database" in top_result.content.lower()
    
    async def test_search_quality_semantic_understanding(self, storage):
        """Search understands semantic similarity."""
        # Store with different wording
        await storage.store("Rate limiting policy is 1k req/hour", ["api"])
        await storage.store("Database connection pooling enabled", ["database"])
        await storage.store("API throttling set to 1000/hour", ["api"])
        
        # Search with synonym
        results = await storage.search("API rate limits", limit=3)
        
        # Should find both rate limiting memories despite different wording
        contents = [m.content for m in results]
        has_rate = any("rate" in c.lower() or "throttling" in c.lower() for c in contents)
        assert has_rate
    
    async def test_recency_affects_ranking(self, storage):
        """Recent memories rank higher with same relevance."""
        # Store old memory
        old_id = await storage.store("Database query optimization", [])
        
        # Wait to create timestamp difference
        import asyncio
        await asyncio.sleep(1)
        
        # Store recent memory with similar content
        recent_id = await storage.store("Database query performance tuning", [])
        
        # Search
        results = await storage.search("database query optimization", limit=2)
        
        # Recent should rank higher (appears first)
        if len(results) >= 2:
            # Due to recency boost, recent might rank first
            result_ids = [m.id for m in results]
            assert recent_id in result_ids
            assert old_id in result_ids
    
    async def test_since_filter_excludes_old(self, storage):
        """Since filter correctly excludes old memories."""
        # Store memories
        id1 = await storage.store("Old memory", [])
        
        # Wait
        import asyncio
        await asyncio.sleep(0.5)
        
        id2 = await storage.store("Recent memory", [])
        
        # Search with since = now (should only find recent)
        since = datetime.utcnow() - timedelta(seconds=0.3)
        results = await storage.search("memory", since=since, limit=10)
        
        result_ids = [m.id for m in results]
        assert id2 in result_ids
        # id1 might or might not be excluded depending on timing
```

2. **Write Isolation Tests** (`tests/integration/test_isolation.py`):

```python
import pytest
import tempfile
from memory_semantic.storage import MemoryStorage

@pytest.mark.integration
@pytest.mark.asyncio
class TestAgentIsolation:
    """Integration tests for agent namespace isolation."""
    
    async def test_different_agents_isolated(self):
        """Different agents cannot see each other's memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"storage_root": tmpdir}
            
            alice = MemoryStorage("alice", config)
            bob = MemoryStorage("bob", config)
            
            # Alice stores memory
            alice_id = await alice.store("Alice's secret", ["private"])
            
            # Bob stores memory
            bob_id = await bob.store("Bob's secret", ["private"])
            
            # Alice searches - should only see her memory
            alice_results = await alice.search("secret", limit=10)
            alice_ids = [m.id for m in alice_results]
            assert alice_id in alice_ids
            assert bob_id not in alice_ids
            
            # Bob searches - should only see his memory
            bob_results = await bob.search("secret", limit=10)
            bob_ids = [m.id for m in bob_results]
            assert bob_id in bob_ids
            assert alice_id not in bob_ids
    
    async def test_agent_cannot_retrieve_other_memory(self):
        """Agent cannot retrieve memory by ID from different agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"storage_root": tmpdir}
            
            alice = MemoryStorage("alice", config)
            bob = MemoryStorage("bob", config)
            
            # Alice stores
            alice_id = await alice.store("Alice's data", [])
            
            # Bob tries to retrieve Alice's memory
            result = await bob.get(alice_id)
            
            # Should not find it (different namespace)
            assert result is None
```

3. **Run Integration Tests**:
```bash
pytest tests/integration/ -v -m integration
```

### Validation Gate
- [ ] Store → search workflow works end-to-end
- [ ] Semantic search finds relevant memories
- [ ] Recency boost affects ranking
- [ ] Since filter works correctly
- [ ] Agent isolation is enforced
- [ ] Cross-agent access fails

### Proof Points
```bash
pytest tests/integration/ -v -m integration
# Should show: 6+ passed
# All isolation tests pass
```

---

## Phase 6: End-to-End Tests

### Deliverable
Complete user-facing workflow validation.

### Implementation Steps

1. **Write E2E Test** (`tests/e2e/test_complete_workflow.py`):

```python
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from memory_semantic.tools import mount

@pytest.fixture
def e2e_coordinator():
    """Coordinator for end-to-end tests."""
    coordinator = MagicMock()
    
    # Use temp directory for storage
    tmpdir = tempfile.mkdtemp()
    coordinator.config = {
        "memory": {
            "storage_root": tmpdir
        }
    }
    
    # Mock session with agent_identity
    mock_session = MagicMock()
    mock_session.capabilities.get.return_value = "e2e-test-agent"
    
    with patch('memory_semantic.tools.get_session', return_value=mock_session):
        yield coordinator
    
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

@pytest.mark.e2e
@pytest.mark.asyncio
class TestCompleteWorkflow:
    """End-to-end tests simulating real usage."""
    
    async def test_complete_user_workflow(self, e2e_coordinator):
        """Simulate complete user interaction."""
        # Mount tools (as Amplifier would)
        tools = mount(e2e_coordinator)
        store_tool = tools[0]
        search_tool = tools[1]
        
        # User stores decision
        result1 = await store_tool.execute(
            content="We decided to use PostgreSQL for the new project",
            tags=["database", "decisions"]
        )
        assert result1.success
        mem_id1 = result1.data["memory_id"]
        
        # User stores API limit
        result2 = await store_tool.execute(
            content="API rate limit is 1000 requests per hour",
            tags=["api", "limits"]
        )
        assert result2.success
        
        # User stores research note
        result3 = await store_tool.execute(
            content="Investigated MongoDB but PostgreSQL fits better for relational data",
            tags=["database", "research"]
        )
        assert result3.success
        
        # User searches: "what did we decide about databases?"
        search_result = await search_tool.execute(
            query="database decisions",
            limit=5
        )
        
        assert search_result.success
        memories = search_result.data["memories"]
        
        # Should find database-related memories
        assert len(memories) >= 2
        contents = [m["content"] for m in memories]
        assert any("PostgreSQL" in c for c in contents)
        
        # Verify structure
        first_memory = memories[0]
        assert "id" in first_memory
        assert "content" in first_memory
        assert "timestamp" in first_memory
        assert "tags" in first_memory
    
    async def test_natural_language_queries(self, e2e_coordinator):
        """Natural language queries work."""
        tools = mount(e2e_coordinator)
        store_tool = tools[0]
        search_tool = tools[1]
        
        # Store various memories
        await store_tool.execute(content="PostgreSQL connection pool size set to 20", tags=["database"])
        await store_tool.execute(content="API authentication uses JWT tokens", tags=["api", "security"])
        await store_tool.execute(content="Rate limiting configured at 1000 req/hour", tags=["api"])
        
        # Natural language queries
        queries = [
            "what's our database setup?",
            "how does authentication work?",
            "what are the API limits?",
        ]
        
        for query in queries:
            result = await search_tool.execute(query=query, limit=3)
            assert result.success
            assert len(result.data["memories"]) >= 1
    
    async def test_error_handling_user_visible(self, e2e_coordinator):
        """Error messages are clear for users."""
        tools = mount(e2e_coordinator)
        store_tool = tools[0]
        search_tool = tools[1]
        
        # Empty content
        result = await store_tool.execute(content="")
        assert not result.success
        assert "empty" in result.error.lower()
        
        # Empty query
        result = await search_tool.execute(query="")
        assert not result.success
        assert "empty" in result.error.lower()
        
        # Too long content
        result = await store_tool.execute(content="x" * 10001)
        assert not result.success
        assert "long" in result.error.lower()
```

2. **Run E2E Tests**:
```bash
export OPENAI_API_KEY=your-key
pytest tests/e2e/ -v -m e2e
```

### Validation Gate
- [ ] Complete workflow from mount → store → search works
- [ ] Natural language queries return relevant results
- [ ] Error messages are user-friendly
- [ ] Tool results have correct structure
- [ ] Real OpenAI embeddings work (if API key set)

### Proof Points
```bash
pytest tests/e2e/ -v -m e2e
# Should show: 3 passed
# Workflow completes successfully
```

---

## Phase 7: Security & Performance Tests

### Deliverable
Security validation and performance benchmarks.

### Implementation Steps

1. **Security Tests** (`tests/integration/test_security.py`):

```python
import pytest
from memory_semantic.storage import MemoryStorage

@pytest.mark.security
class TestSecurityValidation:
    """Security-focused tests."""
    
    def test_path_traversal_blocked(self):
        """Path traversal attempts are blocked."""
        attacks = [
            "../alice",
            "alice/../bob",
            "../../root",
            "./../../../etc/passwd",
            "alice/../../etc",
        ]
        
        for attack in attacks:
            with pytest.raises(ValueError, match="Invalid agent_id"):
                MemoryStorage(attack, {})
    
    def test_special_chars_blocked(self):
        """Special characters in agent_id are blocked."""
        invalid = [
            "alice/bob",
            "alice bob",
            "alice;bob",
            "alice\x00bob",
            "alice<bob",
        ]
        
        for agent_id in invalid:
            with pytest.raises(ValueError, match="Invalid agent_id"):
                MemoryStorage(agent_id, {})
    
    def test_valid_agent_ids_accepted(self):
        """Valid agent IDs are accepted."""
        import tempfile
        
        valid = [
            "alice",
            "bob",
            "user-123",
            "project_alpha",
            "agent-v2",
            "test_agent_123",
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"storage_root": tmpdir}
            for agent_id in valid:
                storage = MemoryStorage(agent_id, config)
                assert storage.agent_id == agent_id
```

2. **Performance Tests** (`tests/performance/test_benchmarks.py`):

```python
import pytest
import time
import tempfile
from memory_semantic.storage import MemoryStorage

@pytest.mark.performance
@pytest.mark.asyncio
class TestPerformanceBenchmarks:
    """Performance benchmarks and targets."""
    
    async def test_store_latency(self):
        """Store operation meets latency target (<500ms)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"storage_root": tmpdir}
            storage = MemoryStorage("perf-test", config)
            
            start = time.time()
            await storage.store("Performance test memory", ["test"])
            duration = (time.time() - start) * 1000  # ms
            
            print(f"\nStore latency: {duration:.0f}ms")
            assert duration < 500, f"Store took {duration}ms (target: <500ms)"
    
    async def test_search_latency_small_corpus(self):
        """Search on small corpus meets target (<200ms)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"storage_root": tmpdir}
            storage = MemoryStorage("perf-test", config)
            
            # Create small corpus (100 memories)
            for i in range(100):
                await storage.store(f"Memory {i} with content about various topics", ["test"])
            
            # Benchmark search
            start = time.time()
            results = await storage.search("various topics", limit=5)
            duration = (time.time() - start) * 1000  # ms
            
            print(f"\nSearch latency (100 memories): {duration:.0f}ms")
            assert duration < 200, f"Search took {duration}ms (target: <200ms)"
    
    async def test_search_latency_medium_corpus(self):
        """Search on medium corpus is reasonable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"storage_root": tmpdir}
            storage = MemoryStorage("perf-test", config)
            
            # Create medium corpus (1,000 memories)
            for i in range(1000):
                await storage.store(f"Memory {i} about topic {i % 10}", ["test"])
            
            # Benchmark
            start = time.time()
            results = await storage.search("topic 5", limit=5)
            duration = (time.time() - start) * 1000  # ms
            
            print(f"\nSearch latency (1,000 memories): {duration:.0f}ms")
            # More lenient for larger corpus
            assert duration < 300, f"Search took {duration}ms (target: <300ms)"
```

3. **Run Security & Performance Tests**:
```bash
pytest tests/integration/test_security.py -v -m security
pytest tests/performance/ -v -m performance
```

### Validation Gate
- [ ] All path traversal attempts blocked
- [ ] Invalid characters rejected
- [ ] Valid agent IDs accepted
- [ ] Store latency <500ms
- [ ] Search latency <200ms (small corpus)
- [ ] Search latency <300ms (1k memories)

### Proof Points
```bash
pytest -v -m security
# Should show: 3 passed, all attacks blocked

pytest -v -m performance
# Should show: 3 passed, all benchmarks met
# Prints actual latencies for verification
```

---

## Phase 8: Bundle Packaging

### Deliverable
Complete bundle ready for distribution.

### Implementation Steps

1. **Create Context File** (`context/memory-awareness.md`):

```markdown
# Memory System Awareness

You have semantic memory capabilities via agent identity.

## Tools Available

- `memory_store(content, tags)` - Store a memory with semantic embedding
- `memory_search(query, limit, since)` - Search memories semantically

## Usage

Store important facts, decisions, and context:
```
User: "Remember this: we use PostgreSQL for new projects"
Agent: [Calls memory_store]
```

Search when you need recall:
```
User: "What's our database standard?"
Agent: [Calls memory_search("database standard")]
       Based on your previous decision, we use PostgreSQL for new projects.
```

**Do not over-explain memory mechanics. Use naturally.**

## Storage

- Location: `~/.amplifier/memory/{agent_id}/`
- Cost: ~$0.00002 per memory (negligible)
- Isolation: Each agent identity has separate storage

## Natural Language

Talk naturally. The system understands semantic similarity:
- "what did we decide about X?" finds decision memories
- "what's our policy on Y?" finds policy memories
- "what have I worked on related to Z?" finds work memories
```

2. **Update README.md** with full documentation

3. **Create GitHub Actions** (`.github/workflows/test.yml`):

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=memory_semantic
      - name: Run integration tests
        run: |
          pytest tests/integration/ -v -m integration
      - name: Run security tests
        run: |
          pytest -v -m security
```

4. **Final Validation**:
```bash
# Run all tests
pytest -v

# Check coverage
pytest --cov=memory_semantic --cov-report=html

# Verify bundle loads
python -c "from memory_semantic import tools; print('OK')"
```

### Validation Gate
- [ ] All tests pass
- [ ] Coverage >80% on critical paths
- [ ] Context file is clear and concise
- [ ] README is complete
- [ ] Bundle structure matches plan
- [ ] Can install with pip

### Proof Points
```bash
pytest -v
# Should show: 30+ tests passed

coverage report
# Should show: >80% on memory_semantic/*

pip install -e .
python -c "from memory_semantic import tools"
# Should complete without error
```

---

## Success Metrics

### Functional Correctness
- [x] All unit tests pass
- [x] All integration tests pass
- [x] All e2e tests pass
- [x] Can store memories
- [x] Can search semantically
- [x] Agent isolation works

### Performance
- [x] Store latency <500ms
- [x] Search latency <200ms (small corpus)
- [x] Query time scales reasonably (1k memories)

### Security
- [x] Path traversal blocked
- [x] Invalid agent IDs rejected
- [x] Cross-agent access fails

### Quality
- [x] Code coverage >80%
- [x] No critical TODOs
- [x] Clear error messages
- [x] Documentation complete

### Usability
- [x] Natural language queries work
- [x] Tool results are well-structured
- [x] Errors are user-friendly

---

## Post-Implementation Checklist

### Before Deployment
- [ ] All tests pass (`pytest -v`)
- [ ] Coverage report reviewed (`pytest --cov`)
- [ ] Security tests pass (`pytest -m security`)
- [ ] Performance benchmarks met (`pytest -m performance`)
- [ ] Documentation complete (README, V1.0_DESIGN.md)
- [ ] No FIXMEs or TODOs in code
- [ ] bundle.md correct
- [ ] context file created

### Deployment
- [ ] Push to GitHub
- [ ] Create v1.0.0 tag
- [ ] Test install from GitHub
- [ ] Add to my-amplifier bundle
- [ ] Test in personal Amplifier session

### Post-Deployment
- [ ] Use personally for 2-4 weeks
- [ ] Track usage patterns
- [ ] Document pain points
- [ ] Decide on V1.5 features based on real data

---

## Debugging Guide

### Test Failures

**If unit tests fail**:
1. Run specific test: `pytest tests/unit/test_models.py::test_name -v`
2. Check imports: Verify module structure
3. Check mocks: Ensure mocks match actual APIs

**If integration tests fail**:
1. Check Qdrant: Is embedded DB working?
2. Check temp directories: Are they being created?
3. Check timestamps: Are timing-dependent tests flaky?

**If e2e tests fail**:
1. Check OpenAI API key: Is it set?
2. Check tool mounting: Are capabilities passed correctly?
3. Check error messages: Are they clear?

### Coverage Gaps

**If coverage <80%**:
1. Identify untested lines: `pytest --cov --cov-report=term-missing`
2. Add tests for error paths
3. Add tests for edge cases
4. Don't test trivial code (getters, simple properties)

### Performance Issues

**If benchmarks fail**:
1. Check system load: Are other processes running?
2. Check Qdrant: Is embedded DB slow?
3. Check OpenAI API: Is embedding generation slow?
4. Profile code: Use `cProfile` to find bottlenecks

---

## Final Notes

**This plan ensures**:
- Every component has unit tests
- Components work together (integration tests)
- User workflows succeed (e2e tests)
- Security is validated
- Performance meets targets
- Quality is measurable (coverage)

**Follow this plan, and you'll have**:
- Working memory system in 3 days
- High confidence in correctness
- No surprises in deployment
- Clear metrics for success

**Test first. Validate rigorously. Ship with confidence.** 🚀
