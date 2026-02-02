# API Reference: amplifier-bundle-agent-memory

## Document Status
**Version**: 0.1.0 (Design Phase)  
**Date**: 2026-01-30  
**Status**: Design Documentation

---

## Table of Contents

1. [Tool APIs](#tool-apis) - What agents/users invoke
2. [Python Module APIs](#python-module-apis) - For developers
3. [Configuration Schemas](#configuration-schemas) - Setup and config
4. [Integration Examples](#integration-examples) - Usage patterns
5. [Error Handling](#error-handling) - Exceptions and recovery
6. [Events](#events) - Observability hooks

---

## Tool APIs

The memory system provides tools that agents can invoke during conversation. All tools follow the Amplifier Tool protocol and return structured `ToolResult` objects.

### memory_search

Semantic search across agent memories with relevance ranking.

**Signature:**
```python
async def memory_search(
    query: str,
    limit: int = 5,
    filters: Optional[dict] = None
) -> ToolResult
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `str` | Yes | - | Natural language query (e.g., "database work last week") |
| `limit` | `int` | No | `5` | Maximum results to return (1-100) |
| `filters` | `dict` | No | `None` | Optional filters (see below) |

**Filters Object:**
```python
{
    "category": str,              # Filter by category (e.g., "preference")
    "tags": list[str],            # Filter by tags (e.g., ["database", "postgresql"])
    "date_range": {
        "start": str,             # ISO 8601 timestamp
        "end": str                # ISO 8601 timestamp
    },
    "min_access_count": int       # Only memories accessed N+ times
}
```

**Returns:**

```python
ToolResult(
    output={
        "results": [
            {
                "id": "mem-a3f2c1b4e5d6",
                "content": "Prefer PostgreSQL for new projects",
                "category": "preference",
                "tags": ["database", "postgresql"],
                "relevance_score": 0.87,
                "match_type": "vector",
                "timestamp": "2026-01-28T12:00:00Z",
                "access_count": 3
            },
            # ... more results
        ],
        "total_found": 12,
        "search_latency_ms": 87.3
    },
    error=None
)
```

**Examples:**

```python
# Basic semantic search
result = await memory_search(
    query="What database preferences do I have?"
)

# Limited results with category filter
result = await memory_search(
    query="recent bugs",
    limit=10,
    filters={"category": "bugfix"}
)

# Date range search
result = await memory_search(
    query="work items",
    filters={
        "date_range": {
            "start": "2026-01-20T00:00:00Z",
            "end": "2026-01-27T23:59:59Z"
        }
    }
)

# Tag-based search
result = await memory_search(
    query="architecture decisions",
    filters={"tags": ["architecture", "design"]}
)
```

**Search Flow:**

1. Check **scratchpad** (hot cache) for keyword matches
2. If insufficient results, query **deep storage** (vector DB)
3. Generate query embedding (if vector storage)
4. Apply freshness decay and access boosting
5. Merge and rank results
6. Update access statistics

**Performance:**
- Scratchpad: ~10ms
- File storage: ~50ms
- Vector storage: ~100-150ms

---

### memory_store

Store a new memory with optional categorization and tagging.

**Signature:**
```python
async def memory_store(
    content: str,
    category: str,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None
) -> ToolResult
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | `str` | Yes | - | Memory content (1-10,000 characters) |
| `category` | `str` | Yes | - | Memory category (see categories below) |
| `tags` | `list[str]` | No | `[]` | Searchable keywords |
| `metadata` | `dict` | No | `{}` | Extensible metadata |

**Valid Categories:**
- `preference` - User preferences and choices
- `decision` - Architectural or implementation decisions
- `discovery` - Learnings and insights
- `pattern` - Code patterns and conventions
- `task` - Active work items
- `bugfix` - Bug resolutions
- `note` - General notes
- `question` - Unanswered questions
- `insight` - Deep understanding

**Returns:**

```python
ToolResult(
    output={
        "memory_id": "mem-a3f2c1b4e5d6",
        "stored_at": "2026-01-30T16:00:00Z",
        "in_scratchpad": True,
        "backend": "file"  # or "vector"
    },
    error=None
)
```

**Examples:**

```python
# Store a preference
result = await memory_store(
    content="Prefer PostgreSQL for new projects due to better tooling",
    category="preference",
    tags=["database", "postgresql", "tools"]
)

# Store an architectural decision
result = await memory_store(
    content="Decided to use Qdrant for vector storage in embedded mode",
    category="decision",
    tags=["architecture", "vector-db", "qdrant"],
    metadata={
        "related_to": "memory-system-design",
        "alternatives_considered": ["Pinecone", "Weaviate", "ChromaDB"]
    }
)

# Store a task
result = await memory_store(
    content="TODO: Complete database migration by end of week",
    category="task",
    tags=["task", "database", "migration"]
)

# Store a bug resolution
result = await memory_store(
    content="Fixed memory leak in scratchpad loader by adding proper cleanup",
    category="bugfix",
    tags=["bugfix", "memory", "performance"],
    metadata={
        "issue_id": "GH-123",
        "commit": "a3f2c1b"
    }
)
```

**Storage Flow:**

1. Validate content and category
2. Normalize and deduplicate tags
3. Generate embedding (if vector backend)
4. Store to backend (file or vector)
5. Add to scratchpad (hot cache)
6. Emit `memory:stored` event
7. Check migration threshold

**Cost:**
- File storage: ~0ms, $0
- Vector storage: ~300ms, ~$0.0001 (OpenAI embedding)

---

### memory_promote

Manually promote a memory to the scratchpad (hot cache).

**Signature:**
```python
async def memory_promote(memory_id: str) -> ToolResult
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `memory_id` | `str` | Yes | Memory ID to promote (e.g., "mem-a3f2c1b4e5d6") |

**Returns:**

```python
ToolResult(
    output={
        "memory_id": "mem-a3f2c1b4e5d6",
        "promoted": True,
        "evicted": "mem-xyz789",  # ID of memory evicted to make room (if any)
        "scratchpad_size": 47
    },
    error=None
)
```

**Examples:**

```python
# Promote an important memory
result = await memory_promote("mem-a3f2c1b4e5d6")

# Check if promotion was successful
if result.output["promoted"]:
    print(f"Memory promoted to scratchpad (size: {result.output['scratchpad_size']})")
```

**Use Cases:**
- Pin important memories for quick access
- Ensure critical context is always available at session start
- Override automatic LRU eviction policy

**Automatic Promotion:**
Memories are also automatically promoted when `access_count >= 3` (configurable).

---

### memory_list_recent

List recent memories from the scratchpad.

**Signature:**
```python
async def memory_list_recent(limit: int = 10) -> ToolResult
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | `int` | No | `10` | Maximum entries to return (1-100) |

**Returns:**

```python
ToolResult(
    output={
        "memories": [
            {
                "id": "mem-a3f2c1b4e5d6",
                "content": "Prefer PostgreSQL for new projects",
                "category": "preference",
                "tags": ["database", "postgresql"],
                "timestamp": "2026-01-28T12:00:00Z",
                "access_count": 3,
                "last_accessed": "2026-01-30T14:00:00Z"
            },
            # ... more memories
        ],
        "scratchpad_size": 47,
        "backend": "file"
    },
    error=None
)
```

**Examples:**

```python
# List all scratchpad entries
result = await memory_list_recent()

# List top 5 most recent
result = await memory_list_recent(limit=5)

# Display formatted list
for mem in result.output["memories"]:
    print(f"[{mem['timestamp']}] {mem['category']}: {mem['content']}")
```

**Use Cases:**
- Display recent memory snapshot
- Debug scratchpad state
- Review what context will be injected at session start

**Performance:** ~10ms (simple file read)

---

## Python Module APIs

Developer APIs for building on top of the memory system.

### Storage Abstraction

Base interface for memory storage backends.

#### MemoryStorage (Abstract Base Class)

```python
from abc import ABC, abstractmethod
from typing import Optional

class MemoryStorage(ABC):
    """Abstract storage interface for memory backends."""
    
    def __init__(self, agent_id: str):
        """
        Initialize storage for an agent.
        
        Args:
            agent_id: Agent namespace (validated, alphanumeric + hyphens/underscores)
        
        Raises:
            ValueError: If agent_id is invalid
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[dict] = None
    ) -> list[MemorySearchResult]:
        """
        Search memories semantically.
        
        Args:
            query: Natural language query
            limit: Maximum results
            filters: Optional filtering criteria
        
        Returns:
            List of MemorySearchResult objects, ranked by relevance
        
        Raises:
            StorageError: If search fails
        """
        pass
    
    @abstractmethod
    async def store(self, memory: Memory) -> str:
        """
        Store a new memory.
        
        Args:
            memory: Memory object to store
        
        Returns:
            Memory ID
        
        Raises:
            StorageError: If storage fails
            ValidationError: If memory is invalid
        """
        pass
    
    @abstractmethod
    async def get(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: Memory identifier
        
        Returns:
            Memory object or None if not found
        """
        pass
    
    @abstractmethod
    async def list_recent(self, limit: int = 10) -> list[Memory]:
        """
        List recent memories (from scratchpad).
        
        Args:
            limit: Maximum memories to return
        
        Returns:
            List of Memory objects, sorted by recency
        """
        pass
    
    @abstractmethod
    async def update_access_stats(self, memory_id: str) -> None:
        """
        Update access statistics for a memory.
        
        Args:
            memory_id: Memory to update
        
        Raises:
            StorageError: If update fails
        """
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """
        Count total memories for this agent.
        
        Returns:
            Total memory count
        """
        pass
```

---

### FileStorage

YAML-based storage implementation (Phase 1).

```python
class FileStorage(MemoryStorage):
    """
    File-based memory storage using YAML.
    
    Features:
    - Simple keyword search
    - Fast for <5k memories
    - Human-readable format
    - No external dependencies
    """
    
    def __init__(self, agent_id: str, storage_root: str = "~/.amplifier/memory"):
        """
        Initialize file-based storage.
        
        Args:
            agent_id: Agent namespace
            storage_root: Root directory for memory storage
        """
        pass
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[dict] = None
    ) -> list[MemorySearchResult]:
        """
        Keyword-based search in YAML.
        
        Implementation:
        1. Load scratchpad.yaml
        2. Tokenize query
        3. Match keywords in content and tags
        4. Rank by recency and access count
        5. Apply filters
        
        Performance: O(n) scan, ~50ms for 5k memories
        """
        pass
    
    async def store(self, memory: Memory) -> str:
        """
        Append memory to scratchpad.yaml.
        
        Implementation:
        1. Load current scratchpad
        2. Add new memory
        3. Atomic write (temp + rename)
        4. Update metadata.json
        
        Performance: ~10ms
        """
        pass
```

**Usage Example:**

```python
from agent_memory.storage import FileStorage
from agent_memory.models import Memory

# Initialize storage
storage = FileStorage(agent_id="bob")

# Store a memory
memory = Memory(
    agent_id="bob",
    content="Prefer PostgreSQL for new projects",
    category="preference",
    tags=["database", "postgresql"],
    created_by_session="session-abc-123"
)
memory_id = await storage.store(memory)

# Search memories
results = await storage.search(query="database preferences", limit=5)
for result in results:
    print(f"[{result.relevance_score:.2f}] {result.memory.content}")
```

---

### VectorStorage

Qdrant-based vector storage implementation (Phase 2).

```python
class VectorStorage(MemoryStorage):
    """
    Vector database memory storage using Qdrant.
    
    Features:
    - Semantic similarity search
    - Scales to 100k+ memories
    - Freshness decay algorithm
    - Access boosting
    """
    
    def __init__(
        self,
        agent_id: str,
        storage_root: str = "~/.amplifier/memory",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize vector storage.
        
        Args:
            agent_id: Agent namespace
            storage_root: Root directory for memory storage
            embedding_model: OpenAI embedding model name
        """
        pass
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[dict] = None
    ) -> list[MemorySearchResult]:
        """
        Semantic vector similarity search.
        
        Implementation:
        1. Generate query embedding
        2. Search Qdrant collection
        3. Apply metadata filters
        4. Calculate relevance scores (similarity * freshness * access_boost)
        5. Rank and return top K
        
        Performance: ~100-150ms for 100k memories
        """
        pass
    
    async def store(self, memory: Memory) -> str:
        """
        Store memory with vector embedding.
        
        Implementation:
        1. Generate embedding (OpenAI API)
        2. Insert to Qdrant collection
        3. Update scratchpad (hot cache)
        4. Update metadata.json
        
        Performance: ~300ms (embedding API latency)
        Cost: ~$0.0001 per memory
        """
        pass
    
    async def migrate_from_file(self, file_storage: FileStorage) -> MigrationProgress:
        """
        Migrate memories from file storage to vector storage.
        
        Args:
            file_storage: Source FileStorage instance
        
        Returns:
            MigrationProgress tracking object
        
        Raises:
            MigrationError: If migration fails
        """
        pass
```

**Usage Example:**

```python
from agent_memory.storage import VectorStorage

# Initialize vector storage
storage = VectorStorage(
    agent_id="bob",
    embedding_model="text-embedding-3-small"
)

# Semantic search
results = await storage.search(
    query="What did we decide about databases?",
    limit=5,
    filters={"category": "decision"}
)

# Results ranked by semantic similarity + freshness + access
for result in results:
    print(f"[Score: {result.relevance_score:.2f}] {result.memory.content}")
```

---

### Storage Detection

Automatically detect and instantiate the correct storage backend.

```python
def detect_storage(agent_id: str) -> MemoryStorage:
    """
    Automatically detect and instantiate correct storage backend.
    
    Detection logic:
    1. Check if memories.db exists → VectorStorage
    2. Check if migration threshold exceeded → migrate and return VectorStorage
    3. Otherwise → FileStorage
    
    Args:
        agent_id: Agent namespace
    
    Returns:
        Appropriate MemoryStorage implementation
    
    Raises:
        ValueError: If agent_id is invalid
        StorageError: If storage initialization fails
    """
    from pathlib import Path
    
    agent_root = Path(f"~/.amplifier/memory/{agent_id}").expanduser()
    
    # Already migrated?
    if (agent_root / "memories.db").exists():
        return VectorStorage(agent_id)
    
    # Should migrate?
    if should_migrate(agent_id):
        if get_config("auto_migrate"):
            return migrate_to_vector(agent_id)
        else:
            emit_event("memory:migration_suggested", {"agent_id": agent_id})
    
    # Use file storage
    return FileStorage(agent_id)
```

**Usage Example:**

```python
from agent_memory.storage import detect_storage

# Automatically use correct backend
storage = detect_storage(agent_id="bob")

# Works regardless of backend
results = await storage.search(query="database work")
```

---

### Scratchpad Management

Hot cache management utilities.

```python
class ScratchpadManager:
    """
    Manage the scratchpad hot cache.
    
    Responsibilities:
    - Load/save scratchpad.yaml
    - LRU eviction policy
    - Promotion logic
    - Token budget management
    """
    
    def __init__(self, agent_id: str, max_size: int = 50):
        """
        Initialize scratchpad manager.
        
        Args:
            agent_id: Agent namespace
            max_size: Maximum scratchpad entries (10-100)
        """
        pass
    
    async def load(self) -> Scratchpad:
        """Load scratchpad from disk."""
        pass
    
    async def save(self, scratchpad: Scratchpad) -> None:
        """Save scratchpad to disk (atomic write)."""
        pass
    
    async def add_memory(self, memory: Memory) -> Optional[str]:
        """
        Add memory to scratchpad, evicting LRU if full.
        
        Args:
            memory: Memory to add
        
        Returns:
            ID of evicted memory (if any)
        """
        pass
    
    async def promote(self, memory_id: str) -> bool:
        """
        Manually promote memory to scratchpad.
        
        Args:
            memory_id: Memory to promote
        
        Returns:
            True if promoted, False if already in scratchpad
        """
        pass
    
    async def format_for_context(self, token_budget: int = 2000) -> str:
        """
        Format scratchpad for context injection.
        
        Args:
            token_budget: Maximum tokens to use
        
        Returns:
            Formatted markdown string
        """
        pass
```

**Usage Example:**

```python
from agent_memory.scratchpad import ScratchpadManager

manager = ScratchpadManager(agent_id="bob", max_size=50)

# Load scratchpad
scratchpad = await manager.load()

# Add a memory
await manager.add_memory(memory)

# Format for context injection
context_text = await manager.format_for_context(token_budget=2000)
print(context_text)
# Output:
# ## Recent Memories
# 
# - [2026-01-28] Preference: PostgreSQL for new projects
# - [2026-01-27] Decision: Use Qdrant for vector storage
# ...
```

---

### Embedding Utilities

Generate embeddings for semantic search.

```python
class EmbeddingGenerator:
    """
    Generate vector embeddings using OpenAI API.
    
    Features:
    - Batch processing
    - Retry logic
    - Cost tracking
    - Caching
    """
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize embedding generator.
        
        Args:
            model: OpenAI embedding model name
        """
        pass
    
    async def generate(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed (max 8191 tokens)
        
        Returns:
            Embedding vector (1536 dimensions)
        
        Raises:
            EmbeddingError: If generation fails
        """
        pass
    
    async def generate_batch(
        self,
        texts: list[str],
        batch_size: int = 100
    ) -> list[list[float]]:
        """
        Generate embeddings in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size (default: 100)
        
        Returns:
            List of embedding vectors
        
        Raises:
            EmbeddingError: If generation fails
        """
        pass
    
    def estimate_cost(self, texts: list[str]) -> float:
        """
        Estimate embedding cost in USD.
        
        Args:
            texts: Texts to embed
        
        Returns:
            Estimated cost (e.g., 0.0001)
        """
        pass
```

**Usage Example:**

```python
from agent_memory.embeddings import EmbeddingGenerator

generator = EmbeddingGenerator(model="text-embedding-3-small")

# Single embedding
embedding = await generator.generate("Prefer PostgreSQL for new projects")
print(f"Dimensions: {len(embedding)}")  # 1536

# Batch embeddings (efficient)
texts = [memory.content for memory in memories]
embeddings = await generator.generate_batch(texts, batch_size=100)

# Estimate cost
cost = generator.estimate_cost(texts)
print(f"Estimated cost: ${cost:.4f}")
```

---

## Configuration Schemas

### Session Configuration

Enable memory for a session with agent identity.

```yaml
# ~/.amplifier/settings.yaml or session config

session:
  capabilities:
    agent_identity: "bob"  # Required: agent namespace
    memory:
      enabled: true        # Enable memory system
      mode: "autonomous"   # Agent decides when to query
```

**Programmatic Access:**

```python
from amplifier_foundation import get_capability

def mount(coordinator):
    agent_id = get_capability(coordinator, "agent_identity")
    
    if not agent_id:
        raise ValueError("Memory system requires agent_identity capability")
    
    # Initialize storage for this agent
    storage = detect_storage(agent_id)
    return MemoryTools(storage)
```

---

### Tool Module Configuration

Configure `tool-memory-semantic` behavior.

```yaml
# bundle.yaml

modules:
  - type: tool
    module: tool-memory-semantic
    config:
      # Storage
      storage_root: "~/.amplifier/memory"
      
      # Search defaults
      default_search_limit: 5
      similarity_threshold: 0.75
      
      # Embedding
      embedding_model: "text-embedding-3-small"
      embedding_dimensions: 1536
      
      # Migration
      migration_threshold: 5000
      auto_migrate: false
      migration_batch_size: 100
      
      # Performance
      enable_caching: true
      cache_ttl_seconds: 300
      
      # Freshness decay
      freshness_decay_rate: 0.1    # 10% decay per day
      access_boost_rate: 0.05       # 5% boost per access
```

---

### Context Module Configuration

Configure `context-memory-scratchpad` behavior.

```yaml
# bundle.yaml

modules:
  - type: context
    module: context-memory-scratchpad
    config:
      # Token budget
      token_budget: 2000
      
      # Scratchpad size
      max_scratchpad_entries: 50
      
      # Promotion thresholds
      promotion_access_threshold: 3
      
      # Formatting
      include_timestamps: true
      include_categories: true
      include_tags: true
      
      # Filtering
      exclude_categories: []  # e.g., ["question", "note"]
```

---

### Hook Module Configuration

Configure `hooks-memory-capture` behavior.

```yaml
# bundle.yaml

modules:
  - type: hook
    module: hooks-memory-capture
    config:
      # Capture patterns (regex)
      capture_patterns:
        - '\b(remember|important|decided to|learned that)\b'
        - '\b(discovered|solved by|preference:)\b'
        - '\b(TODO|FIXME|NOTE):\s'
      
      # Confidence thresholds
      min_confidence: 0.5
      
      # Event hooks
      observe_tool_outputs: true
      observe_turn_completions: true
      observe_session_end: false
      
      # Content filtering
      min_content_length: 10
      max_content_length: 10000
      
      # Rate limiting
      max_captures_per_session: 100
      
      # Async processing
      async_capture: true
```

---

## Integration Examples

### Bundle Inclusion

Include memory behaviors in your bundle.

```yaml
# my-bundle/bundle.yaml

meta:
  name: my-agent-with-memory
  version: 1.0.0
  description: Agent with semantic memory capabilities

includes:
  # Include semantic memory bundle
  - source: amplifier-bundle-agent-memory
    behaviors:
      - memory-semantic        # Semantic search tools
      - memory-capture         # Auto-capture observations

# Optional: Override configurations
modules:
  - type: tool
    module: tool-memory-semantic
    config:
      migration_threshold: 10000  # Custom threshold
```

---

### Session Initialization

Start a memory-enabled session.

**YAML Configuration:**

```yaml
# session-bob.yaml

session:
  capabilities:
    agent_identity: "bob"
    memory:
      enabled: true
      mode: "autonomous"

bundles:
  - my-agent-with-memory
```

**Natural Language Activation:**

```
User: "Start as bob with memory enabled"

Agent: [Recognizes pattern]
       [Injects session capability: agent_identity="bob"]
       [Enables memory tools]
       
       "Memory enabled. I'm bob. I can remember our conversations."
```

**Programmatic Activation:**

```python
from amplifier import Session

session = Session(
    capabilities={
        "agent_identity": "bob",
        "memory": {"enabled": True, "mode": "autonomous"}
    },
    bundles=["my-agent-with-memory"]
)

await session.start()
```

---

### Natural Language Usage

Memory tools are invoked naturally during conversation.

**User Explicit Query:**

```
User: "What database preferences do I have?"

Agent: [Invokes memory_search(query="database preferences")]
       [Receives results]
       
       "Based on your memories:
       - You prefer PostgreSQL for new projects (stored 2 days ago)
       - You decided to use Qdrant for vector storage (stored 3 days ago)
       - You noted that SQLite is good for embedded use cases (stored 1 week ago)"
```

**Agent Autonomous Query:**

```
User: "Help me choose a database for this project"

Agent: [Decides to check memory context]
       [Invokes memory_search(query="database preferences decisions")]
       
       "I recall you prefer PostgreSQL for new projects. Would you like to use
       PostgreSQL for this project as well?"
```

**Explicit Storage:**

```
User: "Remember that I prefer dark mode for all tools"

Agent: [Invokes memory_store(
           content="Prefer dark mode for all tools",
           category="preference",
           tags=["ui", "preferences", "dark-mode"]
       )]
       
       "Stored. I'll remember your dark mode preference."
```

**Auto-Capture:**

```
User: "After debugging, I learned that clearing the cache fixes the issue"

Agent: [Detects pattern "learned that"]
       [Hook auto-captures observation]
       [Stores as category="discovery"]
       
       "Good to know! I've noted that discovery."
```

---

### Multi-Agent Scenarios

Different agents with isolated memory pools.

```python
# Agent Bob
session_bob = Session(
    capabilities={"agent_identity": "bob"},
    bundles=["agent-memory"]
)

# Bob's memories stored in ~/.amplifier/memory/bob/
await memory_store(content="Bob's preference", ...)

# Agent Alice (separate namespace)
session_alice = Session(
    capabilities={"agent_identity": "alice"},
    bundles=["agent-memory"]
)

# Alice's memories stored in ~/.amplifier/memory/alice/
# No access to Bob's memories
await memory_store(content="Alice's preference", ...)
```

**Namespace Isolation Guarantees:**

- Each agent has separate storage directory
- Agent ID validated (alphanumeric + hyphens/underscores only)
- Path traversal prevented
- No cross-agent access without explicit sharing (V2.0)

---

### Context Sink Pattern (Token Efficiency)

Delegate heavy memory queries to sub-agent.

```python
# Main session stays lightweight
delegate(
    agent="agent-memory:memory-assistant",
    instruction="Analyze all database-related memories from last month and summarize key decisions"
)

# Sub-agent performs heavy lifting
# - Loads full memory context (~10k tokens)
# - Performs complex queries
# - Analyzes and summarizes
# - Returns summary (~200 tokens to parent)

# Main session receives only summary
# Token savings: ~98% (10k → 200 tokens)
```

**Memory Assistant Agent:**

```yaml
# agents/memory-assistant.yaml

---
meta:
  name: memory-assistant
  description: Expert at searching and analyzing agent memories

system: |
  You are a memory search specialist. You have access to the full memory store
  and can perform complex queries, aggregations, and analysis.
  
  Always return concise summaries to the parent session, not raw memory dumps.
  Focus on insights and patterns.

tools:
  - memory_search
  - memory_list_recent
---

# Memory Assistant
Ready to search and analyze memories.
```

---

## Error Handling

### Exception Types

```python
class MemoryError(Exception):
    """Base exception for memory system errors."""
    pass

class StorageError(MemoryError):
    """Storage backend error (file I/O, Qdrant, etc.)."""
    pass

class ValidationError(MemoryError):
    """Data validation error (invalid memory, agent_id, etc.)."""
    pass

class EmbeddingError(MemoryError):
    """Embedding generation error (API failure, quota, etc.)."""
    pass

class MigrationError(MemoryError):
    """Migration process error."""
    pass

class ConfigurationError(MemoryError):
    """Configuration error (missing agent_identity, invalid config, etc.)."""
    pass
```

---

### Error Recovery Patterns

#### Storage Failures

```python
try:
    results = await storage.search(query="database work")
except StorageError as e:
    logger.error(f"Storage error: {e}")
    
    # Fallback to scratchpad only
    scratchpad = await scratchpad_manager.load()
    results = scratchpad.search_keywords(query)
    
    # Notify user
    return ToolResult(
        output={"results": results, "degraded": True},
        error=f"Search degraded (storage unavailable): {e}"
    )
```

---

#### Embedding Failures

```python
try:
    embedding = await embedder.generate(content)
except EmbeddingError as e:
    logger.error(f"Embedding failed: {e}")
    
    # Store without embedding (file mode)
    memory.embedding = None
    memory_id = await storage.store(memory)
    
    return ToolResult(
        output={"memory_id": memory_id, "semantic_search_disabled": True},
        error=f"Stored without embedding: {e}"
    )
```

---

#### Migration Failures

```python
try:
    progress = await vector_storage.migrate_from_file(file_storage)
except MigrationError as e:
    logger.error(f"Migration failed: {e}")
    
    # Automatic rollback
    await rollback_migration(agent_id)
    
    # Emit event
    emit_event("memory:migration_failed", {
        "agent_id": agent_id,
        "error": str(e),
        "rolled_back": True
    })
    
    return ToolResult(
        output=None,
        error=f"Migration failed and rolled back: {e}"
    )
```

---

#### Validation Failures

```python
try:
    memory = Memory(**data)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    
    return ToolResult(
        output=None,
        error=f"Invalid memory data: {e}"
    )
```

---

### Graceful Degradation

The system degrades gracefully under failure conditions:

| Failure | Degradation Strategy | User Impact |
|---------|---------------------|-------------|
| Vector DB unavailable | Fall back to scratchpad only | Keyword search instead of semantic |
| Embedding API fails | Store without embedding | File storage continues working |
| Scratchpad corrupted | Rebuild from backup | Recent memories may be lost |
| Migration fails | Automatic rollback | Continue with file storage |
| Invalid agent_id | Reject operation | Clear error message |

---

## Events

The memory system emits events for observability and integration.

### Event: memory:stored

Emitted when a memory is successfully stored.

**Payload:**

```python
{
    "agent_id": "bob",
    "memory_id": "mem-a3f2c1b4e5d6",
    "category": "preference",
    "tags": ["database", "postgresql"],
    "backend": "file",  # or "vector"
    "in_scratchpad": True,
    "session_id": "session-abc-123",
    "timestamp": "2026-01-30T16:00:00Z"
}
```

**Usage:**

```python
# Listen for storage events
coordinator.hooks.register("memory:stored", on_memory_stored)

async def on_memory_stored(event_data: dict):
    logger.info(f"Memory {event_data['memory_id']} stored for {event_data['agent_id']}")
    
    # Update UI, send notification, etc.
    await notify_user(f"Stored: {event_data['category']}")
```

---

### Event: memory:searched

Emitted when a memory search is performed.

**Payload:**

```python
{
    "agent_id": "bob",
    "query": "database preferences",
    "results_count": 5,
    "total_found": 12,
    "latency_ms": 87.3,
    "backend": "vector",
    "session_id": "session-abc-123",
    "timestamp": "2026-01-30T16:00:00Z"
}
```

**Usage:**

```python
# Track search analytics
coordinator.hooks.register("memory:searched", on_memory_searched)

async def on_memory_searched(event_data: dict):
    # Update metrics
    await metrics.record("memory.search.latency", event_data["latency_ms"])
    await metrics.increment("memory.search.count")
```

---

### Event: memory:migration_suggested

Emitted when memory count exceeds migration threshold.

**Payload:**

```python
{
    "agent_id": "bob",
    "memory_count": 5234,
    "threshold": 5000,
    "auto_migrate": False,
    "estimated_duration_seconds": 261.7,
    "estimated_cost_usd": 0.0523,
    "timestamp": "2026-01-30T16:00:00Z"
}
```

**Usage:**

```python
# Prompt user for migration
coordinator.hooks.register("memory:migration_suggested", on_migration_suggested)

async def on_migration_suggested(event_data: dict):
    agent_id = event_data["agent_id"]
    count = event_data["memory_count"]
    cost = event_data["estimated_cost_usd"]
    
    await notify_user(
        f"Agent '{agent_id}' has {count} memories. "
        f"Migrate to vector storage? (Est. ${cost:.4f})"
    )
```

---

### Event: memory:migration_started

Emitted when migration begins.

**Payload:**

```python
{
    "agent_id": "bob",
    "memory_count": 5234,
    "plan": {
        "batch_size": 100,
        "estimated_duration_seconds": 261.7
    },
    "timestamp": "2026-01-30T16:00:00Z"
}
```

---

### Event: memory:migration_progress

Emitted periodically during migration.

**Payload:**

```python
{
    "agent_id": "bob",
    "memories_processed": 1000,
    "total": 5234,
    "percent_complete": 19.1,
    "elapsed_seconds": 50.3,
    "timestamp": "2026-01-30T16:00:50Z"
}
```

---

### Event: memory:migration_completed

Emitted when migration succeeds.

**Payload:**

```python
{
    "agent_id": "bob",
    "memory_count": 5234,
    "duration_seconds": 265.8,
    "cost_usd": 0.0523,
    "backend": "vector",
    "timestamp": "2026-01-30T16:04:25Z"
}
```

---

### Event: memory:migration_failed

Emitted when migration fails.

**Payload:**

```python
{
    "agent_id": "bob",
    "error": "Qdrant connection timeout",
    "memories_processed": 1000,
    "total": 5234,
    "rolled_back": True,
    "timestamp": "2026-01-30T16:00:50Z"
}
```

---

### Event: memory:promoted

Emitted when a memory is promoted to scratchpad.

**Payload:**

```python
{
    "agent_id": "bob",
    "memory_id": "mem-a3f2c1b4e5d6",
    "reason": "manual",  # or "access_threshold"
    "scratchpad_size": 47,
    "evicted": "mem-xyz789",  # if any
    "timestamp": "2026-01-30T16:00:00Z"
}
```

---

## Quick Reference

### Common Tasks

| Task | API Call |
|------|----------|
| Search memories | `memory_search(query="...", limit=5)` |
| Store memory | `memory_store(content="...", category="...", tags=[...])` |
| List recent | `memory_list_recent(limit=10)` |
| Promote memory | `memory_promote(memory_id="...")` |
| Initialize storage | `storage = detect_storage(agent_id)` |
| Generate embedding | `embedding = await embedder.generate(text)` |
| Load scratchpad | `scratchpad = await manager.load()` |

---

### Configuration Quick Start

**Minimal Configuration:**

```yaml
session:
  capabilities:
    agent_identity: "bob"
    memory:
      enabled: true
```

**Production Configuration:**

```yaml
session:
  capabilities:
    agent_identity: "bob"
    memory:
      enabled: true
      mode: "autonomous"

bundles:
  - amplifier-bundle-agent-memory

tool-memory-semantic:
  migration_threshold: 10000
  auto_migrate: false
  embedding_model: "text-embedding-3-small"

context-memory-scratchpad:
  token_budget: 2000
  max_scratchpad_entries: 50
  promotion_access_threshold: 3

hooks-memory-capture:
  min_confidence: 0.6
  observe_tool_outputs: true
  async_capture: true
```

---

### Performance Guidelines

| Operation | Latency Target | Notes |
|-----------|----------------|-------|
| `memory_search` (scratchpad) | <10ms | Hot cache only |
| `memory_search` (file) | <50ms | Keyword scan |
| `memory_search` (vector) | <150ms | Semantic search |
| `memory_store` (file) | <20ms | Simple append |
| `memory_store` (vector) | <300ms | Includes embedding |
| `memory_list_recent` | <10ms | Scratchpad read |
| `memory_promote` | <20ms | Scratchpad update |
| Context injection | <50ms | Session start only |

---

### Cost Estimates

| Operation | Cost | Notes |
|-----------|------|-------|
| Store memory (file) | $0 | Free |
| Store memory (vector) | ~$0.0001 | OpenAI embedding |
| Search (file) | $0 | Free |
| Search (vector) | ~$0.0001 | Query embedding |
| Migration (5k memories) | ~$0.05 | One-time |

---

## Summary

This API reference provides comprehensive documentation for:

✅ **Tool APIs**: All memory tools agents can invoke  
✅ **Python Module APIs**: Developer interfaces for building on the system  
✅ **Configuration Schemas**: Complete setup and configuration options  
✅ **Integration Examples**: Real-world usage patterns  
✅ **Error Handling**: Exception types and recovery strategies  
✅ **Events**: Observability and integration hooks  

**Key Patterns:**

1. **Storage Abstraction**: `FileStorage` → `VectorStorage` hybrid evolution
2. **Context Sink**: Delegate heavy queries to sub-agents for token efficiency
3. **Scratchpad**: Hot cache for fast access and session initialization
4. **Graceful Degradation**: System continues working under failure
5. **Agent Isolation**: Multi-tenant with namespace security

**Next Steps:**

- Implement tool module following this API
- Build storage layer with test coverage
- Create migration engine with progress tracking
- Add comprehensive error handling
- Emit events for observability

---

**End of API Reference Document**
