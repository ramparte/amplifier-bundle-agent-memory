# Architecture: amplifier-bundle-agent-memory

## Document Status
**Version**: 0.1.0 (Design Phase)  
**Date**: 2026-01-30  
**Status**: Validated by foundation:foundation-expert

---

## System Overview

The agent-memory bundle provides semantic memory capabilities for Amplifier, enabling named agent identities to maintain persistent, searchable memory pools across sessions. The system evolves from simple file-based storage to vector databases as scale demands.

### Key Capabilities

- **Named agent identities**: Each agent (e.g., "bob", "alice") has isolated memory namespace
- **Semantic search**: Natural language queries via vector embeddings
- **Tiered storage**: Hot scratchpad (recent) + cold deep storage (historical)
- **Hybrid evolution**: Start file-based, migrate to vector DB at scale (~5k memories)
- **Opt-in stateful sessions**: Memory disabled by default, explicit enablement required
- **Freshness decay**: Stale memories require tighter relevance threshold
- **Token efficiency**: Context sink pattern for heavy queries

---

## High-Level Component Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Amplifier Session                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐ │
│  │  Context Module │    │   Tool Module    │    │  Hook Module    │ │
│  │  ───────────────│    │  ──────────────  │    │  ─────────────  │ │
│  │  Scratchpad     │    │  memory_search() │    │  Auto-capture   │ │
│  │  injection at   │    │  memory_store()  │    │  observations   │ │
│  │  session start  │    │  memory_promote()│    │  from tools     │ │
│  └────────┬────────┘    └────────┬─────────┘    └────────┬────────┘ │
│           │                      │                       │          │
│           └──────────────────────┼───────────────────────┘          │
│                                  │                                  │
└──────────────────────────────────┼──────────────────────────────────┘
                                   │
                                   ▼
            ┌──────────────────────────────────────────┐
            │      Storage Layer (Multi-Tenant)        │
            ├──────────────────────────────────────────┤
            │  ~/.amplifier/memory/                    │
            │  ├── bob/                                │
            │  │   ├── scratchpad.yaml  (hot cache)   │
            │  │   ├── memories.db      (vector DB)   │
            │  │   └── metadata.json                  │
            │  ├── alice/                              │
            │  │   └── scratchpad.yaml                │
            │  └── shared/  (V2.0)                    │
            └──────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Tool Module: tool-memory-semantic

**Purpose**: Storage abstraction layer providing memory CRUD operations

**Protocol**: Amplifier Tool protocol (multiple tools in one module)

**Tools Provided**:
```python
memory_search(query: str, limit: int = 5) -> ToolResult
memory_store(content: str, category: str, tags: list) -> ToolResult
memory_promote(memory_id: str) -> ToolResult
memory_list_recent(limit: int = 10) -> ToolResult
```

**Storage Abstraction**:
```python
class MemoryStorage(ABC):
    """Abstract storage interface."""
    async def search(self, agent_id: str, query: str) -> list[Memory]
    async def store(self, agent_id: str, memory: Memory) -> str
    async def list_recent(self, agent_id: str, limit: int) -> list[Memory]

class FileStorage(MemoryStorage):
    """YAML-based storage (Phase 1)."""
    # Simple file operations, keyword search

class VectorStorage(MemoryStorage):
    """Qdrant-based storage (Phase 2)."""
    # Embedding generation, semantic search, freshness decay
```

**Backend Detection**:
```python
def detect_storage(agent_id: str) -> MemoryStorage:
    """Automatically detect and instantiate correct backend."""
    agent_root = Path(f"~/.amplifier/memory/{agent_id}").expanduser()
    
    if (agent_root / "memories.db").exists():
        return VectorStorage(agent_id)
    elif should_migrate(agent_id):
        return migrate_to_vector(agent_id)
    else:
        return FileStorage(agent_id)
```

**Dependencies**:
- `qdrant-client`: Vector database client
- `openai`: Embedding API
- `pyyaml`: YAML parsing
- `pydantic`: Data validation

---

### 2. Context Module: context-memory-scratchpad

**Purpose**: Inject scratchpad (hot cache) at session initialization

**Protocol**: Amplifier ContextManager protocol

**Interface**:
```python
class MemoryScratchpadContext:
    async def prepare_context(self, messages: list[dict]) -> list[dict]:
        """
        Called once at session start.
        Enriches message list with scratchpad memories.
        """
        agent_id = get_capability(coordinator, "agent_identity")
        if not agent_id:
            return messages  # No agent identity, skip injection
        
        # Load scratchpad (hot cache, 10-50 recent memories)
        scratchpad = await self._load_scratchpad(agent_id)
        
        # Inject as system message
        memory_message = {
            "role": "system",
            "content": self._format_scratchpad(scratchpad)
        }
        
        return [memory_message, *messages]
```

**Token Budget**:
- **Target**: 500-2000 tokens for scratchpad
- **Strategy**: Load 10-50 most recent memories
- **Overflow**: Truncate oldest first, maintain recency

**Format**:
```markdown
## Recent Memories

- [2026-01-28] Preference: PostgreSQL for new projects (database, tools)
- [2026-01-27] Decision: Use Qdrant for vector storage (architecture, tools)
- [2026-01-26] Task: Complete database migration design (active)
```

---

### 3. Hook Module: hooks-memory-capture

**Purpose**: Passive observation and auto-capture of memories

**Protocol**: Amplifier Hook protocol

**Event Observation**:
```python
class MemoryCaptureHook:
    async def handle_event(self, event: str, data: dict) -> HookResult:
        # Primary: Tool outputs (rich content)
        if event == "tool:after_execute":
            result = data.get("result")
            if self._should_capture(result):
                await self._extract_and_store(result, data)
        
        # Secondary: Assistant responses (explicit "remember")
        elif event == "turn:end":
            turn_data = data.get("turn")
            if self._contains_patterns(turn_data):
                await self._store_observation(turn_data, data)
        
        return HookResult(action="continue")
```

**Capture Patterns**:
```python
patterns = [
    "remember",
    "important",
    "decided to",
    "learned that",
    "discovered",
    "solved by",
    "preference:",
    "TODO:",
    "FIXME:"
]
```

**Observation Types** (inspired by reference implementation):
- `preference`: User preferences and choices
- `decision`: Architectural or implementation decisions
- `discovery`: Learnings and insights
- `pattern`: Code patterns and conventions
- `task`: Active work items
- `bugfix`: Bug resolutions

---

### 4. Memory Assistant Agent

**Purpose**: Context sink for heavy memory queries

**Pattern**: Sub-agent delegation absorbs token cost

**Usage**:
```python
# Main agent delegates to memory-assistant
delegate(
    agent="agent-memory:memory-assistant",
    instruction="Search my memories for database-related work from last week"
)
```

**Token Efficiency**:
- **Main session cost**: ~200 tokens (summary only)
- **Sub-agent cost**: ~2000 tokens (full query + results)
- **Savings**: 90% token reduction in main session

**Implementation**:
```markdown
---
meta:
  name: memory-assistant
  description: Expert at searching and analyzing agent memories

system: |
  You are a memory search specialist. You have access to the full memory store
  and can perform complex queries, aggregations, and analysis.
  
  Return concise summaries to the parent session, not raw memory dumps.

tools:
  - memory_search
  - memory_list_recent
---

# Memory Assistant

Heavy memory operations context...
```

---

## Storage Architecture

### Multi-Tenant Namespace Design

```
~/.amplifier/memory/
├── bob/                          # Agent identity namespace
│   ├── scratchpad.yaml           # Hot cache (10-50 memories)
│   ├── memories.db               # Vector DB (after migration)
│   └── metadata.json             # Agent config, stats
│
├── alice/
│   └── scratchpad.yaml           # Not yet migrated
│
└── shared/                       # V2.0: Cross-agent knowledge
    └── memories.db
```

**Namespace Isolation**:
- Each agent identity has separate storage root
- Path construction: `~/.amplifier/memory/{agent_id}/`
- No cross-agent access without explicit sharing (V2.0)

**metadata.json Schema**:
```json
{
  "agent_id": "bob",
  "created_at": "2026-01-15T10:00:00Z",
  "memory_count": 327,
  "storage_backend": "vector",
  "migration_date": "2026-01-20T14:30:00Z",
  "last_accessed": "2026-01-30T16:00:00Z",
  "stats": {
    "total_queries": 145,
    "avg_query_latency_ms": 87,
    "storage_size_mb": 3.2
  }
}
```

---

### Hybrid Evolution: File → Vector

**Phase 1: File-Based** (0-5k memories)

**Storage**: `scratchpad.yaml` only
```yaml
memories:
  - id: "mem-001"
    timestamp: "2026-01-30T12:00:00Z"
    category: "preference"
    content: "Prefer PostgreSQL for new projects"
    tags: ["database", "tools"]
    access_count: 3
    last_accessed: "2026-01-30T14:00:00Z"
```

**Search**: Simple keyword matching, recency ranking
**Performance**: Fast for <5k memories (~10ms)

---

**Phase 2: Vector Migration** (>5k memories)

**Trigger**: Memory count exceeds threshold (default: 5000)

**Migration Process**:
1. **Backup**: Copy `scratchpad.yaml` → `scratchpad.yaml.backup`
2. **Initialize**: Create Qdrant collection at `{agent_id}/memories.db`
3. **Embed**: Generate embeddings for all memories (batch API calls)
4. **Load**: Insert vectors with metadata to Qdrant
5. **Validate**: Verify count matches, spot-check searches
6. **Switch**: Update `metadata.json` storage_backend → "vector"
7. **Preserve**: Keep original YAML (rollback capability)

**Storage**: `scratchpad.yaml` (hot) + `memories.db` (cold)

**Search**: Semantic similarity via embeddings
**Performance**: Scales to 100k+ memories (~100ms)

---

### Scratchpad (Hot Cache)

**Purpose**: Fast access to recent memories without vector search

**Size**: 10-50 memories (~500-2000 tokens)

**Population Strategy**:
1. **Recency**: Last N days (default: 7)
2. **Access frequency**: Frequently queried memories
3. **Manual promotion**: Via `memory_promote()` tool

**Update Triggers**:
- New memory stored → add to scratchpad, evict oldest
- Memory accessed → bump access count, consider promotion
- Session start → load scratchpad into context

**Eviction Policy**: LRU (Least Recently Used)

---

### Deep Storage (Cold)

**Purpose**: Long-term semantic memory with vector search

**Technology**: Qdrant (embedded local database)

**Vector Dimensions**: 1536 (OpenAI text-embedding-3-small)

**Collection Schema**:
```python
{
    "id": "mem-12345",
    "vector": [0.123, -0.456, ...],  # 1536 dimensions
    "payload": {
        "content": "Prefer PostgreSQL for new projects",
        "agent_id": "bob",
        "timestamp": "2026-01-30T12:00:00Z",
        "category": "preference",
        "tags": ["database", "tools"],
        "access_count": 3,
        "last_accessed": "2026-01-30T14:00:00Z",
        "created_by_session": "session-abc-123"
    }
}
```

**Indexing Strategy**:
- HNSW (Hierarchical Navigable Small World) for approximate nearest neighbor
- Metadata filtering (category, tags, date range)
- Hybrid search: vector similarity + metadata filters

---

## Memory Lifecycle

### 1. Creation: Store Operation

```
User/Agent Input
      │
      ▼
memory_store(content, category, tags)
      │
      ├─▶ Generate embedding (OpenAI API)
      │   └─▶ [0.123, -0.456, ...] (1536 dims)
      │
      ├─▶ Create Memory object
      │   └─▶ id, timestamp, metadata
      │
      ├─▶ Store to backend
      │   ├─▶ FileStorage: Append to scratchpad.yaml
      │   └─▶ VectorStorage: Insert to Qdrant + update scratchpad
      │
      └─▶ Return memory_id
```

**Embedding Generation**:
```python
async def generate_embedding(content: str) -> list[float]:
    response = await openai.embeddings.create(
        model="text-embedding-3-small",
        input=content
    )
    return response.data[0].embedding
```

**Cost**: ~$0.0001 per memory (OpenAI pricing)

---

### 2. Retrieval: Search Flow

```
memory_search(query="database work")
      │
      ▼
1. Check Scratchpad (Hot Cache)
      │ Keyword match in recent 10-50 memories
      │ Fast: ~10ms
      │
      ├─▶ Hits found? → Return immediately
      │
      ▼
2. Query Deep Storage (Cold)
      │ Generate query embedding
      │ Vector similarity search (Qdrant)
      │ Apply freshness decay
      │ Apply access boosting
      │ Latency: ~100ms
      │
      ▼
3. Merge & Rank Results
      │ Combine scratchpad + deep storage
      │ Deduplicate by memory_id
      │ Rank by final score
      │
      ▼
4. Update Access Stats
      │ Increment access_count
      │ Update last_accessed timestamp
      │ Consider promotion to scratchpad
      │
      ▼
Return Top K Results (default: 5)
```

---

### 3. Freshness Decay Algorithm

**Purpose**: Stale memories require tighter relevance match

**Formula**:
```python
def calculate_relevance_score(memory, query_embedding):
    # Base similarity (cosine similarity)
    similarity = cosine_similarity(query_embedding, memory.embedding)
    
    # Freshness decay
    age_days = (now() - memory.timestamp).days
    freshness_factor = 1.0 / (1.0 + 0.1 * age_days)
    
    # Access frequency boost
    access_boost = 1.0 + (0.05 * memory.access_count)
    
    # Final score
    final_score = similarity * freshness_factor * access_boost
    
    return final_score
```

**Dynamic Thresholds**:
```python
def get_similarity_threshold(memory):
    # Fresh, accessed memories: easier to recall
    if memory.access_count > 5:
        return 0.70  # Lower threshold
    
    # Stale, never-accessed memories: harder to recall
    elif memory.access_count == 0 and memory.age_days > 30:
        return 0.85  # Higher threshold
    
    # Default
    else:
        return 0.75
```

**Visualization**:
```
Relevance Score vs Age (with decay)
1.0 │ ●●●●●●
    │       ●●●●●
0.8 │            ●●●●●
    │                 ●●●●●
0.6 │                      ●●●●●
    │                           ●●●●
0.4 └──────────────────────────────────
    0     10    20    30    40    50+ days
    
    ● Accessed frequently (boosted)
    ─ Never accessed (decay only)
```

---

### 4. Access Boosting: Promotion to Scratchpad

**Trigger**: Memory accessed N times (default: 3)

**Process**:
```python
async def access_memory(memory_id: str):
    memory = await storage.get(memory_id)
    
    # Update access stats
    memory.access_count += 1
    memory.last_accessed = now()
    
    # Promote if frequently accessed
    if memory.access_count >= 3 and not in_scratchpad(memory_id):
        await scratchpad.add(memory)  # Add to hot cache
        await scratchpad.evict_lru()  # Evict oldest if full
    
    await storage.update(memory)
```

**Scratchpad Benefits**:
- Faster access (no vector search needed)
- Auto-loaded at session start (always available)
- Token-efficient context injection

---

### 5. Migration: File → Vector DB

**Trigger Detection**:
```python
def should_migrate(agent_id: str) -> bool:
    metadata = load_metadata(agent_id)
    
    # Already migrated?
    if metadata.get("storage_backend") == "vector":
        return False
    
    # Threshold exceeded?
    memory_count = count_memories(agent_id)
    threshold = config.get("migration_threshold", 5000)
    
    return memory_count >= threshold
```

**Migration Strategy** (Configurable):
```yaml
memory:
  migration:
    threshold: 5000
    auto_migrate: false  # Manual by default (safe)
    strategy: "background"  # or "immediate"
```

**Migration Process**:
```python
async def migrate_to_vector(agent_id: str):
    # 1. Backup
    backup_path = f"{agent_id}/scratchpad.yaml.backup"
    shutil.copy(scratchpad_path, backup_path)
    
    # 2. Initialize Qdrant
    client = QdrantClient(path=f"{agent_id}/memories.db")
    client.create_collection(
        collection_name="memories",
        vectors_config={"size": 1536, "distance": "Cosine"}
    )
    
    # 3. Load and embed all memories
    memories = load_yaml(scratchpad_path)
    for memory in memories:
        embedding = await generate_embedding(memory["content"])
        await client.upsert(
            collection_name="memories",
            points=[{
                "id": memory["id"],
                "vector": embedding,
                "payload": memory
            }]
        )
    
    # 4. Validate
    assert client.count("memories") == len(memories)
    
    # 5. Update metadata
    metadata["storage_backend"] = "vector"
    metadata["migration_date"] = now()
    save_metadata(agent_id, metadata)
    
    # 6. Keep original YAML (rollback capability)
    # Don't delete scratchpad.yaml!
```

**Rollback Mechanism**:
```python
async def rollback_migration(agent_id: str):
    # Restore from backup
    backup_path = f"{agent_id}/scratchpad.yaml.backup"
    shutil.copy(backup_path, scratchpad_path)
    
    # Delete vector DB
    os.remove(f"{agent_id}/memories.db")
    
    # Update metadata
    metadata["storage_backend"] = "file"
    save_metadata(agent_id, metadata)
```

---

## Agent Identity System

### Session Capability Pattern

**Declaration**:
```yaml
# User's settings.yaml or session config
session:
  capabilities:
    agent_identity: "bob"  # Required for memory operations
    memory:
      enabled: true        # Opt-in stateful behavior
      mode: "autonomous"   # Agent decides when to query
```

**Module Access**:
```python
# In tool or context module
from amplifier_foundation import get_capability

def mount(coordinator):
    agent_id = get_capability(coordinator, "agent_identity")
    
    if not agent_id:
        raise ValueError("Memory system requires agent_identity capability")
    
    # Use agent_id for storage path resolution
    storage = MemoryStorage(agent_id)
    return MemoryTools(storage)
```

**Natural Language Activation**:
```
User: "Start as bob with memory enabled"

Agent: [Recognizes pattern]
       [Injects session capability: agent_identity="bob"]
       [Enables memory tools]
       
       "Memory enabled. I'm bob."
```

---

### Namespace Isolation Guarantees

**Path Construction**:
```python
def resolve_storage_path(agent_id: str, filename: str) -> Path:
    # Validate agent_id (security)
    if not is_valid_agent_id(agent_id):
        raise ValueError(f"Invalid agent_id: {agent_id}")
    
    # Construct isolated path
    base = Path("~/.amplifier/memory").expanduser()
    agent_root = base / agent_id
    
    # Prevent path traversal
    resolved = (agent_root / filename).resolve()
    if not resolved.is_relative_to(agent_root):
        raise ValueError("Path traversal attempt detected")
    
    return resolved
```

**Validation**:
```python
def is_valid_agent_id(agent_id: str) -> bool:
    # Alphanumeric, hyphens, underscores only
    # No path separators, no special characters
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', agent_id))
```

**Security Test**:
```python
# These should fail validation
assert not is_valid_agent_id("../alice")
assert not is_valid_agent_id("bob/../alice")
assert not is_valid_agent_id("bob/../../etc/passwd")
```

---

## Token Efficiency

### Context Sink Pattern

**Problem**: Heavy memory queries consume main session tokens

**Solution**: Delegate to memory-assistant sub-agent

**Comparison**:
```
WITHOUT context sink:
┌─────────────────────────────────────┐
│ Main Session                        │
│ - Load full memory store (10k tok) │
│ - Search and filter (2k tok)       │
│ - Format results (1k tok)          │
│ ─────────────────────────────────── │
│ Total: 13k tokens in main context  │
└─────────────────────────────────────┘

WITH context sink:
┌─────────────────────────────────────┐
│ Main Session                        │
│ - Delegate to memory-assistant      │
│ - Receive summary (200 tokens)     │
│ ─────────────────────────────────── │
│ Total: 200 tokens in main context  │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Sub-Agent (memory-assistant)        │
│ - Load full memory store (10k tok) │
│ - Search and filter (2k tok)       │
│ - Summarize for parent (1k tok)    │
│ ─────────────────────────────────── │
│ Total: 13k tokens in SUB context   │
│ (isolated, doesn't affect main)    │
└─────────────────────────────────────┘
```

**Token Savings**: 98% (13k → 200 tokens in main session)

---

### Scratchpad Injection Budget

**Target**: 500-2000 tokens at session start

**Strategy**:
```python
async def load_scratchpad(agent_id: str, token_budget: int = 2000):
    memories = await storage.list_recent(agent_id, limit=50)
    
    # Prioritize by recency and access
    ranked = sorted(memories, key=lambda m: (
        m.last_accessed,
        m.access_count
    ), reverse=True)
    
    # Fit within budget
    selected = []
    tokens_used = 0
    
    for memory in ranked:
        memory_tokens = estimate_tokens(memory.content)
        if tokens_used + memory_tokens > token_budget:
            break
        selected.append(memory)
        tokens_used += memory_tokens
    
    return selected
```

**Formatting**:
```markdown
## Recent Memories (8 entries, ~1500 tokens)

- [2026-01-28] Preference: PostgreSQL for new projects
- [2026-01-27] Decision: Use Qdrant for vector storage
- [2026-01-26] Task: Complete database migration design
- ...
```

---

### On-Demand Semantic Search

**Explicit queries bypass scratchpad**, go directly to deep storage:

```python
# Agent decides to query
memory_search(query="What database work have I done?")
```

**Token cost**:
- Query execution: ~50 tokens (tool call)
- Results: ~200-500 tokens (5 memories)
- Total: ~300-550 tokens (only when queried)

**Benefit**: Rich semantic search without loading full context

---

## Integration Points

### Session Lifecycle Integration

```
Session Creation
      │
      ▼
1. Load Bundle (includes memory behavior)
      │
      ▼
2. Mount Modules
      ├─▶ tool-memory-semantic → Tools available
      ├─▶ context-memory-scratchpad → prepare_context()
      └─▶ hooks-memory-capture → Registers event observers
      │
      ▼
3. Check Capabilities
      │ agent_identity present?
      ├─▶ YES: Initialize storage for agent_id
      └─▶ NO: Skip memory initialization (opt-in)
      │
      ▼
4. Prepare Context (context module)
      │ Load scratchpad (~500-2000 tokens)
      │ Inject as system message
      │
      ▼
5. Session Ready
      │ Agent has access to:
      │ - Scratchpad (in context)
      │ - memory_search() tool (explicit queries)
      │ - Auto-capture hook (passive)
```

---

### Event Observation Points

**Hook: hooks-memory-capture**

**Observes**:
```python
# Primary: Tool outputs (rich content)
coordinator.hooks.register("tool:after_execute", capture_hook)

# Secondary: Turn completion (assistant responses)
coordinator.hooks.register("turn:end", capture_hook)

# Tertiary: Session end (full turn summary)
coordinator.hooks.register("orchestrator:complete", capture_hook)
```

**Capture Logic**:
```python
async def handle_tool_after_execute(event_data):
    tool_id = event_data["tool_id"]
    result = event_data["result"]
    
    # Extract interesting content
    if contains_patterns(result):
        observation = extract_observation(result)
        await memory_store(
            content=observation["content"],
            category=observation["type"],  # bugfix, discovery, etc.
            tags=observation["keywords"]
        )
```

---

### Context Injection Timing

**When**: Session initialization (after bundle load, before first turn)

**How**: Context module's `prepare_context()` is called once

**Effect**: Scratchpad appears in conversation context for entire session

**Diagram**:
```
Session Start
      │
      ▼
prepare_context([])  ← Empty message list
      │
      ├─▶ Load scratchpad for agent_id
      ├─▶ Format as system message
      │
      ▼
Return enriched message list:
[
  {"role": "system", "content": "## Recent Memories\n..."},
  ... (user messages follow)
]
      │
      ▼
First LLM Request
      │ Agent sees scratchpad immediately
      │ No explicit query needed
```

**Why once, not every turn?**
- Token efficiency (avoid redundant injection)
- Context accumulation (conversation builds on initial context)
- Dynamic refresh via `memory_search()` tool if needed

---

## Data Models

### Memory Object Schema

```python
from pydantic import BaseModel
from datetime import datetime

class Memory(BaseModel):
    id: str                    # "mem-12345"
    agent_id: str              # "bob"
    timestamp: datetime        # When created
    content: str               # The actual memory text
    category: str              # preference, decision, discovery, etc.
    tags: list[str]            # ["database", "postgresql"]
    embedding: list[float]     # [0.123, -0.456, ...] (1536 dims)
    access_count: int = 0      # How many times retrieved
    last_accessed: datetime | None = None
    created_by_session: str    # Session ID that created it
    metadata: dict = {}        # Extensible metadata
```

---

### Scratchpad Schema (YAML)

```yaml
# ~/.amplifier/memory/bob/scratchpad.yaml
agent_id: "bob"
last_updated: "2026-01-30T16:00:00Z"

memories:
  - id: "mem-001"
    timestamp: "2026-01-28T12:00:00Z"
    category: "preference"
    content: "Prefer PostgreSQL for new projects"
    tags: ["database", "tools"]
    access_count: 3
    last_accessed: "2026-01-30T14:00:00Z"
  
  - id: "mem-002"
    timestamp: "2026-01-27T15:30:00Z"
    category: "decision"
    content: "Use Qdrant for vector storage"
    tags: ["architecture", "tools"]
    access_count: 1
    last_accessed: "2026-01-27T16:00:00Z"
```

---

## Performance Characteristics

### Latency Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Scratchpad access | <10ms | Simple file read |
| memory_store() | <300ms | Embedding API + storage |
| memory_search() (file) | <50ms | Keyword scan in YAML |
| memory_search() (vector) | <150ms | Qdrant query + ranking |
| Context injection | <50ms | Load scratchpad at session start |
| Auto-capture | Async | Non-blocking, background |

### Scalability Limits

| Metric | File Storage | Vector Storage |
|--------|--------------|----------------|
| Memories per agent | ~5,000 | 100,000+ |
| Search quality | Keyword match | Semantic similarity |
| Concurrent agents | 10+ | 10+ |
| Storage per 10k memories | ~2MB (YAML) | ~15MB (embeddings) |
| Query latency | O(n) scan | O(log n) HNSW |

### Token Consumption

| Component | Per Session | Per Query |
|-----------|-------------|-----------|
| Scratchpad injection | 500-2000 | N/A |
| memory_search() call | N/A | 50 |
| memory_search() results | N/A | 200-500 |
| memory-assistant (context sink) | 200 (summary) | N/A |

---

## Reliability & Observability

### Data Persistence

- **Write-ahead logging**: All memory operations logged to events.jsonl
- **Atomic writes**: YAML writes are atomic (write temp, rename)
- **Backup on migration**: Original files preserved during vector migration
- **Rollback capability**: Can revert to file-based if migration fails

### Graceful Degradation

```python
async def memory_search(query: str):
    try:
        # Try vector search
        return await vector_storage.search(query)
    except QdrantError:
        logger.warning("Vector DB unavailable, falling back to scratchpad")
        # Fallback to scratchpad only
        return await scratchpad.search(query)
```

### Observability

**Events emitted**:
```python
# Memory operations
coordinator.emit_event("memory:stored", {
    "agent_id": "bob",
    "memory_id": "mem-123",
    "category": "preference"
})

coordinator.emit_event("memory:searched", {
    "agent_id": "bob",
    "query": "database work",
    "results_count": 5,
    "latency_ms": 87
})

coordinator.emit_event("memory:migration_suggested", {
    "agent_id": "bob",
    "memory_count": 5234,
    "threshold": 5000
})
```

**Metrics tracked** (in metadata.json):
- Total memories
- Query count
- Average query latency
- Storage size
- Last accessed timestamp

---

## Security Considerations

### Namespace Isolation

- **Path validation**: Prevent path traversal attacks
- **Agent ID validation**: Alphanumeric + hyphens/underscores only
- **No cross-agent access**: Each agent strictly isolated to their namespace

### Sensitive Data

- **No automatic redaction**: User responsible for not storing secrets
- **Integration point**: Can compose with `hooks-redaction` for sanitization
- **Local storage only**: No cloud uploads, all data stays on disk

### Embedding Privacy

- **API calls**: Content sent to OpenAI for embeddings
- **User consent**: Should be documented and agreed upon
- **Alternative**: Local embeddings (sentence-transformers) for full privacy

---

## Extension Points

### V2.0 Features (Architecture-Ready)

**Shared Memory Namespace**:
- Add `~/.amplifier/memory/shared/` storage
- Extend search to query both agent + shared namespaces
- Access control policies

**Custom Embedding Providers**:
- Plugin interface for embedding backends
- Support Cohere, Voyage AI, local models
- Fallback chain (primary → backup → local)

**Memory Analytics**:
- Visualize memory evolution over time
- Identify knowledge gaps
- Recommend memories to review/archive

**Export/Import**:
- Memory portability between systems
- Backup/restore capabilities
- Format conversion (YAML ↔ JSON ↔ SQLite)

---

## Summary

The agent-memory architecture provides a **multi-tenant semantic memory system** that:

1. **Scales gracefully**: File → Vector hybrid evolution
2. **Token-efficient**: Context sink + scratchpad injection
3. **Privacy-preserving**: Local storage, namespace isolation
4. **Amplifier-native**: Follows Tool + Context + Hook patterns
5. **Production-ready**: Graceful degradation, observability, rollback

**Key architectural wins**:
- ✅ Protocol compliance (validated by foundation expert)
- ✅ Composable behaviors (semantic + capture separate)
- ✅ Multi-tenant isolation (agent identities enforced)
- ✅ Token efficiency (context sink pattern)
- ✅ Hybrid storage (start simple, scale when needed)

**Ready for implementation** per validated design decisions.

---

**End of Architecture Document**
