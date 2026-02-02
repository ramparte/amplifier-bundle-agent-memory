# Implementation Plan: amplifier-bundle-agent-memory

## Document Status
**Version**: 0.1.0  
**Date**: 2026-01-30  
**Phase**: Design complete, ready for implementation  
**Dependencies**: All design documents validated

---

## Overview

This document provides the complete implementation roadmap for the agent-memory system, broken into phases with clear dependencies and validation gates.

**Design validation**: ✅ Approved by foundation:foundation-expert

---

## Phase 1: Core Storage Layer (File-Based)

**Goal**: Implement file-based storage with agent identity namespacing

**Duration Estimate**: Foundation expert guidance - quality over speed, no time constraints

**Dependencies**: None (starting point)

### Task 1.1: Project Structure Setup

**Deliverables**:
```
amplifier-bundle-agent-memory/
├── modules/
│   └── tool-memory-semantic/
│       ├── pyproject.toml
│       ├── README.md
│       └── memory_semantic/
│           ├── __init__.py
│           ├── models.py      # Pydantic models from DATA_MODELS.md
│           ├── storage.py     # Storage abstraction
│           ├── file_storage.py
│           ├── scratchpad.py
│           └── tools.py       # Tool mount point
```

**Actions**:
- [ ] Create module directory structure
- [ ] Set up pyproject.toml with dependencies:
  ```toml
  [project]
  name = "amplifier-module-tool-memory-semantic"
  dependencies = [
      "amplifier-core>=0.1.0",
      "pyyaml>=6.0",
      "pydantic>=2.0",
  ]
  ```
- [ ] Copy Pydantic models from DATA_MODELS.md to `models.py`

**Validation**:
- [ ] `uv pip install -e .` succeeds
- [ ] `python -c "from memory_semantic import models"` works

---

### Task 1.2: Storage Abstraction Interface

**File**: `modules/tool-memory-semantic/memory_semantic/storage.py`

**Deliverables**:
```python
from abc import ABC, abstractmethod
from typing import Optional
from .models import Memory

class MemoryStorage(ABC):
    """Abstract storage interface for memory backends."""
    
    def __init__(self, agent_id: str, config: dict):
        self.agent_id = agent_id
        self.config = config
        self._validate_agent_id(agent_id)
        self.storage_root = self._resolve_storage_root()
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        limit: int = 5,
        filters: Optional[dict] = None
    ) -> list[Memory]:
        """Search memories by query."""
        pass
    
    @abstractmethod
    async def store(self, memory: Memory) -> str:
        """Store a memory, return memory_id."""
        pass
    
    @abstractmethod
    async def get(self, memory_id: str) -> Optional[Memory]:
        """Get specific memory by ID."""
        pass
    
    @abstractmethod
    async def list_recent(self, limit: int = 10) -> list[Memory]:
        """List recent memories."""
        pass
    
    @abstractmethod
    async def update(self, memory: Memory) -> None:
        """Update existing memory (access stats, etc.)."""
        pass
    
    # Common utility methods (implemented in base)
    def _validate_agent_id(self, agent_id: str) -> None:
        """Validate agent ID prevents path traversal."""
        # Implementation from DATA_MODELS.md
    
    def _resolve_storage_root(self) -> Path:
        """Resolve storage path for agent."""
        # Implementation from ARCHITECTURE.md
```

**Actions**:
- [ ] Implement `MemoryStorage` abstract base class
- [ ] Add agent ID validation (security)
- [ ] Add storage root resolution
- [ ] Add docstrings for all methods

**Validation**:
- [ ] Abstract class can be imported
- [ ] Agent ID validation catches path traversal attempts:
  ```python
  assert_raises(ValueError, MemoryStorage, "../alice", {})
  ```

---

### Task 1.3: File Storage Implementation

**File**: `modules/tool-memory-semantic/memory_semantic/file_storage.py`

**Deliverables**:
```python
class FileStorage(MemoryStorage):
    """YAML-based file storage (Phase 1)."""
    
    async def search(self, query: str, limit: int = 5, filters=None):
        """Keyword search in YAML file."""
        memories = await self._load_all_memories()
        
        # Simple keyword matching
        matches = [
            m for m in memories
            if query.lower() in m.content.lower()
        ]
        
        # Rank by recency and access
        ranked = sorted(matches, key=lambda m: (
            m.last_accessed or m.timestamp,
            m.access_count
        ), reverse=True)
        
        return ranked[:limit]
    
    async def store(self, memory: Memory) -> str:
        """Append to YAML file atomically."""
        # Atomic write pattern from DATA_MODELS.md
    
    async def _load_all_memories(self) -> list[Memory]:
        """Load all memories from YAML."""
        # YAML parsing
```

**Actions**:
- [ ] Implement all abstract methods
- [ ] Add atomic file writes (write to temp, rename)
- [ ] Add YAML serialization/deserialization
- [ ] Add keyword search with ranking
- [ ] Handle missing files gracefully (create on first write)

**Validation**:
- [ ] Unit tests:
  ```python
  storage = FileStorage("test-agent", {})
  memory_id = await storage.store(Memory(...))
  results = await storage.search("test query")
  assert len(results) > 0
  ```
- [ ] File is created with correct structure
- [ ] Atomic writes don't corrupt on failure

---

### Task 1.4: Scratchpad Manager

**File**: `modules/tool-memory-semantic/memory_semantic/scratchpad.py`

**Deliverables**:
```python
class ScratchpadManager:
    """Hot cache manager for recent memories."""
    
    def __init__(self, agent_id: str, config: dict):
        self.agent_id = agent_id
        self.max_size = config.get("scratchpad_size", 10)
        self.scratchpad_path = self._resolve_path()
    
    async def load(self) -> list[Memory]:
        """Load scratchpad memories."""
        pass
    
    async def add(self, memory: Memory) -> None:
        """Add memory to scratchpad."""
        pass
    
    async def evict_lru(self) -> None:
        """Evict least recently used if at capacity."""
        pass
    
    async def promote(self, memory_id: str, storage: MemoryStorage) -> None:
        """Promote memory from deep storage to scratchpad."""
        pass
```

**Actions**:
- [ ] Implement scratchpad CRUD operations
- [ ] Add LRU eviction logic
- [ ] Add token budget management (from ARCHITECTURE.md)
- [ ] Add promotion logic with access count checking

**Validation**:
- [ ] Scratchpad respects max_size
- [ ] LRU eviction works correctly
- [ ] Token budget is enforced

---

### Task 1.5: Tool Implementation (File-Based)

**File**: `modules/tool-memory-semantic/memory_semantic/tools.py`

**Deliverables**:
```python
from amplifier_core import Tool, ToolResult

def mount(coordinator):
    """Mount memory tools."""
    from amplifier_foundation import get_capability
    
    agent_id = get_capability(coordinator, "agent_identity")
    if not agent_id:
        raise ValueError("Memory requires agent_identity capability")
    
    config = coordinator.config.get("memory", {})
    storage = FileStorage(agent_id, config)
    scratchpad = ScratchpadManager(agent_id, config)
    
    return [
        Tool(
            name="memory_search",
            description="Search semantic memory with natural language",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 5}
            },
            execute=lambda **kwargs: _memory_search(storage, scratchpad, **kwargs)
        ),
        Tool(
            name="memory_store",
            description="Store a memory with metadata",
            parameters={
                "content": {"type": "string"},
                "category": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            execute=lambda **kwargs: _memory_store(storage, scratchpad, **kwargs)
        ),
        Tool(
            name="memory_promote",
            description="Promote memory to scratchpad",
            parameters={
                "memory_id": {"type": "string"}
            },
            execute=lambda **kwargs: _memory_promote(storage, scratchpad, **kwargs)
        ),
        Tool(
            name="memory_list_recent",
            description="List recent memories",
            parameters={
                "limit": {"type": "integer", "default": 10}
            },
            execute=lambda **kwargs: _memory_list_recent(scratchpad, **kwargs)
        ),
    ]

async def _memory_search(storage, scratchpad, query, limit=5):
    """Execute memory search."""
    # Check scratchpad first
    scratchpad_hits = await scratchpad.search(query)
    
    # Query deep storage
    deep_hits = await storage.search(query, limit)
    
    # Merge and deduplicate
    all_hits = _merge_results(scratchpad_hits, deep_hits)
    
    return ToolResult(
        success=True,
        data={"memories": [m.dict() for m in all_hits[:limit]]}
    )

async def _memory_store(storage, scratchpad, content, category, tags):
    """Store new memory."""
    memory = Memory(
        id=generate_id(),
        agent_id=storage.agent_id,
        content=content,
        category=category,
        tags=tags,
        timestamp=datetime.utcnow()
    )
    
    memory_id = await storage.store(memory)
    await scratchpad.add(memory)
    
    return ToolResult(
        success=True,
        data={"memory_id": memory_id}
    )
```

**Actions**:
- [ ] Implement all four tools
- [ ] Add proper error handling
- [ ] Add capability checking (agent_identity required)
- [ ] Add result formatting
- [ ] Add event emission (memory:stored, memory:searched)

**Validation**:
- [ ] Tools can be mounted
- [ ] memory_search returns results
- [ ] memory_store creates valid memory
- [ ] memory_promote works
- [ ] Error handling works (missing agent_identity)

---

### Phase 1 Integration Test

**Test script**: `tests/test_phase1_integration.py`

**Scenario**:
```python
async def test_file_storage_lifecycle():
    # 1. Create storage for agent
    storage = FileStorage("bob", {})
    
    # 2. Store memories
    mem1 = await storage.store(Memory(content="Prefer PostgreSQL", ...))
    mem2 = await storage.store(Memory(content="Use Qdrant for vectors", ...))
    
    # 3. Search
    results = await storage.search("database")
    assert len(results) == 1
    assert "PostgreSQL" in results[0].content
    
    # 4. Scratchpad operations
    scratchpad = ScratchpadManager("bob", {"scratchpad_size": 10})
    await scratchpad.add(results[0])
    cached = await scratchpad.load()
    assert len(cached) == 1
    
    # 5. Access boosting
    memory = await storage.get(mem1)
    memory.access_count += 1
    await storage.update(memory)
    
    print("✅ Phase 1 integration test passed")
```

**Validation Gates**:
- [ ] Storage CRUD operations work
- [ ] Scratchpad management works
- [ ] Search returns relevant results
- [ ] Files are created correctly
- [ ] Namespace isolation works (bob can't access alice's memories)

---

## Phase 2: Context Module (Scratchpad Injection)

**Goal**: Inject scratchpad at session start

**Dependencies**: Phase 1 complete and validated

### Task 2.1: Context Module Structure

**Deliverables**:
```
modules/
└── context-memory-scratchpad/
    ├── pyproject.toml
    ├── README.md
    └── context_scratchpad/
        ├── __init__.py
        └── manager.py
```

**Actions**:
- [ ] Create module directory
- [ ] Set up pyproject.toml with dependencies
- [ ] Add dependency on tool-memory-semantic (for scratchpad access)

---

### Task 2.2: Context Manager Implementation

**File**: `modules/context-memory-scratchpad/context_scratchpad/manager.py`

**Deliverables**:
```python
class MemoryScratchpadContext:
    """Context manager for scratchpad injection."""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.config = coordinator.config.get("memory_context", {})
    
    async def prepare_context(self, messages: list[dict]) -> list[dict]:
        """
        Called once at session start.
        Inject scratchpad as system message.
        """
        from amplifier_foundation import get_capability
        
        agent_id = get_capability(self.coordinator, "agent_identity")
        if not agent_id:
            return messages  # No agent identity, skip injection
        
        # Load scratchpad
        scratchpad = ScratchpadManager(agent_id, self.config)
        memories = await scratchpad.load()
        
        if not memories:
            return messages  # No memories to inject
        
        # Format for injection
        memory_content = self._format_scratchpad(memories)
        
        # Inject as system message
        memory_message = {
            "role": "system",
            "content": memory_content
        }
        
        return [memory_message, *messages]
    
    def _format_scratchpad(self, memories: list[Memory]) -> str:
        """Format scratchpad for injection."""
        lines = ["## Recent Memories\n"]
        for memory in memories:
            date = memory.timestamp.strftime("%Y-%m-%d")
            lines.append(f"- [{date}] {memory.content}")
        return "\n".join(lines)
```

**Actions**:
- [ ] Implement ContextManager protocol
- [ ] Add scratchpad loading
- [ ] Add formatting for injection
- [ ] Add token budget enforcement
- [ ] Handle missing agent_identity gracefully

**Validation**:
- [ ] Context module can be mounted
- [ ] prepare_context() returns enriched messages
- [ ] System message format is correct
- [ ] Token budget is respected

---

### Phase 2 Integration Test

**Test script**: `tests/test_phase2_integration.py`

**Scenario**:
```python
async def test_context_injection():
    # 1. Populate scratchpad
    storage = FileStorage("bob", {})
    await storage.store(Memory(content="Prefer PostgreSQL", ...))
    
    # 2. Initialize context module
    context_mgr = MemoryScratchpadContext(coordinator)
    
    # 3. Prepare context
    messages = await context_mgr.prepare_context([
        {"role": "user", "content": "What database should I use?"}
    ])
    
    # 4. Verify injection
    assert messages[0]["role"] == "system"
    assert "Recent Memories" in messages[0]["content"]
    assert "PostgreSQL" in messages[0]["content"]
    
    print("✅ Phase 2 integration test passed")
```

**Validation Gates**:
- [ ] Context injection works at session start
- [ ] Scratchpad contents appear in system message
- [ ] Token budget is enforced
- [ ] Missing agent_identity handled gracefully

---

## Phase 3: Auto-Capture Hook

**Goal**: Passive observation and memory capture

**Dependencies**: Phase 1 complete (storage layer needed)

### Task 3.1: Hook Module Structure

**Deliverables**:
```
modules/
└── hooks-memory-capture/
    ├── pyproject.toml
    ├── README.md
    └── hooks_capture/
        ├── __init__.py
        ├── capture.py
        └── patterns.py
```

---

### Task 3.2: Pattern Matching

**File**: `modules/hooks-memory-capture/hooks_capture/patterns.py`

**Deliverables**:
```python
CAPTURE_PATTERNS = [
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

def contains_capture_pattern(text: str) -> bool:
    """Check if text contains capture keywords."""
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in CAPTURE_PATTERNS)

def extract_observation(text: str) -> dict:
    """Extract structured observation from text."""
    # Category detection
    category = detect_category(text)
    
    # Keyword extraction
    keywords = extract_keywords(text)
    
    return {
        "content": text,
        "category": category,
        "tags": keywords
    }
```

**Actions**:
- [ ] Implement pattern matching
- [ ] Add category detection (from DATA_MODELS.md)
- [ ] Add keyword extraction
- [ ] Add content validation (min length, etc.)

---

### Task 3.3: Hook Implementation

**File**: `modules/hooks-memory-capture/hooks_capture/capture.py`

**Deliverables**:
```python
from amplifier_core import Hook, HookResult

class MemoryCaptureHook:
    """Passive memory capture from observations."""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.config = coordinator.config.get("memory_capture", {})
        self.patterns = self.config.get("patterns", CAPTURE_PATTERNS)
    
    async def handle_event(self, event: str, data: dict) -> HookResult:
        """Observe events and capture memories."""
        
        # Primary: Tool outputs
        if event == "tool:after_execute":
            await self._capture_from_tool(data)
        
        # Secondary: Turn completion
        elif event == "turn:end":
            await self._capture_from_turn(data)
        
        return HookResult(action="continue")
    
    async def _capture_from_tool(self, data: dict):
        """Extract memories from tool results."""
        result = data.get("result", "")
        
        if contains_capture_pattern(result):
            observation = extract_observation(result)
            await self._store_observation(observation)
    
    async def _store_observation(self, observation: dict):
        """Store captured observation as memory."""
        from memory_semantic import FileStorage
        
        agent_id = get_capability(self.coordinator, "agent_identity")
        if not agent_id:
            return  # No agent identity, skip storage
        
        storage = FileStorage(agent_id, {})
        memory = Memory(
            content=observation["content"],
            category=observation["category"],
            tags=observation["tags"],
            timestamp=datetime.utcnow()
        )
        await storage.store(memory)
        
        # Emit event
        await self.coordinator.emit_event("memory:captured", {
            "agent_id": agent_id,
            "category": observation["category"]
        })
```

**Actions**:
- [ ] Implement event handling
- [ ] Add pattern-based capture
- [ ] Add async storage (non-blocking)
- [ ] Add event emission
- [ ] Add rate limiting (optional config)

**Validation**:
- [ ] Hook can be registered
- [ ] tool:after_execute triggers capture
- [ ] Patterns are matched correctly
- [ ] Memories are stored
- [ ] Events are emitted

---

### Phase 3 Integration Test

**Test script**: `tests/test_phase3_integration.py`

**Scenario**:
```python
async def test_auto_capture():
    # 1. Register hook
    hook = MemoryCaptureHook(coordinator)
    coordinator.hooks.register("tool:after_execute", hook)
    
    # 2. Simulate tool execution with capture pattern
    await coordinator.emit_event("tool:after_execute", {
        "result": "Remember: PostgreSQL is preferred for new projects"
    })
    
    # 3. Verify memory was captured
    storage = FileStorage("bob", {})
    memories = await storage.list_recent(limit=10)
    
    assert len(memories) >= 1
    assert "PostgreSQL" in memories[-1].content
    
    print("✅ Phase 3 integration test passed")
```

**Validation Gates**:
- [ ] Hook captures memories automatically
- [ ] Patterns trigger storage correctly
- [ ] No blocking of main event loop
- [ ] Events are emitted properly

---

## Phase 4: Bundle & Behavior Composition

**Goal**: Package as thin bundle with composable behaviors

**Dependencies**: Phases 1-3 complete

### Task 4.1: Behavior YAML Files

**File**: `behaviors/memory-semantic.yaml`

**Deliverables**:
```yaml
bundle:
  name: memory-semantic-behavior
  version: 1.0.0
  description: Semantic memory with agent identities - core capability

tools:
  - module: tool-memory-semantic
    source: ./modules/tool-memory-semantic
    config:
      storage_root: ~/.amplifier/memory
      migration_threshold: 5000
      auto_migrate: false

session:
  context:
    module: context-memory-scratchpad
    source: ./modules/context-memory-scratchpad
    config:
      scratchpad_size: 10
      token_budget: 2000

agents:
  include:
    - agent-memory:agents/memory-assistant

context:
  include:
    - agent-memory:context/memory-awareness.md
```

**File**: `behaviors/memory-capture.yaml`

**Deliverables**:
```yaml
bundle:
  name: memory-capture-behavior
  version: 1.0.0
  description: Automatic memory capture (opt-in)

hooks:
  - module: hooks-memory-capture
    source: ./modules/hooks-memory-capture
    config:
      patterns:
        - remember
        - important
        - decided to
      min_content_length: 50
      rate_limit_per_minute: 10
```

**Actions**:
- [ ] Create behavior YAML files
- [ ] Reference all Phase 1-3 modules
- [ ] Add configuration defaults
- [ ] Validate YAML syntax

---

### Task 4.2: Thin Bundle Root

**File**: `bundle.md`

**Deliverables**:
```markdown
---
bundle:
  name: agent-memory
  version: 1.0.0
  description: Semantic memory with named agent identities

includes:
  - bundle: git+https://github.com/microsoft/amplifier-foundation@main
  - bundle: agent-memory:behaviors/memory-semantic
---

# Agent Memory System

Multi-tenant semantic memory for Amplifier with named agent identities.

@agent-memory:context/memory-awareness.md

---

@foundation:context/shared/common-system-base.md
```

**Actions**:
- [ ] Create thin bundle.md
- [ ] Include foundation dependency
- [ ] Reference memory-semantic behavior
- [ ] Add context awareness pointer

---

### Task 4.3: Context Files

**File**: `context/memory-awareness.md` (thin, ~30 lines)

**Deliverables**:
```markdown
# Memory System Awareness

You have access to persistent memory capabilities via agent identity.

## Available Tools

- `memory_search(query)` - Search memories semantically
- `memory_store(content, category, tags)` - Store new memory
- `memory_promote(memory_id)` - Promote to hot cache
- `memory_list_recent(limit)` - List recent memories

## When to Use

- User asks about past work
- Context would help current task
- Important fact needs persistence

Do not over-explain memory mechanics. Use naturally.

For detailed instructions, delegate to memory-assistant agent (context sink).
```

**File**: `context/memory-instructions.md` (heavy, for agent only)

**Deliverables**:
```markdown
# Memory System Instructions (Agent Context)

Complete guide to memory operations, protocols, and patterns.

[Heavy documentation, 200+ lines from API_REFERENCE.md]
```

**Actions**:
- [ ] Create thin awareness pointer
- [ ] Create heavy instructions for agent
- [ ] Keep thin file minimal (token efficiency)

---

### Task 4.4: Memory Assistant Agent

**File**: `agents/memory-assistant.md`

**Deliverables**:
```markdown
---
meta:
  name: memory-assistant
  description: Expert at searching and analyzing agent memories (context sink)

tools:
  - memory_search
  - memory_store
  - memory_list_recent

system: |
  You are a memory search specialist. You have full access to the memory store.
  
  Your role: Perform complex queries, aggregations, and analysis.
  
  Return concise summaries to parent session, not raw dumps.
---

# Memory Assistant

@agent-memory:context/memory-instructions.md

[Heavy memory operation guide loaded here]
```

**Actions**:
- [ ] Create agent definition
- [ ] Add memory tools
- [ ] Add heavy context reference
- [ ] Add system prompt

---

### Phase 4 Integration Test

**Test script**: `tests/test_phase4_bundle.py`

**Scenario**:
```python
async def test_bundle_loading():
    from amplifier_foundation import load_bundle
    
    # 1. Load bundle
    bundle = await load_bundle("./bundle.md")
    
    # 2. Verify tools loaded
    assert "memory_search" in bundle.tools
    assert "memory_store" in bundle.tools
    
    # 3. Verify context module loaded
    assert bundle.context_module is not None
    
    # 4. Verify agent available
    assert "memory-assistant" in bundle.agents
    
    # 5. Test with agent identity
    session = await create_session(
        bundle=bundle,
        capabilities={"agent_identity": "bob"}
    )
    
    # 6. Store and retrieve
    result = await session.invoke_tool("memory_store", {
        "content": "Test memory",
        "category": "test",
        "tags": ["test"]
    })
    assert result.success
    
    print("✅ Phase 4 bundle test passed")
```

**Validation Gates**:
- [ ] Bundle loads without errors
- [ ] All modules mount correctly
- [ ] Tools are available
- [ ] Context injection works
- [ ] Agent can be spawned
- [ ] End-to-end memory lifecycle works

---

## Phase 5: Vector DB & Semantic Search

**Goal**: Add Qdrant backend with embeddings

**Dependencies**: Phase 1-4 complete and validated

### Task 5.1: Add Dependencies

**File**: `modules/tool-memory-semantic/pyproject.toml`

**Changes**:
```toml
[project]
dependencies = [
    "amplifier-core>=0.1.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "qdrant-client>=1.7.0",    # Add
    "openai>=1.0.0",           # Add
]
```

---

### Task 5.2: Embedding Generator

**File**: `modules/tool-memory-semantic/memory_semantic/embeddings.py`

**Deliverables**:
```python
class EmbeddingGenerator:
    """Generate embeddings for semantic search."""
    
    def __init__(self, config: dict):
        self.provider = config.get("embedding_provider", "openai")
        self.model = config.get("embedding_model", "text-embedding-3-small")
        self.dimensions = config.get("embedding_dimensions", 1536)
        
        if self.provider == "openai":
            import openai
            self.client = openai.AsyncOpenAI()
    
    async def generate(self, content: str) -> list[float]:
        """Generate embedding vector."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=content
        )
        return response.data[0].embedding
    
    async def generate_batch(self, contents: list[str]) -> list[list[float]]:
        """Batch embedding generation for efficiency."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=contents
        )
        return [item.embedding for item in response.data]
```

**Actions**:
- [ ] Implement embedding generation
- [ ] Add OpenAI API integration
- [ ] Add batch operations for migration
- [ ] Add error handling (API failures)
- [ ] Add caching (avoid re-embedding same content)

---

### Task 5.3: Vector Storage Implementation

**File**: `modules/tool-memory-semantic/memory_semantic/vector_storage.py`

**Deliverables**:
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class VectorStorage(MemoryStorage):
    """Qdrant-based vector storage with semantic search."""
    
    def __init__(self, agent_id: str, config: dict):
        super().__init__(agent_id, config)
        
        db_path = self.storage_root / "memories.db"
        self.client = QdrantClient(path=str(db_path))
        self.collection = "memories"
        
        self.embeddings = EmbeddingGenerator(config)
        self.scratchpad = ScratchpadManager(agent_id, config)
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if not exists."""
        collections = self.client.get_collections().collections
        if self.collection not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.embeddings.dimensions,
                    distance=Distance.COSINE
                )
            )
    
    async def search(self, query: str, limit: int = 5, filters=None):
        """Semantic search with freshness decay."""
        # 1. Check scratchpad first
        scratchpad_hits = await self.scratchpad.search(query)
        
        # 2. Generate query embedding
        query_embedding = await self.embeddings.generate(query)
        
        # 3. Vector search
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            limit=limit * 2,  # Over-fetch for ranking
            with_payload=True
        )
        
        # 4. Apply freshness decay and access boosting
        ranked = []
        for result in results:
            memory = Memory(**result.payload)
            score = self._calculate_relevance_score(
                memory, 
                result.score  # cosine similarity
            )
            
            # Dynamic threshold
            threshold = self._get_similarity_threshold(memory)
            if score >= threshold:
                ranked.append((score, memory))
        
        # 5. Sort by final score
        ranked.sort(key=lambda x: x[0], reverse=True)
        deep_hits = [m for _, m in ranked]
        
        # 6. Merge with scratchpad
        all_hits = self._merge_results(scratchpad_hits, deep_hits)
        
        return all_hits[:limit]
    
    def _calculate_relevance_score(self, memory: Memory, similarity: float):
        """Apply freshness decay and access boosting."""
        # Implementation from ARCHITECTURE.md
        age_days = (datetime.utcnow() - memory.timestamp).days
        freshness = 1.0 / (1.0 + 0.1 * age_days)
        access_boost = 1.0 + (0.05 * memory.access_count)
        return similarity * freshness * access_boost
    
    def _get_similarity_threshold(self, memory: Memory):
        """Dynamic threshold based on memory characteristics."""
        # Implementation from ARCHITECTURE.md
        if memory.access_count > 5:
            return 0.70
        elif memory.access_count == 0 and memory.age_days > 30:
            return 0.85
        else:
            return 0.75
    
    async def store(self, memory: Memory) -> str:
        """Store with embedding generation."""
        # Generate embedding
        memory.embedding = await self.embeddings.generate(memory.content)
        
        # Store to Qdrant
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=memory.id,
                vector=memory.embedding,
                payload=memory.dict(exclude={"embedding"})
            )]
        )
        
        # Also add to scratchpad
        await self.scratchpad.add(memory)
        
        return memory.id
```

**Actions**:
- [ ] Implement all abstract methods
- [ ] Add Qdrant client initialization
- [ ] Add collection creation
- [ ] Add semantic search with ranking
- [ ] Add freshness decay algorithm
- [ ] Add dynamic threshold calculation
- [ ] Add scratchpad integration

**Validation**:
- [ ] Collection is created on first use
- [ ] Embeddings are generated correctly
- [ ] Semantic search returns relevant results
- [ ] Freshness decay affects ranking
- [ ] Access boosting works

---

### Task 5.4: Migration Implementation

**File**: `modules/tool-memory-semantic/memory_semantic/migration.py`

**Deliverables**:
```python
class MigrationManager:
    """Migrate from file storage to vector storage."""
    
    async def should_migrate(self, agent_id: str) -> bool:
        """Check if migration threshold exceeded."""
        # Implementation from ARCHITECTURE.md
    
    async def migrate(
        self, 
        agent_id: str, 
        progress_callback=None
    ) -> MigrationProgress:
        """Execute migration with progress tracking."""
        # 1. Backup
        # 2. Initialize Qdrant
        # 3. Load all memories
        # 4. Generate embeddings (batch)
        # 5. Insert to Qdrant
        # 6. Validate
        # 7. Update metadata
        # 8. Keep original YAML
    
    async def rollback(self, agent_id: str):
        """Rollback migration if failed."""
        # Implementation from ARCHITECTURE.md
```

**Actions**:
- [ ] Implement migration logic
- [ ] Add backup creation
- [ ] Add batch embedding generation
- [ ] Add progress tracking
- [ ] Add validation
- [ ] Add rollback capability

---

### Task 5.5: Storage Detection & Auto-Migration

**File**: `modules/tool-memory-semantic/memory_semantic/storage.py`

**Updates**:
```python
def detect_storage(agent_id: str, config: dict) -> MemoryStorage:
    """Detect and instantiate correct storage backend."""
    agent_root = Path(f"~/.amplifier/memory/{agent_id}").expanduser()
    
    # Check if vector DB exists
    if (agent_root / "memories.db").exists():
        return VectorStorage(agent_id, config)
    
    # Check if should migrate
    migration_mgr = MigrationManager(config)
    if migration_mgr.should_migrate(agent_id):
        auto_migrate = config.get("auto_migrate", False)
        
        if auto_migrate:
            # Auto-migrate
            await migration_mgr.migrate(agent_id)
            return VectorStorage(agent_id, config)
        else:
            # Emit suggestion event
            emit_event("memory:migration_suggested", {
                "agent_id": agent_id,
                "memory_count": count_memories(agent_id)
            })
            return FileStorage(agent_id, config)
    
    # Default: file storage
    return FileStorage(agent_id, config)
```

**Actions**:
- [ ] Add auto-detection logic
- [ ] Add migration triggering
- [ ] Add event emission for manual migration
- [ ] Update tool mount to use detect_storage()

---

### Phase 5 Integration Test

**Test script**: `tests/test_phase5_vector.py`

**Scenario**:
```python
async def test_vector_search():
    # 1. Create vector storage
    storage = VectorStorage("bob", {
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small"
    })
    
    # 2. Store memories
    await storage.store(Memory(content="PostgreSQL database", ...))
    await storage.store(Memory(content="Python programming", ...))
    await storage.store(Memory(content="Database optimization", ...))
    
    # 3. Semantic search
    results = await storage.search("database work")
    
    # Should find both "PostgreSQL database" and "Database optimization"
    assert len(results) >= 2
    assert any("PostgreSQL" in r.content for r in results)
    assert any("optimization" in r.content for r in results)
    
    # 4. Test freshness decay
    # Create old memory
    old_memory = Memory(
        content="Old database note",
        timestamp=datetime.utcnow() - timedelta(days=60)
    )
    await storage.store(old_memory)
    
    # Search should rank recent memories higher
    results = await storage.search("database")
    recent_ranks = [i for i, r in enumerate(results) 
                   if "PostgreSQL" in r.content]
    old_ranks = [i for i, r in enumerate(results) 
                if "Old" in r.content]
    
    assert min(recent_ranks) < min(old_ranks)  # Recent ranked higher
    
    print("✅ Phase 5 vector search test passed")

async def test_migration():
    # 1. Create file storage with >threshold memories
    file_storage = FileStorage("alice", {})
    for i in range(5100):  # Exceed threshold
        await file_storage.store(Memory(content=f"Memory {i}", ...))
    
    # 2. Trigger migration
    migration_mgr = MigrationManager({"migration_threshold": 5000})
    assert await migration_mgr.should_migrate("alice")
    
    progress = await migration_mgr.migrate("alice")
    assert progress.status == "completed"
    
    # 3. Verify vector storage works
    vector_storage = VectorStorage("alice", {})
    results = await vector_storage.search("Memory")
    assert len(results) > 0
    
    # 4. Verify original YAML preserved
    backup_path = Path("~/.amplifier/memory/alice/scratchpad.yaml.backup")
    assert backup_path.exists()
    
    print("✅ Phase 5 migration test passed")
```

**Validation Gates**:
- [ ] Vector storage works end-to-end
- [ ] Semantic search returns relevant results
- [ ] Freshness decay affects ranking
- [ ] Migration completes successfully
- [ ] Original data preserved
- [ ] Rollback capability works

---

## Phase 6: Documentation & Examples

**Goal**: User-facing documentation and examples

**Dependencies**: All phases complete

### Task 6.1: User Guide

**File**: `docs/USER_GUIDE.md`

**Deliverables**:
- Getting started
- Configuration options
- Usage examples
- Natural language patterns
- Troubleshooting

---

### Task 6.2: Developer Guide

**File**: `docs/DEVELOPER_GUIDE.md`

**Deliverables**:
- Extending storage backends
- Custom embedding providers
- Integration patterns
- Testing strategies

---

### Task 6.3: Example Bundles

**Directory**: `examples/`

**Deliverables**:
- `examples/simple-memory.yaml` - Minimal config
- `examples/multi-agent.yaml` - Multiple identities
- `examples/with-capture.yaml` - Auto-capture enabled
- `examples/vector-only.yaml` - Force vector storage

---

## Phase 7: my-amplifier Integration

**Goal**: Update my-amplifier bundle to include agent-memory

**Dependencies**: Phase 1-6 complete

### Task 7.1: Add to my-amplifier

**File**: `~/.amplifier/bundles/my-amplifier/bundle.md`

**Changes**:
```markdown
includes:
  - bundle: foundation
  - bundle: git+https://github.com/USER/amplifier-bundle-agent-memory@main
  # Or local during development:
  # - bundle: /home/samschillace/dev/ANext/amplifier-bundle-agent-memory/bundle.md
```

**Configuration**:
```yaml
# ~/.amplifier/settings.yaml
session:
  capabilities:
    agent_identity: "sam"  # Your agent identity
    memory:
      enabled: true
      mode: "autonomous"

memory:
  embedding_provider: "openai"
  embedding_model: "text-embedding-3-small"
  migration_threshold: 5000
  auto_migrate: false
```

---

## Testing Strategy

### Unit Tests

**Coverage targets**:
- [ ] Storage abstraction: 90%+
- [ ] File storage: 90%+
- [ ] Vector storage: 85%+
- [ ] Scratchpad: 90%+
- [ ] Tools: 85%+
- [ ] Context module: 85%+
- [ ] Hook module: 85%+

**Framework**: pytest with async support

**Location**: `tests/unit/`

---

### Integration Tests

**Scenarios**:
- [ ] End-to-end memory lifecycle
- [ ] Multi-agent isolation
- [ ] Context injection
- [ ] Auto-capture
- [ ] Migration
- [ ] Bundle loading

**Location**: `tests/integration/`

---

### Manual Testing

**Checklist**:
- [ ] Natural language usage ("remember this: X")
- [ ] Semantic search quality
- [ ] Freshness decay behavior
- [ ] Access boosting promotion
- [ ] Migration user experience
- [ ] Error recovery

---

## Success Criteria (MVP)

### Functional Requirements
- [ ] Single agent can store and retrieve memories
- [ ] Semantic search returns relevant results
- [ ] Scratchpad provides recent context
- [ ] Opt-in enablement works
- [ ] File-based storage operational
- [ ] Vector DB migration works
- [ ] Auto-capture captures correctly
- [ ] Context injection appears at session start

### Non-Functional Requirements
- [ ] Namespace isolation (security tested)
- [ ] Token efficiency (context sink validated)
- [ ] Performance targets met (see ARCHITECTURE.md)
- [ ] Error handling comprehensive
- [ ] Documentation complete
- [ ] All tests passing

### Quality Gates
- [ ] Foundation expert validation ✅ (DONE)
- [ ] Code review by maintainer
- [ ] Security audit (namespace isolation)
- [ ] Performance testing (latency targets)
- [ ] Integration testing (all phases)
- [ ] User acceptance testing

---

## Rollout Plan

### Alpha (Internal Testing)
- [ ] Deploy to personal development environment
- [ ] Test with real workloads
- [ ] Gather feedback
- [ ] Iterate on UX

### Beta (Community Preview)
- [ ] Create GitHub repository
- [ ] Publish to community
- [ ] Gather feedback
- [ ] Address issues

### V1.0 (Production Release)
- [ ] All success criteria met
- [ ] Documentation complete
- [ ] Security audit passed
- [ ] Performance validated
- [ ] Community feedback integrated

---

## Risk Mitigation

### Technical Risks

**Vector DB Migration Failure**
- **Mitigation**: Comprehensive backup, validation, rollback
- **Testing**: Simulate failures in test environment

**Embedding API Outages**
- **Mitigation**: Graceful degradation to file storage
- **Testing**: Mock API failures

**Namespace Isolation Bugs**
- **Mitigation**: Security audit, penetration testing
- **Testing**: Cross-agent access attempt tests

**Performance Degradation**
- **Mitigation**: Performance testing, optimization
- **Testing**: Load testing with 100k+ memories

---

## Implementation Order Summary

**Recommended sequence** (minimizes rework):

1. **Phase 1**: Core storage (file-based) → **Must have foundation**
2. **Phase 2**: Context injection → **Depends on Phase 1**
3. **Phase 3**: Auto-capture → **Depends on Phase 1**
4. **Phase 4**: Bundle composition → **Depends on Phases 1-3**
5. **Phase 5**: Vector DB → **Depends on Phase 1**
6. **Phase 6**: Documentation → **Depends on all**
7. **Phase 7**: my-amplifier integration → **Depends on Phase 6**

**Parallel work opportunities**:
- Phases 2 & 3 can proceed in parallel (both depend only on Phase 1)
- Documentation (Phase 6) can start during Phase 5

---

## Next Actions

1. **Begin Phase 1**: Create project structure and storage layer
2. **Set up testing framework**: pytest, fixtures, CI
3. **Create GitHub repository**: (when ready for community)
4. **Regular validation**: Test each phase before proceeding
5. **Iterate based on findings**: Design is validated, but implementation may reveal refinements needed

---

**Status**: Ready to begin implementation

**Approval**: Design validated by foundation:foundation-expert ✅

**Let's build this! 🚀**
