# Design Decisions: amplifier-bundle-agent-memory

## Document Status
**Version**: 0.1.0 (Draft)  
**Date**: 2026-01-30  
**Phase**: Pre-validation design  
**Next Step**: Foundation expert validation

---

## Executive Summary

A semantic memory system for Amplifier that provides named agent identities with persistent, searchable memory pools. Agents can maintain context across sessions, query their history with natural language, and evolve from simple file-based storage to vector databases as scale demands.

---

## Core Requirements (From User)

### 1. Scale Expectations
- **Volume**: Eventually many thousands of memories per agent
- **Multi-tenancy**: Named agents ("bob", "alice") with isolated memory pools
- **Segregation**: Each agent identity has its own namespace
- **Hybrid evolution**: Start simple, migrate to vector DB when scale demands (~5k memories)

### 2. Search Semantics
- **Semantic understanding required**: "what did we do last week" → retrieve temporal context
- **Natural language queries**: "tell me what I've done on databases" → semantic similarity
- **Rich, human-like memories**: Not just keywords, full context understanding
- **Tiered storage**: Hot scratchpad (recent) + cold deep storage (historical)
- **Freshness management**: Stale memories need tighter cosine similarity threshold
- **Access patterns**: Frequently accessed memories stay relevant longer

### 3. Interaction Model
- **Opt-in by default**: Sessions must explicitly enable memory (no memory by default)
- **Agent autonomy**: Agent decides when to query (not always-on injection)
- **Human override**: User can explicitly request memory operations
- **Presence & persistence**: Natural sense of continuity when enabled

### 4. Development Approach
- **Quality over speed**: Implementation time not a constraint
- **Design-first**: Full design validation before implementation
- **Correctness focus**: Get architecture right, then build

### 5. Privacy/Locality
- **Embeddings**: OK with API calls (OpenAI, Cohere, etc.)
- **Storage**: Local only (no cloud memory storage)
- **Model**: Use embedding APIs for quality, store locally

---

## Architectural Decisions

### Decision 1: Multi-Module Architecture

**Decision**: Implement as combination of Tool + Context + Hook modules

**Rationale**:
- **Tool module**: Storage abstraction, explicit queries, agent-controlled
- **Context module**: Scratchpad injection at session start (when enabled)
- **Hook module**: Passive observation and auto-capture (opt-in behavior)

**Alignment**: Follows proven pattern from `amplifier-bundle-memory` (community reference)

**Module breakdown**:
```
tool-memory-semantic      # Storage layer (file → vector hybrid)
context-memory-scratchpad # Hot cache injection
hooks-memory-capture      # Auto-capture observations
```

---

### Decision 2: Agent Identity as First-Class Concept

**Decision**: Agent identity (`agent_id`) is required for all memory operations

**Rationale**:
- Enforces namespace isolation
- Prevents accidental cross-contamination
- Enables multi-tenant architecture
- Each identity has separate storage root

**Storage structure**:
```
~/.amplifier/memory/
├── bob/
│   ├── scratchpad.yaml       # Hot cache
│   ├── memories.db           # Vector DB (when migrated)
│   └── metadata.json         # Identity config
├── alice/
│   └── scratchpad.yaml
└── shared/                   # Optional: cross-agent knowledge
    └── memories.db
```

**Session declaration**:
```yaml
session:
  agent_identity: "bob"  # This session IS bob
  capabilities:
    memory:
      enabled: true      # Explicit opt-in
```

---

### Decision 3: Tiered Storage (Hot + Cold)

**Decision**: Two-tier memory architecture

**Hot tier (Scratchpad)**:
- **Format**: YAML file
- **Size**: 10-50 recent memories
- **Purpose**: Fast access, session context
- **Queries**: Simple keyword matching
- **Always present**: Even before vector DB migration

**Cold tier (Deep Storage)**:
- **Format**: Vector database (Qdrant)
- **Size**: Thousands to millions of memories
- **Purpose**: Semantic search, long-term storage
- **Queries**: Vector similarity with freshness decay
- **Migration trigger**: When memories exceed ~5k

**Query flow**:
```
1. Check scratchpad first (fast, recent context)
2. If insufficient, query deep storage (semantic search)
3. Merge results, rank by relevance + freshness
4. Promote frequently accessed to scratchpad
```

---

### Decision 4: Hybrid Evolution Path

**Decision**: Start file-based, transparently migrate to vector DB at scale

**Phase 1: File-based** (0-5k memories)
- **Storage**: `scratchpad.yaml` only
- **Search**: Keyword matching, recency ranking
- **Complexity**: Low (no dependencies)
- **Performance**: Fast for small scale

**Phase 2: Vector migration** (>5k memories)
- **Storage**: `scratchpad.yaml` + `memories.db` (Qdrant)
- **Search**: Semantic similarity via embeddings
- **Migration**: Tool detects threshold, migrates automatically
- **Performance**: Scales to millions

**Transparency**: Tool abstraction hides backend from agent/user

**Implementation**:
```python
class MemoryTool:
    def __init__(self, agent_id):
        self.storage = self._detect_storage(agent_id)
    
    def _detect_storage(self, agent_id):
        if vector_db_exists(agent_id):
            return VectorStorage(agent_id)
        elif should_migrate(agent_id):
            return self._migrate_to_vector(agent_id)
        else:
            return FileStorage(agent_id)
```

---

### Decision 5: Freshness Decay & Access Boosting

**Decision**: Memory relevance decays over time, access frequency boosts relevance

**Freshness decay formula**:
```python
freshness_factor = 1.0 / (1.0 + decay_rate * age_days)
```

**Access boost formula**:
```python
access_boost = 1.0 + (boost_rate * access_count)
```

**Final relevance score**:
```python
score = cosine_similarity * freshness_factor * access_boost
```

**Dynamic thresholds**:
- **Fresh, accessed memories**: Lower threshold (0.70) - easier to recall
- **Stale, unaccessed memories**: Higher threshold (0.85) - harder to recall
- **Normal memories**: Default threshold (0.75)

**Rationale**: Prevents stale memory pollution while keeping useful memories accessible

---

### Decision 6: Opt-In Memory System

**Decision**: Memory is disabled by default, must be explicitly enabled per session

**Rationale**:
- Not all sessions need memory (exploratory tasks, one-offs)
- Privacy-preserving (no accidental storage)
- User control over stateful behavior
- Reduces overhead for non-memory sessions

**Enablement mechanisms**:

**1. Configuration file**:
```yaml
# settings.yaml or bundle
session:
  agent_identity: "bob"
  capabilities:
    memory:
      enabled: true
      mode: "autonomous"  # or "explicit"
```

**2. Natural language**:
```
User: "Start as bob with memory enabled"
Agent: [Recognizes pattern, enables memory capability]
```

**3. Bundle inclusion**:
```yaml
# Include memory behavior in bundle
includes:
  - bundle: agent-memory:behaviors/memory-autonomous
```

**When disabled**: No memory tool available, no context injection, no capture hooks

---

### Decision 7: Embedding Provider Strategy

**Decision**: Use OpenAI embedding API as primary, support alternatives

**Primary**: `text-embedding-3-small` (OpenAI)
- **Quality**: Industry-leading semantic understanding
- **Cost**: ~$0.02 per 1M tokens (~$10 for 100k memories)
- **Latency**: ~100-200ms per API call
- **Dimensions**: 1536 (configurable)

**Alternatives** (future):
- Cohere `embed-english-v3.0`
- Voyage AI `voyage-2`
- Local models (sentence-transformers) for offline operation

**Configuration**:
```yaml
memory:
  embedding:
    provider: "openai"
    model: "text-embedding-3-small"
    dimensions: 1536
    fallback: "local"  # Optional: local model if API fails
```

---

### Decision 8: Vector Database Choice

**Decision**: Qdrant as vector store (local embedded mode)

**Rationale**:
- **Local-first**: Embedded database (no server required)
- **Rich querying**: Metadata filtering, hybrid search
- **Performance**: Scales to millions of vectors
- **Persistence**: On-disk storage, resumable
- **Python-friendly**: Clean SDK

**Alternatives considered**:
- ChromaDB: Similar capabilities, slightly simpler
- LanceDB: Good performance, newer ecosystem
- FAISS: Raw vectors only (no metadata richness)

**Storage location**: `~/.amplifier/memory/{agent_id}/memories.db`

---

### Decision 9: Memory Capture Patterns

**Decision**: Hook-based auto-capture as opt-in behavior (separate from core)

**Capture triggers**:
- User says "remember this: {content}"
- Tool outputs contain learnings (pattern matching)
- Assistant makes explicit decisions
- Session ends (capture summary)

**Extraction patterns**:
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

**Behavior separation**:
```yaml
# Core behavior: storage and retrieval
behaviors/memory-core.yaml

# Optional behavior: auto-capture
behaviors/memory-capture.yaml  # User opts in separately
```

**Rationale**: Not all users want auto-capture (explicit control preference)

---

### Decision 10: System Prompt Integration

**Decision**: When memory enabled, inject scratchpad + capability description

**System prompt addition** (when enabled):
```markdown
## Memory System

You are **{agent_identity}**. You have persistent memory across sessions.

### Recent Context (Scratchpad)
{top_10_scratchpad_memories}

### Memory Capabilities
- Query deep storage: Use memory tool with natural language
- Store important facts: Use memory tool to persist
- Examples: "What did we work on last week?" "What's my database preference?"

### Guidelines
- Query when relevant to current conversation
- Don't over-explain memory system mechanics
- Natural integration (user shouldn't notice the machinery)
```

**Injection timing**: Session initialization (after bundle load, before first turn)

---

## Open Questions (To Validate with Foundation Expert)

### 1. Module Type Correctness
- **Question**: Is Tool + Context + Hook the right combination?
- **Concern**: Context module ownership of message list vs memory injection
- **Validate**: Does this align with ContextManager protocol?

### 2. Bundle Composition Pattern
- **Question**: Should memory-core and memory-capture be separate behaviors?
- **Concern**: Composability vs complexity
- **Validate**: Is this the "thin bundle" pattern done correctly?

### 3. Tool Protocol Compliance
- **Question**: Should memory tool expose multiple operations or single entry point?
- **Options**: 
  - Single tool with `operation` parameter: `memory(operation="search", query="...")`
  - Multiple tools: `memory_search(query)`, `memory_store(content)`
- **Validate**: What's the Amplifier convention?

### 4. Hook Emission Points
- **Question**: What events should hooks observe for memory capture?
- **Options**: `tool:after_execute`, `turn:end`, `orchestrator:complete`
- **Validate**: Which events are appropriate for memory capture?

### 5. Context Module Injection Timing
- **Question**: When should scratchpad inject? Session start only, or every turn?
- **Concern**: Token budget management
- **Validate**: What's the recommended pattern?

### 6. Agent Identity Declaration
- **Question**: Is session-level agent_identity the right approach?
- **Alternative**: Bundle-level? Provider-level?
- **Validate**: Where should identity live in the config hierarchy?

### 7. Migration Strategy
- **Question**: Should migration be automatic or manual?
- **Options**: Auto-migrate at threshold, or user-triggered
- **Validate**: Philosophy alignment (mechanism vs policy)

### 8. Shared Memory Namespace
- **Question**: Should there be a `shared/` namespace for cross-agent knowledge?
- **Use case**: Common learnings (e.g., "Amplifier architecture facts")
- **Validate**: Is this scope creep or valid use case?

---

## Non-Functional Requirements

### Performance
- **Scratchpad access**: <10ms (file read)
- **Vector search**: <100ms (local Qdrant query)
- **Embedding generation**: <200ms (API call, cached when possible)
- **Memory storage**: Async, non-blocking

### Scalability
- **Per-agent capacity**: 100k+ memories without degradation
- **Concurrent agents**: 10+ agent identities without interference
- **Storage efficiency**: ~1MB per 10k memories (embeddings + metadata)

### Reliability
- **Graceful degradation**: If vector DB unavailable, fall back to scratchpad
- **Data persistence**: All memory operations durable (write-ahead logging)
- **Migration safety**: Original data preserved during file → vector migration

### Privacy
- **Local storage only**: No cloud memory uploads
- **Namespace isolation**: Agent identities cannot access each other's memories
- **Optional redaction**: Hooks can sanitize before storage

### Observability
- **Memory operations logged**: All store/retrieve in events.jsonl
- **Metrics available**: Memory count, access frequency, search latency
- **Debug mode**: Verbose logging for troubleshooting

---

## Success Criteria

### MVP (Minimum Viable Product)
- [ ] Single agent identity can store memories
- [ ] Natural language queries retrieve relevant memories
- [ ] Scratchpad provides recent context
- [ ] Opt-in enablement works
- [ ] File-based storage operational

### V1.0 (Production Ready)
- [ ] Multiple agent identities with isolation
- [ ] Vector DB migration at scale threshold
- [ ] Freshness decay and access boosting functional
- [ ] Auto-capture hook (opt-in) works
- [ ] Context injection at session start
- [ ] Comprehensive testing and validation

### V2.0 (Advanced Features)
- [ ] Shared knowledge namespace
- [ ] Memory analytics and insights
- [ ] Custom embedding providers
- [ ] Memory export/import
- [ ] Collaborative memory (multi-user agents)

---

## Implementation Dependencies

### External Libraries
- **qdrant-client**: Vector database client
- **openai**: Embedding API (or alternatives: cohere, etc.)
- **pyyaml**: YAML parsing for scratchpad
- **pydantic**: Data validation for memory schemas

### Amplifier Modules (Dependencies)
- **amplifier-core**: Tool, Hook, ContextManager protocols
- **amplifier-foundation**: Bundle primitives, load_bundle()

### Reference Materials
- **amplifier-bundle-memory**: Community implementation (study patterns)
- **amplifier-module-tool-*****: Tool protocol examples
- **amplifier-module-context-*****: Context manager examples

---

## Risk Assessment

### Technical Risks

**Risk**: Vector DB migration complexity
- **Impact**: High (data loss potential)
- **Mitigation**: Preserve original files, validation checksum, rollback mechanism

**Risk**: Embedding API rate limits/costs
- **Impact**: Medium (latency, cost)
- **Mitigation**: Caching, batch operations, local fallback

**Risk**: Context manager protocol misunderstanding
- **Impact**: High (core functionality broken)
- **Mitigation**: Foundation expert validation BEFORE implementation

### Architectural Risks

**Risk**: Multi-tenant isolation bugs
- **Impact**: Critical (memory leakage across agents)
- **Mitigation**: Comprehensive namespace testing, security audit

**Risk**: Freshness decay algorithm too aggressive
- **Impact**: Medium (useful memories forgotten)
- **Mitigation**: Configurable parameters, user feedback loop

### Operational Risks

**Risk**: Storage growth unchecked
- **Impact**: Medium (disk space consumption)
- **Mitigation**: Cleanup policies, archive old memories, user quotas

---

## Next Steps

1. **Validate design** with `foundation:foundation-expert` (address open questions)
2. **Create detailed design docs**:
   - Architecture diagrams
   - Data models and schemas
   - Module protocols and interfaces
   - API specifications
3. **Generate implementation task list** with dependencies
4. **Update my-amplifier bundle** for integration
5. **Begin implementation** (only after design validated)

---

## Appendices

### Appendix A: Comparison to Existing Systems

| Feature | dev-memory (current) | amplifier-bundle-memory | agent-memory (this) |
|---------|---------------------|------------------------|---------------------|
| Storage | YAML files | SQLite + FTS5 | YAML → Qdrant (hybrid) |
| Search | Keyword | Full-text | Semantic (vector) |
| Scope | Single user | Single session | Multi-tenant agents |
| Scale | <1k memories | ~10k memories | 100k+ per agent |
| Identity | Implicit | Session-based | Named agents |
| Auto-capture | No | Yes (hooks) | Yes (opt-in hook) |

### Appendix B: User Experience Examples

**Example 1: Bob's first session**
```
User: Start a session as bob with memory enabled

Agent: Memory enabled. I'm bob.

User: Remember: I prefer PostgreSQL for new projects

Agent: ✓ Stored memory: PostgreSQL preference
```

**Example 2: Bob's second session (days later)**
```
User: Start as bob

Agent: [Scratchpad loads: PostgreSQL preference visible]
       Welcome back. Ready to continue.

User: What database should we use for this project?

Agent: Based on your preference for PostgreSQL, I'd recommend that.
```

**Example 3: Semantic query**
```
User: What have I worked on related to databases?

Agent: [Queries deep storage with "databases"]
       Found 3 relevant memories:
       1. PostgreSQL preference (7 days ago)
       2. Completed migration script for user table (3 days ago)
       3. Researched connection pooling (5 days ago)
```

### Appendix C: Terminology

- **Agent identity**: Named entity (e.g., "bob") with isolated memory pool
- **Scratchpad**: Hot cache of 10-50 recent memories (fast access)
- **Deep storage**: Cold storage of thousands+ memories (semantic search)
- **Freshness decay**: Relevance reduction over time (prevents stale memory)
- **Access boosting**: Relevance increase from frequent retrieval
- **Memory namespace**: Isolated storage root per agent identity
- **Opt-in stateful**: Memory disabled by default, requires explicit enablement

---

**End of Design Decisions Document**
