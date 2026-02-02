# Data Models: amplifier-bundle-agent-memory

## Document Status
**Version**: 0.1.0 (Design Phase)  
**Date**: 2026-01-30  
**Status**: Design Documentation

---

## Overview

This document defines the complete data model specification for the agent-memory system, including:

- **Core Models**: Memory objects and their Pydantic schemas
- **Storage Schemas**: YAML and JSON file formats
- **Configuration Models**: Tool, context, and hook configurations
- **Enums and Constants**: Category types, observation patterns, constraints
- **Validation Rules**: Business logic and data integrity constraints
- **Serialization**: Conversion between formats (dict, JSON, YAML)

All models use **Pydantic v2** for validation and serialization.

---

## Core Data Models

### 1. Memory Object

The fundamental unit of persistent memory.

```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

class Memory(BaseModel):
    """
    A single memory entry with content, metadata, and access tracking.
    
    Represents both file-based and vector-based memories.
    """
    
    # Identity
    id: str = Field(
        default_factory=lambda: f"mem-{uuid4().hex[:12]}",
        description="Unique memory identifier (e.g., 'mem-a3f2c1b4e5d6')"
    )
    agent_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Agent namespace this memory belongs to"
    )
    
    # Content
    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The actual memory text (1-10,000 chars)"
    )
    category: str = Field(
        ...,
        description="Memory category (preference, decision, discovery, etc.)"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Searchable keywords (e.g., ['database', 'postgresql'])"
    )
    
    # Embedding (vector storage only)
    embedding: Optional[list[float]] = Field(
        None,
        description="Vector embedding (1536 dimensions for text-embedding-3-small)"
    )
    
    # Temporal
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the memory was created (UTC)"
    )
    last_accessed: Optional[datetime] = Field(
        None,
        description="When the memory was last retrieved (UTC)"
    )
    
    # Access tracking
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this memory has been retrieved"
    )
    
    # Provenance
    created_by_session: str = Field(
        ...,
        description="Session ID that created this memory"
    )
    
    # Extensibility
    metadata: dict = Field(
        default_factory=dict,
        description="Extensible metadata for custom fields"
    )
    
    # Model configuration
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "mem-a3f2c1b4e5d6",
                    "agent_id": "bob",
                    "content": "Prefer PostgreSQL for new projects due to better tooling support",
                    "category": "preference",
                    "tags": ["database", "postgresql", "tools"],
                    "embedding": None,  # Not yet migrated to vector
                    "timestamp": "2026-01-28T12:00:00Z",
                    "last_accessed": "2026-01-30T14:30:00Z",
                    "access_count": 3,
                    "created_by_session": "session-abc-123",
                    "metadata": {
                        "source": "explicit_user_statement",
                        "confidence": 0.95
                    }
                }
            ]
        }
    }
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Ensure tags are lowercase and unique."""
        return list(set(tag.lower().strip() for tag in v if tag.strip()))
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        """Ensure embedding has correct dimensions (1536 for OpenAI)."""
        if v is not None and len(v) != 1536:
            raise ValueError(f"Embedding must be 1536 dimensions, got {len(v)}")
        return v
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Ensure category is valid."""
        valid_categories = {
            'preference', 'decision', 'discovery', 'pattern', 
            'task', 'bugfix', 'note', 'question', 'insight'
        }
        if v not in valid_categories:
            raise ValueError(f"Category must be one of {valid_categories}, got '{v}'")
        return v
    
    def to_scratchpad_entry(self) -> dict:
        """
        Serialize for scratchpad YAML (lightweight, no embedding).
        """
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category,
            'content': self.content,
            'tags': self.tags,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None
        }
    
    def to_vector_payload(self) -> dict:
        """
        Serialize for Qdrant payload (full metadata, no embedding in payload).
        """
        return {
            'content': self.content,
            'agent_id': self.agent_id,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category,
            'tags': self.tags,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'created_by_session': self.created_by_session,
            'metadata': self.metadata
        }
    
    def calculate_relevance_score(
        self, 
        similarity: float, 
        now: Optional[datetime] = None
    ) -> float:
        """
        Calculate relevance score with freshness decay and access boosting.
        
        Args:
            similarity: Base cosine similarity (0.0-1.0)
            now: Current timestamp (default: utcnow)
        
        Returns:
            Final relevance score (0.0-1.0+)
        """
        if now is None:
            now = datetime.utcnow()
        
        # Freshness decay (older memories decay)
        age_days = (now - self.timestamp).days
        freshness_factor = 1.0 / (1.0 + 0.1 * age_days)
        
        # Access frequency boost (frequently accessed memories boosted)
        access_boost = 1.0 + (0.05 * self.access_count)
        
        # Final score
        return similarity * freshness_factor * access_boost
    
    def should_promote_to_scratchpad(self, threshold: int = 3) -> bool:
        """Check if memory should be promoted to hot cache."""
        return self.access_count >= threshold
```

---

### 2. MemorySearchResult

Result object returned from search operations.

```python
class MemorySearchResult(BaseModel):
    """
    A memory with relevance scoring for search results.
    """
    
    memory: Memory = Field(..., description="The memory object")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=2.0,  # Can exceed 1.0 with boosting
        description="Relevance score (similarity * freshness * access_boost)"
    )
    match_type: str = Field(
        ...,
        description="How this memory was found (scratchpad, vector, keyword)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "memory": {
                        "id": "mem-abc123",
                        "content": "Prefer PostgreSQL for new projects",
                        "category": "preference",
                        "tags": ["database"]
                    },
                    "relevance_score": 0.87,
                    "match_type": "vector"
                }
            ]
        }
    }
```

---

### 3. Observation

Captured observation from hook system (before converting to Memory).

```python
from enum import Enum

class ObservationType(str, Enum):
    """Categories of observations that can be auto-captured."""
    
    PREFERENCE = "preference"      # User preferences and choices
    DECISION = "decision"          # Architectural or implementation decisions
    DISCOVERY = "discovery"        # Learnings and insights
    PATTERN = "pattern"            # Code patterns and conventions
    TASK = "task"                  # Active work items (TODO, FIXME)
    BUGFIX = "bugfix"              # Bug resolutions
    NOTE = "note"                  # General notes
    QUESTION = "question"          # Unanswered questions
    INSIGHT = "insight"            # Deep understanding or realization

class Observation(BaseModel):
    """
    Raw observation captured by hooks before conversion to Memory.
    """
    
    content: str = Field(..., description="Observed content")
    observation_type: ObservationType = Field(..., description="Type of observation")
    source: str = Field(..., description="Where this was observed (tool_id, turn, etc.)")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in auto-capture (0.0-1.0)"
    )
    session_id: str = Field(..., description="Session that observed this")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def to_memory(self, agent_id: str) -> Memory:
        """Convert observation to Memory object."""
        return Memory(
            agent_id=agent_id,
            content=self.content,
            category=self.observation_type.value,
            tags=self.keywords,
            created_by_session=self.session_id,
            timestamp=self.timestamp,
            metadata={
                'source': self.source,
                'confidence': self.confidence,
                'auto_captured': True
            }
        )
```

---

## Storage Schemas

### 1. Scratchpad Schema (YAML)

Hot cache stored at `~/.amplifier/memory/{agent_id}/scratchpad.yaml`

```yaml
# Schema definition
agent_id: string              # Agent namespace (e.g., "bob")
last_updated: datetime        # ISO 8601 timestamp
max_size: integer            # Maximum entries (default: 50)

memories:
  - id: string               # Memory ID
    timestamp: datetime      # ISO 8601
    category: string         # Memory category
    content: string          # Memory text
    tags: [string]           # List of tags
    access_count: integer    # Access count
    last_accessed: datetime  # ISO 8601 or null
```

**Example:**

```yaml
agent_id: "bob"
last_updated: "2026-01-30T16:00:00Z"
max_size: 50

memories:
  - id: "mem-a3f2c1b4e5d6"
    timestamp: "2026-01-28T12:00:00Z"
    category: "preference"
    content: "Prefer PostgreSQL for new projects due to better tooling support"
    tags: ["database", "postgresql", "tools"]
    access_count: 3
    last_accessed: "2026-01-30T14:00:00Z"
  
  - id: "mem-b7e8f9a0c1d2"
    timestamp: "2026-01-27T15:30:00Z"
    category: "decision"
    content: "Use Qdrant for vector storage due to embedded mode"
    tags: ["architecture", "vector-db", "qdrant"]
    access_count: 1
    last_accessed: "2026-01-27T16:00:00Z"
  
  - id: "mem-c3d4e5f6a7b8"
    timestamp: "2026-01-26T10:15:00Z"
    category: "task"
    content: "TODO: Complete database migration design by end of week"
    tags: ["task", "database", "migration"]
    access_count: 0
    last_accessed: null
```

**Pydantic Model:**

```python
class ScratchpadMemoryEntry(BaseModel):
    """Lightweight memory entry for scratchpad (no embeddings)."""
    
    id: str
    timestamp: datetime
    category: str
    content: str
    tags: list[str]
    access_count: int
    last_accessed: Optional[datetime]

class Scratchpad(BaseModel):
    """Scratchpad hot cache schema."""
    
    agent_id: str = Field(..., pattern=r'^[a-zA-Z0-9_-]+$')
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    max_size: int = Field(default=50, ge=10, le=100)
    memories: list[ScratchpadMemoryEntry] = Field(default_factory=list)
    
    @field_validator('memories')
    @classmethod
    def validate_size(cls, v: list[ScratchpadMemoryEntry], info) -> list[ScratchpadMemoryEntry]:
        """Ensure scratchpad doesn't exceed max_size."""
        max_size = info.data.get('max_size', 50)
        if len(v) > max_size:
            # Keep most recent
            return sorted(v, key=lambda m: m.timestamp, reverse=True)[:max_size]
        return v
    
    def add_memory(self, memory: Memory) -> None:
        """Add memory to scratchpad, evicting oldest if full."""
        entry = ScratchpadMemoryEntry(**memory.to_scratchpad_entry())
        self.memories.append(entry)
        self.last_updated = datetime.utcnow()
        
        # Evict LRU if over capacity
        if len(self.memories) > self.max_size:
            self.evict_lru()
    
    def evict_lru(self) -> None:
        """Remove least recently used memory."""
        if not self.memories:
            return
        
        # Sort by last_accessed (None treated as very old)
        sorted_memories = sorted(
            self.memories,
            key=lambda m: m.last_accessed or datetime.min,
            reverse=True
        )
        self.memories = sorted_memories[:self.max_size]
```

---

### 2. Agent Metadata Schema (JSON)

Agent-level configuration and statistics stored at `~/.amplifier/memory/{agent_id}/metadata.json`

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
    "avg_query_latency_ms": 87.3,
    "storage_size_mb": 3.2,
    "embeddings_generated": 327,
    "scratchpad_promotions": 12
  },
  "config": {
    "scratchpad_max_size": 50,
    "migration_threshold": 5000,
    "auto_migrate": false,
    "embedding_model": "text-embedding-3-small",
    "similarity_threshold": 0.75
  },
  "migration": {
    "backup_path": "scratchpad.yaml.backup",
    "can_rollback": true,
    "migration_duration_seconds": 45.2
  }
}
```

**Pydantic Model:**

```python
class AgentStats(BaseModel):
    """Statistics tracked per agent."""
    
    total_queries: int = Field(default=0, ge=0)
    avg_query_latency_ms: float = Field(default=0.0, ge=0.0)
    storage_size_mb: float = Field(default=0.0, ge=0.0)
    embeddings_generated: int = Field(default=0, ge=0)
    scratchpad_promotions: int = Field(default=0, ge=0)

class AgentConfig(BaseModel):
    """Agent-level configuration."""
    
    scratchpad_max_size: int = Field(default=50, ge=10, le=100)
    migration_threshold: int = Field(default=5000, ge=100)
    auto_migrate: bool = Field(default=False)
    embedding_model: str = Field(default="text-embedding-3-small")
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

class MigrationInfo(BaseModel):
    """Migration state tracking."""
    
    backup_path: Optional[str] = None
    can_rollback: bool = False
    migration_duration_seconds: Optional[float] = None

class AgentMetadata(BaseModel):
    """Complete agent metadata schema."""
    
    agent_id: str = Field(..., pattern=r'^[a-zA-Z0-9_-]+$')
    created_at: datetime = Field(default_factory=datetime.utcnow)
    memory_count: int = Field(default=0, ge=0)
    storage_backend: str = Field(
        default="file",
        pattern=r'^(file|vector)$',
        description="Current storage backend (file or vector)"
    )
    migration_date: Optional[datetime] = None
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    stats: AgentStats = Field(default_factory=AgentStats)
    config: AgentConfig = Field(default_factory=AgentConfig)
    migration: Optional[MigrationInfo] = None
    
    def should_migrate(self) -> bool:
        """Check if agent should migrate to vector storage."""
        return (
            self.storage_backend == "file" and
            self.memory_count >= self.config.migration_threshold
        )
```

---

### 3. Vector Storage Schema (Qdrant)

Collection configuration for vector database at `~/.amplifier/memory/{agent_id}/memories.db`

**Collection Configuration:**

```python
from qdrant_client.models import Distance, VectorParams

QDRANT_CONFIG = {
    "collection_name": "memories",
    "vectors_config": VectorParams(
        size=1536,              # OpenAI text-embedding-3-small dimensions
        distance=Distance.COSINE  # Cosine similarity
    )
}
```

**Point Structure:**

```python
from qdrant_client.models import PointStruct

class VectorPoint(BaseModel):
    """Qdrant point structure for memory storage."""
    
    id: str  # Memory ID (e.g., "mem-abc123")
    vector: list[float]  # 1536 dimensions
    payload: dict  # Memory metadata (from Memory.to_vector_payload())

# Example point
point = PointStruct(
    id="mem-a3f2c1b4e5d6",
    vector=[0.123, -0.456, ...],  # 1536 floats
    payload={
        "content": "Prefer PostgreSQL for new projects",
        "agent_id": "bob",
        "timestamp": "2026-01-28T12:00:00Z",
        "category": "preference",
        "tags": ["database", "postgresql"],
        "access_count": 3,
        "last_accessed": "2026-01-30T14:00:00Z",
        "created_by_session": "session-abc-123",
        "metadata": {}
    }
)
```

**Query Structure:**

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

class VectorSearchQuery(BaseModel):
    """Parameters for vector similarity search."""
    
    query_vector: list[float] = Field(..., description="Query embedding (1536 dims)")
    limit: int = Field(default=5, ge=1, le=100, description="Max results")
    score_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )
    
    # Optional filters
    category_filter: Optional[str] = None
    tags_filter: Optional[list[str]] = None
    date_range: Optional[tuple[datetime, datetime]] = None
    
    def to_qdrant_filter(self) -> Optional[Filter]:
        """Convert to Qdrant filter conditions."""
        conditions = []
        
        if self.category_filter:
            conditions.append(
                FieldCondition(
                    key="category",
                    match=MatchValue(value=self.category_filter)
                )
            )
        
        if self.tags_filter:
            for tag in self.tags_filter:
                conditions.append(
                    FieldCondition(
                        key="tags",
                        match=MatchValue(value=tag)
                    )
                )
        
        return Filter(must=conditions) if conditions else None
```

---

## Configuration Models

### 1. Tool Configuration

Configuration for `tool-memory-semantic` module.

```python
class MemoryToolConfig(BaseModel):
    """Configuration for memory tools."""
    
    # Storage
    storage_root: str = Field(
        default="~/.amplifier/memory",
        description="Root directory for memory storage"
    )
    
    # Search defaults
    default_search_limit: int = Field(default=5, ge=1, le=100)
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    
    # Embedding
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536, ge=512)
    
    # Migration
    migration_threshold: int = Field(
        default=5000,
        ge=100,
        description="Memory count that triggers migration suggestion"
    )
    auto_migrate: bool = Field(
        default=False,
        description="Automatically migrate when threshold reached"
    )
    migration_batch_size: int = Field(
        default=100,
        ge=10,
        description="Batch size for embedding generation during migration"
    )
    
    # Performance
    enable_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=300, ge=0)
    
    # Freshness decay parameters
    freshness_decay_rate: float = Field(
        default=0.1,
        ge=0.0,
        description="Decay rate per day (0.1 = 10% per day)"
    )
    access_boost_rate: float = Field(
        default=0.05,
        ge=0.0,
        description="Boost per access (0.05 = 5% per access)"
    )
```

---

### 2. Context Configuration

Configuration for `context-memory-scratchpad` module.

```python
class MemoryContextConfig(BaseModel):
    """Configuration for scratchpad context injection."""
    
    # Token budget
    token_budget: int = Field(
        default=2000,
        ge=500,
        le=5000,
        description="Maximum tokens for scratchpad injection"
    )
    
    # Scratchpad size
    max_scratchpad_entries: int = Field(
        default=50,
        ge=10,
        le=100,
        description="Maximum memories in hot cache"
    )
    
    # Promotion thresholds
    promotion_access_threshold: int = Field(
        default=3,
        ge=1,
        description="Access count to promote to scratchpad"
    )
    
    # Formatting
    include_timestamps: bool = Field(default=True)
    include_categories: bool = Field(default=True)
    include_tags: bool = Field(default=True)
    
    # Filtering
    exclude_categories: list[str] = Field(
        default_factory=list,
        description="Categories to exclude from scratchpad"
    )
```

---

### 3. Hook Configuration

Configuration for `hooks-memory-capture` module.

```python
class MemoryCaptureConfig(BaseModel):
    """Configuration for auto-capture hook."""
    
    # Capture patterns
    capture_patterns: list[str] = Field(
        default_factory=lambda: [
            r'\b(remember|important|decided to|learned that)\b',
            r'\b(discovered|solved by|preference:)\b',
            r'\b(TODO|FIXME|NOTE):\s',
        ],
        description="Regex patterns that trigger capture"
    )
    
    # Confidence thresholds
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to auto-capture"
    )
    
    # Event hooks
    observe_tool_outputs: bool = Field(default=True)
    observe_turn_completions: bool = Field(default=True)
    observe_session_end: bool = Field(default=False)
    
    # Content filtering
    min_content_length: int = Field(default=10, ge=1)
    max_content_length: int = Field(default=10000, ge=100)
    
    # Rate limiting
    max_captures_per_session: int = Field(default=100, ge=1)
    
    # Async processing
    async_capture: bool = Field(
        default=True,
        description="Capture in background to avoid blocking"
    )
```

---

## Enums and Constants

### 1. Memory Categories

```python
class MemoryCategory(str, Enum):
    """Standard memory categories."""
    
    PREFERENCE = "preference"      # User preferences and choices
    DECISION = "decision"          # Architectural or implementation decisions
    DISCOVERY = "discovery"        # Learnings and insights
    PATTERN = "pattern"            # Code patterns and conventions
    TASK = "task"                  # Active work items
    BUGFIX = "bugfix"              # Bug resolutions
    NOTE = "note"                  # General notes
    QUESTION = "question"          # Unanswered questions
    INSIGHT = "insight"            # Deep understanding
    
    @classmethod
    def all_values(cls) -> set[str]:
        """Get all valid category values."""
        return {cat.value for cat in cls}
```

---

### 2. Storage Backend Types

```python
class StorageBackend(str, Enum):
    """Available storage backends."""
    
    FILE = "file"        # YAML-based (Phase 1)
    VECTOR = "vector"    # Qdrant-based (Phase 2)
    
    def __str__(self) -> str:
        return self.value
```

---

### 3. Match Types

```python
class MatchType(str, Enum):
    """How a memory was matched during search."""
    
    SCRATCHPAD = "scratchpad"    # Found in hot cache (keyword)
    VECTOR = "vector"            # Found via semantic similarity
    KEYWORD = "keyword"          # Found via keyword in file storage
    PROMOTED = "promoted"        # Recently promoted to scratchpad
```

---

### 4. System Constants

```python
# Embedding model specifications
EMBEDDING_MODELS = {
    "text-embedding-3-small": {
        "dimensions": 1536,
        "cost_per_1k_tokens": 0.00002,
        "max_tokens": 8191
    },
    "text-embedding-3-large": {
        "dimensions": 3072,
        "cost_per_1k_tokens": 0.00013,
        "max_tokens": 8191
    }
}

# Default paths
DEFAULT_MEMORY_ROOT = "~/.amplifier/memory"
SCRATCHPAD_FILENAME = "scratchpad.yaml"
VECTOR_DB_FILENAME = "memories.db"
METADATA_FILENAME = "metadata.json"
BACKUP_SUFFIX = ".backup"

# Size limits
MAX_MEMORY_CONTENT_LENGTH = 10000  # characters
MAX_SCRATCHPAD_SIZE = 100          # entries
MAX_TAGS_PER_MEMORY = 20
MAX_TAG_LENGTH = 50                # characters

# Performance
DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 100
DEFAULT_SIMILARITY_THRESHOLD = 0.75

# Migration
DEFAULT_MIGRATION_THRESHOLD = 5000  # memories
MIGRATION_BATCH_SIZE = 100          # embeddings per batch
```

---

## Validation Rules

### 1. Agent ID Validation

```python
import re

def validate_agent_id(agent_id: str) -> bool:
    """
    Validate agent ID for security and compatibility.
    
    Rules:
    - Alphanumeric, hyphens, underscores only
    - 1-64 characters
    - No path separators or special characters
    """
    if not agent_id or len(agent_id) > 64:
        return False
    
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, agent_id))

def sanitize_agent_id(agent_id: str) -> str:
    """Sanitize agent ID by removing invalid characters."""
    return re.sub(r'[^a-zA-Z0-9_-]', '', agent_id)[:64]
```

---

### 2. Content Validation

```python
def validate_memory_content(content: str) -> tuple[bool, Optional[str]]:
    """
    Validate memory content.
    
    Returns:
        (is_valid, error_message)
    """
    if not content or not content.strip():
        return False, "Content cannot be empty"
    
    if len(content) > MAX_MEMORY_CONTENT_LENGTH:
        return False, f"Content exceeds {MAX_MEMORY_CONTENT_LENGTH} characters"
    
    if len(content) < 3:
        return False, "Content too short (minimum 3 characters)"
    
    return True, None
```

---

### 3. Tag Validation

```python
def validate_tags(tags: list[str]) -> tuple[bool, Optional[str]]:
    """
    Validate memory tags.
    
    Returns:
        (is_valid, error_message)
    """
    if len(tags) > MAX_TAGS_PER_MEMORY:
        return False, f"Too many tags (maximum {MAX_TAGS_PER_MEMORY})"
    
    for tag in tags:
        if len(tag) > MAX_TAG_LENGTH:
            return False, f"Tag '{tag}' exceeds {MAX_TAG_LENGTH} characters"
        
        if not tag.strip():
            return False, "Empty tags not allowed"
    
    return True, None

def normalize_tags(tags: list[str]) -> list[str]:
    """Normalize tags: lowercase, strip whitespace, deduplicate."""
    return list(set(tag.lower().strip() for tag in tags if tag.strip()))
```

---

## Serialization Examples

### 1. Memory to JSON

```python
memory = Memory(
    agent_id="bob",
    content="Prefer PostgreSQL for new projects",
    category="preference",
    tags=["database", "postgresql"],
    created_by_session="session-abc-123"
)

# Serialize to JSON
json_str = memory.model_dump_json(indent=2)
```

**Output:**
```json
{
  "id": "mem-a3f2c1b4e5d6",
  "agent_id": "bob",
  "content": "Prefer PostgreSQL for new projects",
  "category": "preference",
  "tags": ["database", "postgresql"],
  "embedding": null,
  "timestamp": "2026-01-30T16:00:00Z",
  "last_accessed": null,
  "access_count": 0,
  "created_by_session": "session-abc-123",
  "metadata": {}
}
```

---

### 2. Scratchpad to YAML

```python
import yaml

scratchpad = Scratchpad(
    agent_id="bob",
    memories=[
        ScratchpadMemoryEntry(
            id="mem-001",
            timestamp=datetime.utcnow(),
            category="preference",
            content="Prefer PostgreSQL",
            tags=["database"],
            access_count=3,
            last_accessed=datetime.utcnow()
        )
    ]
)

# Serialize to YAML
yaml_str = yaml.dump(
    scratchpad.model_dump(mode='json'),
    sort_keys=False,
    allow_unicode=True
)
```

**Output:**
```yaml
agent_id: bob
last_updated: '2026-01-30T16:00:00Z'
max_size: 50
memories:
- id: mem-001
  timestamp: '2026-01-30T16:00:00Z'
  category: preference
  content: Prefer PostgreSQL
  tags:
  - database
  access_count: 3
  last_accessed: '2026-01-30T16:00:00Z'
```

---

### 3. Loading from Storage

```python
from pathlib import Path

def load_scratchpad(agent_id: str) -> Scratchpad:
    """Load scratchpad from YAML file."""
    path = Path(f"~/.amplifier/memory/{agent_id}/scratchpad.yaml").expanduser()
    
    if not path.exists():
        return Scratchpad(agent_id=agent_id)
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    return Scratchpad(**data)

def save_scratchpad(scratchpad: Scratchpad) -> None:
    """Save scratchpad to YAML file (atomic write)."""
    path = Path(f"~/.amplifier/memory/{scratchpad.agent_id}/scratchpad.yaml").expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Atomic write: temp file + rename
    temp_path = path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        yaml.dump(
            scratchpad.model_dump(mode='json'),
            f,
            sort_keys=False,
            allow_unicode=True
        )
    temp_path.replace(path)
```

---

## Migration Data Models

### 1. Migration Plan

```python
class MigrationPlan(BaseModel):
    """Plan for file -> vector migration."""
    
    agent_id: str
    current_backend: StorageBackend
    target_backend: StorageBackend
    memory_count: int
    estimated_duration_seconds: float
    estimated_cost_usd: float
    batch_size: int = 100
    backup_path: str
    
    @classmethod
    def create(cls, agent_id: str, metadata: AgentMetadata) -> "MigrationPlan":
        """Create migration plan from agent metadata."""
        memory_count = metadata.memory_count
        
        # Estimate duration (100 memories per batch, ~5 seconds per batch)
        batches = (memory_count + 99) // 100
        estimated_duration = batches * 5.0
        
        # Estimate cost (OpenAI embeddings)
        # Assume ~50 tokens per memory, $0.00002 per 1k tokens
        estimated_tokens = memory_count * 50
        estimated_cost = (estimated_tokens / 1000) * 0.00002
        
        return cls(
            agent_id=agent_id,
            current_backend=StorageBackend(metadata.storage_backend),
            target_backend=StorageBackend.VECTOR,
            memory_count=memory_count,
            estimated_duration_seconds=estimated_duration,
            estimated_cost_usd=estimated_cost,
            backup_path=f"{agent_id}/scratchpad.yaml.backup"
        )
```

---

### 2. Migration Progress

```python
class MigrationProgress(BaseModel):
    """Track migration progress."""
    
    plan: MigrationPlan
    status: str = Field(pattern=r'^(pending|running|completed|failed|rolled_back)$')
    
    # Progress tracking
    memories_processed: int = 0
    embeddings_generated: int = 0
    vectors_stored: int = 0
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Error handling
    errors: list[str] = Field(default_factory=list)
    can_retry: bool = True
    
    def percent_complete(self) -> float:
        """Calculate completion percentage."""
        if self.plan.memory_count == 0:
            return 100.0
        return (self.memories_processed / self.plan.memory_count) * 100.0
    
    def elapsed_seconds(self) -> Optional[float]:
        """Calculate elapsed time."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
```

---

## Summary

This document defines the complete data model specification for amplifier-bundle-agent-memory, including:

✅ **Core Models**: Memory, MemorySearchResult, Observation with full validation  
✅ **Storage Schemas**: YAML (scratchpad), JSON (metadata), Qdrant (vector)  
✅ **Configuration Models**: Tool, context, and hook configurations  
✅ **Enums & Constants**: Categories, storage types, system limits  
✅ **Validation Rules**: Security and data integrity constraints  
✅ **Serialization**: Conversion between Python, JSON, and YAML  
✅ **Migration Models**: Planning and progress tracking  

**Implementation Notes:**

1. All models use **Pydantic v2** for type safety and validation
2. Timestamps are **UTC-based** for consistency
3. Agent IDs are **validated** to prevent path traversal
4. Embeddings are **optional** (only for vector storage)
5. Scratchpad is **size-limited** with LRU eviction
6. All writes are **atomic** (temp file + rename)

**Next Steps:**

- Implement storage layer using these schemas
- Create serialization utilities for file I/O
- Build migration engine with progress tracking
- Add comprehensive unit tests for validation rules

---

**End of Data Models Document**
