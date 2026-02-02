---
bundle:
  name: agent-memory
  version: 1.0.0
  description: Semantic memory system for Amplifier agents with vector similarity search
  repository: https://github.com/ramparte/amplifier-bundle-agent-memory
  license: MIT

config:
  tools:
    - module: tool-memory-semantic
      source: git+https://github.com/ramparte/amplifier-bundle-agent-memory@main#subdirectory=modules/tool-memory-semantic
      config:
        # Agent namespace for memory isolation
        agent_id: "{{ session.agent_id | default('default-agent') }}"
        
        # Storage location (default: ~/.amplifier/memory)
        # storage_root: "~/.amplifier/memory"
        
        # OpenAI embedding model
        # embedding_model: "text-embedding-3-small"
        
        # Maximum memories per agent (prevents unbounded growth)
        # max_memories_per_agent: 10000

includes:
  # Include context awareness by default
  - "context/memory-system-overview.md"
---

# Agent Memory Bundle

Persistent semantic memory for Amplifier agents using vector similarity search.

## What This Provides

**Tools:**
- `memory_store` - Store memories with automatic embedding generation
- `memory_search` - Search memories semantically with relevance + recency ranking

**Architecture:**
- OpenAI embeddings for semantic understanding
- Qdrant vector database for fast similarity search
- Per-agent namespace isolation
- Local-first storage (~/.amplifier/memory)

## Quick Start

**Add to your bundle:**
```yaml
includes:
  - bundle: git+https://github.com/microsoft/amplifier-bundle-agent-memory@main
```

**Or install standalone:**
```bash
amplifier bundle add git+https://github.com/microsoft/amplifier-bundle-agent-memory
amplifier bundle use agent-memory
```

## Configuration

Customize in your bundle or settings:

```yaml
config:
  tools:
    - module: tool-memory-semantic
      config:
        agent_id: "my-custom-agent"
        storage_root: "/custom/path"
        embedding_model: "text-embedding-3-large"
        max_memories_per_agent: 5000
```

## Environment Variables

**Required:**
- `OPENAI_API_KEY` - Your OpenAI API key for embeddings

**Optional:**
- Set in config instead of relying on defaults

## Usage Patterns

### Storing Memories

```
"Remember this: I prefer PostgreSQL for new projects"
"Store this learning: Repository pattern separates data access from business logic"
```

The assistant will:
1. Generate semantic embedding
2. Store with timestamp and tags
3. Confirm storage with memory ID

### Searching Memories

```
"What do I remember about database choices?"
"Search my memories for architecture patterns"
```

The assistant will:
1. Search semantically (not just keywords)
2. Rank by relevance + recency
3. Return top matches with context

### Natural Integration

The memory system is designed to feel natural in conversation:

```
User: Remember that I prefer FastAPI for APIs
Assistant: ✓ Stored memory: FastAPI preference
          Category: preference
          Memory ID: abc123...

User: What do you remember about my API preferences?
Assistant: I found 2 memories:
          
          1. "I prefer FastAPI for APIs"
             Stored: 2026-02-01
             Tags: preference, fastapi, api
          
          2. "FastAPI uses Pydantic for validation"
             Stored: 2026-01-28
             Tags: fastapi, validation
```

## Features

### Semantic Search
Not just keyword matching - understands meaning and context.

**Example:**
- Store: "I prefer PostgreSQL for new projects"
- Search: "database choices" ✓ Found!

### Recency Boost
Recent memories (last 7 days) get 20% relevance boost.

### Agent Isolation
Each agent has isolated storage - no cross-contamination.

### Cost Efficient
- Uses OpenAI's most cost-effective embedding model
- Embeddings cached in Qdrant (no re-embedding on search)
- ~$0.00002 per memory stored

### Security Hardened
- Path traversal protection
- Input validation (content, tags, memory IDs)
- Memory count limits (prevents unbounded growth)
- Rate limiting ready (configurable)

## Limits

**Per memory:**
- Content: 10,000 characters
- Tags: 50 max, 100 chars each

**Per agent:**
- Total memories: 10,000 (configurable)
- Storage: ~1GB typical (depends on usage)

## Architecture

```
User prompt → Assistant
              ↓
         memory_store tool
              ↓
    OpenAI Embedding API (text-embedding-3-small)
              ↓
    Qdrant Vector DB (local)
    ~/.amplifier/memory/{agent_id}/
              ↓
         Success ✓

Search query → memory_search tool
              ↓
    OpenAI Embedding API
              ↓
    Qdrant Similarity Search
              ↓
    Rank by relevance + recency
              ↓
    Return top N results
```

## Dependencies

**Python packages:**
- `openai` - Embedding generation
- `qdrant-client` - Vector database
- `pydantic` - Data validation

**External services:**
- OpenAI API (for embeddings only, not chat)

## Costs

**OpenAI Embeddings:**
- Model: `text-embedding-3-small`
- Cost: ~$0.02 per 1M tokens
- Typical memory: ~100 tokens
- **Cost per memory: ~$0.000002** (very cheap!)

**Storage:**
- Local disk only (no cloud costs)
- ~100KB per memory (including vector)

## Development

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for:
- Local development setup
- Running tests
- Adding features
- Security guidelines

## License

MIT - See [LICENSE](LICENSE)