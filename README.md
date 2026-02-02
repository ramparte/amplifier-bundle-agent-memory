# Amplifier Agent Memory Bundle

Persistent semantic memory for Amplifier agents using vector similarity search.

## Features

- **Semantic Search** - Understands meaning, not just keywords
- **Automatic Embeddings** - OpenAI embeddings generated automatically
- **Recency Ranking** - Recent memories get relevance boost
- **Agent Isolation** - Per-agent namespaces prevent cross-contamination
- **Local-First** - All storage on your machine (~/.amplifier/memory)
- **Cost Efficient** - ~$0.000002 per memory stored

## Quick Start

### Installation

**Option 1: Include in your bundle**
```yaml
includes:
  - bundle: git+https://github.com/microsoft/amplifier-bundle-agent-memory@main
```

**Option 2: Standalone**
```bash
amplifier bundle add git+https://github.com/microsoft/amplifier-bundle-agent-memory
amplifier bundle use agent-memory
```

### Prerequisites

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### Usage

**Store memories:**
```
User: Remember this: I prefer PostgreSQL for new projects
Assistant: ✓ Stored memory: PostgreSQL preference (ID: mem-abc123)
```

**Search memories:**
```
User: What do you remember about database choices?
Assistant: Found 2 memories:
          1. "I prefer PostgreSQL for new projects" (2 days ago)
          2. "PostgreSQL has better ACID guarantees" (5 days ago)
```

## How It Works

1. **Storage**: Content → OpenAI Embedding → Qdrant Vector DB
2. **Search**: Query → OpenAI Embedding → Similarity Search → Ranked Results

Memories are ranked by **semantic similarity + recency** (20% boost for last 7 days).

## Configuration

Customize in your bundle or settings:

```yaml
config:
  tools:
    - module: tool-memory-semantic
      config:
        agent_id: "my-agent"                    # Agent namespace
        storage_root: "~/.amplifier/memory"     # Storage location
        embedding_model: "text-embedding-3-small"  # OpenAI model
        max_memories_per_agent: 10000           # Memory limit
```

## Architecture

```
┌─────────────────────────────────────────────┐
│ Amplifier Agent                             │
│   ↓                                         │
│ memory_store / memory_search tools          │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│ OpenAI Embeddings API                       │
│ (text-embedding-3-small)                    │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│ Qdrant Vector Database (local)              │
│ ~/.amplifier/memory/{agent_id}/             │
└─────────────────────────────────────────────┘
```

## Tools Provided

### memory_store

Store a memory with automatic embedding generation.

**Input:**
```json
{
  "content": "Memory text to store",
  "tags": ["optional", "tags"]  // max 50 tags
}
```

**Returns:**
```json
{
  "success": true,
  "memory_id": "uuid-here",
  "agent_id": "agent-name"
}
```

### memory_search

Search memories semantically with relevance + recency ranking.

**Input:**
```json
{
  "query": "Search query",
  "limit": 5,                              // optional, max 20
  "since": "2026-01-01T00:00:00Z"         // optional
}
```

**Returns:**
```json
{
  "success": true,
  "count": 2,
  "memories": [
    {
      "id": "uuid",
      "content": "Memory text",
      "timestamp": "2026-02-01T10:00:00Z",
      "tags": ["tag1", "tag2"],
      "score": 0.89
    }
  ]
}
```

## Limits

**Per Memory:**
- Content: 10,000 characters max
- Tags: 50 max, 100 characters each

**Per Agent:**
- Total memories: 10,000 (configurable)
- Storage: ~1GB typical

## Costs

**OpenAI Embeddings:**
- Model: `text-embedding-3-small`
- Cost: ~$0.02 per 1M tokens
- **Per memory: ~$0.000002** (very cheap!)

**Storage:**
- Local disk only (no cloud costs)
- ~100KB per memory

## Security

Built-in protections:
- ✓ Agent ID validation (no path traversal)
- ✓ Content length limits (prevents DoS)
- ✓ Tag validation
- ✓ Memory count limits
- ✓ API key never logged or exposed

## Development

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for:
- Local development setup
- Running tests
- Contributing guidelines

## Performance

**Storage:** ~200ms per memory (OpenAI API latency)  
**Search:** ~155ms (embedding + vector search)  
**Scalability:** Up to 10,000 memories per agent

## License

MIT License - see [LICENSE](LICENSE)

## Support

- **Issues:** [GitHub Issues](https://github.com/microsoft/amplifier-bundle-agent-memory/issues)
- **Discussions:** [GitHub Discussions](https://github.com/microsoft/amplifier-bundle-agent-memory/discussions)
- **Amplifier Docs:** [microsoft.github.io/amplifier](https://microsoft.github.io/amplifier)

## Related Bundles

- [amplifier-foundation](https://github.com/microsoft/amplifier-foundation) - Core Amplifier library
- [amplifier-core](https://github.com/microsoft/amplifier-core) - Amplifier kernel

## Acknowledgments

Built on:
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Qdrant](https://qdrant.tech/) - Vector similarity search engine
- [Amplifier](https://github.com/microsoft/amplifier) - AI agent framework
