# Amplifier Memory Tool Module

Semantic memory tool for Amplifier using OpenAI embeddings and Qdrant vector database.

## Installation

### As part of the agent-memory bundle

```yaml
includes:
  - bundle: git+https://github.com/microsoft/amplifier-bundle-agent-memory@main
```

### Standalone module installation

```bash
pip install git+https://github.com/microsoft/amplifier-bundle-agent-memory@main#subdirectory=modules/tool-memory-semantic
```

## Usage in Amplifier

Configure in your bundle:

```yaml
config:
  tools:
    - module: tool-memory-semantic
      source: git+https://github.com/microsoft/amplifier-bundle-agent-memory@main#subdirectory=modules/tool-memory-semantic
      config:
        agent_id: "your-agent-name"  # Optional - auto-detected from session
        storage_root: "~/.amplifier/memory"  # Optional
        embedding_model: "text-embedding-3-small"  # Optional
        max_memories_per_agent: 10000  # Optional
```

## Agent ID Resolution

The module automatically determines the agent ID using this priority:

1. **Explicit config**: `config.agent_id` (if provided)
2. **Runtime detection**: `coordinator.session.agent_id` (from active session)
3. **Fallback**: `"default-agent"`

This allows the bundle to work without explicit configuration in most cases.

## Tools Provided

### memory_store

Store a memory with automatic embedding generation.

**Input:**
```json
{
  "content": "Memory text to store",
  "tags": ["optional", "tags"]
}
```

**Returns:**
```json
{
  "success": true,
  "memory_id": "uuid",
  "agent_id": "agent-name"
}
```

### memory_search

Search memories semantically with relevance + recency ranking.

**Input:**
```json
{
  "query": "Search query",
  "limit": 5,
  "since": "2026-01-01T00:00:00Z"
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
      "tags": ["tag1"],
      "score": 0.89
    }
  ]
}
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `agent_id` | string | auto-detected | Agent namespace for memory isolation |
| `storage_root` | string | `~/.amplifier/memory` | Base directory for storage |
| `embedding_model` | string | `text-embedding-3-small` | OpenAI embedding model |
| `max_memories_per_agent` | int | 10000 | Maximum memories per agent |

## Environment Variables

**Required:**
- `OPENAI_API_KEY` - Your OpenAI API key for embedding generation

## Architecture

```
Tool calls → OpenAI Embeddings API → Qdrant Vector DB → Results
                                      (local storage)
```

**Storage location:** `~/.amplifier/memory/{agent_id}/qdrant.db/`

## Security Features

- ✅ Agent ID validation (prevents path traversal)
- ✅ Content length limits (max 10,000 chars)
- ✅ Tag validation (max 50 tags, 100 chars each)
- ✅ Memory count limits (max 10,000 per agent)
- ✅ API key never logged or exposed

## Performance

- **Storage:** ~200ms per memory (OpenAI API latency)
- **Search:** ~155ms (embedding + vector search)
- **Scalability:** Linear up to 10,000 memories

## Development

### Running Tests

```bash
# From bundle root
pytest tests/ -v

# With coverage
pytest tests/ --cov=memory_semantic
```

### Dependencies

- `amplifier-core>=2.0.0` - Amplifier kernel
- `openai>=1.0.0` - OpenAI API client
- `qdrant-client>=1.7.0` - Qdrant vector database
- `pydantic>=2.0.0` - Data validation

## Full Documentation

See the [main bundle README](../../README.md) for complete documentation including:
- Detailed usage examples
- Bundle composition patterns
- Cost analysis
- Troubleshooting

## License

MIT License - see [LICENSE](../../LICENSE)
