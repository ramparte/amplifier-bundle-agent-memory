# Memory System Overview

This bundle provides semantic memory capabilities for Amplifier agents.

## What You Can Do

### Store Memories
Store information for later recall using semantic understanding:

```
User: Remember this: I prefer PostgreSQL for new projects
Assistant: ✓ Stored memory with ID: mem-abc123
```

### Search Memories
Find relevant memories through semantic search (not just keywords):

```
User: What do you remember about database choices?
Assistant: I found 2 relevant memories:
          1. "I prefer PostgreSQL for new projects" (stored 2 days ago)
          2. "PostgreSQL has better ACID guarantees" (stored 5 days ago)
```

## How It Works

**Storage:**
1. You provide memory content and optional tags
2. System generates semantic embedding via OpenAI
3. Memory stored in local Qdrant vector database
4. Organized by agent namespace for isolation

**Search:**
1. You provide search query
2. System generates query embedding
3. Vector similarity search finds relevant memories
4. Results ranked by relevance + recency (20% boost for last 7 days)
5. Top matches returned with context

## Architecture

```
Memory Storage:
  Content → OpenAI Embedding API → Vector → Qdrant DB → Success

Memory Search:
  Query → OpenAI Embedding API → Vector → Qdrant Search → Ranked Results
```

## Memory Isolation

Each agent has isolated storage:
- Stored at: `~/.amplifier/memory/{agent_id}/`
- No cross-agent access
- Separate Qdrant collections per agent

## Limits and Constraints

**Per Memory:**
- Maximum content length: 10,000 characters
- Maximum tags: 50
- Maximum tag length: 100 characters

**Per Agent:**
- Maximum memories: 10,000 (configurable)
- Storage size: ~1GB typical usage

**Rate Limits:**
- OpenAI API rate limits apply to embedding generation
- Local storage has no rate limits

## Costs

**OpenAI Embeddings:**
- Model: `text-embedding-3-small` (1536 dimensions)
- Cost: ~$0.000002 per memory
- Very cost-effective for persistent knowledge

**Storage:**
- Local disk only (no cloud costs)
- ~100KB per memory (including vector)

## Security

**Built-in protections:**
- Agent ID validation (no path traversal)
- Content length limits (prevents DoS)
- Tag validation
- Memory count limits

**API Key Handling:**
- OpenAI API key from environment variable
- Never logged or exposed
- Used only for embedding generation

## Tool Interface

**memory_store:**
```json
{
  "content": "The memory text to store",
  "tags": ["optional", "categorization", "tags"]
}
```

**memory_search:**
```json
{
  "query": "What to search for",
  "limit": 5,
  "since": "2026-01-01T00:00:00Z"  // optional
}
```

## Configuration

Customize in your bundle:

```yaml
config:
  tools:
    - module: tool-memory-semantic
      config:
        agent_id: "custom-agent-id"
        storage_root: "/custom/path"
        embedding_model: "text-embedding-3-large"  # for better quality
        max_memories_per_agent: 5000
```

## Environment Requirements

**Required:**
- `OPENAI_API_KEY` environment variable

**Optional:**
- Custom storage paths via config
- Custom embedding models via config

## Use Cases

**Learning Preferences:**
```
Store: "I prefer functional programming over OOP"
Search: "programming style preferences"
```

**Architectural Decisions:**
```
Store: "We chose microservices for team scalability"
Search: "why did we choose microservices"
```

**Code Patterns:**
```
Store: "Use repository pattern for data access"
Search: "data access patterns"
```

**Project Context:**
```
Store: "Project deadline is March 15th"
Search: "project timeline"
```

## Troubleshooting

**"No memories found":**
- Check that agent_id matches between store and search
- Verify memories were successfully stored
- Try broader search terms

**"OpenAI API error":**
- Verify OPENAI_API_KEY is set
- Check API key has sufficient credits
- Verify network connectivity

**"Storage error":**
- Check disk space at ~/.amplifier/memory/
- Verify write permissions
- Check memory limit hasn't been reached

## Performance

**Storage speed:**
- ~200ms per memory (OpenAI API latency)
- Local disk write is negligible

**Search speed:**
- ~150ms for embedding generation
- ~5ms for vector search (local)
- Total: ~155ms typical

**Scalability:**
- Up to 10,000 memories per agent
- Linear search time (Qdrant HNSW index)
- Memory footprint: ~1GB per 10,000 memories
