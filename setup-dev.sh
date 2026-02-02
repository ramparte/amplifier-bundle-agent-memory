#!/bin/bash
# Development setup script

# Install dependencies (without amplifier-core for now)
.venv/bin/pip install qdrant-client openai pydantic pytest pytest-asyncio pytest-cov pytest-mock

# Add amplifier-core to PYTHONPATH from local checkout
export PYTHONPATH="${PYTHONPATH}:/home/samschillace/dev/ANext/amplifier-core/src"

echo "✓ Development environment ready"
echo "Run tests with: PYTHONPATH=/home/samschillace/dev/ANext/amplifier-core/src .venv/bin/pytest tests/"
