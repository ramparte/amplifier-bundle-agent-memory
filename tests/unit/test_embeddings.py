"""Unit tests for embedding generation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from memory_semantic.embeddings import EmbeddingGenerator


@pytest.mark.asyncio
class TestEmbeddingGenerator:
    """Unit tests for embedding generation."""

    async def test_generate_single_embedding(self):
        """Can generate embedding for single text."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with patch("memory_semantic.embeddings.AsyncOpenAI") as mock_client:
            mock_client.return_value.embeddings.create = AsyncMock(
                return_value=mock_response
            )

            embedder = EmbeddingGenerator()
            result = await embedder.generate("test content")

            assert len(result) == 1536
            assert all(isinstance(x, float) for x in result)

    async def test_generate_batch_embeddings(self):
        """Can generate embeddings for batch."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]

        with patch("memory_semantic.embeddings.AsyncOpenAI") as mock_client:
            mock_client.return_value.embeddings.create = AsyncMock(
                return_value=mock_response
            )

            embedder = EmbeddingGenerator()
            results = await embedder.generate_batch(["text1", "text2"])

            assert len(results) == 2
            assert len(results[0]) == 1536
            assert len(results[1]) == 1536

    async def test_uses_correct_model(self):
        """Uses text-embedding-3-small by default."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with patch("memory_semantic.embeddings.AsyncOpenAI") as mock_client:
            mock_create = AsyncMock(return_value=mock_response)
            mock_client.return_value.embeddings.create = mock_create

            embedder = EmbeddingGenerator()
            await embedder.generate("test")

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["model"] == "text-embedding-3-small"

    def test_custom_model(self):
        """Can specify custom embedding model."""
        with patch("memory_semantic.embeddings.AsyncOpenAI"):
            embedder = EmbeddingGenerator(model="text-embedding-3-large")
            assert embedder.model == "text-embedding-3-large"
