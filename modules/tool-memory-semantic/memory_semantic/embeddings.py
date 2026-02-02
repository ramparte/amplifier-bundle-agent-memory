"""OpenAI embedding generation wrapper."""

import os
from typing import Optional
from openai import AsyncOpenAI


class EmbeddingGenerator:
    """OpenAI embedding API wrapper for semantic memory.

    Uses text-embedding-3-small by default:
    - 1536 dimensions
    - ~$0.02/year for 1,000 memories
    - Good quality for semantic search
    """

    def __init__(
        self, model: str = "text-embedding-3-small", api_key: Optional[str] = None
    ):
        self.model = model
        self.dimensions = 1536

        # Validate API key is provided
        final_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not final_api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = AsyncOpenAI(api_key=final_api_key)

    async def generate(self, content: str) -> list[float]:
        """Generate embedding for single content.

        Args:
            content: Text to embed (max 100,000 chars)

        Returns:
            Vector of 1536 floats

        Raises:
            ValueError: If content exceeds size limit
            openai.OpenAIError: If API call fails
        """
        # Validate content length (OpenAI has 8191 token limit for this model)
        # 100k chars ~= 25k tokens, well within limits but prevents abuse
        MAX_CONTENT_LENGTH = 100_000
        if len(content) > MAX_CONTENT_LENGTH:
            raise ValueError(
                f"Content too long: {len(content)} chars (max {MAX_CONTENT_LENGTH})"
            )

        response = await self.client.embeddings.create(model=self.model, input=content)
        return response.data[0].embedding

    async def generate_batch(self, contents: list[str]) -> list[list[float]]:
        """Generate embeddings for batch of content.

        More efficient than multiple individual calls.

        Args:
            contents: List of texts to embed

        Returns:
            List of vectors

        Raises:
            openai.OpenAIError: If API call fails
        """
        response = await self.client.embeddings.create(model=self.model, input=contents)
        return [item.embedding for item in response.data]
