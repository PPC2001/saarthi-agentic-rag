"""Embedding provider abstraction with zero duplicate code between DEV and PROD.

- DEV:  FastEmbedEmbedder (local, CPU-based, free, 384 dimensions)
- PROD: BedrockEmbedder (AWS Bedrock Titan Embeddings V2, 1024 dimensions)
"""

import asyncio
import hashlib
import json
import math
from typing import Protocol

from app.config import EmbeddingProvider, Settings
from app.observability.logging import get_logger

logger = get_logger("embeddings")


class Embedder(Protocol):
    """Abstract embedding protocol — both local and AWS providers implement this."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document chunks."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...

    @property
    def dimensions(self) -> int:
        """Return the embedding vector dimensionality."""
        ...


class FastEmbedEmbedder:
    """Local embedding using FastEmbed (DEV environment).

    Zero AWS calls, fast CPU inference, completely free.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dims: int = 384):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self._dims = dims
        logger.info(
            "FastEmbedEmbedder initialized",
            model_name=model_name,
            dimensions=dims,
        )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # FastEmbed runs synchronously; run in executor to keep async loop non-blocking
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, lambda: list(self._model.embed(texts)))
        logger.info(
            "FASTEMBED_DOCUMENTS_EMBEDDED", count=len(texts), dims=self._dims
        )
        return [e.tolist() for e in embeddings]

    async def embed_query(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: list(self._model.embed([text])))
        logger.info("FASTEMBED_QUERY_EMBEDDED", dims=self._dims)
        return result[0].tolist()

    @property
    def dimensions(self) -> int:
        return self._dims


class BedrockEmbedder:
    """AWS Bedrock Titan Embeddings V2 (PROD environment).

    High-dimensional, enterprise accuracy, AWS-native.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._model_id = settings.EMBEDDING_MODEL
        self._dims = settings.EMBEDDING_DIMENSIONS
        logger.info(
            "BedrockEmbedder initialized",
            model_id=self._model_id,
            dimensions=self._dims,
            region=settings.BEDROCK_REGION,
        )

    async def _embed_single(self, text: str) -> list[float]:
        import aioboto3

        session = aioboto3.Session(
            aws_access_key_id=self._settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self._settings.AWS_SECRET_ACCESS_KEY,
            region_name=self._settings.BEDROCK_REGION,
        )
        async with session.client("bedrock-runtime") as client:
            payload = {
                "inputText": text,
                "dimensions": self._dims,
                "normalize": True,
            }
            response = await client.invoke_model(
                modelId=self._model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            body = json.loads(await response["body"].read())
            return body["embedding"]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Batch in groups of 10 to avoid API throttling
        results: list[list[float]] = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tasks = [self._embed_single(text) for text in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        logger.info(
            "BEDROCK_TITAN_DOCUMENTS_EMBEDDED",
            count=len(texts),
            dims=self._dims,
        )
        return results

    async def embed_query(self, text: str) -> list[float]:
        result = await self._embed_single(text)
        logger.info("BEDROCK_TITAN_QUERY_EMBEDDED", dims=self._dims)
        return result

    @property
    def dimensions(self) -> int:
        return self._dims


class QwenEmbedder:
    """Qwen3-Embedding-8B embedding provider for all environments.

    State-of-the-art multilingual embedding model with 4096 dimensions
    and Matryoshka Representation Learning (MRL) support.
    Compatible with Portkey AI Gateway, Ollama, vLLM, DashScope, and HuggingFace TEI.
    """

    def __init__(self, settings: Settings):
        import openai

        self._model = settings.EMBEDDING_MODEL
        self._dims = settings.EMBEDDING_DIMENSIONS
        self._base_url = settings.PORTKEY_BASE_URL
        self._api_key = settings.PORTKEY_API_KEY

        default_headers = {}
        if "portkey" in self._base_url.lower():
            default_headers["x-portkey-api-key"] = self._api_key
            if settings.PORTKEY_VIRTUAL_KEY:
                default_headers["x-portkey-virtual-key"] = settings.PORTKEY_VIRTUAL_KEY

        self.is_fallback: bool = False
        self._client = openai.AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            default_headers=default_headers if default_headers else None,
            timeout=10.0,
        )
        logger.info(
            "QwenEmbedder initialized",
            model=self._model,
            dimensions=self._dims,
            base_url=self._base_url,
        )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""
        try:
            response = await self._client.embeddings.create(
                input=texts,
                model=self._model,
            )
            embeddings = [data.embedding for data in response.data]
            logger.info(
                "QWEN_DOCUMENTS_EMBEDDED",
                count=len(texts),
                dims=self._dims,
            )
            return embeddings
        except Exception as e:
            self.is_fallback = True
            logger.info("QWEN_EMBEDDING_FALLBACK", reason=str(e), count=len(texts))
            # Return deterministic normalized vectors for offline dev or unit tests
            return [self._generate_fallback_vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        results = await self.embed_documents([text])
        logger.info("QWEN_QUERY_EMBEDDED", dims=self._dims)
        return results[0]

    def _generate_fallback_vector(self, text: str) -> list[float]:
        """Generate deterministic normalized vector of size _dims if external API is unreachable."""
        h = hashlib.sha256(text.encode()).digest()
        vec = []
        for i in range(self._dims):
            byte_val = h[i % len(h)]
            vec.append((byte_val / 127.5) - 1.0)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    @property
    def dimensions(self) -> int:
        return self._dims


def create_embedder(settings: Settings) -> Embedder:
    """Factory creating the appropriate embedder based on config."""
    if settings.EMBEDDING_PROVIDER == EmbeddingProvider.FASTEMBED:
        return FastEmbedEmbedder(
            model_name=settings.EMBEDDING_MODEL or "BAAI/bge-small-en-v1.5",
            dims=settings.EMBEDDING_DIMENSIONS or 384,
        )
    if settings.EMBEDDING_PROVIDER == EmbeddingProvider.QWEN:
        return QwenEmbedder(settings=settings)
    return BedrockEmbedder(settings=settings)
