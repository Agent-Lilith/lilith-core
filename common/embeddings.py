import logging

import httpx

logger = logging.getLogger(__name__)


class Embedder:
    """Shared embedding service client."""

    def __init__(self, endpoint_url: str, dim: int) -> None:
        self.endpoint_url = endpoint_url
        self.dim = dim

    def encode_sync(self, text: str) -> list[float]:
        """Synchronous wrapper for single text encoding."""
        if not text or not text.strip():
            return [0.0] * self.dim

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(self.endpoint_url, json={"texts": [text]})
                resp.raise_for_status()
                data = resp.json()

                # Handle different common response formats
                if isinstance(data, dict):
                    if "embeddings" in data:
                        return data["embeddings"][0]
                    if "data" in data and isinstance(data["data"], list):
                        # Some APIs return {"data": [{"embedding": [...]}]}
                        first = data["data"][0]
                        if isinstance(first, dict) and "embedding" in first:
                            return first["embedding"]
                        return first
                if isinstance(data, list) and len(data) > 0:
                    return data[0] if isinstance(data[0], list) else data

                return [0.0] * self.dim
        except Exception as e:
            logger.error("encode_sync failed: %s", e)
            return [0.0] * self.dim

    async def encode(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Asynchronous encoding."""
        texts = [text] if isinstance(text, str) else text
        if not texts:
            return [] if isinstance(text, list) else [0.0] * self.dim

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.endpoint_url, json={"texts": texts})
                resp.raise_for_status()
                data = resp.json()

                # Normalize response format
                embeddings = []
                if isinstance(data, dict) and "embeddings" in data:
                    embeddings = data["embeddings"]
                elif isinstance(data, list):
                    embeddings = data

                if isinstance(text, str):
                    return embeddings[0] if embeddings else [0.0] * self.dim
                return embeddings
        except Exception as e:
            logger.error("encode failed: %s", e)
            if isinstance(text, str):
                return [0.0] * self.dim
            return [[0.0] * self.dim] * len(texts)

    def encode_batch(
        self, texts: list[str], batch_size: int = 4, **kwargs
    ) -> list[list[float]]:
        """Synchronous batch encoding."""
        if not texts:
            return []

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(self.endpoint_url, json={"texts": batch})
                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, dict) and "embeddings" in data:
                        results.extend(data["embeddings"])
                    elif isinstance(data, list):
                        results.extend(data)
            except Exception as e:
                logger.error("encode_batch failed for batch %d: %s", i, e)
                results.extend([[0.0] * self.dim] * len(batch))
        return results
