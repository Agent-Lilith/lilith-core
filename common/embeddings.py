import asyncio
import httpx
import logging
from typing import List, Union, Optional

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, endpoint_url: str, dim: int) -> None:
        self.endpoint_url = endpoint_url
        self.dim = dim

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 4,
        **kwargs
    ) -> List[List[float]]:
        # This is an abstract base; implementation should be overridden or moved here if fully shared.
        # Based on previous session, this should have the robust logic.
        raise NotImplementedError("Implement encode_batch in subclass or move implementation to common")

    async def encode(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        raise NotImplementedError()
