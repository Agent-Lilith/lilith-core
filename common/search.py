import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from common.ranking import MCPScoreRanker

T = TypeVar("T")


class BaseHybridSearchEngine(ABC, Generic[T]):
    """Base class for hybrid search engines (Email, Browser, WhatsApp)."""

    def __init__(self, db: Any, embedder: Any = None):
        self.db = db
        self.embedder = embedder
        self._ranker = MCPScoreRanker()

    @abstractmethod
    def _get_item_id(self, item: T) -> Any:
        """Return a unique ID for the item."""
        pass

    @abstractmethod
    def _format_result(
        self, item: T, scores: dict[str, float], methods: list[str]
    ) -> dict[str, Any]:
        """Format the raw item into a SearchResultV1 compatible dict."""
        pass

    def _structured(
        self, filters: list[dict] | None, limit: int
    ) -> list[tuple[T, float]]:
        """Perform structured search using metadata filters."""
        return []

    def _fulltext(
        self, query: str, filters: list[dict] | None, limit: int
    ) -> list[tuple[T, float]]:
        """Perform keyword search."""
        return []

    def _vector(
        self, query: str, filters: list[dict] | None, limit: int
    ) -> list[tuple[T, float]]:
        """Perform semantic search."""
        return []

    def search(
        self,
        query: str = "",
        methods: list[str] | None = None,
        filters: list[dict] | None = None,
        top_k: int = 10,
    ) -> tuple[list[dict[str, Any]], dict[str, float], list[str]]:
        """Main hybrid search entry point."""
        t_start = time.monotonic()
        methods = methods or ["structured", "fulltext", "vector"]
        timing = {}
        methods_executed = []

        all_results: dict[Any, dict[str, Any]] = {}  # id -> {item, scores, methods}

        def add_batch(batch: list[tuple[T, float]], method_name: str):
            if not batch:
                return
            methods_executed.append(method_name)
            for item, score in batch:
                item_id = self._get_item_id(item)
                if item_id not in all_results:
                    all_results[item_id] = {"item": item, "scores": {}, "methods": []}
                all_results[item_id]["scores"][method_name] = score
                if method_name not in all_results[item_id]["methods"]:
                    all_results[item_id]["methods"].append(method_name)

        # 1. Structured
        if "structured" in methods:
            t0 = time.monotonic()
            try:
                batch = self._structured(filters, top_k * 2)
                add_batch(batch, "structured")
            except Exception as e:
                import logging

                logging.getLogger("common.search").warning(
                    "Structured search failed: %s", e
                )
            timing["structured"] = round((time.monotonic() - t0) * 1000, 1)

        # 2. Fulltext
        if "fulltext" in methods and query and query.strip():
            t0 = time.monotonic()
            try:
                batch = self._fulltext(query, filters, top_k * 2)
                add_batch(batch, "fulltext")
            except Exception as e:
                import logging

                logging.getLogger("common.search").warning(
                    "Fulltext search failed: %s", e
                )
            timing["fulltext"] = round((time.monotonic() - t0) * 1000, 1)

        # 3. Vector
        if "vector" in methods and query and query.strip():
            t0 = time.monotonic()
            try:
                batch = self._vector(query, filters, top_k * 2)
                add_batch(batch, "vector")
            except Exception as e:
                import logging

                logging.getLogger("common.search").warning(
                    "Vector search failed: %s", e
                )
            timing["vector"] = round((time.monotonic() - t0) * 1000, 1)

        # 4. Fusion & Format
        fusion_results: list[dict[str, Any]] = []
        for res in all_results.values():
            formatted = self._format_result(res["item"], res["scores"], res["methods"])
            fusion_results.append(formatted)
        t_fusion = time.monotonic()
        results = self._ranker.rank_results(fusion_results, top_k=top_k)
        timing["fusion"] = round((time.monotonic() - t_fusion) * 1000, 1)

        timing["total"] = round((time.monotonic() - t_start) * 1000, 1)
        return results, timing, methods_executed

    def rrf_fusion(
        self, results_groups: list[list[dict[str, Any]]], k: int = 60
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF) to combine multiple search result sets."""
        scores: dict[Any, float] = {}
        for group in results_groups:
            for rank, result in enumerate(group):
                doc_id = result.get("id")
                if doc_id is None:
                    continue
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

        return sorted(
            [{"id": doc_id, "score": score} for doc_id, score in scores.items()],
            key=lambda x: x["score"],
            reverse=True,
        )
