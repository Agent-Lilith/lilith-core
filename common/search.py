import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generic, TypeVar, Tuple, Optional

T = TypeVar("T")

class BaseHybridSearchEngine(ABC, Generic[T]):
    """Base class for hybrid search engines (Email, Browser, WhatsApp)."""

    def __init__(self, db: Any, embedder: Any = None):
        self.db = db
        self.embedder = embedder

    @abstractmethod
    def _get_item_id(self, item: T) -> Any:
        """Return a unique ID for the item."""
        pass

    @abstractmethod
    def _format_result(self, item: T, scores: Dict[str, float], methods: List[str]) -> Dict[str, Any]:
        """Format the raw item into a SearchResultV1 compatible dict."""
        pass

    def _structured(self, filters: List[Dict] | None, limit: int) -> List[Tuple[T, float]]:
        """Perform structured search using metadata filters."""
        return []

    def _fulltext(self, query: str, filters: List[Dict] | None, limit: int) -> List[Tuple[T, float]]:
        """Perform keyword search."""
        return []

    def _vector(self, query: str, filters: List[Dict] | None, limit: int) -> List[Tuple[T, float]]:
        """Perform semantic search."""
        return []

    def search(
        self,
        query: str = "",
        methods: List[str] | None = None,
        filters: List[Dict] | None = None,
        top_k: int = 10,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float], List[str]]:
        """Main hybrid search entry point."""
        t_start = time.monotonic()
        methods = methods or ["structured", "fulltext", "vector"]
        timing = {}
        methods_executed = []

        all_results: Dict[Any, Dict[str, Any]] = {} # id -> {item, scores, methods}

        def add_batch(batch: List[Tuple[T, float]], method_name: str):
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
                logging.getLogger("common.search").warning("Structured search failed: %s", e)
            timing["structured"] = round((time.monotonic() - t0) * 1000, 1)

        # 2. Fulltext
        if "fulltext" in methods and query and query.strip():
            t0 = time.monotonic()
            try:
                batch = self._fulltext(query, filters, top_k * 2)
                add_batch(batch, "fulltext")
            except Exception as e:
                import logging
                logging.getLogger("common.search").warning("Fulltext search failed: %s", e)
            timing["fulltext"] = round((time.monotonic() - t0) * 1000, 1)

        # 3. Vector
        if "vector" in methods and query and query.strip():
            t0 = time.monotonic()
            try:
                batch = self._vector(query, filters, top_k * 2)
                add_batch(batch, "vector")
            except Exception as e:
                import logging
                logging.getLogger("common.search").warning("Vector search failed: %s", e)
            timing["vector"] = round((time.monotonic() - t0) * 1000, 1)

        # 4. Fusion & Format
        fusion_results = []
        for res in all_results.values():
            # Use max score for now
            final_score = max(res["scores"].values())
            formatted = self._format_result(res["item"], res["scores"], res["methods"])
            fusion_results.append((formatted, final_score))

        # Sort by score descending
        fusion_results.sort(key=lambda x: x[1], reverse=True)
        results = [x[0] for x in fusion_results[:top_k]]

        timing["total"] = round((time.monotonic() - t_start) * 1000, 1)
        return results, timing, methods_executed

    def rrf_fusion(self, results_groups: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF) to combine multiple search result sets."""
        scores = {}
        for group in results_groups:
            for rank, result in enumerate(group):
                doc_id = result.get("id")
                if doc_id is None:
                    continue
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        return sorted([{"id": doc_id, "score": score} for doc_id, score in scores.items()], key=lambda x: x["score"], reverse=True)
