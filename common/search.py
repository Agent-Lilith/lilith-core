from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generic, TypeVar

T = TypeVar("T")

class BaseHybridSearchEngine(ABC, Generic[T]):
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform a hybrid search combining multiple methods."""
        pass

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
