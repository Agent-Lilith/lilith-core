from common.config import BaseAgentSettings
from common.search import BaseHybridSearchEngine


class _DummyEngine(BaseHybridSearchEngine[dict]):
    def __init__(self, settings: BaseAgentSettings):
        super().__init__(db=None, embedder=None)
        # Override ranker settings for deterministic tests.
        from common.ranking import MCPScoreRanker

        self._ranker = MCPScoreRanker(settings=settings)

    def _get_item_id(self, item: dict):
        return item["id"]

    def _format_result(self, item: dict, scores: dict[str, float], methods: list[str]):
        return {
            "id": str(item["id"]),
            "source": "test_source",
            "timestamp": item.get("timestamp"),
            "scores": scores,
            "methods_used": methods,
            "metadata": {},
        }

    def _structured(self, filters, limit):
        return [({"id": 1}, 0.4), ({"id": 2}, 0.2)]

    def _vector(self, query, filters, limit):
        return [({"id": 2}, 0.8)]


def test_base_search_uses_ranker_and_emits_fusion_trace(tmp_path):
    settings = BaseAgentSettings(
        LILITH_SCORE_CALIBRATION_PATH=str(tmp_path / "calibration.json"),
    )
    engine = _DummyEngine(settings=settings)
    results, timing, methods = engine.search(query="hello", top_k=2)

    assert len(results) == 2
    assert "fusion" in timing
    assert set(methods) == {"structured", "vector"}
    assert "fusion_trace" in results[0]["metadata"]
