from datetime import UTC, datetime, timedelta

from common.config import BaseAgentSettings
from common.ranking import MCPScoreRanker


def _settings(tmp_path, **kwargs) -> BaseAgentSettings:
    return BaseAgentSettings(
        LILITH_SCORE_CALIBRATION_PATH=str(tmp_path / "calibration.json"),
        **kwargs,
    )


def test_rank_results_adds_fusion_trace_and_sorts(tmp_path):
    ranker = MCPScoreRanker(settings=_settings(tmp_path))
    now = datetime.now(UTC)
    results = [
        {
            "id": "1",
            "source": "email",
            "timestamp": now.isoformat(),
            "scores": {"vector": 0.9},
            "metadata": {},
        },
        {
            "id": "2",
            "source": "email",
            "timestamp": (now - timedelta(days=700)).isoformat(),
            "scores": {"vector": 0.5},
            "metadata": {},
        },
    ]

    ranked = ranker.rank_results(results, top_k=2)

    assert ranked[0]["id"] == "1"
    trace = ranked[0]["metadata"]["fusion_trace"]
    assert "raw_scores" in trace
    assert "normalized_scores" in trace
    assert "final_score" in trace


def test_rolling_window_is_capped(tmp_path):
    ranker = MCPScoreRanker(
        settings=_settings(tmp_path, LILITH_SCORE_WINDOW_SIZE=100),
    )
    results = [
        {
            "id": str(i),
            "source": "email",
            "timestamp": None,
            "scores": {"fulltext": float((i % 10) / 10)},
            "metadata": {},
        }
        for i in range(150)
    ]
    ranker.rank_results(results, top_k=10)
    stats = ranker._get_stats("email", "fulltext")
    assert stats["count"] == 100
    assert len(stats["scores"]) == 100


def test_reliability_prior_from_settings_affects_trace(tmp_path):
    ranker = MCPScoreRanker(
        settings=_settings(
            tmp_path,
            LILITH_SOURCE_RELIABILITY_PRIORS={"email": 1.08},
        ),
    )
    ranked = ranker.rank_results(
        [
            {
                "id": "x",
                "source": "email",
                "timestamp": None,
                "scores": {"structured": 0.6},
                "metadata": {},
            }
        ],
        top_k=1,
    )
    trace = ranked[0]["metadata"]["fusion_trace"]
    assert trace["reliability_prior"] == 1.08


def test_recency_is_gentle(tmp_path):
    ranker = MCPScoreRanker(settings=_settings(tmp_path))
    recent = ranker._compute_recency_boost(datetime.now(UTC).isoformat())
    old = ranker._compute_recency_boost(
        (datetime.now(UTC) - timedelta(days=3650)).isoformat()
    )
    assert recent <= 1.05
    assert old >= 1.0
    assert recent >= old
