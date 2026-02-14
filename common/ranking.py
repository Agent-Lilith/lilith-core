"""MCP-side ranking calibration and fusion.

Implements:
- rolling-window score calibration (min/max + mean/std)
- per-query drift detection
- gentle recency boost
- source reliability prior
- optional learned ranker flag (placeholder, off by default)
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.config import BaseAgentSettings, get_base_settings

logger = logging.getLogger(__name__)

_METHOD_WEIGHTS: dict[str, float] = {
    "structured": 1.0,
    "fulltext": 0.85,
    "vector": 0.7,
    "graph": 0.9,
}


def _parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class MCPScoreRanker:
    """Calibrates and ranks results on MCP server side."""

    def __init__(self, settings: BaseAgentSettings | None = None) -> None:
        cfg = settings or get_base_settings()
        self._window_size = max(100, int(cfg.LILITH_SCORE_WINDOW_SIZE))
        self._drift_z = max(0.5, float(cfg.LILITH_SCORE_DRIFT_Z))
        self._recency_half_life_days = max(
            1.0, float(cfg.LILITH_SCORE_RECENCY_HALF_LIFE_DAYS)
        )
        self._learned_enabled = bool(cfg.LILITH_ENABLE_LEARNED_RANKING)
        self._artifact_path = Path(cfg.LILITH_SCORE_CALIBRATION_PATH)
        self._reliability_priors = {
            str(k): float(v)
            for k, v in (cfg.LILITH_SOURCE_RELIABILITY_PRIORS or {}).items()
        }
        self._state: dict[str, Any] = self._load_state()
        self._dirty_count = 0

    def rank_results(
        self,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not results:
            return []

        method_groups: dict[tuple[str, str], list[float]] = {}
        for r in results:
            source = str(r.get("source", "") or "")
            scores = r.get("scores", {}) or {}
            if not source or not isinstance(scores, dict):
                continue
            for method, score in scores.items():
                try:
                    s = float(score)
                except (TypeError, ValueError):
                    continue
                method_groups.setdefault((source, str(method)), []).append(_clamp01(s))

        drift_flags: dict[tuple[str, str], bool] = {}
        for key, values in method_groups.items():
            source, method = key
            stats = self._get_stats(source, method)
            mean = float(stats.get("mean", 0.0))
            std = float(stats.get("std", 0.0))
            query_mean = sum(values) / len(values) if values else 0.0
            drift = bool(
                stats.get("count", 0) >= 50
                and std > 1e-6
                and abs(query_mean - mean) > self._drift_z * std
            )
            drift_flags[key] = drift

        scored: list[tuple[float, dict[str, Any]]] = []
        for r in results:
            score, trace = self._score_result(r, drift_flags)
            meta = dict(r.get("metadata") or {})
            meta["fusion_trace"] = trace
            r["metadata"] = meta
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        ranked = [r for _, r in scored[:top_k]]

        self._record_observations(method_groups, drift_flags)
        return ranked

    def _score_result(
        self,
        result: dict[str, Any],
        drift_flags: dict[tuple[str, str], bool],
    ) -> tuple[float, dict[str, Any]]:
        source = str(result.get("source", "") or "")
        scores = result.get("scores", {}) or {}

        total_weight = 0.0
        weighted_signal = 0.0
        normalized_scores: dict[str, float] = {}
        method_signals: dict[str, float] = {}
        drifted_methods: list[str] = []
        raw_scores: dict[str, float] = {}

        for method, score in scores.items():
            try:
                raw = _clamp01(float(score))
            except (TypeError, ValueError):
                continue
            method_str = str(method)
            raw_scores[method_str] = raw
            norm = self._normalize_score(source, method_str, raw)
            normalized_scores[method_str] = norm

            drift = drift_flags.get((source, method_str), False)
            raw_weight = 0.85 if drift else 0.7
            if drift:
                drifted_methods.append(method_str)
            signal = raw_weight * raw + (1.0 - raw_weight) * norm
            method_signals[method_str] = signal

            w = _METHOD_WEIGHTS.get(method_str, 0.5)
            total_weight += w
            weighted_signal += signal * w

        base_score = weighted_signal / total_weight if total_weight > 0 else 0.0
        recency_boost = self._compute_recency_boost(result.get("timestamp"))
        reliability_prior = self._get_reliability_prior(source)

        # Keep relevance dominant. Recency/reliability are gentle multipliers.
        final_score = _clamp01(
            base_score * (0.9 + 0.05 * recency_boost + 0.05 * reliability_prior)
        )

        trace = {
            "raw_scores": raw_scores,
            "normalized_scores": normalized_scores,
            "method_signals": method_signals,
            "base_score": round(base_score, 4),
            "recency_boost": round(recency_boost, 4),
            "reliability_prior": round(reliability_prior, 4),
            "drift_detected_methods": drifted_methods,
            "learned_ranking_enabled": self._learned_enabled,
            "final_score": round(final_score, 4),
        }
        return final_score, trace

    def _normalize_score(self, source: str, method: str, raw_score: float) -> float:
        stats = self._get_stats(source, method)
        count = int(stats.get("count", 0))
        if count < 20:
            return raw_score
        min_s = float(stats.get("min", raw_score))
        max_s = float(stats.get("max", raw_score))
        if max_s - min_s < 1e-6:
            return raw_score
        return _clamp01((raw_score - min_s) / (max_s - min_s))

    def _compute_recency_boost(self, timestamp: str | None) -> float:
        ts = _parse_timestamp(timestamp)
        if not ts:
            return 1.0
        now = datetime.now(UTC)
        age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
        # Gentle boost: [1.0, 1.05], decays slowly.
        return 1.0 + 0.05 * math.exp(-age_days / self._recency_half_life_days)

    def _get_reliability_prior(self, source: str) -> float:
        env_prior = self._reliability_priors.get(source)
        if env_prior is not None:
            return max(0.9, min(1.1, env_prior))
        source_state = self._state.get("sources", {}).get(source, {})
        prior = source_state.get("reliability_prior", 1.0)
        try:
            return max(0.9, min(1.1, float(prior)))
        except (TypeError, ValueError):
            return 1.0

    def _load_state(self) -> dict[str, Any]:
        if not self._artifact_path.exists():
            return {"version": 1, "sources": {}}
        try:
            data = json.loads(self._artifact_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load calibration artifact: %s", e)
            return {"version": 1, "sources": {}}
        if not isinstance(data, dict):
            return {"version": 1, "sources": {}}
        data.setdefault("version", 1)
        data.setdefault("sources", {})
        return data

    def _save_state(self) -> None:
        self._state["updated_at"] = datetime.now(UTC).isoformat()
        self._artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifact_path.write_text(
            json.dumps(self._state, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _get_stats(self, source: str, method: str) -> dict[str, Any]:
        source_state = self._state.setdefault("sources", {}).setdefault(source, {})
        methods = source_state.setdefault("methods", {})
        stats = methods.setdefault(
            method,
            {
                "scores": [],
                "count": 0,
                "min": 0.0,
                "max": 1.0,
                "mean": 0.0,
                "std": 0.0,
                "drift_events": 0,
            },
        )
        return stats

    def _record_observations(
        self,
        method_groups: dict[tuple[str, str], list[float]],
        drift_flags: dict[tuple[str, str], bool],
    ) -> None:
        if not method_groups:
            return
        for (source, method), values in method_groups.items():
            stats = self._get_stats(source, method)
            existing = [
                float(x) for x in stats.get("scores", []) if isinstance(x, int | float)
            ]
            merged = existing + [_clamp01(v) for v in values]
            if len(merged) > self._window_size:
                merged = merged[-self._window_size :]

            count = len(merged)
            mean = sum(merged) / count if count else 0.0
            variance = sum((x - mean) ** 2 for x in merged) / count if count else 0.0
            std = math.sqrt(variance)

            stats["scores"] = merged
            stats["count"] = count
            stats["min"] = min(merged) if merged else 0.0
            stats["max"] = max(merged) if merged else 1.0
            stats["mean"] = mean
            stats["std"] = std
            if drift_flags.get((source, method), False):
                stats["drift_events"] = int(stats.get("drift_events", 0)) + 1

        self._dirty_count += 1
        if self._dirty_count >= 10:
            try:
                self._save_state()
            except Exception as e:
                logger.warning("Failed to persist calibration artifact: %s", e)
            self._dirty_count = 0
