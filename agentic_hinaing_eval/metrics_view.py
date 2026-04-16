from __future__ import annotations

from typing import Any

from .models import JsonDict


class MetricsView:
    """Typed read-only view over Hinaing's PipelineMetrics dict.

    Wraps `get_metrics_collector()._completed_runs[-1].to_dict()` so evaluator
    code reads like the upstream schema (backend/app/services/metrics/collector.py)
    rather than sprinkling `.get("x", 0.0) or 0.0` everywhere.
    """

    __slots__ = ("_data",)

    def __init__(self, data: JsonDict | None):
        self._data = data or {}

    def f(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self._data.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    def i(self, key: str, default: int = 0) -> int:
        try:
            return int(self._data.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def per_node_latencies_ms(self) -> dict[str, float]:
        keys = (
            "query_orchestrator_ms",
            "external_retrieval_ms",
            "internal_retrieval_ms",
            "sentiment_analysis_ms",
            "credibility_analysis_ms",
            "theme_routing_ms",
            "memory_consolidation_ms",
            "theme_agents_ms",
            "coordinator_ms",
        )
        return {key: self.f(key) for key in keys if key in self._data}

    def ablation_config(self) -> JsonDict:
        cfg = self._data.get("ablation_config")
        return dict(cfg) if isinstance(cfg, dict) else {}

    def theme_distribution(self) -> dict[str, int]:
        td = self._data.get("theme_distribution")
        return dict(td) if isinstance(td, dict) else {}

    def hallucination_types(self) -> dict[str, int]:
        ht = self._data.get("hallucination_types")
        return dict(ht) if isinstance(ht, dict) else {}

    def query_strategy(self) -> str:
        return str(self._data.get("query_strategy") or "")

    def is_empty(self) -> bool:
        return not self._data
