"""CAIR-style counterfactual Agent Influence Ranking.

Implements the core metric math from "Counterfactual-based Agent Influence
Ranker" (Fujitsu Research, arXiv:2510.25612):

- FOC (Final Output Change): cosine distance between baseline and patched
  workflow output embeddings.
- AOC (Agent Output Change): cosine distance between baseline and patched
  target-node outputs.
- WC (Workflow Change): Levenshtein distance between baseline and patched
  node-order sequences, normalized by sequence length.
- AF (Amplification Factor): normalizes impact by the fraction of nodes
  remaining downstream of the target.
- OC (Overall Change): convex combination of the above.

This module does NOT invoke Hinaing itself. The adapter layer is responsible
for producing pairs of (baseline, patched) artifacts; this module computes the
rankings from those artifacts.

Optional dependency: sentence-transformers. When unavailable, falls back to a
deterministic token-overlap distance (Jaccard) so tests can run offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .models import JsonDict, RunArtifact


def _token_set(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}


def _jaccard_distance(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta and not tb:
        return 0.0
    intersection = len(ta & tb)
    union = len(ta | tb)
    return 0.0 if union == 0 else 1.0 - (intersection / union)


# Default OC weights match CAIR's binary split (arXiv:2510.25612, lines 492-528):
#   0.6 * output_metric + 0.4 * workflow_metric
# Our four-component breakdown maps to: FOC is the primary output metric,
# AOC is a secondary output metric, WC is the workflow metric, and AF
# amplifies FOC for early-graph nodes.
DEFAULT_OC_WEIGHTS = (0.4, 0.2, 0.3, 0.1)
CAIR_PAPER_OC_WEIGHTS = (0.6, 0.0, 0.4, 0.0)

# SBERT model for cosine distance. CAIR uses all-mpnet-base-v2 (768d);
# fall back to all-MiniLM-L6-v2 (384d) if the heavier model is unavailable.
DEFAULT_SBERT_MODEL = "all-mpnet-base-v2"


class _Embedder:
    """Lazy SBERT embedder with a Jaccard fallback."""

    def __init__(self, model_name: str = DEFAULT_SBERT_MODEL) -> None:
        self._model: Any = None
        self._tried_load = False
        self._model_name = model_name

    def _ensure(self) -> None:
        if self._tried_load:
            return
        self._tried_load = True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self._model_name)
        except Exception:
            self._model = None

    def cosine_distance(self, a: str, b: str) -> float:
        self._ensure()
        if self._model is None:
            return _jaccard_distance(a, b)
        try:
            import numpy as np  # type: ignore
            vecs = self._model.encode([a, b], normalize_embeddings=True)
            sim = float(np.dot(vecs[0], vecs[1]))
            return max(0.0, min(2.0, 1.0 - sim))
        except Exception:
            return _jaccard_distance(a, b)


_embedder = _Embedder()


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, aa in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, bb in enumerate(b, 1):
            cost = 0 if aa == bb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


@dataclass(slots=True)
class InfluenceMetrics:
    node: str
    foc: float
    aoc: float
    wc: float
    af: float
    oc: float

    def to_dict(self) -> JsonDict:
        return {
            "node": self.node,
            "foc": round(self.foc, 4),
            "aoc": round(self.aoc, 4),
            "wc": round(self.wc, 4),
            "af": round(self.af, 4),
            "oc": round(self.oc, 4),
        }


def _response_text(artifact: RunArtifact) -> str:
    payload = artifact.snapshot_response or artifact.chat_response or {}
    return _flatten(payload)


def _flatten(payload: Any) -> str:
    if isinstance(payload, dict):
        return " ".join(_flatten(v) for v in payload.values())
    if isinstance(payload, list):
        return " ".join(_flatten(v) for v in payload)
    return "" if payload is None else str(payload)


def _node_output_text(artifact: RunArtifact, node: str) -> str:
    for event in artifact.tool_events:
        if event.get("stage") == node:
            return str(event.get("output") or event.get("message") or "")
    for event in artifact.progress_events:
        if event.get("stage") == node:
            return str(event.get("message") or event.get("detail") or "")
    return ""


def compute_metrics(
    baseline: RunArtifact,
    patched: RunArtifact,
    target_node: str,
    total_nodes: int,
    node_index: int,
    weights: tuple[float, float, float, float] = DEFAULT_OC_WEIGHTS,
    embedder: _Embedder | None = None,
) -> InfluenceMetrics:
    """Compute FOC/AOC/WC/AF/OC for one (baseline, patched) pair.

    ``node_index`` is the 0-based position of the target node in the baseline
    trajectory. Used to compute the amplification factor — perturbing an early
    node has more downstream propagation opportunity than perturbing a late one.
    """
    emb = embedder or _embedder
    foc = emb.cosine_distance(_response_text(baseline), _response_text(patched))
    aoc = emb.cosine_distance(
        _node_output_text(baseline, target_node),
        _node_output_text(patched, target_node),
    )
    baseline_seq = baseline.node_order_observed or baseline.trajectory
    patched_seq = patched.node_order_observed or patched.trajectory
    max_len = max(len(baseline_seq), len(patched_seq), 1)
    wc = _levenshtein(baseline_seq, patched_seq) / max_len
    remaining = max(total_nodes - node_index - 1, 0)
    af = remaining / max(total_nodes - 1, 1) if total_nodes > 1 else 0.0
    w_foc, w_aoc, w_wc, w_af = weights
    oc = (w_foc * foc) + (w_aoc * aoc) + (w_wc * wc) + (w_af * af * foc)
    return InfluenceMetrics(node=target_node, foc=foc, aoc=aoc, wc=wc, af=af, oc=oc)


def rank_nodes(
    baseline: RunArtifact,
    patched_by_node: dict[str, list[RunArtifact]],
    node_sequence: Sequence[str],
) -> list[JsonDict]:
    """Produce a ranked list of nodes by Overall Change.

    ``patched_by_node`` maps each node name to a list of counterfactual artifacts
    produced by patching that node (N patches per node; averaged).
    """
    total = len(node_sequence)
    aggregates: list[JsonDict] = []
    for idx, node in enumerate(node_sequence):
        variants = patched_by_node.get(node) or []
        if not variants:
            continue
        per_variant = [
            compute_metrics(baseline, patched, node, total, idx)
            for patched in variants
        ]
        avg = InfluenceMetrics(
            node=node,
            foc=_mean(m.foc for m in per_variant),
            aoc=_mean(m.aoc for m in per_variant),
            wc=_mean(m.wc for m in per_variant),
            af=_mean(m.af for m in per_variant),
            oc=_mean(m.oc for m in per_variant),
        )
        aggregates.append(avg.to_dict())
    aggregates.sort(key=lambda e: e["oc"], reverse=True)
    return aggregates


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def stability_1_sfd(ranking_a: Sequence[JsonDict], ranking_b: Sequence[JsonDict]) -> float:
    """1 minus normalized Spearman's Footrule Distance over the shared node set.

    Returns 1.0 when the two rankings are identical, 0.0 when they are
    maximally shuffled. Used as a determinism / replay-stability signal
    per arXiv:2510.25612.
    """
    nodes = [e["node"] for e in ranking_a]
    index_b = {e["node"]: idx for idx, e in enumerate(ranking_b)}
    if not nodes or len(nodes) < 2:
        return 1.0
    total = 0
    norm = 0
    for i, node in enumerate(nodes):
        j = index_b.get(node, len(nodes) - 1)
        total += abs(i - j)
        norm += max(i, len(nodes) - 1 - i)
    return 1.0 - (total / norm) if norm else 1.0


# --- Stage-aware patch strategies -------------------------------------------
# Maps each Hinaing stage to the state keys it writes (from graph.py:200-319).
# Used by patch strategies to target the correct output fields per node.

STAGE_OUTPUT_KEYS: dict[str, list[str]] = {
    "query_orchestrator": ["retrieval_plan"],
    "retrieval":          ["external_documents", "documents"],
    "recall":             ["internal_documents", "documents", "rag_relevance_scores"],
    "analyze":            ["enriched", "credibility_notes", "theme_documents"],
    "memory":             ["memory_stored"],
    "themes":             ["themed_insights"],
    "snapshot":           ["snapshot"],
}

PatchFn = Callable[[Any], Any]
StatePatchFn = Callable[[dict, str], None]


def patch_drop_for_stage(state: dict, stage: str) -> None:
    """Zero out the state keys written by ``stage`` (CAIR 'drop' strategy)."""
    for key in STAGE_OUTPUT_KEYS.get(stage, []):
        val = state.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            state[key] = []
        elif isinstance(val, dict):
            state[key] = {}
        elif isinstance(val, str):
            state[key] = ""
        else:
            state[key] = None


def patch_shuffle_for_stage(state: dict, stage: str) -> None:
    """Reverse list-typed output keys of ``stage`` (CAIR 'shuffle' strategy)."""
    for key in STAGE_OUTPUT_KEYS.get(stage, []):
        val = state.get(key)
        if isinstance(val, list):
            state[key] = list(reversed(val))


# Legacy generic patch fns (still used by rank_nodes tests)
def patch_drop(output: Any) -> Any:
    if isinstance(output, list):
        return []
    if isinstance(output, dict):
        return {}
    if isinstance(output, str):
        return ""
    return None


def patch_shuffle(output: Any) -> Any:
    if isinstance(output, list):
        return list(reversed(output))
    return output


PATCH_STRATEGIES: dict[str, StatePatchFn] = {
    "drop": patch_drop_for_stage,
    "shuffle": patch_shuffle_for_stage,
}
