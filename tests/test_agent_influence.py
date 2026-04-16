from agentic_hinaing_eval.agent_influence import (
    STAGE_OUTPUT_KEYS,
    InfluenceMetrics,
    _jaccard_distance,
    _levenshtein,
    compute_metrics,
    patch_drop,
    patch_drop_for_stage,
    patch_shuffle,
    patch_shuffle_for_stage,
    rank_nodes,
    stability_1_sfd,
)
from agentic_hinaing_eval.models import RunArtifact


def _artifact(text: str, nodes: list[str]) -> RunArtifact:
    return RunArtifact(
        scenario_id="CF-001",
        run_id="cf-1",
        mode="counterfactual",
        started_at="2026-04-15T00:00:00Z",
        snapshot_response={"summary": text},
        trajectory=list(nodes),
        node_order_observed=list(nodes),
        tool_events=[{"stage": n, "output": f"output-{n}"} for n in nodes],
    )


def test_jaccard_distance_basic() -> None:
    assert _jaccard_distance("baguio traffic", "baguio traffic") == 0.0
    assert 0.0 < _jaccard_distance("baguio traffic", "manila traffic") < 1.0
    assert _jaccard_distance("foo", "bar") == 1.0


def test_levenshtein_ordering() -> None:
    assert _levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert _levenshtein(["a", "b", "c"], ["a", "c", "b"]) == 2
    assert _levenshtein([], ["a", "b"]) == 2


def test_compute_metrics_detects_downstream_drift() -> None:
    baseline = _artifact("baguio traffic monitor today", ["query_orchestrator", "retrieval", "snapshot"])
    patched = _artifact("completely unrelated cebu beach vendors", ["query_orchestrator", "retrieval", "snapshot"])
    metrics = compute_metrics(baseline, patched, "retrieval", total_nodes=3, node_index=1)
    assert metrics.foc > 0.3  # large semantic change
    assert metrics.oc > 0.0


def test_rank_nodes_orders_by_oc() -> None:
    baseline = _artifact("alpha beta gamma delta", ["a", "b", "c"])
    patched_a = _artifact("zzz zzz zzz zzz", ["a", "b", "c"])
    patched_b = _artifact("alpha beta gamma delta", ["a", "b", "c"])  # no-op patch
    ranking = rank_nodes(
        baseline,
        {
            "a": [patched_a],
            "b": [patched_b],
        },
        ["a", "b", "c"],
    )
    # Node 'a' had a destructive patch — it should rank above node 'b' whose patch was a no-op.
    node_order = [entry["node"] for entry in ranking]
    assert node_order.index("a") < node_order.index("b")


def test_patch_strategies_are_pure() -> None:
    assert patch_drop([1, 2, 3]) == []
    assert patch_drop("hello") == ""
    assert patch_shuffle([1, 2, 3]) == [3, 2, 1]


def test_stability_1_sfd_identical_rankings() -> None:
    ranking = [{"node": "a", "oc": 0.5}, {"node": "b", "oc": 0.3}, {"node": "c", "oc": 0.1}]
    assert stability_1_sfd(ranking, ranking) == 1.0


def test_patch_drop_for_stage_clears_output_keys() -> None:
    state = {
        "documents": [{"title": "a"}, {"title": "b"}],
        "external_documents": [{"title": "c"}],
        "enriched": [{"x": 1}],
        "unrelated": "keep this",
    }
    patch_drop_for_stage(state, "retrieval")
    assert state["documents"] == []
    assert state["external_documents"] == []
    assert state["enriched"] == [{"x": 1}]
    assert state["unrelated"] == "keep this"


def test_patch_shuffle_for_stage_reverses_lists() -> None:
    state = {
        "enriched": [1, 2, 3],
        "credibility_notes": {"a": "b"},
        "theme_documents": [4, 5],
    }
    patch_shuffle_for_stage(state, "analyze")
    assert state["enriched"] == [3, 2, 1]
    assert state["credibility_notes"] == {"a": "b"}
    assert state["theme_documents"] == [5, 4]


def test_stage_output_keys_covers_all_hinaing_stages() -> None:
    from agentic_hinaing_eval.adapter import HINAING_STAGES
    for stage in HINAING_STAGES:
        assert stage in STAGE_OUTPUT_KEYS, f"STAGE_OUTPUT_KEYS missing stage {stage!r}"


def test_stability_1_sfd_reversed_rankings() -> None:
    ranking_a = [{"node": "a", "oc": 0.5}, {"node": "b", "oc": 0.3}, {"node": "c", "oc": 0.1}]
    ranking_b = list(reversed(ranking_a))
    identical = stability_1_sfd(ranking_a, ranking_a)
    reversed_stability = stability_1_sfd(ranking_a, ranking_b)
    assert reversed_stability < identical
    assert reversed_stability <= 0.25
