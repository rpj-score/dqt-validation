"""Unit tests for each _score_* evaluator function.

Tests boundary conditions: max-path scores stay ≤ 5.0, empty/None inputs,
error artifacts, missing_data deduplication, adversarial judge paths.
"""

from agentic_hinaing_eval.evaluators import (
    _score_agent_attribution,
    _score_efficiency_readiness,
    _score_groundedness,
    _score_memory_cache,
    _score_objective_quality,
    _score_robustness_safety,
    _score_temporal_hyperlocal,
    _score_trajectory,
    score_scenario,
)
from agentic_hinaing_eval.models import RunArtifact, ValidationScenario


def _s(**kw) -> ValidationScenario:
    base = {"id": "X", "name": "x", "persona": "x", "request": {"time_window": "24h"}}
    base.update(kw)
    return ValidationScenario.from_dict(base)


def _a(**kw) -> RunArtifact:
    base = RunArtifact(scenario_id="X", run_id="r1", mode="unit", started_at="2026-01-01T00:00:00Z")
    for k, v in kw.items():
        setattr(base, k, v)
    return base


# --- Clamping: no evaluator returns raw_score > 5.0 -----------------------

def test_memory_cache_max_path_capped() -> None:
    s = _s(tags=["cache"], family="cache")
    a = _a(metrics={
        "smart_reuse_rate": 1.0, "api_cost_reduction_rate": 1.0,
        "internal_docs_count": 5, "documents_cached": 3,
        "documents_fresh": 2, "vsee_api_calls_avoided": 10,
    })
    sec = _score_memory_cache(s, a)
    assert sec.raw_score <= 5.0, f"memory_cache raw={sec.raw_score}"


def test_groundedness_max_path_judge_capped() -> None:
    s = _s(frozen_documents=[{"url": "http://a"}])
    a = _a(
        snapshot_response={
            "sources": [{"url": "http://a", "title": "t"}],
            "verification": {
                "faithfulness_score": 1.0,
                "hallucination_analysis": {"hallucination_count": 0},
                "citation_verification": {"citation_accuracy_rate": 1.0},
                "misattribution_analysis": {"misattribution_count": 0},
                "numerical_hallucinations": {"count": 0},
                "claim_details": [{"verified": True}],
            },
        },
        independent_grading={
            "groundedness": {"support_rate": 1.0, "hallucination_count": 0},
            "source_provenance": {
                "trust_profile": {"n_sources": 1, "summary": {"flagged": 0}, "agent_reasoning_excerpts": []},
                "cross_check": {"fabricated_source_detected": False, "credibility_inflation": [],
                                "all_inputs_cited": True, "dropped_urls": []},
            },
            "source_quality": {"source_quality_score": 1.0, "fabricated_sources": [], "low_trust_citations": []},
        },
    )
    sec = _score_groundedness(s, a)
    assert sec.raw_score <= 5.0, f"groundedness raw={sec.raw_score}"


def test_groundedness_max_path_self_report_capped() -> None:
    s = _s(frozen_documents=[{"url": "http://a"}])
    a = _a(
        snapshot_response={
            "sources": [{"url": "http://a", "title": "t"}],
            "verification": {
                "faithfulness_score": 1.0,
                "hallucination_analysis": {"hallucination_count": 0},
                "citation_verification": {"citation_accuracy_rate": 1.0},
                "misattribution_analysis": {"misattribution_count": 0},
                "numerical_hallucinations": {"count": 0},
                "claim_details": [{"verified": True}],
            },
        },
        independent_grading={
            "source_provenance": {
                "trust_profile": {"n_sources": 1, "summary": {"flagged": 0}, "agent_reasoning_excerpts": []},
                "cross_check": {"fabricated_source_detected": False, "credibility_inflation": [],
                                "all_inputs_cited": True, "dropped_urls": []},
            },
            "source_quality": {"source_quality_score": 1.0, "fabricated_sources": [], "low_trust_citations": []},
        },
    )
    sec = _score_groundedness(s, a)
    assert sec.raw_score <= 5.0, f"groundedness self-report raw={sec.raw_score}"


def test_robustness_safety_max_path_capped() -> None:
    s = _s(tags=["adversarial"], family="adversarial",
           semantic_forbidden=["bad"], expected_refusal="refuse")
    a = _a(
        snapshot_response={"overall_sentiment": {"label": "x", "summary": "clean output", "scores": {}}},
        independent_grading={
            "adversarial": {"violations": [], "refusal_quality": 1.0},
            "source_provenance": {
                "trust_profile": {"n_sources": 1, "summary": {"has_credibility_breakdown": 2},
                                  "agent_reasoning_excerpts": [
                                      {"url": "x", "reasoning": "not official suspicious", "score": 0.2},
                                      {"url": "y", "reasoning": "unverified caution", "score": 0.3},
                                  ]},
                "cross_check": {},
            },
        },
    )
    sec = _score_robustness_safety(s, a)
    assert sec.raw_score <= 5.0, f"robustness raw={sec.raw_score}"


def test_objective_quality_max_path_capped() -> None:
    s = _s(expected={"must_mention": ["baguio", "traffic"], "must_not_claim": []})
    a = _a(snapshot_response={
        "overall_sentiment": {"label": "x", "summary": "baguio traffic monitor coordinate alert", "scores": {}},
        "actionable_insights": [{"title": "x", "detail": "monitor baguio traffic"}],
    }, metrics={"sentiment_agreement_rate": 0.9, "sentiment_positive": 3, "theme_distribution": {"x": 1}})
    sec = _score_objective_quality(s, a)
    assert sec.raw_score <= 5.0, f"objective_quality raw={sec.raw_score}"


def test_efficiency_readiness_max_path_capped() -> None:
    a = _a(metrics={
        "total_latency_ms": 5000, "errors": [], "fallbacks_used": [],
    })
    sec = _score_efficiency_readiness(_s(), a)
    assert sec.raw_score <= 5.0, f"efficiency raw={sec.raw_score}"


# --- Error paths: all evaluators handle errors gracefully ------------------

def test_all_sections_handle_errors_gracefully() -> None:
    s = _s()
    a = _a(errors=["ModuleNotFoundError: langgraph"])
    result = score_scenario(s, a)
    for sec in result.sections:
        assert 0.0 <= sec.raw_score <= 5.0, f"{sec.id} raw={sec.raw_score}"


# --- Empty/None inputs: no crashes ----------------------------------------

def test_all_sections_handle_none_response() -> None:
    s = _s()
    a = _a(snapshot_response=None, metrics={}, trajectory=[], independent_grading={})
    result = score_scenario(s, a)
    for sec in result.sections:
        assert 0.0 <= sec.raw_score <= 5.0, f"{sec.id} raw={sec.raw_score}"


# --- Missing_data deduplication: objective_quality defers to robustness -----

def test_missing_data_objective_quality_defers() -> None:
    s = _s(family="missing_data", expected={"must_mention": ["no data"]})
    a = _a(snapshot_response={
        "overall_sentiment": {"label": "x", "summary": "no data available", "scores": {}},
        "actionable_insights": [],
    })
    sec = _score_objective_quality(s, a)
    assert any("Missing-data scenario" in e for e in sec.evidence)
    # Score should NOT get the 1.25 must_mention bonus
    assert sec.raw_score < 4.0


# --- Trajectory: HTTP snapshot marked not applicable ----------------------

def test_trajectory_http_snapshot_not_applicable() -> None:
    a = _a(mode="http_snapshot", trajectory=[])
    sec = _score_trajectory(_s(), a)
    assert sec.applicable is False


# --- Trajectory: supplemented from metrics --------------------------------

def test_trajectory_uses_observed_nodes() -> None:
    s = _s(reference_trajectory=["query_orchestrator", "retrieval", "snapshot"])
    a = _a(
        node_order_observed=["query_orchestrator", "retrieval", "snapshot"],
        trajectory=["query_orchestrator", "retrieval", "snapshot"],
    )
    sec = _score_trajectory(s, a)
    assert sec.raw_score > 3.0


# --- Agent attribution: not applicable without data -----------------------

def test_agent_attribution_not_applicable_without_ranking() -> None:
    sec = _score_agent_attribution(_s(), _a())
    assert sec.applicable is False


# --- Provenance-related groundedness penalties -----------------------------

def test_groundedness_penalizes_fabricated_sources() -> None:
    s = _s(frozen_documents=[{"url": "http://legit"}])
    a = _a(
        snapshot_response={"sources": [{"url": "http://fake", "title": "t"}]},
        independent_grading={
            "source_provenance": {
                "trust_profile": {"n_sources": 1, "summary": {"flagged": 0}, "agent_reasoning_excerpts": []},
                "cross_check": {
                    "fabricated_source_detected": True,
                    "fabricated_urls": ["fake"],
                    "credibility_inflation": [],
                    "all_inputs_cited": False,
                    "dropped_urls": ["legit"],
                },
            },
        },
    )
    sec = _score_groundedness(s, a)
    assert any("Fabricated" in i for i in sec.issues)
