from agentic_hinaing_eval.evaluators import score_scenario
from agentic_hinaing_eval.models import RunArtifact, ValidationScenario
from agentic_hinaing_eval.scoring import aggregate_scores


def _scenario() -> ValidationScenario:
    return ValidationScenario.from_dict(
        {
            "id": "T-001",
            "name": "Test",
            "persona": "city_official",
            "request": {"time_window": "24h", "focus_areas": ["infrastructure"]},
            "expected": {
                "must_mention": ["Baguio", "traffic"],
                "must_not_claim": ["casualties confirmed"],
            },
            "reference_trajectory": ["query_orchestrator", "retrieval", "snapshot"],
        }
    )


def _artifact() -> RunArtifact:
    artifact = RunArtifact(
        scenario_id="T-001",
        run_id="abc123",
        mode="unit",
        started_at="2026-04-15T00:00:00Z",
        completed_at="2026-04-15T00:00:01Z",
        snapshot_response={
            "overall_sentiment": {
                "label": "Mixed Sentiment",
                "summary": "Baguio traffic should be monitored and coordinated with traffic aides.",
                "scores": {"negative": 0.5, "neutral": 0.5, "positive": 0.0},
            },
            "actionable_insights": [
                {"category": "Infrastructure", "title": "Monitor traffic", "detail": "Coordinate traffic aides.", "evidence": ["https://example.org"]}
            ],
            "sources": [
                {"title": "Baguio traffic", "snippet": "traffic", "url": "https://example.org", "published_at": "2026-04-15T00:00:00+00:00"}
            ],
            "verification": {
                "faithfulness_score": 1.0,
                "hallucination_analysis": {"hallucination_count": 0},
                "citation_verification": {"citation_accuracy_rate": 1.0},
            },
        },
        metrics={"total_latency_ms": 1000, "smart_reuse_rate": 0.5, "api_cost_reduction_rate": 0.5, "documents_cached": 1, "documents_fresh": 1, "internal_docs_count": 1},
        trajectory=["query_orchestrator", "retrieval", "recall", "snapshot"],
        node_order_observed=["query_orchestrator", "retrieval", "recall", "snapshot"],
    )
    return artifact


def test_score_scenario_produces_100_point_total() -> None:
    score = score_scenario(_scenario(), _artifact())
    assert 0 <= score.total_score <= 100
    assert len(score.sections) == 8
    assert any(section.id == "trajectory" for section in score.sections)
    assert any(section.id == "agent_attribution" for section in score.sections)


def test_aggregate_scores_has_readiness_label() -> None:
    card = aggregate_scores([_scenario()], [_artifact()])
    assert 0 <= card.total_score <= 100
    assert card.readiness_label
    assert len(card.section_averages) == 8


def test_section_weights_sum_to_100() -> None:
    from agentic_hinaing_eval.research_basis import SECTION_WEIGHTS

    assert round(sum(weight for _, weight in SECTION_WEIGHTS.values()), 2) == 100.0

