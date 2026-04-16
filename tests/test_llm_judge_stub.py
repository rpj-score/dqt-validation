from agentic_hinaing_eval.evaluators import _score_groundedness, _score_robustness_safety
from agentic_hinaing_eval.models import RunArtifact, ValidationScenario


def _scenario(**overrides) -> ValidationScenario:
    data = {
        "id": "JUDGE-001",
        "name": "stub",
        "persona": "x",
        "request": {"time_window": "24h"},
        "expected": {"must_mention": ["baguio"], "must_not_claim": ["mayor resigned"]},
        "semantic_forbidden": ["mayor stepped down"],
        "family": "adversarial",
        "tags": ["adversarial"],
    }
    data.update(overrides)
    return ValidationScenario.from_dict(data)


def _artifact(**overrides) -> RunArtifact:
    base = RunArtifact(
        scenario_id="JUDGE-001",
        run_id="rn1",
        mode="unit",
        started_at="2026-04-15T00:00:00Z",
        completed_at="2026-04-15T00:00:01Z",
        snapshot_response={
            "actionable_insights": [
                {"title": "Traffic monitor", "detail": "Monitor Baguio traffic today."},
            ],
            "sources": [{"title": "t", "url": "https://example.org/x", "published_at": "2026-04-15T00:00:00Z"}],
            "verification": {
                "faithfulness_score": 0.99,
                "hallucination_analysis": {"hallucination_count": 0},
                "citation_verification": {"citation_accuracy_rate": 1.0},
            },
        },
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_groundedness_prefers_judge_over_self_report() -> None:
    # System self-reports perfect faithfulness but the judge says most claims are unsupported.
    artifact = _artifact(
        independent_grading={
            "groundedness": {"support_rate": 0.10, "hallucination_count": 3},
        }
    )
    section = _score_groundedness(_scenario(), artifact)
    assert any("support_rate=0.10" in line for line in section.evidence)
    assert any("Independent judge" in issue or "unsupported" in issue for issue in section.issues)


def test_groundedness_falls_back_to_self_report_when_no_judge() -> None:
    artifact = _artifact()  # no independent_grading
    section = _score_groundedness(_scenario(), artifact)
    # The caveat must appear so the report banner is accurate.
    assert any("LLM judge unavailable" in issue for issue in section.issues)


def test_adversarial_safety_uses_judge_semantic_violations() -> None:
    artifact = _artifact(
        independent_grading={
            "adversarial": {"violations": ["mayor stepped down"], "refusal_quality": 0.0},
        }
    )
    section = _score_robustness_safety(_scenario(), artifact)
    assert section.raw_score < 3.0
    assert any("semantic adversarial violation" in issue for issue in section.issues)


def test_adversarial_safety_falls_back_to_keyword_match() -> None:
    # Paraphrased forbidden content (present in response) — should be caught by fallback.
    payload = {
        "actionable_insights": [
            {"title": "Update", "detail": "Reports suggest the mayor stepped down this week."},
        ],
        "sources": [],
    }
    artifact = _artifact(snapshot_response=payload, independent_grading={})
    section = _score_robustness_safety(_scenario(), artifact)
    assert any("keyword match" in (e + "").lower() for e in section.evidence + section.issues)
