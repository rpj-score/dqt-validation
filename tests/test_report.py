from agentic_hinaing_eval.models import (
    RunArtifact,
    ScenarioScore,
    SectionScore,
    ValidationScenario,
    ValidationScorecard,
)
from agentic_hinaing_eval.report import render_validation_form


def test_blank_validation_form_contains_signature_and_total() -> None:
    text = render_validation_form()
    assert "Signature" in text
    assert "Total" in text
    assert "AgentDiagnose" in text


def test_form_no_longer_advertises_readiness_label_or_cv_appendix() -> None:
    text = render_validation_form()
    # Operational-readiness vocabulary must not appear in the user-facing form.
    for phrase in (
        "Operationally Ready",
        "Ready With Minor Revisions",
        "Conditionally Ready",
        "Prototype Needs Revision",
        "Not Yet Operationally Ready",
        "Final readiness label",
    ):
        assert phrase not in text, f"form still contains removed phrase: {phrase!r}"
    assert "CV Appendix" not in text
    assert "Expert CV Appendix" not in text
    assert "CV is provided as a separate attachment" in text


def test_form_contains_attestation_fields() -> None:
    text = render_validation_form()
    assert "Signature" in text
    assert "Expert name" in text
    assert "Date signed" in text


def _scenario(**overrides) -> ValidationScenario:
    data = {
        "id": "T-001",
        "name": "Test scenario",
        "persona": "tester",
        "request": {"time_window": "24h", "focus_areas": ["safety"]},
        "expected": {"must_mention": ["baguio"], "must_not_claim": ["road closed"]},
        "family": "hyperlocal",
        "reference_trajectory": ["query_orchestrator", "snapshot"],
    }
    data.update(overrides)
    return ValidationScenario.from_dict(data)


def _run() -> RunArtifact:
    return RunArtifact(
        scenario_id="T-001",
        run_id="rn1",
        mode="unit",
        started_at="2026-04-15T00:00:00Z",
        completed_at="2026-04-15T00:00:01Z",
        snapshot_response={
            "actionable_insights": [{"title": "x", "detail": "Baguio update."}],
            "sources": [{"title": "s", "url": "https://example.org/x", "published_at": "2026-04-15T00:00:00Z"}],
        },
        trajectory=["query_orchestrator", "snapshot"],
        node_order_observed=["query_orchestrator", "snapshot"],
    )


def _scorecard(scenarios, runs) -> ValidationScorecard:
    section = SectionScore(
        id="objective_quality",
        label="Objective Quality",
        weight=18.0,
        raw_score=4.0,
        weighted_score=14.4,
        evidence=["ok"],
        issues=["Minor groundedness gap"],
    )
    score = ScenarioScore(
        scenario_id="T-001",
        run_id="rn1",
        total_score=80.0,
        sections=[section],
        passed=True,
        critical_failures=[],
        family="hyperlocal",
        failure_modes=[],
    )
    return ValidationScorecard(
        generated_at="2026-04-15T00:00:00Z",
        total_score=80.0,
        readiness_label="hidden",
        scenario_scores=[score],
        section_averages=[section],
        research_basis=[],
        assumptions=[],
        bootstrap_ci={"iterations": 1000, "mean": 80.0, "low": 78.0, "high": 82.0, "alpha": 0.05},
        family_breakdown=[{"family": "hyperlocal", "n": 1, "pass_rate": 1.0, "mean_score": 80.0, "min_score": 80.0, "max_score": 80.0}],
        failure_modes={"hallucination": 0},
        observed_issue_categories={"hallucination": 0},
        agent_influence=[],
        ablation_deltas=[],
    )


def test_micro_view_has_per_scenario_structured_blocks() -> None:
    scenarios = [_scenario()]
    runs = [_run()]
    card = _scorecard(scenarios, runs)
    text = render_validation_form(card, scenarios=scenarios, runs=runs)

    # New structured headings must appear for every scenario
    assert "### T-001 — Test scenario" in text
    assert "**What was tested**" in text
    assert "**What happened**" in text
    assert "**Issues found**" in text
    assert "**Critical failures**" in text
    # "Scenario intent" banner for a non-adversarial family should not appear
    # by default, but if it does the text is informative (no assertion needed).
    # Issue severity tagging:
    assert "(low)" in text or "(medium)" in text or "(info)" in text or "(high)" in text or "(critical)" in text


def test_adversarial_scenario_receives_intent_note() -> None:
    adv = _scenario(
        id="ADV-XY",
        name="Adversarial test",
        family="adversarial",
        tags=["adversarial"],
        semantic_forbidden=["mayor stepped down"],
        expected_refusal="refuse politely",
    )
    card = _scorecard([adv], [_run()])
    # Rewrite the scenario score to match ADV-XY
    card.scenario_scores[0].scenario_id = "ADV-XY"
    card.scenario_scores[0].family = "adversarial"
    text = render_validation_form(card, scenarios=[adv], runs=[])
    assert "Scenario intent" in text
    assert "adversarial by design" in text
