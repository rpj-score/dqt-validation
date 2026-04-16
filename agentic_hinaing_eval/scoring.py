from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean

from .agent_influence import stability_1_sfd
from .evaluators import FAILURE_MODES, score_scenario
from .models import JsonDict, RunArtifact, ScenarioScore, SectionScore, ValidationScenario, ValidationScorecard
from .research_basis import RESEARCH_BASIS, SECTION_WEIGHTS


def readiness_label(score: float) -> str:
    if score >= 90:
        return "Operationally Ready"
    if score >= 80:
        return "Ready With Minor Revisions"
    if score >= 70:
        return "Conditionally Ready"
    if score >= 60:
        return "Prototype Needs Revision"
    return "Not Yet Operationally Ready"


def _bootstrap_ci(
    totals: list[float], iterations: int = 1000, alpha: float = 0.05, seed: int = 1337
) -> JsonDict:
    if len(totals) < 2:
        return {
            "iterations": 0,
            "mean": totals[0] if totals else 0.0,
            "low": totals[0] if totals else 0.0,
            "high": totals[0] if totals else 0.0,
            "note": "Not enough scenario runs for a bootstrap CI.",
        }
    rng = random.Random(seed)
    means: list[float] = []
    n = len(totals)
    for _ in range(iterations):
        sample = [totals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_idx = int((alpha / 2) * iterations)
    high_idx = int((1 - alpha / 2) * iterations) - 1
    return {
        "iterations": iterations,
        "mean": round(sum(means) / iterations, 2),
        "low": round(means[max(low_idx, 0)], 2),
        "high": round(means[min(high_idx, iterations - 1)], 2),
        "alpha": alpha,
    }


def _family_breakdown(scenario_scores: list[ScenarioScore]) -> list[JsonDict]:
    by_family: dict[str | None, list[ScenarioScore]] = defaultdict(list)
    for score in scenario_scores:
        by_family[score.family].append(score)
    rows: list[JsonDict] = []
    for family, scores in sorted(by_family.items(), key=lambda kv: (kv[0] or "")):
        if not scores:
            continue
        totals = [s.total_score for s in scores]
        passed = sum(1 for s in scores if s.passed)
        rows.append(
            {
                "family": family or "unspecified",
                "n": len(scores),
                "pass_rate": round(passed / len(scores), 2),
                "mean_score": round(mean(totals), 2),
                "min_score": round(min(totals), 2),
                "max_score": round(max(totals), 2),
            }
        )
    return rows


def _failure_mode_counts(scenario_scores: list[ScenarioScore]) -> JsonDict:
    counts: dict[str, int] = {mode: 0 for mode in FAILURE_MODES}
    for score in scenario_scores:
        for mode in score.failure_modes:
            counts[mode] = counts.get(mode, 0) + 1
    return counts


def _ablation_deltas(
    scenarios: list[ValidationScenario], scenario_scores: list[ScenarioScore]
) -> list[JsonDict]:
    by_id = {s.scenario_id: s for s in scenario_scores}
    pairs: dict[tuple[str, str], dict[str, ScenarioScore | None]] = {}
    for scenario in scenarios:
        if not scenario.ablation_pair:
            continue
        full_id = scenario.ablation_pair.get("full_id")
        ablated_id = scenario.ablation_pair.get("ablated_id")
        if not full_id or not ablated_id:
            continue
        key = (full_id, ablated_id)
        pairs.setdefault(key, {"full": None, "ablated": None})
        if scenario.id == full_id:
            pairs[key]["full"] = by_id.get(full_id)
        elif scenario.id == ablated_id:
            pairs[key]["ablated"] = by_id.get(ablated_id)
    deltas: list[JsonDict] = []
    for (full_id, ablated_id), entry in sorted(pairs.items()):
        full = entry.get("full")
        ablated = entry.get("ablated")
        if not full or not ablated:
            continue
        deltas.append(
            {
                "full_id": full_id,
                "ablated_id": ablated_id,
                "full_score": full.total_score,
                "ablated_score": ablated.total_score,
                "delta": round(full.total_score - ablated.total_score, 2),
                "full_passed": full.passed,
                "ablated_passed": ablated.passed,
            }
        )
    return deltas


def _aggregate_influence(runs: list[RunArtifact]) -> list[JsonDict]:
    by_node: dict[str, list[float]] = defaultdict(list)
    rankings: list[list[JsonDict]] = []
    for run in runs:
        if not run.influence_ranking:
            continue
        rankings.append(run.influence_ranking)
        for entry in run.influence_ranking:
            node = entry.get("node")
            oc = entry.get("oc")
            if node is None or oc is None:
                continue
            by_node[node].append(float(oc))
    rows: list[JsonDict] = []
    for node, values in sorted(by_node.items(), key=lambda kv: -sum(kv[1]) / max(len(kv[1]), 1)):
        rows.append(
            {
                "node": node,
                "n": len(values),
                "mean_oc": round(sum(values) / len(values), 4),
                "max_oc": round(max(values), 4),
            }
        )
    stability = None
    if len(rankings) >= 2:
        pairs = [stability_1_sfd(rankings[i], rankings[i + 1]) for i in range(len(rankings) - 1)]
        stability = round(sum(pairs) / len(pairs), 3)
    if stability is not None and rows:
        rows[0]["stability_1_sfd_pairwise_mean"] = stability
    return rows


def aggregate_scores(
    scenarios: list[ValidationScenario],
    runs: list[RunArtifact],
    preflight: dict | None = None,
) -> ValidationScorecard:
    scenario_by_id = {scenario.id: scenario for scenario in scenarios}
    scenario_scores: list[ScenarioScore] = []
    for run in runs:
        scenario = scenario_by_id.get(run.scenario_id)
        if not scenario:
            continue
        scenario_scores.append(score_scenario(scenario, run))

    section_rows: dict[str, list[SectionScore]] = defaultdict(list)
    for score in scenario_scores:
        for section in score.sections:
            if section.applicable:
                section_rows[section.id].append(section)

    section_averages: list[SectionScore] = []
    for section_id, (label, weight) in SECTION_WEIGHTS.items():
        rows = section_rows.get(section_id, [])
        if rows:
            raw = sum(row.raw_score for row in rows) / len(rows)
            weighted = sum(row.weighted_score for row in rows) / len(rows)
            evidence = [f"Averaged over {len(rows)} scenario run(s)."]
            issues = sorted({issue for row in rows for issue in row.issues})[:8]
        else:
            raw = 0.0
            weighted = 0.0
            evidence = []
            issues = ["No applicable scenario runs."]
        section_averages.append(
            SectionScore(
                id=section_id,
                label=label,
                weight=weight,
                raw_score=round(raw, 2),
                weighted_score=round(weighted, 2),
                evidence=evidence,
                issues=issues,
            )
        )

    total = round(sum(section.weighted_score for section in section_averages), 2)
    assumptions = [
        "Frozen fixture scores are the official reproducible scores; live HTTP scores are operational smoke evidence.",
        "The expert validation form must be reviewed and signed by a qualified human expert.",
        "If Hinaing backend dependencies are missing, import-adapter failures are counted as implementation-readiness failures.",
        "Groundedness reflects an independent Claude judge when available; otherwise it falls back to the system's self-reported verifier.",
        "Agent Attribution uses CAIR-style counterfactual patching; the section is marked not-applicable when counterfactual runs were not produced.",
    ]
    totals = [score.total_score for score in scenario_scores]
    bootstrap = _bootstrap_ci(totals) if totals else {}
    failure_modes = _failure_mode_counts(scenario_scores)
    return ValidationScorecard(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        total_score=total,
        readiness_label=readiness_label(total),
        scenario_scores=scenario_scores,
        section_averages=section_averages,
        preflight=preflight or {},
        research_basis=RESEARCH_BASIS,
        assumptions=assumptions,
        bootstrap_ci=bootstrap,
        family_breakdown=_family_breakdown(scenario_scores),
        failure_modes=failure_modes,
        observed_issue_categories=dict(failure_modes),
        agent_influence=_aggregate_influence(runs),
        ablation_deltas=_ablation_deltas(scenarios, scenario_scores),
    )
