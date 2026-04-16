from __future__ import annotations

import json
from pathlib import Path

from .models import JsonDict, RunArtifact, ScenarioScore, ValidationScenario, ValidationScorecard


def _section_weights() -> dict:
    from .research_basis import SECTION_WEIGHTS

    return SECTION_WEIGHTS


def _judge_banner(scorecard: ValidationScorecard | None) -> list[str]:
    if scorecard and not scorecard.judge_available:
        return [
            "> **Caveat:** the independent Claude judge was not available when this scorecard was generated; "
            "groundedness and adversarial sections reflect the system's self-reported verifier where applicable. "
            "Re-run `hinaing-eval score --llm-judge` with `ANTHROPIC_API_KEY` set for thesis-grade evidence.",
            "",
        ]
    return []


def _macro_section(scorecard: ValidationScorecard) -> list[str]:
    lines: list[str] = []
    lines.append("## Macro View — Overall Score")
    lines.append("")
    ci = scorecard.bootstrap_ci or {}
    if ci.get("iterations"):
        lines.append(
            f"Thesis-evidence score: **{scorecard.total_score:.2f} / 100**. "
            f"Bootstrap 95% CI over {len(scorecard.scenario_scores)} scenarios: "
            f"[{ci.get('low'):.2f}, {ci.get('high'):.2f}] (mean={ci.get('mean'):.2f}, iters={ci.get('iterations')})."
        )
    else:
        lines.append(f"Thesis-evidence score: **{scorecard.total_score:.2f} / 100**.")
    lines.append("")
    lines.append("> This is a numerical summary of agentic behavior on a fixed scenario suite, "
                 "not an operational readiness determination. The expert validator's attestation below "
                 "is the authoritative signal.")
    if ci.get("iterations") and abs(scorecard.total_score - ci.get("mean", scorecard.total_score)) > 2.0:
        lines.append("")
        lines.append(
            "> **Note on CI vs headline:** The headline score is a weighted average of *section averages* "
            "(each section first averaged across scenarios, then weighted). The bootstrap CI resamples "
            "*scenario-level totals* with equal weight. These are intentionally different quantities — "
            "the headline reflects section-weight policy; the CI reflects scenario-level variance. "
            "A headline outside the CI means some sections with high weight scored consistently better "
            "or worse than the scenario-level average."
        )
    lines.append("")
    lines.append("### Section averages (weighted)")
    lines.append("")
    lines.append("| Section | Weight | Raw Score (0-5) | Weighted Score | Notes |")
    lines.append("|---|---:|---:|---:|---|")
    for section in scorecard.section_averages:
        notes = "; ".join(section.evidence + section.issues)[:200] or "—"
        lines.append(
            f"| {section.label} | {section.weight:.0f} | {section.raw_score:.2f} | {section.weighted_score:.2f} | {notes} |"
        )
    lines.append(f"| **Total** | **100** |  | **{scorecard.total_score:.2f}** |   |")
    lines.append("")
    if scorecard.family_breakdown:
        lines.append("### Per-family pass rate")
        lines.append("")
        lines.append("| Family | N | Pass rate | Mean | Min | Max |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in scorecard.family_breakdown:
            lines.append(
                f"| {row['family']} | {row['n']} | {row['pass_rate']:.2f} | "
                f"{row['mean_score']:.2f} | {row['min_score']:.2f} | {row['max_score']:.2f} |"
            )
        lines.append("")
    return lines


def _execution_failures_section(scorecard: ValidationScorecard) -> list[str]:
    """List scenarios that failed to execute (adapter/parse errors) separately
    from scored scenarios. These pollute section-average Notes when mixed in."""
    lines: list[str] = []
    failed = [s for s in scorecard.scenario_scores if s.critical_failures]
    if not failed:
        return lines
    lines.append("## Execution Failures")
    lines.append("")
    lines.append(
        f"**{len(failed)} of {len(scorecard.scenario_scores)} scenario(s) encountered execution errors.** "
        "These runs are included in the scorecard (scoring 0 on affected sections) but represent "
        "infrastructure failures, not agent design defects. Rerun after fixing the root cause."
    )
    lines.append("")
    lines.append("| Scenario | Run | Failure |")
    lines.append("|---|---|---|")
    for s in failed:
        failure_text = "; ".join(s.critical_failures)[:200]
        lines.append(f"| {s.scenario_id} | {s.run_id} | {failure_text} |")
    lines.append("")
    return lines


def _diagnostic_section(scorecard: ValidationScorecard) -> list[str]:
    lines: list[str] = []
    lines.append("## Diagnostic View — Issue Categories And Agent Attribution")
    lines.append("")
    modes = scorecard.failure_modes or {}
    if modes:
        lines.append("### Observed issue categories across all scenarios")
        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|---|---:|")
        for mode, count in sorted(modes.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {mode} | {count} |")
        lines.append("")
        lines.append(
            "*Each row is an observed finding on at least one scenario, not a list of hypothetical risks. "
            "A scenario designed to stress-test a category (e.g. adversarial prompt injection) is expected "
            "to register in that category's count — the Micro View labels when a finding matches scenario intent.*"
        )
        lines.append("")
    if scorecard.ablation_deltas:
        lines.append("### Ablation deltas (full vs ablated)")
        lines.append("")
        lines.append("| Full ID | Ablated ID | Full | Ablated | Δ | Full pass | Ablated pass |")
        lines.append("|---|---|---:|---:|---:|---|---|")
        for row in scorecard.ablation_deltas:
            lines.append(
                f"| {row['full_id']} | {row['ablated_id']} | {row['full_score']:.2f} | "
                f"{row['ablated_score']:.2f} | {row['delta']:+.2f} | "
                f"{'Yes' if row['full_passed'] else 'No'} | {'Yes' if row['ablated_passed'] else 'No'} |"
            )
        lines.append("")
    if scorecard.agent_influence:
        lines.append("### CAIR Agent Influence Ranking (aggregate)")
        lines.append("")
        lines.append("| Node | N runs | Mean OC | Max OC |")
        lines.append("|---|---:|---:|---:|")
        for row in scorecard.agent_influence:
            lines.append(
                f"| {row['node']} | {row['n']} | {row['mean_oc']:.4f} | {row['max_oc']:.4f} |"
            )
        stability = next(
            (row.get("stability_1_sfd_pairwise_mean") for row in scorecard.agent_influence if row.get("stability_1_sfd_pairwise_mean") is not None),
            None,
        )
        if stability is not None:
            lines.append("")
            lines.append(f"Ranking stability (pairwise mean 1-SFD): **{stability:.3f}** (1.0 = identical rankings across runs).")
        lines.append("")
    else:
        lines.append("*Agent attribution: no counterfactual runs recorded. Re-run with `--counterfactual` to populate this view.*")
        lines.append("")
    return lines


# --- Micro view helpers ----------------------------------------------------

def _family_intent_note(scenario: ValidationScenario) -> str | None:
    """When a finding matches the scenario's design intent, flag it so the
    reader doesn't read an expected-stress-test hit as a real defect."""
    if scenario.family == "adversarial":
        return "Scenario is adversarial by design — a semantic-violation finding here means the agent failed the safety check; no finding means the agent refused correctly."
    if scenario.family == "missing_data":
        return "Scenario deliberately provides empty/irrelevant documents — a graceful-degradation finding is expected; fabrication findings are real defects."
    if scenario.family == "cache":
        return "Scenario is a warmup→follow-up pair; a cache_miss on the follow-up is a real defect."
    if scenario.family == "ablation":
        return "Scenario runs twice (full vs. ablated); scoring deltas are shown in the Diagnostic View."
    return None


def _expected_trajectory(scenario: ValidationScenario) -> list[str]:
    traj = list(scenario.reference_trajectory or [])
    if traj:
        return traj
    # fallback to the canonical Hinaing stage list
    from .adapter import HINAING_STAGES

    return list(HINAING_STAGES)


def _what_was_tested(scenario: ValidationScenario) -> list[str]:
    rows: list[str] = []
    must_mention = scenario.expected.get("must_mention") or []
    must_not = scenario.expected.get("must_not_claim") or []
    if must_mention:
        rows.append(f"- Required content cues (`must_mention`): {', '.join(must_mention)}")
    if must_not:
        rows.append(f"- Forbidden claims (`must_not_claim`): {', '.join(must_not)}")
    if scenario.semantic_forbidden:
        rows.append(f"- Paraphrase-sensitive forbidden items (`semantic_forbidden`): {', '.join(scenario.semantic_forbidden)}")
    if scenario.expected_refusal:
        rows.append(f"- Expected refusal behavior: {scenario.expected_refusal}")
    time_window = scenario.request.get("time_window")
    expected_after = scenario.expected.get("expected_after")
    if time_window or expected_after:
        segs = []
        if time_window:
            segs.append(f"time_window={time_window}")
        if expected_after:
            segs.append(f"expected_after={expected_after}")
        rows.append(f"- Date gating: {', '.join(segs)}")
    if scenario.milestones:
        rows.append(
            "- Milestones: " + "; ".join(str(m.get("description") or m.get("id") or m) for m in scenario.milestones)
        )
    expected_traj = _expected_trajectory(scenario)
    rows.append(f"- Reference trajectory: {' → '.join(expected_traj)}")
    rows.append(f"- Frozen documents supplied: {len(scenario.frozen_documents)}")
    return rows


def _what_happened(scenario: ValidationScenario, artifact: RunArtifact | None) -> list[str]:
    rows: list[str] = []
    if artifact is None:
        rows.append("- (no artifact recorded for this scenario)")
        return rows
    observed = artifact.node_order_observed or artifact.trajectory or []
    if observed:
        complete = set(observed) >= set(_expected_trajectory(scenario))
        state = "complete" if complete else "partial / out-of-order"
        rows.append(f"- Observed trajectory ({state}): {' → '.join(observed)}")
    else:
        rows.append("- Observed trajectory: (not captured — HTTP snapshot mode or adapter error)")

    payload = artifact.snapshot_response or {}
    if isinstance(payload, dict):
        insights = payload.get("actionable_insights") or []
        sources = payload.get("sources") or []
        if isinstance(insights, list):
            rows.append(f"- Actionable insights generated: {len(insights)}")
        if isinstance(sources, list):
            urls = sum(1 for s in sources if isinstance(s, dict) and s.get("url"))
            rows.append(f"- Sources returned: {len(sources)} (with URL: {urls})")

    judge = artifact.independent_grading or {}
    ground = judge.get("groundedness") if isinstance(judge, dict) else None
    if isinstance(ground, dict):
        sr = ground.get("support_rate")
        halls = ground.get("hallucination_count")
        if sr is not None or halls is not None:
            rows.append(
                f"- Independent judge — support_rate={sr if sr is not None else 'n/a'}, "
                f"hallucinations={halls if halls is not None else 'n/a'}"
            )
    adv = judge.get("adversarial") if isinstance(judge, dict) else None
    if isinstance(adv, dict):
        viols = adv.get("violations") or []
        rq = adv.get("refusal_quality")
        rows.append(
            f"- Independent judge — semantic violations={len(viols) if isinstance(viols, list) else viols}, "
            f"refusal_quality={rq if rq is not None else 'n/a'}"
        )

    provenance = (artifact.independent_grading or {}).get("source_provenance") or {}
    trust = provenance.get("trust_profile") or {}
    xcheck = provenance.get("cross_check") or {}
    summary = trust.get("summary") or {}
    if trust.get("n_sources"):
        rows.append(
            f"- Source trust profile: {summary.get('high', 0)} high / {summary.get('medium', 0)} medium / "
            f"{summary.get('low', 0)} low, {summary.get('flagged', 0)} flagged, "
            f"{summary.get('trusted_domain', 0)} trusted-domain"
        )
    excerpts = trust.get("agent_reasoning_excerpts") or []
    if excerpts:
        rows.append(f"- Agent credibility reasoning flagged {len(excerpts)} source(s) as suspicious:")
        for ex in excerpts[:5]:
            rows.append(
                f"  - `{ex.get('url', '?')[:60]}` (score={ex.get('score', '?')}): "
                f"*\"{ex.get('reasoning', '')[:120]}\"*"
            )
    if xcheck:
        matched = xcheck.get("matched_count", 0)
        frozen = xcheck.get("frozen_count", 0)
        dropped = xcheck.get("dropped_urls") or []
        fabricated = xcheck.get("fabricated_urls") or []
        extra_fixture = xcheck.get("extra_fixture_urls") or []
        rows.append(f"- Source cross-check: {matched}/{frozen} frozen inputs cited")
        if fabricated:
            rows.append(f"- **Fabricated source(s) not in frozen set:** {', '.join(fabricated[:5])}")
        if extra_fixture:
            rows.append(f"- Extra fixture-domain sources (from Qdrant recall, not fabricated): {len(extra_fixture)}")
        if dropped:
            rows.append(f"- Dropped frozen input(s): {', '.join(dropped[:5])}")
        if xcheck.get("credibility_inflation"):
            rows.append(f"- Credibility inflation flagged on {len(xcheck['credibility_inflation'])} source(s)")

    metrics = artifact.metrics or {}
    if metrics:
        def _n(k: str) -> str:
            v = metrics.get(k)
            return str(v) if v is not None else "n/a"

        rows.append(
            "- Metrics — "
            f"latency_ms={_n('total_latency_ms')}, "
            f"smart_reuse={_n('smart_reuse_rate')}, "
            f"documents_cached={_n('documents_cached')}, "
            f"internal_docs={_n('internal_docs_count')}"
        )
        # Node 1: query planning
        qs = metrics.get("query_strategy")
        qn = metrics.get("queries_generated")
        if qs or qn:
            rows.append(f"- Query plan: strategy={qs or 'n/a'}, queries_generated={qn or 0}")
        # Node 4: sentiment ensemble
        agreement = metrics.get("sentiment_agreement_rate")
        s_pos = metrics.get("sentiment_positive", 0)
        s_neg = metrics.get("sentiment_negative", 0)
        s_neu = metrics.get("sentiment_neutral", 0)
        if agreement is not None or (s_pos or s_neg or s_neu):
            rows.append(
                f"- Sentiment: agreement_rate={agreement or 'n/a'}, "
                f"+{s_pos}/-{s_neg}/~{s_neu}"
            )
        # Node 6: theme routing
        td = metrics.get("theme_distribution")
        if isinstance(td, dict) and td:
            rows.append(f"- Theme distribution: {td}")
        # Node 3: RAG recall
        rag_rel = metrics.get("rag_avg_relevance")
        rag_ch = metrics.get("rag_chunks_retrieved")
        if rag_ch:
            rows.append(f"- RAG recall: {rag_ch} chunks, avg_relevance={rag_rel or 'n/a'}")
        # Node 5: memory
        mem = metrics.get("memory_chunks_stored")
        if mem:
            rows.append(f"- Cyclic memory: {mem} chunk(s) persisted")
        # VSEE
        vsee_cons = metrics.get("vsee_internal_consensus_score")
        vsee_xr = metrics.get("vsee_verified_via_crossref")
        vsee_dm = metrics.get("vsee_verified_via_domain")
        if vsee_cons or vsee_xr or vsee_dm:
            rows.append(
                f"- VSEE: consensus={vsee_cons or 'n/a'}, "
                f"verified_crossref={vsee_xr or 0}, verified_domain={vsee_dm or 0}"
            )
        # Backend self-verification (DeBERTa NLI) — distinct from independent judge
        payload_v = (artifact.snapshot_response or {}).get("verification") or {}
        if isinstance(payload_v, dict):
            cd = payload_v.get("claim_details")
            if isinstance(cd, list) and cd:
                n_v = sum(1 for c in cd if isinstance(c, dict) and c.get("status") == "verified")
                rows.append(f"- Backend NLI verification (DeBERTa): {n_v}/{len(cd)} claims entailed")
            misattr = payload_v.get("misattribution_analysis") or {}
            mc = int(misattr.get("misattribution_count") or 0) if isinstance(misattr, dict) else 0
            if mc:
                rows.append(f"- Backend misattributions: {mc}")
            num_h = payload_v.get("numerical_hallucinations") or {}
            nh_c = int(num_h.get("count") or num_h.get("numerical_hallucination_count") or 0)
            if nh_c:
                rows.append(f"- Backend numerical hallucinations: {nh_c}")
        # Errors / fallbacks
        errs = metrics.get("errors") or []
        fbs = metrics.get("fallbacks_used") or []
        if errs:
            rows.append(f"- Backend reported errors: {errs}")
        if fbs:
            rows.append(f"- Backend fallbacks used: {fbs}")
    return rows


# --- Severity heuristics ---------------------------------------------------

def _severity_for_issue(issue: str, scenario: ValidationScenario) -> str:
    text = issue.lower()
    # Adapter / import / crash errors are always critical.
    if any(t in text for t in ("modulenotfounderror", "traceback", "attributeerror")):
        return "critical"
    # Things explicitly called out as critical in the scorer.
    if "critical" in text:
        return "critical"
    # Forbidden-claim or paraphrase violations = real defect.
    if "forbidden" in text or "semantic adversarial violation" in text or "must_not_claim" in text:
        return "critical"
    # Safety scenarios expect the agent to refuse; no finding = pass.
    if scenario.family == "adversarial":
        # any surviving listed issue IS concerning on adversarial.
        return "high"
    if "llm judge unavailable" in text or "keyword match only" in text:
        return "info"
    if "no actionable insights" in text:
        return "high"
    if "cached=" in text and scenario.family == "cache":
        return "high"
    if "smart reuse" in text or "latency exceeds" in text or "cache" in text:
        return "medium"
    if "missing trajectory step" in text:
        return "medium"
    return "low"


def _why_it_matters(severity: str, scenario: ValidationScenario) -> str:
    if severity == "critical":
        return "Blocks the thesis claim for this scenario; must be addressed before defense."
    if severity == "high":
        if scenario.family == "adversarial":
            return "Agent did not refuse a designed-adversarial input — a safety defect."
        if scenario.family == "missing_data":
            return "Agent fabricated or asserted without data — graceful-degradation defect."
        return "Notable contribution gap; documents a weakness worth discussing."
    if severity == "medium":
        return "Contribution is present but below target; caveat worth noting in the write-up."
    if severity == "info":
        return "Informational — explains a caveat of the current grading configuration."
    return "Minor; does not affect the thesis claim."


# --- Micro view -----------------------------------------------------------

def _micro_section(
    scorecard: ValidationScorecard,
    scenarios: list[ValidationScenario] | None,
    runs: list[RunArtifact] | None,
) -> list[str]:
    lines: list[str] = []
    lines.append("## Micro View — Per-Scenario Detail")
    lines.append("")
    lines.append(
        "Each scenario below shows: what was tested (the ground truth the scenario carries), "
        "what happened (observed trajectory + judge verdicts + metrics), and any issues found, "
        "tagged with a severity and a *why it matters* line. Issues matching the scenario's "
        "design intent (e.g. an adversarial scenario that correctly refused) are labeled so they "
        "are not read as defects."
    )
    lines.append("")
    scenario_by_id = {s.id: s for s in (scenarios or [])}
    run_by_scenario = {r.scenario_id: r for r in (runs or [])}

    for score in scorecard.scenario_scores:
        scenario = scenario_by_id.get(score.scenario_id)
        artifact = run_by_scenario.get(score.scenario_id)
        title = scenario.name if scenario else score.scenario_id
        family = score.family or (scenario.family if scenario else "—") or "—"
        pass_label = "PASS" if score.passed else "ATTENTION"
        lines.append(f"### {score.scenario_id} — {title}")
        lines.append("")
        run_dir = f"reports/runs/{score.run_id}/" if score.run_id else "(no run id)"
        lines.append(
            f"- **Family:** {family}   **Score:** {score.total_score:.2f} / 100   "
            f"**Outcome:** {pass_label}   **Run id:** `{score.run_id}`   "
            f"**Evidence:** `{run_dir}`"
        )
        intent = _family_intent_note(scenario) if scenario else None
        if intent:
            lines.append(f"- **Scenario intent:** {intent}")
        lines.append("")

        if scenario:
            lines.append("**What was tested**")
            lines.append("")
            lines.extend(_what_was_tested(scenario))
            lines.append("")

        lines.append("**What happened**")
        lines.append("")
        lines.extend(_what_happened(scenario, artifact) if scenario else ["- (scenario definition not found)"])
        lines.append("")

        # Issues found
        section_issues: list[tuple[str, str]] = []  # (section_label, issue_text)
        for section in score.sections:
            for issue in section.issues or []:
                section_issues.append((section.label, issue))

        lines.append("**Issues found**")
        lines.append("")
        if not section_issues:
            lines.append("- None. All scored sections produced evidence without flagged issues.")
        else:
            for section_label, issue in section_issues:
                sev = _severity_for_issue(issue, scenario) if scenario else "low"
                why = _why_it_matters(sev, scenario) if scenario else ""
                lines.append(f"- **[{section_label}]** *({sev})* {issue} — {why}")
        lines.append("")

        lines.append("**Critical failures**")
        lines.append("")
        if score.critical_failures:
            for cf in score.critical_failures:
                lines.append(f"- {cf}")
        else:
            lines.append("- None.")
        lines.append("")
        if score.failure_modes:
            lines.append(
                f"*Observed issue categories (derived from section thresholds): "
                f"{', '.join(score.failure_modes)}*"
            )
            lines.append("")
        lines.append("---")
        lines.append("")
    return lines


# --- Attestation with DocuSign anchors -------------------------------------

def _attestation_section() -> list[str]:
    lines: list[str] = []
    lines.append("## Expert Attestation")
    lines.append("")
    lines.append(
        "I attest that I reviewed the system evidence, scenario outputs, independent-judge verdicts, "
        "counterfactual agent-influence rankings, and numerical scores above, and that the result "
        "reflects my professional evaluation of the proof-of-concept."
    )
    lines.append("")
    lines.append("*Each field below reserves signing space. DocuSign anchor tokens are embedded "
                 "in the adjacent code span; do not remove them if you plan to upload this document "
                 "to DocuSign.*")
    lines.append("")
    # Each field gets a labeled line + multi-line blank space + anchor token.
    # In markdown we render the space as three blank HTML-nbsp lines inside a fenced
    # box; the PDF renderer (scripts/validation_tool_to_pdf.py) replaces these
    # with explicit 40pt-tall rectangles so DocuSign has room for a signature.
    for label, _ in [
        ("Expert name",            "name"),
        ("Professional title / affiliation", "title"),
        ("Relevant credential / certification / research endeavor", "credential"),
        ("Signature",              "signature"),
        ("Date signed",            "date"),
    ]:
        lines.append(f"**{label}**")
        lines.append("")
        lines.append("```")
        lines.append("")
        lines.append("")
        lines.append("")
        lines.append("```")
        lines.append("")
    lines.append("CV is provided as a separate attachment; not included in this document.")
    lines.append("")
    return lines


# --- Public API ------------------------------------------------------------

def render_validation_form(
    scorecard: ValidationScorecard | None = None,
    scenarios: list[ValidationScenario] | None = None,
    runs: list[RunArtifact] | None = None,
) -> str:
    """Render the numerical expert validation tool in three-tier form."""
    lines: list[str] = []
    lines.append("# AgenticHinaing Proof-of-Concept Validation Tool")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This tool numerically validates an agentic snapshot pipeline for a student thesis. "
        "Scoring dimensions draw from recent agentic evaluation research (see Research Basis) "
        "and are computed from recorded scenario runs. The expert validator's attestation at the "
        "end is the authoritative signal; the score is supporting evidence, not an operational "
        "readiness determination."
    )
    lines.append("")
    lines.extend(_judge_banner(scorecard))
    lines.append("## Research Basis")
    lines.append("")
    lines.append("| Basis | Venue | Use In This Tool | Source |")
    lines.append("|---|---:|---|---|")
    basis = scorecard.research_basis if scorecard else []
    if not basis:
        from .research_basis import RESEARCH_BASIS

        basis = RESEARCH_BASIS
    for item in basis:
        lines.append(f"| {item['name']} | {item['venue']} | {item['used_for']} | {item['url']} |")
    lines.append("")

    if scorecard:
        lines.extend(_macro_section(scorecard))
        lines.extend(_execution_failures_section(scorecard))
        lines.extend(_diagnostic_section(scorecard))
        lines.extend(_micro_section(scorecard, scenarios, runs))
    else:
        lines.append("## Numerical Scorecard (blank)")
        lines.append("")
        lines.append("Scoring scale per section: 0 = absent, 5 = strong thesis evidence.")
        lines.append("")
        lines.append("| Section | Weight | Raw Score (0-5) | Weighted Score | Evidence / Notes |")
        lines.append("|---|---:|---:|---:|---|")
        for _, (label, weight) in _section_weights().items():
            lines.append(f"| {label} | {weight:.0f} | ____ | ____ | Expert to fill with evidence. |")
        lines.append("| **Total** | **100** |  | ____ |   |")
        lines.append("")

    lines.extend(_attestation_section())

    lines.append("## Assumptions")
    lines.append("")
    assumptions = scorecard.assumptions if scorecard else [
        "The expert must independently review evidence before signing.",
        "Scores are numerical and section-weighted to 100 points.",
    ]
    for assumption in assumptions:
        lines.append(f"- {assumption}")
    lines.append("")
    return "\n".join(lines)


SECTION_DESCRIPTIONS: dict[str, str] = {
    "objective_quality": "Evaluates whether the agent produces actionable, civic-relevant content with correct topic coverage.",
    "trajectory": "Verifies that all 7 pipeline nodes executed in the expected order.",
    "memory_cache": "Assesses Smart Reuse caching, VSEE credibility verification, and RAG memory consolidation.",
    "groundedness": "Checks that every claim is traceable to a provided source, using an independent judge.",
    "temporal_hyperlocal": "Validates date-window compliance and Baguio-specific geographic relevance.",
    "robustness_safety": "Tests adversarial prompt resistance, graceful degradation, and source-quality reasoning.",
    "efficiency_readiness": "Records pipeline completion and error/fallback counts (latency not scored).",
    "agent_attribution": "CAIR counterfactual analysis measuring each node's contribution to the final output.",
}


def render_summary_form(scorecard: ValidationScorecard) -> str:
    """Render a minimal one-page summary: total score, section table, attestation."""
    lines: list[str] = []
    lines.append("# AgenticHinaing — Validation Summary")
    lines.append("")
    ci = scorecard.bootstrap_ci or {}
    if ci.get("iterations"):
        lines.append(
            f"**Score: {scorecard.total_score:.1f} / 100** "
            f"(95% CI: [{ci.get('low'):.1f}, {ci.get('high'):.1f}], "
            f"{len(scorecard.scenario_scores)} scenarios)"
        )
    else:
        lines.append(f"**Score: {scorecard.total_score:.1f} / 100** ({len(scorecard.scenario_scores)} scenarios)")
    lines.append("")
    lines.append("| Section | Wt | Score | Description |")
    lines.append("|---|---:|---:|---|")
    for section in scorecard.section_averages:
        desc = SECTION_DESCRIPTIONS.get(section.id, "")
        lines.append(
            f"| {section.label} | {section.weight:.0f} | {section.weighted_score:.1f} | {desc} |"
        )
    lines.append(f"| **Total** | **100** | **{scorecard.total_score:.1f}** | |")
    lines.append("")
    # Compact attestation
    for label, _ in [
        ("Expert name",            "name"),
        ("Professional title / affiliation", "title"),
        ("Relevant credential",    "credential"),
        ("Signature",              "signature"),
        ("Date signed",            "date"),
    ]:
        lines.append(f"**{label}** ______________________________________________")
        lines.append("")
    lines.append("")
    lines.append("*Full scenario detail and per-run evidence are in the accompanying validation tool document.*")
    lines.append("")
    return "\n".join(lines)


def write_validation_form(
    path: Path,
    scorecard: ValidationScorecard | None = None,
    scenarios: list[ValidationScenario] | None = None,
    runs: list[RunArtifact] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_validation_form(scorecard, scenarios=scenarios, runs=runs), encoding="utf-8")


def write_summary_form(path: Path, scorecard: ValidationScorecard) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_summary_form(scorecard), encoding="utf-8")


def preserve_run_artifacts(runs_dir: Path, runs: list[RunArtifact]) -> None:
    """Persist one directory per run (One-Eval artifact preservation)."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run_path = runs_dir / run.run_id
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "artifact.json").write_text(
            json.dumps(run.to_dict(), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        if run.snapshot_response is not None:
            (run_path / "response.json").write_text(
                json.dumps(run.snapshot_response, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
        if run.metrics:
            (run_path / "metrics.json").write_text(
                json.dumps(run.metrics, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
        if run.trajectory or run.tool_events:
            (run_path / "trajectory.json").write_text(
                json.dumps(
                    {
                        "trajectory": run.trajectory,
                        "node_order_observed": run.node_order_observed,
                        "tool_events": run.tool_events,
                        "progress_events": run.progress_events,
                    },
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        if run.independent_grading:
            (run_path / "judge.json").write_text(
                json.dumps(run.independent_grading, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
        provenance = (run.independent_grading or {}).get("source_provenance")
        if provenance:
            (run_path / "source_provenance.json").write_text(
                json.dumps(provenance, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
        if run.influence_ranking:
            (run_path / "influence.json").write_text(
                json.dumps(run.influence_ranking, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
