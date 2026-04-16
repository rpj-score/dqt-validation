from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import httpx

from .adapter import HinaingHttpAdapter, HinaingImportAdapter, HinaingSubprocessAdapter, detect_rate_limit
from .agentevals_bridge import optional_agentevals_status
from .io import DEFAULT_SCENARIO_PATH, load_runs, load_scenarios, write_json, write_jsonl
from .llm_judge import JudgeConfig, judge_artifact, judge_available
from .log import kv, log, set_quiet
from .models import RunArtifact, ValidationScenario
from .preflight import _backend_env_import_check as _preflight_env_check, run_preflight
from .report import preserve_run_artifacts, write_summary_form, write_validation_form
from .scoring import aggregate_scores


DEFAULT_HINAING_ROOT = Path("../Hinaing")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the AgenticHinaing proof-of-concept.")
    parser.add_argument("--hinaing-root", type=Path, default=DEFAULT_HINAING_ROOT)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIO_PATH)
    parser.add_argument("--quiet", action="store_true", help="Suppress timestamped progress logs on stderr.")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight_cmd = sub.add_parser("preflight")
    preflight_cmd.add_argument(
        "--python-cmd",
        default=None,
        help="Optional Python command for Hinaing's backend environment; if set, will attempt a real import.",
    )

    check_http = sub.add_parser("check-http")
    check_http.add_argument("--api-base", default="https://donyelqt-hinaing-backend.hf.space")
    check_http.add_argument("--output", type=Path, default=Path("reports/http_health.json"))

    form = sub.add_parser("generate-form")
    form.add_argument("--output", type=Path, default=Path("reports/validation_tool.md"))

    run = sub.add_parser("run-fixtures")
    run.add_argument("--output", type=Path, default=Path("reports/fixture_runs.jsonl"))
    run.add_argument(
        "--python-cmd",
        default=None,
        help="Optional Python command for Hinaing's backend environment, e.g. 'poetry run python'.",
    )
    run.add_argument(
        "--counterfactual",
        action="store_true",
        help="Run CAIR-style counterfactual patches per scenario for Agent Attribution scoring. Costs ~8x more graph executions per targeted scenario.",
    )
    run.add_argument(
        "--counterfactual-scenarios",
        type=str,
        default=None,
        help="Comma-separated scenario ids to counterfactually patch. Default: one per family.",
    )
    run.add_argument(
        "--counterfactual-patches",
        type=str,
        default="drop,shuffle",
        help="Comma-separated patch strategies (drop|shuffle). Default: drop,shuffle.",
    )
    run.add_argument(
        "--cair-weights",
        type=str,
        default=None,
        help="OC weights as 4 comma-separated floats: FOC,AOC,WC,AF. "
             "Default: 0.4,0.2,0.3,0.1. Use '0.6,0.0,0.4,0.0' to match the CAIR paper exactly.",
    )

    http = sub.add_parser("run-http")
    http.add_argument("--api-base", default="http://localhost:8000")
    http.add_argument("--output", type=Path, default=Path("reports/http_runs.jsonl"))
    http.add_argument("--chat", action="store_true", help="Use chat analyze route instead of snapshot route.")
    http.add_argument("--timeout", type=float, default=600.0, help="Read timeout for long-running Hinaing requests.")

    score = sub.add_parser("score")
    score.add_argument("--runs", type=Path, required=True)
    score.add_argument("--output", type=Path, default=Path("reports/scorecard.json"))
    score.add_argument("--form", type=Path, default=Path("reports/validation_tool_filled.md"))
    score.add_argument("--preflight", type=Path, default=None)
    score.add_argument("--llm-judge", action="store_true", help="Use independent Claude judge for groundedness and safety.")
    score.add_argument("--no-llm-judge", dest="llm_judge", action="store_false")
    score.set_defaults(llm_judge=None)
    score.add_argument("--judge-model", default="claude-sonnet-4-6")
    score.add_argument("--judge-cache-dir", type=Path, default=Path("reports/judge_cache"))
    score.add_argument("--runs-dir", type=Path, default=Path("reports/runs"), help="Where to persist per-run artifact folders (One-Eval style).")
    score.add_argument("--pdf", type=Path, default=None, help="Optional path to also render the full form as a signable PDF (requires `uv sync --extra pdf`).")
    score.add_argument("--summary", type=Path, default=None, help="Path for the one-page summary markdown.")
    score.add_argument("--summary-pdf", type=Path, default=None, help="Path for the one-page summary PDF.")
    score.add_argument("--no-preserve-artifacts", dest="preserve_artifacts", action="store_false")
    score.set_defaults(preserve_artifacts=True)

    all_cmd = sub.add_parser("all")
    all_cmd.add_argument("--runs", type=Path, default=Path("reports/fixture_runs.jsonl"))
    all_cmd.add_argument("--scorecard", type=Path, default=Path("reports/scorecard.json"))
    all_cmd.add_argument("--form", type=Path, default=Path("reports/validation_tool_filled.md"))

    return parser


async def _run_fixtures(args: argparse.Namespace) -> None:
    scenarios = load_scenarios(args.scenarios)
    python_cmd = getattr(args, "python_cmd", None)
    if python_cmd:
        log(f"preflight env check via {python_cmd!r}", prefix="preflight")
        ok, detail = _preflight_env_check(args.hinaing_root, python_cmd)
        if not ok:
            raise SystemExit(
                f"Backend env preflight failed: {detail}\n"
                f"Fix by installing Hinaing dependencies (e.g. `cd {args.hinaing_root}/backend && poetry install`)."
            )
        log(f"preflight OK: {detail}", prefix="preflight")
        adapter = HinaingSubprocessAdapter(args.hinaing_root, python_cmd, Path.cwd())
        adapter_label = f"subprocess({python_cmd!r})"
    else:
        adapter = HinaingImportAdapter(args.hinaing_root)
        adapter_label = "import"
    log(f"run-fixtures {kv(adapter=adapter_label, n=len(scenarios), out=str(args.output))}", prefix="cli")

    cf_enabled = bool(getattr(args, "counterfactual", False))
    cf_targets: set[str] = set()
    if cf_enabled:
        if not isinstance(adapter, (HinaingImportAdapter, HinaingSubprocessAdapter)):
            log("counterfactual: only supported by the import or subprocess adapters. Disabling.", prefix="cli")
            cf_enabled = False
        else:
            import fnmatch

            explicit = getattr(args, "counterfactual_scenarios", None)
            all_ids = [s.id for s in scenarios]
            if explicit and explicit.strip().lower() == "all":
                cf_targets = set(all_ids)
            elif explicit:
                cf_targets = set()
                for pattern in explicit.split(","):
                    pattern = pattern.strip()
                    if not pattern:
                        continue
                    if "*" in pattern or "?" in pattern or "[" in pattern:
                        matched = fnmatch.filter(all_ids, pattern)
                        cf_targets.update(matched)
                    else:
                        cf_targets.add(pattern)
            else:
                # one scenario per family by default
                by_family: dict[str, str] = {}
                for s in scenarios:
                    if s.family and s.family not in by_family:
                        by_family[s.family] = s.id
                cf_targets = set(by_family.values())
            log(f"counterfactual: enabled for {len(cf_targets)} scenarios={sorted(cf_targets)} patches={args.counterfactual_patches}", prefix="cli")

    # --- Pass 1: all baselines ------------------------------------------------
    artifacts: list[RunArtifact] = []
    scenario_by_id = {s.id: s for s in scenarios}
    t0 = time.monotonic()
    ok_count = err_count = 0
    rate_limit_hit = False
    for idx, scenario in enumerate(scenarios, 1):
        log(f"[{idx}/{len(scenarios)}] scenario={scenario.id} family={scenario.family}", prefix="cli")
        s0 = time.monotonic()
        artifact = await adapter.run_fixture(scenario)
        artifacts.append(artifact)
        if artifact.errors:
            err_count += 1
        else:
            ok_count += 1
        log(
            f"[{idx}/{len(scenarios)}] scenario={scenario.id} "
            + kv(status="ok" if not artifact.errors else "err", elapsed_s=round(time.monotonic() - s0, 2), stages=len(artifact.node_order_observed)),
            prefix="cli",
        )
        # Rate-limit detection: halt early so we don't waste time on runs
        # that will produce degraded output and pollute the scorecard.
        rl = detect_rate_limit(artifact)
        if rl:
            rate_limit_hit = True
            log(f"RATE LIMIT DETECTED on scenario {scenario.id}: {rl}", prefix="cli")
            log(
                f"STOPPING — {len(scenarios) - idx} scenario(s) remaining. "
                f"Completed {idx}/{len(scenarios)} before hitting the limit. "
                f"Results so far are saved; re-run after the rate limit resets or upgrade your API tier.",
                prefix="cli",
            )
            break
    baseline_elapsed = round(time.monotonic() - t0, 2)
    log(f"baselines {'PARTIAL' if rate_limit_hit else 'complete'} {kv(ok=ok_count, err=err_count, completed=len(artifacts), total=len(scenarios), total_s=baseline_elapsed)}", prefix="cli")
    # Tag rate-limited artifacts so they're classified as infrastructure failures
    # in the report, not scored as agent defects.
    for artifact in artifacts:
        rl = detect_rate_limit(artifact)
        if rl and not any("rate limit" in e.lower() for e in artifact.errors):
            artifact.errors.append(f"API rate limit detected (infrastructure failure, not agent defect): {rl}")

    # Write baselines immediately so results are safe if counterfactual crashes.
    write_jsonl(args.output, [a.to_dict() for a in artifacts])
    log(f"baseline results saved to {args.output}", prefix="cli")

    # --- Pass 2: counterfactual (after all baselines) -----------------------
    if cf_enabled and not rate_limit_hit:
        cf_candidates = [a for a in artifacts if a.scenario_id in cf_targets and not a.errors]
        log(
            f"counterfactual pass starting: {len(cf_candidates)} of {len(cf_targets)} targets have clean baselines",
            prefix="cli",
        )
        t1 = time.monotonic()
        for idx, artifact in enumerate(cf_candidates, 1):
            scenario = scenario_by_id.get(artifact.scenario_id)
            if not scenario:
                continue
            log(f"[cf {idx}/{len(cf_candidates)}] counterfactual for {scenario.id}", prefix="cli")
            try:
                await _run_counterfactual_for(adapter, scenario, artifact, args.counterfactual_patches)
            except Exception as exc:
                log(f"counterfactual: failed for {scenario.id}: {type(exc).__name__}: {exc}", prefix="cli")
        cf_elapsed = round(time.monotonic() - t1, 2)
        log(f"counterfactual pass complete {kv(n=len(cf_candidates), total_s=cf_elapsed)}", prefix="cli")
        # Re-write with influence_ranking attached.
        write_jsonl(args.output, [a.to_dict() for a in artifacts])
        log(f"results (with influence rankings) saved to {args.output}", prefix="cli")

    if cf_enabled and rate_limit_hit:
        log("counterfactual pass SKIPPED — rate limit was hit during baselines", prefix="cli")

    total_elapsed = round(time.monotonic() - t0, 2)
    log(f"run-fixtures done {kv(total_s=total_elapsed, n=len(artifacts), out=str(args.output))}", prefix="cli")
    if rate_limit_hit:
        print(f"PARTIAL RUN: {len(artifacts)}/{len(scenarios)} scenarios completed before rate limit. Results saved to {args.output}")
    else:
        print(f"Wrote {len(artifacts)} fixture run artifact(s) to {args.output}")


async def _run_counterfactual_for(
    adapter: "HinaingImportAdapter | HinaingSubprocessAdapter",
    scenario: ValidationScenario,
    baseline: RunArtifact,
    patches_spec: str,
) -> None:
    """Run CAIR-style counterfactual patches and attach influence_ranking to baseline."""
    from .agent_influence import PATCH_STRATEGIES, rank_nodes

    patch_names = [p.strip() for p in patches_spec.split(",") if p.strip() in PATCH_STRATEGIES]
    if not patch_names:
        patch_names = ["drop"]
    observed_nodes = baseline.node_order_observed or baseline.trajectory
    if not observed_nodes:
        log("counterfactual: no observed nodes on baseline — skipping", prefix="cli")
        return
    patched_by_node: dict[str, list[RunArtifact]] = {}
    use_subprocess = isinstance(adapter, HinaingSubprocessAdapter)
    for node in observed_nodes:
        variants: list[RunArtifact] = []
        for pname in patch_names:
            log(f"counterfactual: running patch={pname} target={node} scenario={scenario.id}", prefix="cli")
            try:
                if use_subprocess:
                    cf_artifact = await adapter.run_counterfactual_subprocess(
                        scenario, target_stage=node, patch_name=pname,
                    )
                else:
                    patch_fn = PATCH_STRATEGIES[pname]
                    cf_artifact = await adapter.run_fixture_with_patch(
                        scenario, target_stage=node, patch_fn=patch_fn,
                    )
                variants.append(cf_artifact)
            except Exception as exc:
                log(f"counterfactual: patch {pname!r}@{node} failed: {type(exc).__name__}: {exc}", prefix="cli")
        if variants:
            patched_by_node[node] = variants
    if patched_by_node:
        baseline.influence_ranking = rank_nodes(baseline, patched_by_node, observed_nodes)
        log(
            f"counterfactual: ranking complete for {scenario.id} — "
            f"{len(patched_by_node)} nodes, {sum(len(v) for v in patched_by_node.values())} variants",
            prefix="cli",
        )


async def _run_http(args: argparse.Namespace) -> None:
    scenarios = load_scenarios(args.scenarios)
    adapter = HinaingHttpAdapter(args.api_base, timeout_seconds=args.timeout)
    mode = "chat" if args.chat else "snapshot"
    log(
        f"run-http {kv(mode=mode, api_base=args.api_base, n=len(scenarios), timeout_s=int(args.timeout), out=str(args.output))}",
        prefix="cli",
    )
    rows = []
    t0 = time.monotonic()
    ok_count = err_count = 0
    for idx, scenario in enumerate(scenarios, 1):
        log(f"[{idx}/{len(scenarios)}] scenario={scenario.id} family={scenario.family} mode={mode}", prefix="cli")
        s0 = time.monotonic()
        artifact = await (adapter.run_chat_analyze(scenario) if args.chat else adapter.run_snapshot(scenario))
        rows.append(artifact.to_dict())
        if artifact.errors:
            err_count += 1
            status = "err"
        else:
            ok_count += 1
            status = "ok"
        log(
            f"[{idx}/{len(scenarios)}] scenario={scenario.id} "
            + kv(
                status=status,
                elapsed_s=round(time.monotonic() - s0, 2),
                stages=len(artifact.node_order_observed),
                insights=(
                    len((artifact.snapshot_response or {}).get("actionable_insights") or [])
                    if isinstance(artifact.snapshot_response, dict)
                    else 0
                ),
            ),
            prefix="cli",
        )
    write_jsonl(args.output, rows)
    log(
        "run-http done "
        + kv(
            total_s=round(time.monotonic() - t0, 2),
            ok=ok_count,
            err=err_count,
            n=len(rows),
            out=str(args.output),
        ),
        prefix="cli",
    )
    print(f"Wrote {len(rows)} HTTP run artifact(s) to {args.output}")


async def _check_http(args: argparse.Namespace) -> None:
    api_base = args.api_base.rstrip("/")
    log(f"check-http {kv(api_base=api_base)}", prefix="cli")
    result = {"api_base": api_base, "checks": []}
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        for path, method in [
            ("/health", "GET"),
            ("/insights/snapshot", "GET"),
            ("/chat/analyze/start", "GET"),
        ]:
            url = api_base + path
            log(f"probe {method} {url}", prefix="http")
            t0 = time.monotonic()
            try:
                response = await client.request(method, url)
                body = response.text[:300]
                ok = response.status_code == 200 if path == "/health" else response.status_code in {405, 422}
                log(
                    f"probe-result {kv(path=path, status=response.status_code, ok=ok, elapsed_s=round(time.monotonic() - t0, 2))}",
                    prefix="http",
                )
                result["checks"].append(
                    {
                        "path": path,
                        "method": method,
                        "status_code": response.status_code,
                        "content_type": response.headers.get("content-type"),
                        "body_preview": body,
                        "passed": ok,
                    }
                )
            except Exception as exc:
                log(
                    f"probe-FAILED {kv(path=path, err=type(exc).__name__, elapsed_s=round(time.monotonic() - t0, 2))}: {exc}",
                    prefix="http",
                )
                result["checks"].append(
                    {
                        "path": path,
                        "method": method,
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    result["passed"] = all(check["passed"] for check in result["checks"])
    write_json(args.output, result)
    print(f"HTTP health passed: {result['passed']}")
    for check in result["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        detail = check.get("status_code", check.get("error"))
        print(f"{status} {check['method']} {check['path']}: {detail}")
    print(f"Wrote {args.output}")


def _apply_source_provenance(
    scenarios: list[ValidationScenario], runs: list[RunArtifact]
) -> None:
    """Compute source provenance (trust profile + cross-check) for every run."""
    from .source_provenance import compute_source_provenance

    by_id = {s.id: s for s in scenarios}
    for run in runs:
        scenario = by_id.get(run.scenario_id)
        if not scenario:
            continue
        prov = compute_source_provenance(scenario, run)
        if not run.independent_grading:
            run.independent_grading = {}
        run.independent_grading["source_provenance"] = prov
    log(f"source provenance computed for {len(runs)} runs", prefix="cli")


def _apply_llm_judge(
    scenarios: list[ValidationScenario], runs: list[RunArtifact], args: argparse.Namespace
) -> None:
    want = args.llm_judge
    if want is None:
        want = judge_available()
    if not want:
        log("LLM judge disabled (heuristic grading only)", prefix="judge")
        return
    if not judge_available():
        import os
        reasons = []
        if os.environ.get("ANTHROPIC_API_KEY") is None:
            reasons.append("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic  # noqa: F401
        except Exception as exc:
            reasons.append(f"`anthropic` SDK not importable ({type(exc).__name__}: {exc}) — install via `uv sync --extra judge` or `uv pip install anthropic`")
        detail = "; ".join(reasons) or "judge_available() returned False"
        print(f"LLM judge requested but unavailable: {detail}. Falling back to heuristic grading.")
        log(f"LLM judge UNAVAILABLE — {detail}", prefix="judge")
        return
    config = JudgeConfig(model=args.judge_model, cache_dir=args.judge_cache_dir, enabled=True)
    log(
        f"llm-judge start {kv(model=args.judge_model, n=len(runs), cache_dir=str(args.judge_cache_dir))}",
        prefix="judge",
    )
    by_id = {scenario.id: scenario for scenario in scenarios}
    graded = 0
    t0 = time.monotonic()
    for idx, run in enumerate(runs, 1):
        scenario = by_id.get(run.scenario_id)
        if not scenario:
            log(f"[{idx}/{len(runs)}] judge SKIP (unknown scenario {run.scenario_id})", prefix="judge")
            continue
        s0 = time.monotonic()
        verdict = judge_artifact(scenario, run, config)
        elapsed = round(time.monotonic() - s0, 2)
        if verdict is not None:
            run.independent_grading = verdict
            graded += 1
            meta = verdict.get("_meta", {}) if isinstance(verdict, dict) else {}
            cache_hit = meta.get("cache_read_input_tokens") not in (None, 0)
            support_rate = None
            violations = None
            try:
                support_rate = verdict.get("groundedness", {}).get("support_rate")
                violations = len(verdict.get("adversarial", {}).get("violations", []) or [])
            except Exception:
                pass
            log(
                f"[{idx}/{len(runs)}] judge OK "
                + kv(
                    scenario=run.scenario_id,
                    elapsed_s=elapsed,
                    cache_hit=cache_hit,
                    support_rate=(round(support_rate, 3) if isinstance(support_rate, (int, float)) else support_rate),
                    violations=violations,
                ),
                prefix="judge",
            )
        elif isinstance(verdict, dict) and "_meta" in verdict:
            meta = verdict.get("_meta", {})
            err = meta.get("error") or meta.get("parse_error") or "unknown"
            attempts = meta.get("attempts", "?")
            run.independent_grading = verdict
            log(
                f"[{idx}/{len(runs)}] judge FAILED after {attempts} attempt(s) "
                + kv(scenario=run.scenario_id, elapsed_s=elapsed, error=str(err)[:120]),
                prefix="judge",
            )
        else:
            log(
                f"[{idx}/{len(runs)}] judge no-verdict {kv(scenario=run.scenario_id, elapsed_s=elapsed)}",
                prefix="judge",
            )
    log(
        f"llm-judge done {kv(graded=graded, n=len(runs), total_s=round(time.monotonic() - t0, 2))}",
        prefix="judge",
    )
    print(f"LLM judge graded {graded}/{len(runs)} runs using {args.judge_model} (cache: {args.judge_cache_dir}).")


def _read_preflight(path: Path | None, hinaing_root: Path) -> dict:
    if path and path.exists():
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    return run_preflight(hinaing_root)


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    set_quiet(getattr(args, "quiet", False))
    log(f"cli start command={args.command}", prefix="cli")

    if args.command == "preflight":
        result = run_preflight(args.hinaing_root, python_cmd=getattr(args, "python_cmd", None))
        result["agentevals"] = optional_agentevals_status()
        write_json(Path("reports/preflight.json"), result)
        print(f"Preflight passed: {result['passed']}")
        for check in result["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"{status} [{check['severity']}] {check['id']}: {check['detail']}")
        print("Wrote reports/preflight.json")
        return

    if args.command == "check-http":
        asyncio.run(_check_http(args))
        return

    if args.command == "generate-form":
        write_validation_form(args.output)
        print(f"Wrote blank validation form to {args.output}")
        return

    if args.command == "run-fixtures":
        asyncio.run(_run_fixtures(args))
        return

    if args.command == "run-http":
        asyncio.run(_run_http(args))
        return

    if args.command == "score":
        scenarios = load_scenarios(args.scenarios)
        runs = load_runs(args.runs)
        log(f"score load {kv(scenarios=len(scenarios), runs=len(runs))}", prefix="cli")
        preflight = _read_preflight(args.preflight, args.hinaing_root)
        _apply_source_provenance(scenarios, runs)
        _apply_llm_judge(scenarios, runs, args)
        log("score aggregate", prefix="cli")
        scorecard = aggregate_scores(scenarios, runs, preflight=preflight)
        scorecard.judge_available = judge_available()
        write_json(args.output, scorecard.to_dict())
        write_validation_form(args.form, scorecard, scenarios=scenarios, runs=runs)
        if getattr(args, "preserve_artifacts", True):
            preserve_run_artifacts(args.runs_dir, runs)
            log(f"preserved per-run artifacts {kv(runs_dir=str(args.runs_dir), n=len(runs))}", prefix="cli")
            print(f"Preserved per-run artifacts in {args.runs_dir}")
        summary_path = getattr(args, "summary", None)
        summary_pdf = getattr(args, "summary_pdf", None)
        if summary_path or summary_pdf:
            if not summary_path:
                summary_path = args.form.parent / (args.form.stem + "_summary.md")
            write_summary_form(summary_path, scorecard)
            log(f"summary: wrote {summary_path}", prefix="cli")
            print(f"Wrote summary {summary_path}")
            if summary_pdf:
                try:
                    import sys as _sys
                    import subprocess as _sp
                    script = Path(__file__).resolve().parent.parent / "scripts" / "validation_tool_to_pdf.py"
                    proc = _sp.run(
                        [_sys.executable, str(script), "--input", str(summary_path), "--output", str(summary_pdf)],
                        capture_output=True, text=True, check=False,
                    )
                    if proc.returncode == 0:
                        log(f"summary-pdf: wrote {summary_pdf}", prefix="cli")
                        print(f"Wrote summary PDF {summary_pdf}")
                    else:
                        log(f"summary-pdf: failed — {proc.stderr.strip()[-300:]}", prefix="cli")
                except Exception as exc:
                    log(f"summary-pdf: {type(exc).__name__}: {exc}", prefix="cli")
        pdf_target = getattr(args, "pdf", None)
        if pdf_target:
            try:
                import sys
                script = Path(__file__).resolve().parent.parent / "scripts" / "validation_tool_to_pdf.py"
                import subprocess
                proc = subprocess.run(
                    [sys.executable, str(script), "--input", str(args.form), "--output", str(pdf_target)],
                    capture_output=True, text=True, check=False,
                )
                if proc.returncode == 0:
                    log(f"pdf: wrote {pdf_target}", prefix="cli")
                    print(f"Wrote PDF {pdf_target}")
                else:
                    log(f"pdf: rendering failed — {proc.stderr.strip()[-500:]}", prefix="cli")
                    print(f"PDF rendering failed: {proc.stderr.strip()[-500:]}")
            except Exception as exc:
                log(f"pdf: unavailable — {type(exc).__name__}: {exc}", prefix="cli")
        print(f"Total score: {scorecard.total_score:.2f} / 100")
        print(f"Wrote {args.output} and {args.form}")
        return

    if args.command == "all":
        preflight = run_preflight(args.hinaing_root)
        write_json(Path("reports/preflight.json"), preflight)
        args.output = args.runs
        asyncio.run(_run_fixtures(args))
        scenarios = load_scenarios(args.scenarios)
        runs = load_runs(args.runs)
        scorecard = aggregate_scores(scenarios, runs, preflight=preflight)
        write_json(args.scorecard, scorecard.to_dict())
        write_validation_form(args.form, scorecard, scenarios=scenarios, runs=runs)
        print(f"Total score: {scorecard.total_score:.2f} / 100")
        print(f"Wrote {args.runs}, {args.scorecard}, and {args.form}")
        return


if __name__ == "__main__":
    main()
