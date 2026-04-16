from __future__ import annotations

import asyncio
import os
import json
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .log import heartbeat, kv, log
from .models import JsonDict, RunArtifact, ValidationScenario


# Canonical stage names emitted by Hinaing's progress_callback in
# backend/app/services/insights/graph.py:130-171. Used only as a reference list
# for trajectory scoring — NEVER assigned as an observed trajectory.
HINAING_STAGES = [
    "query_orchestrator",
    "retrieval",
    "recall",
    "analyze",
    "memory",
    "themes",
    "snapshot",
]

# Maps stage names to the PipelineMetrics per-node latency keys. When the
# progress_callback misses a stage (e.g. Node 6 runs in asyncio.to_thread
# and the callback doesn't fire), we can still detect that the node ran if
# its latency is non-zero.
_STAGE_TO_LATENCY_KEY: dict[str, str] = {
    "query_orchestrator": "query_orchestrator_ms",
    "retrieval":          "external_retrieval_ms",
    "recall":             "internal_retrieval_ms",
    "analyze":            "sentiment_analysis_ms",
    "memory":             "memory_consolidation_ms",
    "themes":             "theme_agents_ms",
    "snapshot":           "coordinator_ms",
}


def _supplement_trajectory_from_metrics(artifact: "RunArtifact") -> None:
    """Fill gaps in the observed trajectory using PipelineMetrics latency keys.

    The progress_callback may not fire for all 7 stages (e.g. Node 6 runs in
    asyncio.to_thread and Node 7's callback races with the return). If the
    per-node latency is > 0 in the metrics, the node definitely executed and
    should appear in the trajectory.
    """
    if not artifact.metrics:
        return
    observed = set(artifact.node_order_observed or artifact.trajectory or [])
    appended: list[str] = []
    for stage in HINAING_STAGES:
        if stage in observed:
            continue
        latency_key = _STAGE_TO_LATENCY_KEY.get(stage)
        if not latency_key:
            continue
        latency = artifact.metrics.get(latency_key)
        try:
            if latency is not None and float(latency) > 0:
                appended.append(stage)
        except (TypeError, ValueError):
            continue
    if appended:
        artifact.trajectory = list(artifact.trajectory or []) + appended
        artifact.node_order_observed = list(artifact.node_order_observed or []) + appended


_RATE_LIMIT_INDICATORS = ("rate limit", "rate_limit", "429", "too many requests", "tokens per day", "tpd")


def detect_rate_limit(artifact: "RunArtifact") -> str | None:
    """Return a description if the artifact shows signs of API rate limiting.

    Checks errors and warnings for Groq/OpenRouter/Gemini 429 patterns.
    Returns None if no rate limit detected.
    """
    for source in (artifact.errors, artifact.warnings):
        for msg in source:
            lower = msg.lower()
            if any(indicator in lower for indicator in _RATE_LIMIT_INDICATORS):
                # Extract the key detail
                for line in msg.split("\n"):
                    ll = line.lower()
                    if any(ind in ll for ind in _RATE_LIMIT_INDICATORS):
                        return line.strip()[:300]
                return msg[:300]
    return None


class _ProgressRecorder:
    """Accumulates stage events and per-stage latencies from progress_callback."""

    def __init__(self) -> None:
        self.events: list[JsonDict] = []
        self.tool_events: list[JsonDict] = []
        self.trajectory: list[str] = []
        self.node_order: list[str] = []
        self._last_stage: str | None = None
        self._last_stage_started: float | None = None

    async def __call__(self, stage: str, message: str, progress: float) -> None:
        now = time.monotonic()
        now_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        event = {"stage": stage, "message": message, "progress": progress, "at": now_iso}
        self.events.append(event)
        self.trajectory.append(stage)
        if stage not in self.node_order:
            self.node_order.append(stage)
        if self._last_stage is not None and self._last_stage_started is not None:
            self.tool_events.append(
                {
                    "stage": self._last_stage,
                    "latency_ms": round((now - self._last_stage_started) * 1000.0, 2),
                    "completed_at": now_iso,
                }
            )
        self._last_stage = stage
        self._last_stage_started = now

    def finalize(self) -> None:
        if self._last_stage is None or self._last_stage_started is None:
            return
        now = time.monotonic()
        self.tool_events.append(
            {
                "stage": self._last_stage,
                "latency_ms": round((now - self._last_stage_started) * 1000.0, 2),
                "completed_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            }
        )


class HinaingImportAdapter:
    """Run Hinaing directly through backend Python imports."""

    def __init__(self, hinaing_root: Path):
        self.hinaing_root = hinaing_root.resolve()
        self.backend_root = self.hinaing_root / "backend"

    # Maps eval stage names → (graph.py module attribute, is_sync).
    # generate_snapshot calls these as `state = await attr(state)` (or
    # asyncio.to_thread for sync). We monkeypatch the attribute to inject
    # a post-call patch function for CAIR counterfactuals.
    NODE_MAP: dict[str, tuple[str, bool]] = {
        "query_orchestrator": ("orchestrate_queries",         False),
        "retrieval":          ("fetch_documents",              False),
        "recall":             ("retrieve_internal_knowledge",  False),
        "analyze":            ("label_sentiment_and_analyze",  False),
        "memory":             ("consolidate_memory",           False),
        "themes":             ("theme_agents",                 True),
        "snapshot":           ("build_snapshot",               False),
    }

    async def run_fixture_with_patch(
        self,
        scenario: ValidationScenario,
        target_stage: str,
        patch_fn: "Any",
    ) -> RunArtifact:
        """Run ``scenario`` with a monkeypatched node for CAIR counterfactual eval.

        After the target node writes to state, ``patch_fn(state, target_stage)``
        is called to mutate state in-place. Downstream nodes then see the
        patched state, causing the final SnapshotResponse to diverge from the
        baseline — which is exactly what CAIR's FOC/AOC/WC metrics measure.

        The patch is applied by temporarily swapping the node function on the
        ``app.services.insights.graph`` module. This avoids any changes to the
        Hinaing backend; the original function is restored in a finally block.
        """
        artifact = RunArtifact.start(
            scenario=scenario,
            mode="fixture_counterfactual",
            run_id=str(uuid.uuid4())[:8],
            adapter="import",
        )
        recorder = _ProgressRecorder()
        old_cwd = Path.cwd()
        old_path = list(sys.path)

        if target_stage not in self.NODE_MAP:
            artifact.errors.append(f"Unknown target_stage={target_stage!r}; valid: {sorted(self.NODE_MAP)}")
            artifact.finish()
            return artifact

        attr_name, is_sync = self.NODE_MAP[target_stage]
        log(
            f"counterfactual start {kv(scenario=scenario.id, run_id=artifact.run_id, target=target_stage, attr=attr_name)}",
            prefix="cf",
        )
        original_fn = None
        graph_module = None
        try:
            os.chdir(self.backend_root)
            sys.path.insert(0, str(self.backend_root))

            from app.schemas.snapshot import SnapshotRequest, WebDocument  # type: ignore
            import app.services.insights.graph as _graph_module  # type: ignore
            from app.services.metrics import get_metrics_collector  # type: ignore

            graph_module = _graph_module
            original_fn = getattr(graph_module, attr_name)

            if is_sync:
                def wrapped(state, _orig=original_fn, _pfn=patch_fn, _stage=target_stage):
                    result = _orig(state)
                    try:
                        _pfn(result, _stage)
                    except Exception as exc:
                        log(f"counterfactual patch_fn raised: {type(exc).__name__}: {exc}", prefix="cf")
                    return result
            else:
                async def wrapped(state, _orig=original_fn, _pfn=patch_fn, _stage=target_stage):
                    result = await _orig(state)
                    try:
                        _pfn(result, _stage)
                    except Exception as exc:
                        log(f"counterfactual patch_fn raised: {type(exc).__name__}: {exc}", prefix="cf")
                    return result

            setattr(graph_module, attr_name, wrapped)
            log(f"counterfactual patched {attr_name} on graph module", prefix="cf")

            request = SnapshotRequest(**scenario.request)
            docs = [WebDocument(**doc) for doc in scenario.frozen_documents]
            async with heartbeat(
                f"counterfactual generate_snapshot(scenario={scenario.id}, target={target_stage})",
                every=15.0,
                prefix="cf",
            ):
                response = await graph_module.generate_snapshot(
                    request,
                    progress_callback=recorder,
                    pre_retrieved_documents=docs,
                )
            recorder.finalize()
            artifact.snapshot_response = response.model_dump(mode="json")
            artifact.progress_events = recorder.events
            artifact.trajectory = recorder.trajectory
            artifact.node_order_observed = recorder.node_order
            artifact.tool_events = recorder.tool_events

            collector = get_metrics_collector()
            completed = getattr(collector, "_completed_runs", [])
            if completed:
                artifact.metrics = completed[-1].to_dict()
            log(
                f"counterfactual DONE {kv(scenario=scenario.id, target=target_stage, stages=len(recorder.node_order))}",
                prefix="cf",
            )
        except Exception as exc:
            artifact.errors.append(f"{type(exc).__name__}: {exc}")
            recorder.finalize()
            artifact.progress_events = recorder.events
            artifact.trajectory = recorder.trajectory
            artifact.node_order_observed = recorder.node_order
            artifact.tool_events = recorder.tool_events
            log(f"counterfactual FAILED {kv(scenario=scenario.id, target=target_stage, err=type(exc).__name__)}: {exc}", prefix="cf")
        finally:
            if graph_module is not None and original_fn is not None:
                setattr(graph_module, attr_name, original_fn)
                log(f"counterfactual restored {attr_name}", prefix="cf")
            sys.path = old_path
            os.chdir(old_cwd)
            artifact.finish()
        return artifact

    async def run_fixture(self, scenario: ValidationScenario) -> RunArtifact:
        artifact = RunArtifact.start(
            scenario=scenario,
            mode="fixture_import",
            run_id=str(uuid.uuid4())[:8],
            adapter="import",
        )
        recorder = _ProgressRecorder()
        log(
            f"fixture start {kv(scenario=scenario.id, run_id=artifact.run_id, docs=len(scenario.frozen_documents))}",
            prefix="import",
        )
        started = time.monotonic()

        old_cwd = Path.cwd()
        old_path = list(sys.path)
        try:
            os.chdir(self.backend_root)
            sys.path.insert(0, str(self.backend_root))

            from app.schemas.snapshot import SnapshotRequest, WebDocument  # type: ignore
            from app.services.insights.graph import generate_snapshot  # type: ignore
            from app.services.metrics import get_metrics_collector  # type: ignore

            request = SnapshotRequest(**scenario.request)
            docs = [WebDocument(**doc) for doc in scenario.frozen_documents]
            async with heartbeat(
                f"generate_snapshot(scenario={scenario.id})", every=15.0, prefix="import"
            ):
                response = await generate_snapshot(
                    request,
                    progress_callback=recorder,
                    pre_retrieved_documents=docs,
                )
            recorder.finalize()
            artifact.snapshot_response = response.model_dump(mode="json")
            artifact.progress_events = recorder.events
            artifact.trajectory = recorder.trajectory
            artifact.node_order_observed = recorder.node_order
            artifact.tool_events = recorder.tool_events

            collector = get_metrics_collector()
            completed = getattr(collector, "_completed_runs", [])
            if completed:
                artifact.metrics = completed[-1].to_dict()
            _supplement_trajectory_from_metrics(artifact)
            log(
                f"fixture DONE {kv(scenario=scenario.id, elapsed_s=round(time.monotonic() - started, 2), stages=len(artifact.node_order_observed), has_metrics=bool(artifact.metrics))}",
                prefix="import",
            )
        except Exception as exc:
            artifact.errors.append(f"{type(exc).__name__}: {exc}")
            artifact.warnings.append(
                "Import adapter failed. Install Hinaing backend dependencies or use HTTP mode with a running backend."
            )
            recorder.finalize()
            artifact.progress_events = recorder.events
            artifact.trajectory = recorder.trajectory
            artifact.node_order_observed = recorder.node_order
            artifact.tool_events = recorder.tool_events
            log(
                f"fixture FAILED {kv(scenario=scenario.id, err=type(exc).__name__)}: {exc}",
                prefix="import",
            )
        finally:
            sys.path = old_path
            os.chdir(old_cwd)
            artifact.finish()
        return artifact


class HinaingHttpAdapter:
    """Run Hinaing through its frontend-facing HTTP API."""

    def __init__(self, api_base: str, timeout_seconds: float = 600.0):
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.http_timeout = httpx.Timeout(timeout_seconds, connect=min(5.0, timeout_seconds))

    async def run_snapshot(self, scenario: ValidationScenario) -> RunArtifact:
        artifact = RunArtifact.start(
            scenario=scenario,
            mode="http_snapshot",
            run_id=str(uuid.uuid4())[:8],
            adapter="http",
        )
        log(
            f"snapshot start {kv(scenario=scenario.id, run_id=artifact.run_id, api_base=self.api_base)}",
            prefix="http",
        )
        try:
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                label = (
                    f"POST {self.api_base}/insights/snapshot (scenario={scenario.id}, "
                    f"timeout={self.timeout_seconds:.0f}s)"
                )
                async with heartbeat(label, every=15.0, prefix="http"):
                    response = await client.post(
                        f"{self.api_base}/insights/snapshot", json=scenario.request
                    )
                log(
                    f"snapshot http-response {kv(scenario=scenario.id, status=response.status_code, bytes=len(response.content))}",
                    prefix="http",
                )
                response.raise_for_status()
                artifact.snapshot_response = response.json()
                insights = (
                    len(artifact.snapshot_response.get("actionable_insights") or [])
                    if isinstance(artifact.snapshot_response, dict)
                    else 0
                )
                sources = (
                    len(artifact.snapshot_response.get("sources") or [])
                    if isinstance(artifact.snapshot_response, dict)
                    else 0
                )
                log(
                    f"snapshot parsed {kv(scenario=scenario.id, insights=insights, sources=sources)}",
                    prefix="http",
                )
                artifact.warnings.append(
                    "HTTP snapshot endpoint is non-streaming; trajectory is not observable. "
                    "Section 2 (Trajectory) will be marked not applicable for this run."
                )
        except httpx.HTTPStatusError as exc:
            artifact.errors.append(f"HTTPStatusError: {exc.response.status_code} {exc.response.text[:300]}")
            log(
                f"snapshot FAILED {kv(scenario=scenario.id, status=exc.response.status_code)}",
                prefix="http",
            )
        except httpx.TimeoutException as exc:
            artifact.errors.append(f"TimeoutException: {exc}")
            log(f"snapshot TIMEOUT {kv(scenario=scenario.id, timeout_s=self.timeout_seconds)}", prefix="http")
        except Exception as exc:
            artifact.errors.append(f"{type(exc).__name__}: {exc}")
            log(f"snapshot FAILED {kv(scenario=scenario.id, err=type(exc).__name__)}: {exc}", prefix="http")
        finally:
            artifact.finish()
        return artifact

    async def run_chat_analyze(self, scenario: ValidationScenario) -> RunArtifact:
        artifact = RunArtifact.start(
            scenario=scenario,
            mode="http_chat_analyze",
            run_id=str(uuid.uuid4())[:8],
            adapter="http",
        )
        chat_request = {
            "message": scenario.expected.get("chat_message") or scenario.name,
            "platforms": scenario.request.get("platforms", ["web"]),
            "time_window": scenario.request.get("time_window", "24h"),
            "mode": scenario.request.get("mode", "full"),
            "system_mode": "agentic_hinaing",
            "ablation_preset": scenario.request.get("ablation_preset", "full"),
        }
        log(
            f"chat start {kv(scenario=scenario.id, run_id=artifact.run_id, api_base=self.api_base, msg_len=len(chat_request['message']))}",
            prefix="http",
        )
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                async with heartbeat(
                    f"POST {self.api_base}/chat/analyze/start (scenario={scenario.id})",
                    every=10.0,
                    prefix="http",
                ):
                    start_response = await client.post(
                        f"{self.api_base}/chat/analyze/start", json=chat_request
                    )
                log(
                    f"chat start-response {kv(scenario=scenario.id, status=start_response.status_code)}",
                    prefix="http",
                )
                start_response.raise_for_status()
                start_payload = start_response.json()
                artifact.chat_response = start_payload

                if start_payload.get("immediate_result"):
                    artifact.trajectory = ["chat_analyze_start"]
                    artifact.node_order_observed = ["chat_analyze_start"]
                    log(
                        f"chat immediate_result {kv(scenario=scenario.id, elapsed_s=round(time.monotonic() - started, 2))}",
                        prefix="http",
                    )
                    return artifact

                task_id = start_payload.get("task_id")
                if not task_id:
                    artifact.errors.append("No task_id returned from chat analyze start route.")
                    log(f"chat NO-TASK-ID {kv(scenario=scenario.id)}", prefix="http")
                    return artifact

                log(
                    f"chat polling {kv(scenario=scenario.id, task_id=task_id, timeout_s=int(self.timeout_seconds))}",
                    prefix="http",
                )
                deadline = datetime.now(timezone.utc).timestamp() + self.timeout_seconds
                tick = 0
                last_reported_stage: str | None = None
                last_reported_progress: float | None = None
                while datetime.now(timezone.utc).timestamp() < deadline:
                    tick += 1
                    try:
                        status_response = await client.get(
                            f"{self.api_base}/chat/analyze/status/{task_id}"
                        )
                    except httpx.TimeoutException as exc:
                        log(
                            f"chat poll TIMEOUT {kv(scenario=scenario.id, tick=tick, err=type(exc).__name__)}",
                            prefix="http",
                        )
                        await asyncio.sleep(2.0)
                        continue
                    status_response.raise_for_status()
                    status = status_response.json()
                    artifact.progress_events.append(status)
                    stage = status.get("stage") or status.get("current_stage")
                    progress = status.get("progress")
                    state = status.get("status")
                    message = status.get("message") or status.get("detail") or ""
                    # Always emit the first tick; after that only when something moves.
                    moved = (
                        tick == 1
                        or stage != last_reported_stage
                        or (progress is not None and progress != last_reported_progress)
                        or state in {"completed", "failed"}
                    )
                    if moved:
                        elapsed = time.monotonic() - started
                        log(
                            "chat status "
                            + kv(
                                scenario=scenario.id,
                                tick=tick,
                                state=state,
                                stage=stage,
                                progress=(round(progress, 3) if isinstance(progress, (int, float)) else progress),
                                elapsed_s=round(elapsed, 1),
                            )
                            + (f" | {message[:120]}" if message else ""),
                            prefix="http",
                        )
                        last_reported_stage = stage
                        last_reported_progress = progress if isinstance(progress, (int, float)) else last_reported_progress
                    if stage and (not artifact.trajectory or artifact.trajectory[-1] != stage):
                        artifact.trajectory.append(stage)
                        if stage not in artifact.node_order_observed:
                            artifact.node_order_observed.append(stage)
                    if state == "completed":
                        artifact.chat_response = status.get("result", status)
                        log(
                            f"chat DONE {kv(scenario=scenario.id, ticks=tick, elapsed_s=round(time.monotonic() - started, 2), stages=len(artifact.node_order_observed))}",
                            prefix="http",
                        )
                        break
                    if state == "failed":
                        err = str(status.get("error") or status.get("message") or "chat task failed")
                        artifact.errors.append(err)
                        log(f"chat FAILED {kv(scenario=scenario.id, ticks=tick)}: {err}", prefix="http")
                        break
                    await asyncio.sleep(2.0)
                else:
                    artifact.errors.append("Timed out waiting for chat analysis task.")
                    log(
                        f"chat TIMEOUT {kv(scenario=scenario.id, ticks=tick, timeout_s=int(self.timeout_seconds))}",
                        prefix="http",
                    )
        except httpx.HTTPStatusError as exc:
            artifact.errors.append(f"HTTPStatusError: {exc.response.status_code} {exc.response.text[:300]}")
            log(f"chat HTTP-ERROR {kv(scenario=scenario.id, status=exc.response.status_code)}", prefix="http")
        except Exception as exc:
            artifact.errors.append(f"{type(exc).__name__}: {exc}")
            log(f"chat FAILED {kv(scenario=scenario.id, err=type(exc).__name__)}: {exc}", prefix="http")
        finally:
            artifact.finish()
        return artifact


class HinaingSubprocessAdapter:
    """Run fixture scenarios through a specified Python command.

    Use this when Hinaing has a separate virtualenv, e.g.
    `poetry -C ../Hinaing/backend run python`.
    """

    def __init__(self, hinaing_root: Path, python_cmd: str, eval_root: Path):
        self.hinaing_root = hinaing_root.resolve()
        self.backend_root = self.hinaing_root / "backend"
        self.python_cmd = shlex.split(python_cmd)
        self.eval_root = eval_root.resolve()

    async def run_fixture(self, scenario: ValidationScenario) -> RunArtifact:
        # Start from a copy of the current env, then strip venv-selection vars
        # so Poetry resolves the backend's own venv instead of inheriting
        # `uv run`'s VIRTUAL_ENV pointer at the eval's uv venv.
        from .preflight import _subprocess_env_for_poetry
        env = _subprocess_env_for_poetry()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.eval_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        cmd = [
            *self.python_cmd,
            "-m",
            "agentic_hinaing_eval.runner_import",
            "--hinaing-root",
            str(self.hinaing_root),
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=json.dumps(scenario.to_dict()),
                text=True,
                capture_output=True,
                cwd=self.backend_root,
                env=env,
                timeout=900,
                check=False,
            )
        except Exception as exc:
            artifact = RunArtifact.start(scenario, "fixture_subprocess", str(uuid.uuid4())[:8], "subprocess")
            artifact.errors.append(f"{type(exc).__name__}: {exc}")
            artifact.finish()
            return artifact

        if proc.returncode != 0:
            artifact = RunArtifact.start(scenario, "fixture_subprocess", str(uuid.uuid4())[:8], "subprocess")
            artifact.errors.append(f"Subprocess failed with exit code {proc.returncode}: {proc.stderr.strip()}")
            artifact.finish()
            return artifact

        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            artifact = RunArtifact.from_dict(data)
            artifact.mode = "fixture_subprocess"
            artifact.adapter = "subprocess"
            _supplement_trajectory_from_metrics(artifact)
            if proc.stderr.strip():
                artifact.warnings.append(proc.stderr.strip()[-1000:])
            return artifact
        except Exception as exc:
            artifact = RunArtifact.start(scenario, "fixture_subprocess", str(uuid.uuid4())[:8], "subprocess")
            artifact.errors.append(f"Could not parse subprocess artifact: {type(exc).__name__}: {exc}")
            artifact.warnings.append(proc.stdout.strip()[-1000:])
            artifact.finish()
            return artifact

    async def run_counterfactual_subprocess(
        self,
        scenario: ValidationScenario,
        target_stage: str,
        patch_name: str,
    ) -> RunArtifact:
        """Run a CAIR counterfactual variant inside the backend's poetry venv.

        Delegates to ``runner_import.py --counterfactual --target-stage X --patch Y``
        which does the monkeypatch on ``app.services.insights.graph`` inside the
        subprocess. This works with ``--python-cmd`` because the monkeypatch
        happens in the same process that imports the graph module.
        """
        from .preflight import _subprocess_env_for_poetry
        env = _subprocess_env_for_poetry()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.eval_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        cmd = [
            *self.python_cmd,
            "-m",
            "agentic_hinaing_eval.runner_import",
            "--hinaing-root",
            str(self.hinaing_root),
            "--counterfactual",
            "--target-stage",
            target_stage,
            "--patch",
            patch_name,
        ]
        log(
            f"counterfactual subprocess {kv(scenario=scenario.id, target=target_stage, patch=patch_name)}",
            prefix="cf",
        )
        try:
            proc = subprocess.run(
                cmd,
                input=json.dumps(scenario.to_dict()),
                text=True,
                capture_output=True,
                cwd=self.backend_root,
                env=env,
                timeout=900,
                check=False,
            )
        except Exception as exc:
            artifact = RunArtifact.start(scenario, "fixture_counterfactual", str(uuid.uuid4())[:8], "subprocess")
            artifact.errors.append(f"{type(exc).__name__}: {exc}")
            artifact.finish()
            return artifact

        if proc.returncode != 0:
            artifact = RunArtifact.start(scenario, "fixture_counterfactual", str(uuid.uuid4())[:8], "subprocess")
            artifact.errors.append(f"Counterfactual subprocess failed (exit {proc.returncode}): {proc.stderr.strip()[-500:]}")
            artifact.finish()
            return artifact

        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            artifact = RunArtifact.from_dict(data)
            artifact.mode = "fixture_counterfactual"
            artifact.adapter = "subprocess"
            if proc.stderr.strip():
                artifact.warnings.append(proc.stderr.strip()[-1000:])
            log(
                f"counterfactual subprocess DONE {kv(scenario=scenario.id, target=target_stage, patch=patch_name)}",
                prefix="cf",
            )
            return artifact
        except Exception as exc:
            artifact = RunArtifact.start(scenario, "fixture_counterfactual", str(uuid.uuid4())[:8], "subprocess")
            artifact.errors.append(f"Could not parse counterfactual artifact: {type(exc).__name__}: {exc}")
            artifact.warnings.append(proc.stdout.strip()[-1000:])
            artifact.finish()
            return artifact
