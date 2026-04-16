from __future__ import annotations

import ast
import importlib.util
import os
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Env vars that would redirect Poetry away from the backend's own venv.
# When the eval runs under `uv run`, uv exports VIRTUAL_ENV pointing at the
# eval's uv-managed venv — Poetry 2.x honors an already-activated VIRTUAL_ENV
# and silently runs python from that venv instead of the Hinaing backend's
# venv, which is why `poetry run python` can't see langgraph.
_POETRY_ENV_SCRUB = (
    "VIRTUAL_ENV",
    "VIRTUAL_ENV_PROMPT",
    "PYTHONHOME",
    "POETRY_ACTIVE",
    "UV_PROJECT_ENVIRONMENT",
    "PYVENV_LAUNCHER",
)


def _subprocess_env_for_poetry(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of os.environ with venv-confounding vars stripped."""
    env = os.environ.copy()
    for key in _POETRY_ENV_SCRUB:
        env.pop(key, None)
    # Strip uv's venv bin from PATH so `python` / `poetry run` don't
    # accidentally resolve to the eval's uv venv. uv appends its venv bin
    # as the first PATH entry when you're inside `uv run`.
    virtual_env_bin = os.environ.get("VIRTUAL_ENV")
    if virtual_env_bin:
        parts = env.get("PATH", "").split(os.pathsep)
        parts = [p for p in parts if not p.startswith(virtual_env_bin)]
        env["PATH"] = os.pathsep.join(parts)
    if extra:
        env.update(extra)
    return env


@dataclass(slots=True)
class PreflightCheck:
    id: str
    label: str
    passed: bool
    severity: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _query_orchestrator_accepts_ablation(hinaing_root: Path) -> tuple[bool, str]:
    source_path = hinaing_root / "backend/app/services/agents/query_orchestrator.py"
    source = _read(source_path)
    if not source:
        return False, f"Missing {source_path}"
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
            arg_names = [arg.arg for arg in node.args.args]
            has_kwargs = node.args.kwarg is not None
            accepts = "ablation_config" in arg_names or has_kwargs
            return accepts, f"QueryOrchestratorAgent.run args={arg_names}, kwargs={has_kwargs}"
    return False, "Could not find async run() in QueryOrchestratorAgent"


def _backend_env_import_check(hinaing_root: Path, python_cmd: str) -> tuple[bool, str]:
    """Shell out to the configured Python and verify `generate_snapshot` imports."""
    backend = hinaing_root / "backend"
    if not backend.exists():
        return False, f"Backend path missing: {backend}"
    try:
        cmd = shlex.split(python_cmd) + [
            "-c",
            "from app.services.insights.graph import generate_snapshot; print('ok')",
        ]
        proc = subprocess.run(
            cmd,
            cwd=backend,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=_subprocess_env_for_poetry(),
        )
    except FileNotFoundError as exc:
        return False, f"Interpreter not found for --python-cmd={python_cmd!r}: {exc}"
    except subprocess.TimeoutExpired:
        return False, f"--python-cmd={python_cmd!r} timed out while importing generate_snapshot"
    if proc.returncode == 0 and "ok" in proc.stdout:
        return True, f"{python_cmd} imported generate_snapshot successfully"
    stderr_tail = (proc.stderr or "").strip().splitlines()[-3:]
    return False, f"{python_cmd} could not import generate_snapshot: {' | '.join(stderr_tail) or 'unknown error'}"


def run_preflight(hinaing_root: Path, python_cmd: str | None = None) -> dict[str, Any]:
    """Run static and environment readiness checks.

    If ``python_cmd`` is given, additionally shells out to that interpreter and
    asserts that the Hinaing backend module graph is importable. This catches
    the common mistake of running the eval in a venv that does not have the
    Hinaing dependencies installed (e.g. langgraph missing).
    """
    checks: list[PreflightCheck] = []

    required_paths = {
        "architecture_doc": hinaing_root / "docs/ARCHITECTURE.md",
        "docs_readme": hinaing_root / "docs/README.md",
        "snapshot_router": hinaing_root / "backend/app/routers/snapshot.py",
        "chat_analyze_router": hinaing_root / "backend/app/routers/chat_analyze.py",
        "graph_entrypoint": hinaing_root / "backend/app/services/insights/graph.py",
        "frontend_snapshot": hinaing_root / "frontend/src/features/sentiment/components/sentiment-generator-page.tsx",
        "frontend_chat": hinaing_root / "frontend/src/features/chat/chat-analyze-page.tsx",
    }
    for check_id, path in required_paths.items():
        checks.append(
            PreflightCheck(
                id=check_id,
                label=f"Required path exists: {path.relative_to(hinaing_root)}",
                passed=path.exists(),
                severity="critical",
                detail=str(path),
            )
        )

    graph_source = _read(required_paths["graph_entrypoint"])
    checks.append(
        PreflightCheck(
            id="pre_retrieved_documents_path",
            label="Fixture execution path exists",
            passed="pre_retrieved_documents" in graph_source and "generate_snapshot" in graph_source,
            severity="critical",
            detail="generate_snapshot supports pre_retrieved_documents" if "pre_retrieved_documents" in graph_source else "No fixture bypass found",
        )
    )

    snapshot_frontend = _read(required_paths["frontend_snapshot"])
    checks.append(
        PreflightCheck(
            id="frontend_snapshot_route",
            label="Sentiment Generator calls /insights/snapshot",
            passed='"/insights/snapshot"' in snapshot_frontend or "'/insights/snapshot'" in snapshot_frontend,
            severity="critical",
            detail="Frontend trigger verified" if "/insights/snapshot" in snapshot_frontend else "Route call not found",
        )
    )

    chat_frontend = _read(required_paths["frontend_chat"])
    checks.append(
        PreflightCheck(
            id="frontend_chat_routes",
            label="Chat Analyze uses start/status polling routes",
            passed="/chat/analyze/start" in chat_frontend and "/chat/analyze/status/" in chat_frontend,
            severity="critical",
            detail="Background route and polling verified",
        )
    )

    accepts_ablation, detail = _query_orchestrator_accepts_ablation(hinaing_root)
    checks.append(
        PreflightCheck(
            id="query_orchestrator_signature",
            label="QueryOrchestratorAgent.run accepts graph ablation_config",
            passed=accepts_ablation,
            severity="critical",
            detail=detail,
        )
    )

    dependency_names = ["fastapi", "langgraph", "langchain", "qdrant_client", "langchain_google_genai"]
    for name in dependency_names:
        checks.append(
            PreflightCheck(
                id=f"dependency_{name}",
                label=f"Python dependency importable: {name}",
                passed=importlib.util.find_spec(name) is not None,
                severity="warning",
                detail="Available in current Python" if importlib.util.find_spec(name) else "Not available in current Python; use Hinaing backend environment or HTTP mode",
            )
        )

    if python_cmd:
        env_ok, env_detail = _backend_env_import_check(hinaing_root, python_cmd)
        checks.append(
            PreflightCheck(
                id="backend_env_import",
                label=f"Backend env imports generate_snapshot via {python_cmd!r}",
                passed=env_ok,
                severity="critical",
                detail=env_detail,
            )
        )

    metrics_source = _read(hinaing_root / "backend/app/services/metrics/collector.py")
    metrics_path_ok = bool(re.search(r"backend.*data.*metrics", metrics_source, flags=re.DOTALL))
    checks.append(
        PreflightCheck(
            id="metrics_persistence",
            label="Metrics collector persists JSONL metrics",
            passed=metrics_path_ok and "jsonl" in metrics_source,
            severity="critical",
            detail="Metrics JSONL persistence detected" if metrics_path_ok else "Metrics persistence path not detected",
        )
    )

    critical_failed = [check for check in checks if check.severity == "critical" and not check.passed]
    return {
        "hinaing_root": str(hinaing_root),
        "passed": len(critical_failed) == 0,
        "critical_failures": [check.to_dict() for check in critical_failed],
        "checks": [check.to_dict() for check in checks],
    }

