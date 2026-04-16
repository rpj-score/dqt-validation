from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import JsonDict, RunArtifact, ValidationScenario


DEFAULT_SCENARIO_DIR = Path("data/scenarios")
# Backwards-compatible single-file path for the legacy CLI flag.
DEFAULT_SCENARIO_PATH = DEFAULT_SCENARIO_DIR


def read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _scenario_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(target.glob("*.jsonl"))
    # Fallback: legacy single-file default no longer exists — try the default dir.
    if target == DEFAULT_SCENARIO_PATH:
        legacy = target / "hyperlocal_baguio_v1.jsonl"
        if legacy.exists():
            return [legacy]
    return []


def load_scenarios(path: Path = DEFAULT_SCENARIO_PATH) -> list[ValidationScenario]:
    scenarios: list[ValidationScenario] = []
    seen_ids: set[str] = set()
    for file_path in _scenario_files(path):
        for row in read_jsonl(file_path):
            scenario = ValidationScenario.from_dict(row)
            if scenario.id in seen_ids:
                continue
            seen_ids.add(scenario.id)
            scenarios.append(scenario)
    return scenarios


def load_runs(path: Path) -> list[RunArtifact]:
    return [RunArtifact.from_dict(row) for row in read_jsonl(path)]
