from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


JsonDict = dict[str, Any]

SCENARIO_FAMILIES = ("hyperlocal", "cache", "adversarial", "missing_data", "ablation")


@dataclass(slots=True)
class ValidationScenario:
    """A frozen evaluation scenario for the Hinaing webapp."""

    id: str
    name: str
    persona: str
    request: JsonDict
    frozen_documents: list[JsonDict] = field(default_factory=list)
    expected: JsonDict = field(default_factory=dict)
    rubrics: JsonDict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    reference_trajectory: list[str] = field(default_factory=list)
    family: str | None = None
    tool_call_expectations: list[JsonDict] = field(default_factory=list)
    milestones: list[JsonDict] = field(default_factory=list)
    warmup_ref: str | None = None
    semantic_forbidden: list[str] = field(default_factory=list)
    expected_refusal: str | None = None
    ablation_pair: JsonDict | None = None
    representative_query_id: str | None = None

    @classmethod
    def from_dict(cls, data: JsonDict) -> "ValidationScenario":
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            persona=str(data.get("persona", "expert_validator")),
            request=dict(data.get("request", {})),
            frozen_documents=list(data.get("frozen_documents", [])),
            expected=dict(data.get("expected", {})),
            rubrics=dict(data.get("rubrics", {})),
            tags=list(data.get("tags", [])),
            reference_trajectory=list(data.get("reference_trajectory", [])),
            family=data.get("family"),
            tool_call_expectations=list(data.get("tool_call_expectations", [])),
            milestones=list(data.get("milestones", [])),
            warmup_ref=data.get("warmup_ref"),
            semantic_forbidden=list(data.get("semantic_forbidden", [])),
            expected_refusal=data.get("expected_refusal"),
            ablation_pair=data.get("ablation_pair"),
            representative_query_id=data.get("representative_query_id"),
        )

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class RunArtifact:
    """Captured output from one Hinaing evaluation run."""

    scenario_id: str
    run_id: str
    mode: str
    started_at: str
    completed_at: str | None = None
    request: JsonDict = field(default_factory=dict)
    snapshot_response: JsonDict | None = None
    chat_response: JsonDict | None = None
    metrics: JsonDict = field(default_factory=dict)
    progress_events: list[JsonDict] = field(default_factory=list)
    trajectory: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    adapter: str = "unknown"
    tool_events: list[JsonDict] = field(default_factory=list)
    node_order_observed: list[str] = field(default_factory=list)
    independent_grading: JsonDict = field(default_factory=dict)
    influence_ranking: list[JsonDict] = field(default_factory=list)

    @classmethod
    def start(cls, scenario: ValidationScenario, mode: str, run_id: str, adapter: str) -> "RunArtifact":
        return cls(
            scenario_id=scenario.id,
            run_id=run_id,
            mode=mode,
            adapter=adapter,
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            request=scenario.request,
        )

    @classmethod
    def from_dict(cls, data: JsonDict) -> "RunArtifact":
        return cls(
            scenario_id=str(data["scenario_id"]),
            run_id=str(data["run_id"]),
            mode=str(data.get("mode", "unknown")),
            started_at=str(data.get("started_at", "")),
            completed_at=data.get("completed_at"),
            request=dict(data.get("request", {})),
            snapshot_response=data.get("snapshot_response"),
            chat_response=data.get("chat_response"),
            metrics=dict(data.get("metrics", {})),
            progress_events=list(data.get("progress_events", [])),
            trajectory=list(data.get("trajectory", [])),
            errors=list(data.get("errors", [])),
            warnings=list(data.get("warnings", [])),
            adapter=str(data.get("adapter", "unknown")),
            tool_events=list(data.get("tool_events", [])),
            node_order_observed=list(data.get("node_order_observed", [])),
            independent_grading=dict(data.get("independent_grading", {})),
            influence_ranking=list(data.get("influence_ranking", [])),
        )

    def finish(self) -> None:
        self.completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class SectionScore:
    """One scorecard section, rated numerically."""

    id: str
    label: str
    weight: float
    raw_score: float
    weighted_score: float
    evidence: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    applicable: bool = True

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class ScenarioScore:
    """All section scores for one scenario run."""

    scenario_id: str
    run_id: str
    total_score: float
    sections: list[SectionScore]
    passed: bool
    critical_failures: list[str] = field(default_factory=list)
    family: str | None = None
    failure_modes: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        data["sections"] = [section.to_dict() for section in self.sections]
        return data


@dataclass(slots=True)
class ValidationScorecard:
    """Aggregate thesis validation scorecard."""

    generated_at: str
    total_score: float
    readiness_label: str
    scenario_scores: list[ScenarioScore]
    section_averages: list[SectionScore]
    preflight: JsonDict = field(default_factory=dict)
    research_basis: list[JsonDict] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    bootstrap_ci: JsonDict = field(default_factory=dict)
    family_breakdown: list[JsonDict] = field(default_factory=list)
    failure_modes: JsonDict = field(default_factory=dict)  # kept for back-compat
    observed_issue_categories: JsonDict = field(default_factory=dict)
    agent_influence: list[JsonDict] = field(default_factory=list)
    ablation_deltas: list[JsonDict] = field(default_factory=list)
    judge_available: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "generated_at": self.generated_at,
            "total_score": self.total_score,
            "readiness_label": self.readiness_label,
            "scenario_scores": [score.to_dict() for score in self.scenario_scores],
            "section_averages": [score.to_dict() for score in self.section_averages],
            "preflight": self.preflight,
            "research_basis": self.research_basis,
            "assumptions": self.assumptions,
            "bootstrap_ci": self.bootstrap_ci,
            "family_breakdown": self.family_breakdown,
            "failure_modes": self.failure_modes,
            "observed_issue_categories": self.observed_issue_categories,
            "agent_influence": self.agent_influence,
            "ablation_deltas": self.ablation_deltas,
            "judge_available": self.judge_available,
        }
