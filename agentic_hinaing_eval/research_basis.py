from __future__ import annotations

from .models import JsonDict


RESEARCH_BASIS: list[JsonDict] = [
    {
        "name": "AgentDiagnose",
        "venue": "EMNLP 2025 Demo",
        "url": "https://aclanthology.org/2025.emnlp-demos.15/",
        "used_for": "Agent competency diagnosis: objective quality, decomposition, observation reading, self-verification, and exploration/backtracking.",
    },
    {
        "name": "TRAJECT-Bench",
        "venue": "ICLR 2026",
        "url": "https://openreview.net/forum?id=TZWnWvsQ0X",
        "used_for": "Tool-use trajectory scoring: tool selection, argument correctness, dependency order, and route correctness.",
    },
    {
        "name": "ToolSandbox",
        "venue": "NAACL Findings 2025",
        "url": "https://aclanthology.org/2025.findings-naacl.65/",
        "used_for": "Stateful tool execution, implicit state dependencies, intermediate milestones, and final milestone evaluation.",
    },
    {
        "name": "AgentHarm",
        "venue": "ICLR 2025",
        "url": "https://openreview.net/forum?id=AC5n7xHuR1",
        "used_for": "Multi-step agent safety, adversarial prompt robustness, and harmful-task refusal behavior.",
    },
    {
        "name": "Agentic AI Survey",
        "venue": "Artificial Intelligence Review 2026",
        "url": "https://link.springer.com/article/10.1007/s10462-025-11422-4",
        "used_for": "Composite agent evaluation coverage: task success, memory management, tool proficiency, robustness, cost, latency, and auditability.",
    },
    {
        "name": "LangChain AgentEvals",
        "venue": "Implementation framework",
        "url": "https://docs.langchain.com/langsmith/trajectory-evals",
        "used_for": "Optional trajectory-match and LLM-as-judge implementation support.",
    },
    {
        "name": "One-Eval",
        "venue": "OpenDCAI / arXiv:2603.09821",
        "url": "https://github.com/OpenDCAI/One-Eval",
        "used_for": "Hierarchical diagnostic reporting (macro / diagnostic / micro), artifact preservation for auditable evidence trails, and per-failure-mode classifiers.",
    },
    {
        "name": "CAIR (Counterfactual Agent Influence Ranker)",
        "venue": "Fujitsu Research / arXiv:2510.25612",
        "url": "https://github.com/FujitsuResearch/CAIR",
        "used_for": "Per-agent attribution via counterfactual patching with FOC / AOC / WC / AF / OC metrics; Agent Attribution section.",
    },
    {
        "name": "AAW-Zoo",
        "venue": "arXiv:2510.25612",
        "url": "https://arxiv.org/abs/2510.25612",
        "used_for": "Disciplined scenario construction: representative query mapping and dedicated adversarial/guardrail families.",
    },
]


# 100-pt scorecard weights. Adjusted to include CAIR-style Agent Attribution.
SECTION_WEIGHTS: dict[str, tuple[str, float]] = {
    "objective_quality": ("Objective Quality And Civic Usefulness", 18.0),
    "trajectory": ("Trajectory And Tool Correctness", 18.0),
    "memory_cache": ("State, Memory, And Cache Behavior", 13.0),
    "groundedness": ("Groundedness And Self-Verification", 14.0),
    "temporal_hyperlocal": ("Temporal And Hyperlocal Constraint Handling", 9.0),
    "robustness_safety": ("Robustness And Safety", 10.0),
    "efficiency_readiness": ("Efficiency And Implementation Readiness", 8.0),
    "agent_attribution": ("Agent Attribution (CAIR Counterfactual)", 10.0),
}
