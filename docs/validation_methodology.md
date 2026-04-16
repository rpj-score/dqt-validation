# Validation Methodology

## Intent

This framework validates whether AgenticHinaing is operationally successful as a proof-of-concept for a thesis defense. It is not a new psychometric instrument. It is an applied scorecard that maps Hinaing evidence to recent agentic evaluation research and produces a numerical expert-signable validation tool.

## Why A 100-Point Scorecard

The school requirement asks for a validation tool that:

- is based on existing research recommended by an expert,
- is numerical,
- is filled out and signed by the expert.

The scorecard therefore uses a weighted 100-point system. Each section is rated from 0 to 5, then converted into its section weight. The expert may adjust the raw score after reviewing generated evidence, but should preserve the section definitions and weights so results remain comparable across runs.

## Section Mapping

| Scorecard Section | Weight | Research Basis | Hinaing Evidence |
|---|---:|---|---|
| Objective Quality And Civic Usefulness | 20 | AgentDiagnose objective quality and task decomposition | Summary, insights, alerts, persona-specific usefulness |
| Trajectory And Tool Correctness | 20 | TRAJECT-Bench and AgentEvals trajectory matching | Node/tool trajectory, route used, query plan, progress events |
| State, Memory, And Cache Behavior | 15 | ToolSandbox statefulness and agentic AI memory evaluation | Smart Reuse metrics, Qdrant memory recall, cached/fresh document counts |
| Groundedness And Self-Verification | 15 | AgentDiagnose self-verification plus agentic AI auditability | Faithfulness, citation accuracy, hallucination metrics, source evidence |
| Temporal And Hyperlocal Constraint Handling | 10 | Tool argument correctness and state-dependent constraints | `time_window`, `after:YYYY-MM-DD`, source dates, Baguio/locality terms |
| Robustness And Safety | 10 | AgentHarm and robustness evaluation | Prompt-injection resistance, stale data, missing data, harmful/unsupported claims |
| Efficiency And Implementation Readiness | 10 | Agentic AI cost/latency/readiness evaluation | Latency, errors, fallbacks, dependency readiness, metrics persistence |

## Evidence Levels

- **Reproducible fixture evidence**: frozen documents passed through `pre_retrieved_documents`; use this for official scoring.
- **Live HTTP evidence**: webapp route tests against a running backend; use this for operational smoke evidence.
- **Preflight evidence**: static and environment checks; use this to explain readiness blockers.
- **Expert judgment**: final human review and signature; required by the thesis grading criteria.

## Current Readiness Gate

The current Hinaing tree exposes the expected frontend and backend entrypoints, but the preflight detects one critical integration issue:

`backend/app/services/insights/nodes.py::orchestrate_queries()` calls `QueryOrchestratorAgent.run(request, ablation_config=ablation)`, while `QueryOrchestratorAgent.run()` accepts only `request`.

Until that is fixed in Hinaing, the validation tool should treat ReAct query orchestration as not operationally demonstrated, even if downstream fallback behavior still produces output.

