# AgenticHinaing Evaluation Framework

External evaluation framework for the Hinaing webapp in `../Hinaing`.

The framework observes real agent execution and scores it against a 100-point, research-grounded scorecard.

## What it evaluates

- **Sentiment Generator flow:** `POST /insights/snapshot`
- **Chat Analyze flow:** `POST /chat/analyze/start` + `/chat/analyze/status/{task_id}` polling
- **Fixture-mode backend execution** through `generate_snapshot(..., pre_retrieved_documents=...)`
- **Real trajectory capture** from Hinaing's `progress_callback(stage, message, progress)` hook
- **Independent groundedness** via a Claude judge (optional) instead of the system's self-reported verifier
- **CAIR-style counterfactual Agent Influence Ranking** for per-node attribution
- **Ablation deltas** comparing `ablation_preset="full"` vs `"ablated"`
- **Smart Reuse / cache reuse**, date-window gating, hyperlocal specificity, robustness, cost, and implementation readiness

## Why a custom harness?

Agentic system validation is not a solved problem with a drop-in tool. Independent surveys and first-party engineering posts from 2025–2026 converge on the same conclusion: evaluation must be tailored to the specific agent's architecture, failure modes, and deployment surface.

- **General-purpose benchmarks fall short.** [Evaluation and Benchmarking of LLM Agents: A Survey (arXiv:2507.21504, Jul 2025)](https://arxiv.org/html/2507.21504v1) and [General Agent Evaluation (arXiv:2602.22953)](https://arxiv.org/html/2602.22953v1) both document that existing benchmarks target narrow domains (SWE-Bench, WebArena, τ-bench, BrowserGym, etc.) and that unifying frameworks like HAL still require per-benchmark agent adapters. [Benchmark Test-Time Scaling of General LLM Agents (arXiv:2602.18998)](https://arxiv.org/html/2602.18998) quantifies a substantial robustness gap when agents are evaluated outside the domain they were tuned for.
- **Rigorous agentic benchmarks are a research topic in their own right.** [Establishing Best Practices for Building Rigorous Agentic Benchmarks (arXiv:2507.02825, Jul 2025)](https://arxiv.org/pdf/2507.02825) reviews 20+ recent agent benchmarks and finds systematic methodological gaps — flaky grading, under-specified tasks, insufficient adversarial coverage — concluding that practitioners should expect to build custom evaluation infrastructure rather than reuse existing suites unchanged. [CocoaBench (arXiv:2604.11201)](https://arxiv.org/html/2604.11201) and [MCP-AgentBench (arXiv:2509.09734)](https://arxiv.org/pdf/2509.09734) echo this: existing benchmarks are single-domain or single-interaction-mode and cannot systematically evaluate composed agentic systems.
- **Industry engineering guidance explicitly prescribes custom harnesses.** Anthropic's [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) and [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps) both frame the harness as a first-class engineering artifact specific to the application. OpenAI takes the same position in [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/). Martin Fowler's [Harness engineering](https://martinfowler.com/articles/harness-engineering.html) summarises the emerging consensus that agentic harnesses are the new integration point between model providers and real applications.
- **In practice, harnesses are rewritten as models change.** Field reports cited in [The importance of Agent Harness in 2026 (Schmid)](https://www.philschmid.de/agent-harness-2026) — including [Manus rewriting their agent harness five times in six months](https://aakashgupta.medium.com/2025-was-agents-2026-is-agent-harnesses-heres-why-that-changes-everything-073e9877655e) — show that even teams who do not change the underlying model must keep the harness malleable to track architectural shifts. A frozen, generic benchmark cannot follow those shifts.

This framework exists because the Hinaing agentic snapshot pipeline is a specific LangGraph topology (7 named nodes, a custom `progress_callback` contract, a `pre_retrieved_documents` fixture bypass, a `PipelineMetrics` collector, VSEE + Smart Reuse + faithfulness verifier novelties, and an `ablation_preset` switch) that no general-purpose benchmark can observe at the granularity the thesis requires. The scorecard wires six existing research instruments (AgentDiagnose, TRAJECT-Bench, ToolSandbox, AgentHarm, CAIR, One-Eval) into a harness that is specific to this system, rather than trying to force the system onto a benchmark that was designed for something else.

## Research basis

The 100-pt scorecard maps Hinaing evidence onto recent agentic evaluation research:

- **AgentDiagnose** (EMNLP 2025): agent trajectory diagnosis and competencies
- **TRAJECT-Bench** (ICLR 2026): tool-use trajectory selection, arguments, and order
- **ToolSandbox** (NAACL Findings 2025): stateful tool execution and milestone evaluation
- **AgentHarm** (ICLR 2025): multi-step agent safety and adversarial robustness
- **Agentic AI Survey** (AIR 2026): composite coverage (success, memory, tools, robustness, cost, latency, auditability)
- **One-Eval** (arXiv:2603.09821): hierarchical macro / diagnostic / micro reporting with artifact preservation
- **CAIR** (arXiv:2510.25612): counterfactual Agent Influence Ranking via FOC / AOC / WC / AF / OC
- **AAW-Zoo** (arXiv:2510.25612): disciplined scenario construction with representative-query mappings
- **LangChain AgentEvals**: optional trajectory-match implementation support

### Direct citations

These are the load-bearing references the harness is built on. BibTeX is provided so you can drop these into a thesis or appendix verbatim.

#### Scorecard methodology — hierarchical diagnostic reporting + artifact preservation

```bibtex
@misc{shen2026oneevalagenticautomatedtraceable,
      title={One-Eval: An Agentic System for Automated and Traceable LLM Evaluation},
      author={Chengyu Shen and Yanheng Hou and Minghui Pan and Runming He and Zhen Hao Wong and Meiyi Qiang and Zhou Liu and Hao Liang and Peichao Lai and Zeang Sheng and Wentao Zhang},
      year={2026},
      eprint={2603.09821},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.09821},
}
```

Used for: three-tier macro / diagnostic / micro views in `report.py`, failure-mode classifier (`FAILURE_MODES` in `evaluators.py`), and per-run artifact preservation under `reports/runs/<run_id>/`.

#### Agent Attribution — counterfactual per-node influence ranking

```bibtex
@misc{giloni2025counterfactualbasedagentinfluenceranker,
      title={Counterfactual-based Agent Influence Ranker for Agentic AI Workflows},
      author={Amit Giloni and Chiara Picardi and Roy Betser and Shamik Bose and Aishvariya Priya Rathina Sabapathy and Roman Vainshtein},
      year={2025},
      eprint={2510.25612},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.25612},
}
```

Used for: `agentic_hinaing_eval/agent_influence.py` (FOC / AOC / WC / AF / OC metrics), the CAIR counterfactual patch strategies, 1-SFD stability check, and the Agent Attribution scorecard section (10 of the 100 pts). The same paper introduces the AAW-Zoo discipline that informed the 46-scenario split.

**Adaptation notes vs. the reference implementation ([FujitsuResearch/CAIR](https://github.com/FujitsuResearch/CAIR)):**

- **Patching strategy.** CAIR uses an LLM (Azure OpenAI, seed=42) to generate *plausible* counterfactual node outputs. Our implementation uses deterministic structural mutations (`drop` = zero output keys, `shuffle` = reverse list order). Deterministic patches are more reproducible and don't require an additional LLM API, but produce more extreme perturbations — raw FOC/AOC magnitudes are not directly comparable to CAIR's published numbers. Rankings (the thesis signal) are comparable since they measure relative node importance.
- **OC weights.** CAIR uses a binary (0.6 output, 0.4 workflow) split. Our default is a four-component (0.4 FOC, 0.2 AOC, 0.3 WC, 0.1 AF·FOC) decomposition that gives finer granularity. Pass `--cair-weights 0.6,0.0,0.4,0.0` to match the paper's weights exactly.
- **Embedding model.** We use the same `all-mpnet-base-v2` SBERT model as CAIR for cosine distance (FOC/AOC). Falls back to `all-MiniLM-L6-v2` if the larger model is unavailable.
- **Baseline capture.** CAIR uses Langfuse callbacks; we use Hinaing's native `progress_callback` — no external tracing infrastructure needed.
- **Online phase.** CAIR includes an online phase that maps runtime queries to training-set representatives via cosine similarity + t-SNE. This is a deployment-monitoring feature; we omit it as the thesis requires offline evaluation only.

#### Trajectory-first evaluation

```bibtex
@inproceedings{agentdiagnose2025,
  title     = {AgentDiagnose: Competency-Based Trajectory Diagnosis for LLM Agents},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  year      = {2025},
  url       = {https://aclanthology.org/2025.emnlp-demos.15/},
}

@inproceedings{trajectbench2026,
  title     = {TRAJECT-Bench: A Benchmark for Tool-Use Trajectories},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=TZWnWvsQ0X},
}
```

Used for: `_score_trajectory` in `evaluators.py`, `agentevals_bridge.agentevals_trajectory_score`, and the reference-trajectory invariants asserted in `tests/test_scenarios_schema.py`.

#### Stateful tool execution + milestones

```bibtex
@inproceedings{toolsandbox2025,
  title     = {ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use},
  booktitle = {Findings of the Association for Computational Linguistics: NAACL 2025},
  year      = {2025},
  url       = {https://aclanthology.org/2025.findings-naacl.65/},
}
```

Used for: the `milestones` field on `ValidationScenario` (predicate-style intermediate checks) and the cache-family warmup→hit pair structure in `data/scenarios/cache_reuse_v1.jsonl`.

#### Adversarial safety families

```bibtex
@inproceedings{agentharm2025,
  title     = {AgentHarm: Measuring Harm and Refusal Behavior in LLM Agents},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=AC5n7xHuR1},
}
```

Used for: `data/scenarios/adversarial_v1.jsonl` (10 prompt-injection / impersonation / exfiltration / jailbreak / stale-as-current / geographic-hijack scenarios with paraphrase-sensitive `semantic_forbidden` and `expected_refusal` fields).

#### Composite evaluation coverage

```bibtex
@article{agenticaisurvey2026,
  title   = {Agentic AI: A Survey of Capabilities, Evaluation, and Open Challenges},
  journal = {Artificial Intelligence Review},
  year    = {2026},
  doi     = {10.1007/s10462-025-11422-4},
  url     = {https://link.springer.com/article/10.1007/s10462-025-11422-4},
}
```

Used for: the seven-axis coverage of the scorecard (task success, memory, tool proficiency, robustness, cost, latency, auditability) and the assumption list rendered in the expert validation form.

#### Optional implementation support

```bibtex
@misc{langchain_agentevals,
  title  = {LangChain AgentEvals},
  author = {{LangChain}},
  year   = {2024},
  url    = {https://docs.langchain.com/langsmith/trajectory-evals},
}
```

Used for: the fallback `agentevals` trajectory-match evaluator inside `agentic_hinaing_eval/agentevals_bridge.py`; deterministic fallback is used when the optional dep is absent.

#### Why a custom harness (see section above)

```bibtex
@misc{llmagenteval_survey2025,
  title         = {Evaluation and Benchmarking of LLM Agents: A Survey},
  year          = {2025},
  eprint        = {2507.21504},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2507.21504},
}

@misc{rigorous_agentic_benchmarks2025,
  title         = {Establishing Best Practices for Building Rigorous Agentic Benchmarks},
  year          = {2025},
  eprint        = {2507.02825},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2507.02825},
}
```

## Section weights (100 pts)

| Section | Weight |
|---|---:|
| Objective Quality & Civic Usefulness | 18 |
| Trajectory & Tool Correctness | 18 |
| State / Memory / Cache Behavior | 13 |
| Groundedness & Self-Verification | 14 |
| Temporal / Hyperlocal Constraint Handling | 9 |
| Robustness & Safety | 10 |
| Efficiency & Implementation Readiness | 8 |
| Agent Attribution (CAIR Counterfactual) | 10 |

## Install the Hinaing backend env

Fixture mode imports Hinaing's graph code directly. The orchestrator script (`scripts/test_local_env.sh`) handles this automatically, but for manual setup:

```bash
poetry -C ../Hinaing/backend env use python3.11
scripts/setup_backend_env.sh ../Hinaing
```

Use Python 3.11. Python 3.14 currently fails in Hinaing's `qdrant-client` / `protobuf` import path.

## Running the evaluation

The orchestrator script `scripts/test_local_env.sh` is the recommended way to run everything. It handles Qdrant startup, backend warmup (HuggingFace model downloads, Qdrant collection creation), deep readiness checks, and teardown.

### Prerequisites

1. A `.keys.env` file in the eval root (sourced automatically):
   ```
   GROQ_API_KEY=gsk_...
   GEMINI_API_KEY=AI...
   ANTHROPIC_API_KEY=sk-ant-...   # optional, enables independent Claude judge
   ```
   To bypass Groq's 500K TPD free-tier limit, add `LLM_FORCE_OPENROUTER=true` and `OPENROUTER_API_KEY=sk-or-...` to the backend's `.env` — this transparently routes all Groq calls through OpenRouter using the same Llama models.

2. Docker or Podman, Poetry, uv, Python 3.11.

### Commands

```bash
# Start Qdrant + backend, run warmup, verify health
scripts/test_local_env.sh up

# Run the full 46-scenario fixture suite + scoring
scripts/test_local_env.sh eval

# With CAIR counterfactual Agent Attribution (slower, runs patched variants)
COUNTERFACTUAL=1 scripts/test_local_env.sh eval

# Limit counterfactual to specific scenarios (supports wildcards)
COUNTERFACTUAL=1 COUNTERFACTUAL_SCENARIOS="BG-001,ADV-*" scripts/test_local_env.sh eval

# Check platform health / re-run warmup
scripts/test_local_env.sh verify
scripts/test_local_env.sh warmup

# Tail backend logs
scripts/test_local_env.sh logs

# Print endpoint summary
scripts/test_local_env.sh status

# Stop everything (Qdrant container + backend)
scripts/test_local_env.sh down
```

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `HINAING_ROOT` | `../Hinaing` | Path to the Hinaing project |
| `QDRANT_PORT` | `6333` | Qdrant container port |
| `BACKEND_PORT` | `8000` | Uvicorn port |
| `LLM_PROVIDER` | `groq` | `groq`, `openrouter`, or `gemini` |
| `KEYS_FILE` | `.keys.env` | API keys file |
| `COUNTERFACTUAL` | `0` | `1` to enable CAIR counterfactual runs |
| `COUNTERFACTUAL_SCENARIOS` | *(one per family)* | Comma-separated IDs or wildcards (`all` for every scenario) |
| `COUNTERFACTUAL_PATCHES` | `drop,shuffle` | Patch strategies for counterfactual |
| `DOCKER_NETWORK` | `podman` | Set to `""` for native Docker |

For individual `hinaing-eval` subcommands (preflight, run-fixtures, score, run-http, etc.), see [docs/command_reference.md](docs/command_reference.md).

## Scenario suite (46 scenarios across 5 families)

All JSONL files under `data/scenarios/` are loaded automatically:

- `hyperlocal_baguio_v2.jsonl` (10) — canonical civic/tourism/safety scenarios
- `cache_reuse_v1.jsonl` (8) — warmup + cache-hit pairs; follow-ups expect `metrics.documents_cached > 0`
- `adversarial_v1.jsonl` (10) — AgentHarm-style families: prompt injection, impersonation, exfiltration, PII invention, jailbreak, stale-as-current, contradictory numerics, geographic hijack
- `missing_data_v1.jsonl` (6) — empty / irrelevant / stale-only corpora; rewards graceful degradation
- `ablation_pairs_v1.jsonl` (12) — 6 pairs executed as `ablation_preset="full"` vs `"ablated"`; delta table rendered in the form

Every scenario declares a `family`, optional `milestones`, `warmup_ref`, `semantic_forbidden`, `expected_refusal`, and `ablation_pair` so evaluators can target them without keyword matching.

## Tests

```bash
uv run pytest
```

Coverage includes trajectory capture, LLM-judge fallback behavior, scenario-schema invariants (no legacy trajectories, valid families, ablation-pair integrity), CAIR metric math, and the 100-pt weight invariant.
