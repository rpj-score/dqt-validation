# Command Reference

Individual `hinaing-eval` subcommands for when you need fine-grained control. For most workflows, the orchestrator script `scripts/test_local_env.sh` (documented in the main README) is the recommended entry point — it handles Qdrant, backend lifecycle, warmup, and teardown automatically.

## Preflight

```bash
# Static preflight (fast, checks files/paths). Add --python-cmd to also verify
# the Hinaing backend venv can import the graph.
uv run hinaing-eval preflight \
  --python-cmd "poetry run python"
```

## HTTP health check

```bash
# Against a running backend (no graph execution)
uv run hinaing-eval check-http --api-base https://donyelqt-hinaing-backend.hf.space
```

## Generate blank form

```bash
# Generate a blank validation form (no runs yet)
uv run hinaing-eval generate-form --output reports/validation_tool.md
```

## Run fixtures

```bash
# Run the 46-scenario fixture suite against Hinaing's real graph
uv run hinaing-eval run-fixtures \
  --python-cmd "poetry run python" \
  --output reports/fixture_runs.jsonl
```

## Score

```bash
# Score runs with the independent Claude judge (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=... uv run hinaing-eval score \
  --runs reports/fixture_runs.jsonl \
  --llm-judge \
  --judge-model claude-sonnet-4-6 \
  --output reports/scorecard.json \
  --form reports/validation_tool_filled.md \
  --runs-dir reports/runs

# Score without the judge (heuristic groundedness + keyword safety only)
uv run hinaing-eval score \
  --runs reports/fixture_runs.jsonl \
  --no-llm-judge \
  --output reports/scorecard.json \
  --form reports/validation_tool_filled.md
```

## Re-score from cache (no new runs)

```bash
# Use existing runs + judge cache to regenerate reports
uv run hinaing-eval score \
  --runs reports/fixture_runs.jsonl \
  --llm-judge \
  --output reports/scorecard.json \
  --form reports/validation_tool_filled.md \
  --runs-dir reports/runs
```

## Generate summary (one-page) from existing scorecard

```bash
uv run hinaing-eval score \
  --runs reports/fixture_runs.jsonl \
  --llm-judge \
  --output reports/scorecard.json \
  --form reports/validation_tool_filled.md \
  --summary reports/validation_summary.md \
  --summary-pdf reports/validation_summary.pdf
```

## Live HTTP mode

With Hinaing running locally or on the HF Space:

```bash
uv run hinaing-eval run-http --api-base http://localhost:8000 --output reports/http_runs.jsonl

uv run hinaing-eval check-http --api-base https://donyelqt-hinaing-backend.hf.space
uv run hinaing-eval run-http \
  --api-base https://donyelqt-hinaing-backend.hf.space \
  --output reports/hf_http_runs.jsonl
```

Keep live HTTP results separate from frozen-fixture results — live retrieval changes over time. The HTTP snapshot endpoint is non-streaming, so the Trajectory section is marked not-applicable for HTTP snapshot runs rather than silently passing.

Use `--timeout` for long-running full-pipeline requests; the connect timeout stays short so unavailable backends fail quickly.

## All-in-one

```bash
# Run preflight, fixtures, and scoring in one pass
uv run hinaing-eval all \
  --python-cmd "poetry run python" \
  --llm-judge \
  --output reports/scorecard.json \
  --form reports/validation_tool_filled.md \
  --runs-dir reports/runs
```
