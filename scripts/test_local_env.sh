#!/usr/bin/env bash
# Spin up a bulletproof local environment for the Hinaing backend and run the
# eval against it.
#
# Boot sequence this script guarantees:
#   1) Preflight — binaries, ports, API keys, backend source tree.
#   2) Qdrant — in-memory (tmpfs) container, ready-waited.
#   3) Warmup — pre-create the `baguio_documents` Qdrant collection via the
#      backend's VectorStore; pre-download HuggingFace weights so the first
#      scenario doesn't pay a 3 GB cold-download tax.
#   4) Backend — uvicorn with deep readiness: wait for /health AND for the
#      "[startup] Model preloading complete" log line (BGE + RoBERTa + NLI).
#   5) Verify — smoke-test /health, /metrics/summary, and run the eval's
#      preflight + check-http.
#   6) Ready — drop into status summary.
#
# Supabase: INTENTIONALLY NOT USED. The Hinaing backend imports the supabase
# package transitively but never calls it; `app/core/config.py` defines
# supabase_url/service_role/anon_key but no code path reads them. The backend
# persists only to Qdrant and local-disk JSONL under `backend/data/metrics/`.
#
# LangSmith: NOT required. backend/app/core/config.py only reads the key as
# an optional field; no tracer is wired up. Leave blank.
#
# LLM Provider Reality:
#   The LLM_PROVIDER / LLM_PROVIDER_* env vars are written to the backend .env
#   and consumed by the factory in backend/app/services/llm/factory.py. HOWEVER,
#   as of the current backend code, NO AGENT ACTUALLY CALLS THE FACTORY.
#   Each agent hardcodes its own provider:
#
#     Node 1 (QueryOrchestrator) → ChatGoogleGenerativeAI (GEMINI_API_KEY)
#     Node 4 (SentimentAgent)    → GroqProvider directly   (GROQ_API_KEY)
#     Node 4 (CredibilityAgent)  → GroqProvider directly   (GROQ_API_KEY)
#     Node 6 (ThemeAgent)        → GroqProvider directly   (GROQ_API_KEY)
#     Node 7 (CoordinatorAgent)  → LLMNarrativeClient→Groq (GROQ_API_KEY)
#
#   This means:
#     - GROQ_API_KEY is REQUIRED for Nodes 4, 6, 7 (the core pipeline)
#     - GEMINI_API_KEY is needed for Node 1 only; without it, Node 1 fails
#       gracefully (0 queries generated, pipeline uses default focus-area keywords)
#     - OPENROUTER_API_KEY is currently unused by any agent code — the factory
#       supports it but nothing calls the factory
#     - Setting LLM_PROVIDER=openrouter has NO EFFECT on the actual LLM calls
#
#   The script still writes LLM_PROVIDER_* to .env for forward-compatibility
#   (the student may wire the factory in a future iteration), but the preflight
#   key check is based on the ACTUAL provider requirements above.
#
# Usage:
#   scripts/test_local_env.sh up        # start Qdrant + backend (end-to-end)
#   scripts/test_local_env.sh eval      # run-fixtures + score (starts stack if needed)
#   scripts/test_local_env.sh verify    # re-check platform health
#   scripts/test_local_env.sh warmup    # re-run the warmup (idempotent)
#   scripts/test_local_env.sh status    # print endpoints
#   scripts/test_local_env.sh logs      # tail backend log
#   scripts/test_local_env.sh down      # stop everything
#   scripts/test_local_env.sh restart   # down + up
#
# Optional env (defaults in parens):
#   HINAING_ROOT             (../Hinaing)
#   QDRANT_CONTAINER_NAME    (hinaing-qdrant-test)
#   QDRANT_PORT              (6333)
#   BACKEND_PORT             (8000)
#   LLM_PROVIDER             (groq)    — groq|openrouter|gemini
#   DOCKER_NETWORK           (podman)  — set to "" on native Docker
#   KEYS_FILE                (<eval_root>/.keys.env)
#   MODEL_WAIT_TIMEOUT       (600)     — seconds to wait for model preload
#   BACKEND_READY_TIMEOUT    (120)     — seconds to wait for /health
#   COUNTERFACTUAL           (0)       — 1 to enable CAIR counterfactual Agent Attribution
#   COUNTERFACTUAL_SCENARIOS (unset)   — comma-separated scenario ids (default: one per family)
#   COUNTERFACTUAL_PATCHES   (drop,shuffle) — comma-separated patch strategies

set -euo pipefail

# --- Configuration ----------------------------------------------------------
EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HINAING_ROOT="${HINAING_ROOT:-$(cd "${EVAL_ROOT}/../Hinaing" 2>/dev/null && pwd || echo "${EVAL_ROOT}/../Hinaing")}"
BACKEND_ROOT="${HINAING_ROOT}/backend"
QDRANT_CONTAINER_NAME="${QDRANT_CONTAINER_NAME:-hinaing-qdrant-test}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
LOG_DIR="${EVAL_ROOT}/reports/local_env_logs"
BACKEND_PID_FILE="${LOG_DIR}/backend.pid"
BACKEND_LOG="${LOG_DIR}/backend.log"
WARMUP_LOG="${LOG_DIR}/warmup.log"
LLM_PROVIDER="${LLM_PROVIDER:-groq}"
BACKEND_READY_TIMEOUT="${BACKEND_READY_TIMEOUT:-120}"
MODEL_WAIT_TIMEOUT="${MODEL_WAIT_TIMEOUT:-600}"
COUNTERFACTUAL="${COUNTERFACTUAL:-0}"
COUNTERFACTUAL_SCENARIOS="${COUNTERFACTUAL_SCENARIOS:-}"
COUNTERFACTUAL_PATCHES="${COUNTERFACTUAL_PATCHES:-drop,shuffle}"

# Works on both docker and podman. Default is podman (author's env); docker
# users should export DOCKER_NETWORK="" or omit the flag upstream.
DOCKER_NETWORK="${DOCKER_NETWORK:-podman}"

# --- Keys file (optional) ---------------------------------------------------
KEYS_FILE="${KEYS_FILE:-${EVAL_ROOT}/.keys.env}"
load_keys_file() {
  [ -f "${KEYS_FILE}" ] || return 0
  set -a
  # shellcheck disable=SC1090
  source "${KEYS_FILE}"
  set +a
}

# --- Pretty logging ---------------------------------------------------------
ts() { date -u +"%H:%M:%SZ"; }
log()   { echo -e "\033[1;36m[$(ts)] $*\033[0m" >&2; }
warn()  { echo -e "\033[1;33m[$(ts)] WARN: $*\033[0m" >&2; }
fail()  { echo -e "\033[1;31m[$(ts)] FAIL: $*\033[0m" >&2; exit 1; }
ok()    { echo -e "\033[1;32m[$(ts)] OK:   $*\033[0m" >&2; }
step()  { echo -e "\033[1;35m\n[$(ts)] ▶ $*\033[0m" >&2; }

# --- Signal trap ------------------------------------------------------------
CLEANUP_ON_EXIT=0
on_exit() {
  local code=$?
  if [ ${code} -ne 0 ] && [ ${CLEANUP_ON_EXIT} -eq 1 ]; then
    warn "non-zero exit (${code}) during boot — tearing down partial state"
    backend_stop >/dev/null 2>&1 || true
    qdrant_stop  >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

# --- Preflight --------------------------------------------------------------
need() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is required on PATH. $2"
}

port_in_use() {
  local port="$1"
  # ss first (most modern), then lsof, then nc
  if command -v ss >/dev/null 2>&1; then
    ss -lnt "sport = :${port}" 2>/dev/null | grep -q ":${port}"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${port}" -sTCP:LISTEN -Pn >/dev/null 2>&1
  elif command -v nc >/dev/null 2>&1; then
    nc -z localhost "${port}" >/dev/null 2>&1
  else
    return 1  # can't tell; assume free
  fi
}

preflight() {
  step "preflight: checking prerequisites"
  need docker   "Install Docker or Podman (Podman alias must be 'docker')."
  need poetry   "Install Poetry: https://python-poetry.org/docs/#installation"
  need uv       "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
  need curl     "Install curl."
  need python3  "Install Python 3.11+."

  [ -d "${BACKEND_ROOT}" ] || fail "Hinaing backend not found at ${BACKEND_ROOT}. Set HINAING_ROOT."
  [ -f "${BACKEND_ROOT}/pyproject.toml" ] || fail "Missing ${BACKEND_ROOT}/pyproject.toml"

  local missing=()
  case "${LLM_PROVIDER}" in
    groq)       [ -n "${GROQ_API_KEY:-}"       ] || missing+=("GROQ_API_KEY") ;;
    openrouter) [ -n "${OPENROUTER_API_KEY:-}" ] || missing+=("OPENROUTER_API_KEY") ;;
    gemini)     [ -n "${GEMINI_API_KEY:-}"     ] || missing+=("GEMINI_API_KEY") ;;
    *) fail "Unsupported LLM_PROVIDER=${LLM_PROVIDER} (expected groq|openrouter|gemini)" ;;
  esac
  if [ ${#missing[@]} -gt 0 ]; then
    cat >&2 <<EOF

  Required API key(s) for LLM_PROVIDER=${LLM_PROVIDER} are missing: ${missing[*]}

  Two ways to provide them:

  1) ${EVAL_ROOT}/.keys.env (gitignored). A template exists at .keys.env.example:
         cp .keys.env.example .keys.env
         \$EDITOR .keys.env

  2) Export in your shell:
         export ${missing[0]}=...
         # optional: export ANTHROPIC_API_KEY=... to enable --llm-judge

  Or change LLM_PROVIDER:
         export LLM_PROVIDER=openrouter

EOF
    fail "missing: ${missing[*]}"
  fi

  # optional keys — summarise what we found
  local optional=()
  [ -n "${OPENROUTER_API_KEY:-}" ] && [ "${LLM_PROVIDER}" != "openrouter" ] && optional+=("OPENROUTER_API_KEY")
  [ -n "${GROQ_API_KEY:-}"       ] && [ "${LLM_PROVIDER}" != "groq"       ] && optional+=("GROQ_API_KEY")
  [ -n "${GEMINI_API_KEY:-}"     ] && [ "${LLM_PROVIDER}" != "gemini"     ] && optional+=("GEMINI_API_KEY")
  [ -n "${ANTHROPIC_API_KEY:-}"  ] && optional+=("ANTHROPIC_API_KEY (enables --llm-judge)")
  [ -n "${TAVILY_API_KEY:-}"     ] && optional+=("TAVILY_API_KEY")
  [ -n "${LANGSEARCH_API_KEY:-}" ] && optional+=("LANGSEARCH_API_KEY")
  [ ${#optional[@]} -gt 0 ] && ok "keys: optional present — ${optional[*]}"

  # Ports — bail early instead of silently failing uvicorn later.
  if port_in_use "${BACKEND_PORT}" && ! backend_running; then
    fail "port ${BACKEND_PORT} is in use by another process. Set BACKEND_PORT=... or stop the other process."
  fi
  if port_in_use "${QDRANT_PORT}" && ! qdrant_running; then
    fail "port ${QDRANT_PORT} is in use by another process. Set QDRANT_PORT=... or stop the other process."
  fi

  mkdir -p "${LOG_DIR}"
  ok "preflight clean (LLM_PROVIDER=${LLM_PROVIDER}, hinaing=${HINAING_ROOT})"
}

# --- Qdrant (in-memory via tmpfs) ------------------------------------------
qdrant_running() {
  docker ps --filter "name=^${QDRANT_CONTAINER_NAME}$" --filter "status=running" --format '{{.Names}}' | grep -q "${QDRANT_CONTAINER_NAME}"
}

qdrant_start() {
  step "qdrant: start"
  if qdrant_running; then
    ok "qdrant: already running (${QDRANT_CONTAINER_NAME})"
    return
  fi
  docker rm -f "${QDRANT_CONTAINER_NAME}" >/dev/null 2>&1 || true

  log "qdrant: starting in-memory container on :${QDRANT_PORT} (tmpfs — data evaporates on stop)"
  docker run -d \
    --name "${QDRANT_CONTAINER_NAME}" \
    -p "${QDRANT_PORT}:6333" \
    ${DOCKER_NETWORK:+--network "${DOCKER_NETWORK}"} \
    --tmpfs /qdrant/storage:rw,noexec,nosuid,size=512m \
    -e QDRANT__SERVICE__GRPC_PORT=6334 \
    qdrant/qdrant:latest >/dev/null

  for attempt in $(seq 1 30); do
    # Qdrant exposes /healthz and /readyz on some versions, / on all
    if curl -fsS "http://localhost:${QDRANT_PORT}/readyz" >/dev/null 2>&1 \
      || curl -fsS "http://localhost:${QDRANT_PORT}/healthz" >/dev/null 2>&1 \
      || curl -fsS "http://localhost:${QDRANT_PORT}/" >/dev/null 2>&1; then
      # Deeper check: the REST /collections route actually returns a list
      if curl -fsS "http://localhost:${QDRANT_PORT}/collections" >/dev/null 2>&1; then
        ok "qdrant: ready (http://localhost:${QDRANT_PORT})"
        return
      fi
    fi
    sleep 1
  done
  docker logs --tail 50 "${QDRANT_CONTAINER_NAME}" >&2 || true
  fail "qdrant did not become ready within 30s"
}

qdrant_stop() {
  if qdrant_running; then
    log "qdrant: stopping ${QDRANT_CONTAINER_NAME}"
    docker rm -f "${QDRANT_CONTAINER_NAME}" >/dev/null
  else
    log "qdrant: not running"
  fi
}

# --- Write backend .env -----------------------------------------------------
write_env_file() {
  local env_file="${BACKEND_ROOT}/.env"
  step "config: write ${env_file}"
  if [ -f "${env_file}" ]; then
    cp "${env_file}" "${env_file}.bak.$(date +%s)"
    log "config: backed up existing .env"
  fi

  cat > "${env_file}" <<EOF
# Generated by scripts/test_local_env.sh — edit at your own risk.
APP_NAME=Hinaing (local-eval)
ENVIRONMENT=development
FRONTEND_ORIGIN=http://localhost:3000,http://localhost:3001

# --- Supabase: not used by the backend at runtime. Left blank. ---
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_ANON_KEY=

# --- Qdrant (in-memory via tmpfs container) ---
QDRANT_URL=http://localhost:${QDRANT_PORT}
QDRANT_API_KEY=

# --- LLM provider selection (every node pinned to LLM_PROVIDER=${LLM_PROVIDER}) ---
LLM_PROVIDER_QUERY_ORCHESTRATOR=${LLM_PROVIDER}
LLM_PROVIDER_SENTIMENT=${LLM_PROVIDER}
LLM_PROVIDER_CREDIBILITY=${LLM_PROVIDER}
LLM_PROVIDER_THEME_AGENTS=${LLM_PROVIDER}
LLM_PROVIDER_COORDINATOR=${LLM_PROVIDER}
LLM_ENABLE_FALLBACK=true
LLM_FALLBACK_PROVIDER=${LLM_PROVIDER}
LLM_FORCE_OPENROUTER=${LLM_FORCE_OPENROUTER:-false}

# --- Provider keys — only the active provider's key needs a value ---
GROQ_API_KEY=${GROQ_API_KEY:-}
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
GEMINI_API_KEY=${GEMINI_API_KEY:-}

# --- LangSmith: NOT required; blank unless user explicitly provided one ---
LANGSMITH_API_KEY=${LANGSMITH_API_KEY:-}
LANGSMITH_PROJECT=${LANGSMITH_PROJECT:-pr-local-eval}

# --- Optional web search + claim verification ---
LANGSEARCH_API_KEY=${LANGSEARCH_API_KEY:-}
TAVILY_API_KEY=${TAVILY_API_KEY:-}
GOOGLE_FACT_CHECK_API_KEY=${GOOGLE_FACT_CHECK_API_KEY:-}

# --- Ingestion / socials: disabled for local eval ---
FACEBOOK_APP_ID=
FACEBOOK_APP_SECRET=
FACEBOOK_ACCESS_TOKEN=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=Hinaing/1.0 (local-eval)
APIFY_API_TOKEN=

# --- Concerns memory ---
CONCERNS_MEMORY_ENABLED=true
CONCERNS_MEMORY_TTL_DAYS=7

# --- Ingestion loop ---
INGESTION_INTERVAL_SECONDS=120
INGESTION_REGION_KEYWORDS=["baguio","baguio city","cordillera"]
EOF
  ok "config: wrote ${env_file}"
}

# --- Poetry install (idempotent) -------------------------------------------
backend_install() {
  step "backend: poetry install (idempotent)"
  # Use a marker file keyed on poetry.lock hash so we skip reinstall on repeat.
  local marker="${LOG_DIR}/.poetry_install.sha"
  local lock_hash
  lock_hash="$(sha256sum "${BACKEND_ROOT}/poetry.lock" 2>/dev/null | awk '{print $1}' || echo "")"
  if [ -n "${lock_hash}" ] && [ -f "${marker}" ] && [ "$(cat "${marker}")" = "${lock_hash}" ]; then
    ok "backend: deps already installed (poetry.lock unchanged since last install)"
    return
  fi
  (cd "${BACKEND_ROOT}" && poetry install --no-interaction --no-ansi)
  [ -n "${lock_hash}" ] && echo "${lock_hash}" > "${marker}"
  ok "backend: poetry install complete"
}

# --- Warmup: Qdrant collection + HF model downloads -----------------------
warmup() {
  step "warmup: pre-create Qdrant collection + download HF model weights"
  log "warmup: log → ${WARMUP_LOG} (this downloads ~3GB of BGE + RoBERTa + NLI weights on first run)"

  # Run inside the backend's poetry env. cwd=backend so .env is picked up.
  local py_code
  py_code=$(cat <<'PYEOF'
import sys
import time

t0 = time.monotonic()
print(f"[warmup] python={sys.version.split()[0]}", flush=True)

print("[warmup] loading EmbeddingService (BGE-large, ~3GB, downloads once)…", flush=True)
from app.services.rag.embeddings import get_embedding_service
es = get_embedding_service()
print(f"[warmup] embedding service ready dim={es.embedding_dim}", flush=True)

print("[warmup] loading sentiment model (RoBERTa)…", flush=True)
from app.services.agents.sentiment_agent import get_sentiment_model
get_sentiment_model()
print("[warmup] sentiment model ready", flush=True)

print("[warmup] loading NLI entailment model…", flush=True)
from app.services.verification.entailment_checker import get_entailment_checker
get_entailment_checker(use_gpu=False)
print("[warmup] entailment model ready", flush=True)

print("[warmup] pre-creating Qdrant collection 'baguio_documents'…", flush=True)
from app.services.rag.vector_store import get_vector_store
vs = get_vector_store()
stats = vs.get_stats()
print(f"[warmup] vector store ready: {stats}", flush=True)

dt = time.monotonic() - t0
print(f"[warmup] DONE in {dt:.1f}s", flush=True)
PYEOF
)
  if ! (cd "${BACKEND_ROOT}" && poetry run python -c "${py_code}") | tee "${WARMUP_LOG}"; then
    tail -n 40 "${WARMUP_LOG}" >&2 || true
    fail "warmup failed — see ${WARMUP_LOG}"
  fi
  ok "warmup: models downloaded + Qdrant collection created"
}

# --- Backend launch with deep readiness -----------------------------------
backend_running() {
  [ -f "${BACKEND_PID_FILE}" ] && kill -0 "$(cat "${BACKEND_PID_FILE}" 2>/dev/null)" >/dev/null 2>&1
}

backend_start() {
  step "backend: start uvicorn"
  if backend_running; then
    ok "backend: already running (pid=$(cat "${BACKEND_PID_FILE}"))"
    return
  fi
  : > "${BACKEND_LOG}"
  (cd "${BACKEND_ROOT}" && \
    nohup poetry run uvicorn app.main:app --host 0.0.0.0 --port "${BACKEND_PORT}" \
      >"${BACKEND_LOG}" 2>&1 & echo $! > "${BACKEND_PID_FILE}")
  log "backend: pid=$(cat "${BACKEND_PID_FILE}"), log=${BACKEND_LOG}"

  log "backend: waiting for /health (timeout=${BACKEND_READY_TIMEOUT}s)"
  local t0=${SECONDS}
  while true; do
    if curl -fsS "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
      ok "backend: /health OK after $((SECONDS - t0))s"
      break
    fi
    if ! backend_running; then
      tail -n 80 "${BACKEND_LOG}" >&2 || true
      fail "backend process died — see ${BACKEND_LOG}"
    fi
    if [ $((SECONDS - t0)) -ge "${BACKEND_READY_TIMEOUT}" ]; then
      tail -n 80 "${BACKEND_LOG}" >&2 || true
      fail "backend failed /health within ${BACKEND_READY_TIMEOUT}s"
    fi
    sleep 1
  done

  log "backend: waiting for '[startup] Model preloading complete' (timeout=${MODEL_WAIT_TIMEOUT}s)"
  # Because warmup preloaded HF caches, this is fast on re-runs; only the
  # first warmup suffered the 3 GB download.
  local t1=${SECONDS}
  while true; do
    if grep -qF "[startup] Model preloading complete" "${BACKEND_LOG}" 2>/dev/null; then
      ok "backend: models fully loaded after $((SECONDS - t1))s"
      break
    fi
    if ! backend_running; then
      tail -n 80 "${BACKEND_LOG}" >&2 || true
      fail "backend process died during model preload"
    fi
    if [ $((SECONDS - t1)) -ge "${MODEL_WAIT_TIMEOUT}" ]; then
      warn "backend: model preload did not finish within ${MODEL_WAIT_TIMEOUT}s"
      warn "        inspect ${BACKEND_LOG} — first /insights/snapshot may be slow"
      return
    fi
    # Heartbeat every 15s so the operator sees life.
    if [ $(( (SECONDS - t1) % 15 )) -eq 0 ] && [ $((SECONDS - t1)) -gt 0 ]; then
      log "backend: still preloading models ($((SECONDS - t1))s elapsed)…"
    fi
    sleep 1
  done
}

backend_stop() {
  if backend_running; then
    local pid
    pid="$(cat "${BACKEND_PID_FILE}")"
    log "backend: stopping (pid=${pid})"
    kill "${pid}" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${BACKEND_PID_FILE}"
  ok "backend: stopped"
}

# --- End-to-end verification ------------------------------------------------
verify_platform() {
  step "verify: end-to-end platform health"

  # 1. Qdrant collection exists (created by warmup).
  local col_url="http://localhost:${QDRANT_PORT}/collections/baguio_documents"
  if curl -fsS "${col_url}" >/dev/null 2>&1; then
    ok "verify: Qdrant collection 'baguio_documents' exists"
  else
    fail "verify: Qdrant collection missing — run 'scripts/test_local_env.sh warmup'"
  fi

  # 2. Backend health + metrics summary (no LLM, no Qdrant).
  curl -fsS "http://localhost:${BACKEND_PORT}/health" >/dev/null \
    || fail "verify: backend /health failed"
  ok "verify: backend /health OK"
  if curl -fsS "http://localhost:${BACKEND_PORT}/metrics/summary" >/dev/null; then
    ok "verify: backend /metrics/summary OK"
  else
    warn "verify: /metrics/summary unavailable (non-fatal)"
  fi

  # 3. Eval-harness static preflight.
  if (cd "${EVAL_ROOT}" && uv run --extra judge --extra agent --extra pdf hinaing-eval --hinaing-root "${HINAING_ROOT}" preflight \
        --python-cmd "poetry run python" >/dev/null 2>&1); then
    ok "verify: eval preflight clean (backend venv imports generate_snapshot)"
  else
    warn "verify: eval preflight reported issues — rerun with 'uv run --extra judge --extra agent --extra pdf hinaing-eval preflight --python-cmd \"poetry run python\"' from ${EVAL_ROOT} for details"
  fi

  # 4. Eval check-http end-to-end smoke.
  (cd "${EVAL_ROOT}" && uv run --extra judge --extra agent --extra pdf hinaing-eval --hinaing-root "${HINAING_ROOT}" check-http \
      --api-base "http://localhost:${BACKEND_PORT}" >/dev/null 2>&1) \
    && ok "verify: eval check-http passed" \
    || warn "verify: eval check-http failed — see reports/http_health.json"
}

# --- Eval harness wiring ----------------------------------------------------
run_eval() {
  step "eval: run-fixtures against real graph"
  # Build optional counterfactual flags from env.
  local cf_args=()
  case "${COUNTERFACTUAL}" in
    1|true|TRUE|yes|on)
      cf_args+=(--counterfactual)
      [ -n "${COUNTERFACTUAL_SCENARIOS}" ] && cf_args+=(--counterfactual-scenarios "${COUNTERFACTUAL_SCENARIOS}")
      [ -n "${COUNTERFACTUAL_PATCHES}"   ] && cf_args+=(--counterfactual-patches   "${COUNTERFACTUAL_PATCHES}")
      log "eval: counterfactual ENABLED (scenarios=${COUNTERFACTUAL_SCENARIOS:-one-per-family}, patches=${COUNTERFACTUAL_PATCHES})"
      ;;
    *)
      log "eval: counterfactual disabled (set COUNTERFACTUAL=1 to enable CAIR Agent Attribution)"
      ;;
  esac

  (cd "${EVAL_ROOT}" && uv run --extra judge --extra agent --extra pdf hinaing-eval \
    --hinaing-root "${HINAING_ROOT}" \
    run-fixtures \
    --python-cmd "poetry run python" \
    --output reports/fixture_runs.jsonl \
    "${cf_args[@]}")

  local judge_flag="--no-llm-judge"
  if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    judge_flag="--llm-judge"
    log "eval: scoring with Claude judge (ANTHROPIC_API_KEY detected)"
  else
    warn "eval: scoring without Claude judge (set ANTHROPIC_API_KEY to enable)"
  fi

  step "eval: score"
  (cd "${EVAL_ROOT}" && uv run --extra judge --extra agent --extra pdf hinaing-eval \
    --hinaing-root "${HINAING_ROOT}" \
    score \
    --runs reports/fixture_runs.jsonl \
    ${judge_flag} \
    --output reports/scorecard.json \
    --form reports/validation_tool_filled.md \
    --pdf reports/validation_tool_filled.pdf \
    --summary reports/validation_summary.md \
    --summary-pdf reports/validation_summary.pdf)

  ok "eval: done — see reports/scorecard.json and reports/validation_tool_filled.md"
}

# --- Commands ---------------------------------------------------------------
cmd_up() {
  CLEANUP_ON_EXIT=1
  preflight
  qdrant_start
  write_env_file
  backend_install
  warmup
  backend_start
  verify_platform
  CLEANUP_ON_EXIT=0
  cmd_status
}

cmd_down() {
  backend_stop
  qdrant_stop
  ok "all components stopped"
}

cmd_status() {
  echo
  echo "────────────── Local eval environment ──────────────"
  printf "  %-12s %s\n" "Qdrant:"    "http://localhost:${QDRANT_PORT} (container=${QDRANT_CONTAINER_NAME}, in-memory tmpfs)"
  printf "  %-12s %s\n" "Backend:"   "http://localhost:${BACKEND_PORT} (log=${BACKEND_LOG})"
  printf "  %-12s %s\n" "LLM:"       "provider=${LLM_PROVIDER}"
  printf "  %-12s %s\n" "Supabase:"  "not used (backend does not touch it)"
  printf "  %-12s %s\n" "LangSmith:" "disabled (not required)"
  echo "─────────────────────────────────────────────────────"
  echo
  echo "Next steps:"
  echo "    scripts/test_local_env.sh eval      # run the 46-scenario suite"
  echo "    scripts/test_local_env.sh verify    # re-check platform health"
  echo "    scripts/test_local_env.sh logs      # tail backend log"
  echo "    scripts/test_local_env.sh down      # tear everything down"
  echo
}

cmd_logs() {
  if backend_running; then
    log "backend log tail (${BACKEND_LOG}) — Ctrl-C to stop:"
    tail -f "${BACKEND_LOG}"
  else
    warn "backend is not running"
  fi
}

cmd_eval() {
  if ! backend_running || ! qdrant_running; then
    warn "stack is not fully up — running 'up' first"
    cmd_up
  fi
  run_eval
}

cmd_verify() {
  qdrant_running  || fail "qdrant is not running — run 'scripts/test_local_env.sh up'"
  backend_running || fail "backend is not running — run 'scripts/test_local_env.sh up'"
  verify_platform
}

cmd_warmup() {
  qdrant_running  || fail "qdrant must be running first (scripts/test_local_env.sh up)"
  warmup
}

main() {
  local cmd="${1:-up}"
  # Load API keys before any subcommand runs so `eval`, `verify`, `warmup`,
  # etc. see ANTHROPIC_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY without
  # depending on whether the user went through `up` first.
  if [ -f "${KEYS_FILE}" ]; then
    load_keys_file
    ok "keys: loaded ${KEYS_FILE}"
  else
    log "keys: ${KEYS_FILE} not found — using shell environment only"
    log "      copy .keys.env.example → .keys.env to persist your keys"
  fi
  case "${cmd}" in
    up)       cmd_up ;;
    down)     cmd_down ;;
    status)   cmd_status ;;
    logs)     cmd_logs ;;
    eval)     cmd_eval ;;
    verify)   cmd_verify ;;
    warmup)   cmd_warmup ;;
    restart)  cmd_down; cmd_up ;;
    *)
      echo "usage: $0 {up|down|status|logs|eval|verify|warmup|restart}" >&2
      exit 2
      ;;
  esac
}

main "$@"
