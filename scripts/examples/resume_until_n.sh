#!/usr/bin/env bash
# Run the dynamic track CLI in a loop until OUT_DIR/results has TARGET_N JSON files.
#
# Required env:
#   OUT_DIR          -- e.g. runs/mini_router_a
#   ROUTER_IMPORT    -- e.g. swerouter.routers.always_model:AlwaysModelRouter
#   ROUTER_LABEL     -- human label for eval_summary / score
#
# Optional env: (same as historical resume scripts; see inline comments in prior revision)
#
# Usage:
#   bash scripts/examples/example_router_a.sh
#   export OUT_DIR=... ROUTER_IMPORT=... ROUTER_LABEL=...
#   bash scripts/examples/resume_until_n.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.inc.sh"

TRB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${TRB_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${OUT_DIR:?set OUT_DIR}"
: "${ROUTER_IMPORT:?set ROUTER_IMPORT}"
: "${ROUTER_LABEL:?set ROUTER_LABEL}"

RUN_ID="${RUN_ID:-${ROUTER_LABEL}}"
TARGET_N="${TARGET_N:-500}"
WORKERS="${WORKERS:-2}"
MAX_STEPS="${MAX_STEPS:-250}"
BUDGET_USD="${BUDGET_USD:-3.0}"
MAX_ROUNDS="${MAX_ROUNDS:-200}"
STALL_LIMIT="${STALL_LIMIT:-4}"
ROUTER_EXTRA="${ROUTER_EXTRA:-}"

count_results() {
  mkdir -p "${OUT_DIR}/results"
  find "${OUT_DIR}/results" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l
}

DATA_ARGS=()
if [[ -n "${POOL:-}" ]]; then DATA_ARGS+=(--pool "${POOL}"); fi
if [[ -n "${PRICING:-}" ]]; then DATA_ARGS+=(--pricing "${PRICING}"); fi
if [[ -n "${TTL:-}" ]]; then DATA_ARGS+=(--ttl "${TTL}"); fi
if [[ -n "${TIER_MAP:-}" ]]; then DATA_ARGS+=(--tier-map "${TIER_MAP}"); fi

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then LIMIT_ARGS+=(--limit "${LIMIT}"); fi

INSTANCE_ARGS=()
if [[ -n "${INSTANCE_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  INSTANCE_ARGS=(--instances ${INSTANCE_IDS})
fi

STALL_ROUNDS=0
PREV=-1

cd "${TRB_ROOT}"

for round in $(seq 1 "${MAX_ROUNDS}"); do
  n="$(count_results)"
  echo "$(date -Is) resume_until_n: round=${round} results=${n}/${TARGET_N} out=${OUT_DIR}"
  if [[ "${n}" -ge "${TARGET_N}" ]]; then
    echo "$(date -Is) resume_until_n: finished (${n} results)."
    exit 0
  fi
  if [[ "${n}" -eq "${PREV}" ]] && [[ "${round}" -gt 1 ]]; then
    STALL_ROUNDS=$((STALL_ROUNDS + 1))
    if [[ "${STALL_ROUNDS}" -ge "${STALL_LIMIT}" ]]; then
      echo "$(date -Is) resume_until_n: no new results for ${STALL_ROUNDS} rounds; giving up." >&2
      exit 1
    fi
  else
    STALL_ROUNDS=0
  fi
  PREV="${n}"

  # shellcheck disable=SC2086
  set +e
  python3 -m miniswerouter.cli run \
    --router-import "${ROUTER_IMPORT}" \
    ${ROUTER_EXTRA} \
    --router-label "${ROUTER_LABEL}" \
    --output-dir "${OUT_DIR}" \
    --workers "${WORKERS}" \
    --max-steps "${MAX_STEPS}" \
    --budget-usd "${BUDGET_USD}" \
    --run-id "${RUN_ID}" \
    "${LIMIT_ARGS[@]}" \
    "${INSTANCE_ARGS[@]}" \
    "${DATA_ARGS[@]}"
  rc=$?
  set -e
  if [[ "${rc}" != 0 ]]; then
    echo "$(date -Is) resume_until_n: dynamic CLI exited ${rc}; retrying after short sleep." >&2
  fi
  sleep 3
done

echo "$(date -Is) resume_until_n: exceeded MAX_ROUNDS=${MAX_ROUNDS}" >&2
exit 1
