#!/usr/bin/env bash
# Load API credentials for TwinRouterBench example wrappers (dynamic track).
#
# Layout:
#   <checkout-parent>/TwinRouterBench/scripts/examples/env.inc.sh
#   <checkout-parent>/TwinRouterBench/.env   <-- OPENROUTER_* / SWEROUTER_* aliases

TRB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "${TRB_ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${TRB_ROOT}/.env"
  set +a
else
  echo "env.inc.sh: warning: ${TRB_ROOT}/.env not found; set OPENROUTER_* / SWEROUTER_* in the shell." >&2
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]] && [[ -n "${OPENROUTER_API_KEY_EXP:-}" ]]; then
  export OPENROUTER_API_KEY="${OPENROUTER_API_KEY_EXP}"
fi

export SWEROUTER_BASE_URL="${SWEROUTER_BASE_URL:-${OPENROUTER_BASE_URL:-}}"
export SWEROUTER_API_KEY="${SWEROUTER_API_KEY:-${OPENROUTER_API_KEY:-}}"
