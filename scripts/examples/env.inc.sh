#!/usr/bin/env bash
# Load API credentials for TwinRouterBench example wrappers (dynamic track).
#
# Layout:
#   <monorepo>/TwinRouterBench/scripts/examples/env.inc.sh
#   <monorepo>/TwinRouterBench/.env   <-- OPENROUTER_* / CommonStack, etc.

TRB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "${TRB_ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${TRB_ROOT}/.env"
  set +a
else
  echo "env.inc.sh: warning: ${TRB_ROOT}/.env not found; set OPENROUTER_* / SWEROUTER_* in the shell." >&2
fi

if [[ -n "${COMMONSTACK_API_BASE:-}" ]]; then
  export OPENROUTER_BASE_URL="${COMMONSTACK_API_BASE}"
  export SWEROUTER_BASE_URL="${COMMONSTACK_API_BASE}"
fi
if [[ -n "${COMMONSTACK_API_KEY:-}" ]]; then
  export OPENROUTER_API_KEY_EXP="${COMMONSTACK_API_KEY}"
  export OPENROUTER_API_KEY="${COMMONSTACK_API_KEY}"
  export SWEROUTER_API_KEY="${COMMONSTACK_API_KEY}"
fi

if [[ -z "${COMMONSTACK_API_KEY:-}" ]] && [[ -n "${OPENROUTER_API_KEY_EXP:-}" ]]; then
  export OPENROUTER_API_KEY="${OPENROUTER_API_KEY_EXP}"
fi

# Set OPENROUTER_BASE_URL / OPENROUTER_API_KEY_EXP in TwinRouterBench/.env
export SWEROUTER_BASE_URL="${SWEROUTER_BASE_URL:-${OPENROUTER_BASE_URL:-}}"
export SWEROUTER_API_KEY="${SWEROUTER_API_KEY:-${OPENROUTER_API_KEY:-}}"
