#!/usr/bin/env bash
# Portable example B: GoldTierRouter (oracle) — static-track labels + locked tier map.
# Intended for pipeline checks, not leaderboard.
#
# Usage:
#   cd TwinRouterBench
#   export INSTANCE_IDS="django__django-11133"
#   bash scripts/examples/example_router_b.sh
#
# Override paths if needed:
#   export QUESTION_BANK_PATH=/abs/path/to/question_bank.jsonl
#   export TIER_TO_MODEL_PATH=/abs/path/to/tier_to_model.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

_resolve_question_bank() {
  if [[ -n "${QUESTION_BANK_PATH:-}" ]]; then
    echo "${QUESTION_BANK_PATH}"
    return 0
  fi
  TRB_ROOT="${TRB_ROOT}" python3 - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

trb = Path(os.environ["TRB_ROOT"]).resolve()
candidates = [
    trb / "data" / "static" / "question_bank.jsonl",
    trb / "main" / "data" / "question_bank.jsonl",
]
for p in candidates:
    if p.is_file():
        print(p)
        raise SystemExit(0)
raise SystemExit(
    "question_bank.jsonl not found under TwinRouterBench/data/static; set QUESTION_BANK_PATH."
)
PY
}

_resolve_tier_map() {
  if [[ -n "${TIER_TO_MODEL_PATH:-}" ]]; then
    echo "${TIER_TO_MODEL_PATH}"
    return 0
  fi
  python3 - <<'PY'
from __future__ import annotations

from pathlib import Path

import swerouter

p = Path(swerouter.__file__).resolve().parent.parent / "data" / "dynamic" / "tier_to_model.json"
if p.is_file():
    print(p)
    raise SystemExit(0)
raise SystemExit("tier_to_model.json not found; set TIER_TO_MODEL_PATH.")
PY
}

QB="$(_resolve_question_bank)"
TM="$(_resolve_tier_map)"

export INSTANCE_IDS="${INSTANCE_IDS:-django__django-11133}"

export OUT_DIR="${OUT_DIR:-${TRB_ROOT}/runs/mini_gold_tier_smoke}"
export ROUTER_LABEL="${ROUTER_LABEL:-mini_gold_tier_smoke}"
export RUN_ID="${RUN_ID:-${ROUTER_LABEL}}"
export ROUTER_IMPORT="swerouter.routers.gold_tier:GoldTierRouter.from_cli_args"
export ROUTER_EXTRA="--router-arg question_bank_path=${QB} --router-arg tier_to_model_path=${TM} --router-arg allowed_instance_ids=${INSTANCE_IDS// /,} --router-arg label=${ROUTER_LABEL}"

export TARGET_N="${TARGET_N:-1}"
export WORKERS="${WORKERS:-1}"
export LIMIT="${LIMIT:-1}"

exec bash "${SCRIPT_DIR}/resume_until_n.sh"
