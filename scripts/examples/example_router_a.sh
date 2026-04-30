#!/usr/bin/env bash
# Portable example A: fixed pool model every step (AlwaysModelRouter).
# Use for smoke tests or cheap baselines. Pick MODEL_ID from the locked pool
# (see ``data/dynamic/model_pool.json`` under TwinRouterBench).
#
# Quick smoke:
#   cd TwinRouterBench
#   export INSTANCE_IDS="django__django-11133"
#   bash scripts/examples/example_router_a.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export OUT_DIR="${OUT_DIR:-${TRB_ROOT}/runs/mini_always_model}"
export ROUTER_LABEL="${ROUTER_LABEL:-mini_always_model}"
export RUN_ID="${RUN_ID:-${ROUTER_LABEL}}"
export ROUTER_IMPORT="swerouter.routers.always_model:AlwaysModelRouter"
export MODEL_ID="${MODEL_ID:-deepseek/deepseek-v3.2}"
export ROUTER_EXTRA="--router-arg model_id=${MODEL_ID} --router-arg label=${ROUTER_LABEL}"

export TARGET_N="${TARGET_N:-1}"
export WORKERS="${WORKERS:-1}"
export LIMIT="${LIMIT:-1}"
export INSTANCE_IDS="${INSTANCE_IDS:-django__django-11133}"

exec bash "${SCRIPT_DIR}/resume_until_n.sh"
