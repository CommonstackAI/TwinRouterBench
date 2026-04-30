# Twin Router Bench

**Twin Router Bench** is a single benchmark suite for **per-step LLM routing**: a *router* chooses which pooled `model_id` to use on every agent step, under locked pricing and cache rules. The suite ships in one Python distribution (**`twinrouterbench`**) and one source tree (**`TwinRouterBench/`**).

It contains **two tracks** inside the same product—same protocol and locked tables—not two separate benchmarks:

| Track | Role | CLI entry |
|-------|------|-----------|
| **Static** | Fast validation on a fixed supervision bank (tier labels + nominal cost metrics). | `twinrouterbench static …` |
| **Dynamic** | End-to-end evaluation on **SWE-bench Verified** with real tool use—**mini-swe-agent** scaffold or **editor** scaffold. | `twinrouterbench dynamic …` / `twinrouterbench swe …` |

---

## Where to run commands (important)

Many examples use **paths relative to the monorepo root** (parent of `TwinRouterBench/`), for example:

- `semantic-router/...` — checkout of the **semantic-router** repo (KNN weights and `models.py`).
- `TwinRouterBench/data/static/...` — static track: `question_bank.jsonl`, `manifest.json`.
- `TwinRouterBench/data/dynamic/...` — dynamic track: locked pool, pricing, TTL, tier map, SR-KNN mapping.

**Recommended:** `cd` to the monorepo root, `pip install -e "./TwinRouterBench[dynamic]"`, then run `twinrouterbench …` from there so those relative paths resolve.

The dynamic CLI also loads **`TwinRouterBench/.env`** via `miniswerouter.cli` (see [Configuration](#configuration)). That file is resolved from the **TwinRouterBench package directory**, not from your shell `cwd`, so keeping `.env` next to `pyproject.toml` under `TwinRouterBench/` is the supported layout.

---

## Install

**Static track only** (lightweight dependencies):

```bash
pip install -e ./TwinRouterBench
# or from PyPI once published:
# pip install twinrouterbench
```

**Full suite** (adds Docker, SWE-bench harness, mini-swe-agent, LiteLLM, etc.):

```bash
pip install -e "./TwinRouterBench[dynamic]"
```

If you run `twinrouterbench dynamic` or `twinrouterbench swe` without `[dynamic]`, the CLI exits with an explicit message to install the extra.

---

## Configuration

### `TwinRouterBench/.env`

1. Copy `TwinRouterBench/.env.example` to `TwinRouterBench/.env`.
2. Fill in real credentials (never commit `.env`).

**Load order (dynamic / mini CLI):**

1. **`_load_mini_dotenv`** — reads `TwinRouterBench/.env` and sets any key that is **missing or empty** in the process environment (already-set variables win).
2. **`_apply_gateway_aliases`** — if CommonStack variables are set, they **override** chat-related OpenRouter/SWERouter URL and API key (see table below).

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_BASE_URL` | OpenAI-compatible root URL (must end with `/v1`; clients append `/chat/completions`). Example: `https://openrouter.ai/api/v1`. |
| `OPENROUTER_API_KEY` | Bearer token for the gateway above. |
| `OPENROUTER_API_KEY_EXP` | Optional alternate key; if `OPENROUTER_API_KEY` is empty after dotenv, it is copied from this. |
| `SWEROUTER_BASE_URL` / `SWEROUTER_API_KEY` | Optional explicit names; `miniswerouterbench run` defaults fall back to `OPENROUTER_*` when unset. |
| `COMMONSTACK_API_BASE` | If set, **replaces** `OPENROUTER_BASE_URL` and `SWEROUTER_BASE_URL` after bootstrap. |
| `COMMONSTACK_API_KEY` | If set, **replaces** `OPENROUTER_API_KEY`, `OPENROUTER_API_KEY_EXP`, and `SWEROUTER_API_KEY` after bootstrap. |

**Common pitfall:** `OPENROUTER_BASE_URL` pointing at **CommonStack** while `OPENROUTER_API_KEY` is still an **OpenRouter** `sk-or-v1-…` key (or the reverse) yields **401** or “invalid access key”. Use a **consistent** pair: either full CommonStack base + CommonStack access key, or OpenRouter base + OpenRouter key.

Shell snippets under `TwinRouterBench/scripts/examples/env.inc.sh` mirror the Python alias logic for bash-driven runs.

### Prerequisites (dynamic track)

- **Docker** running locally; enough disk for SWE-bench images.
- **Network** for image pulls and the chat gateway.
- **API keys** as above. The dynamic CLIs load `TwinRouterBench/.env` before parsing arguments.

Static `metrics` subcommand does not need Docker or network if you only aggregate local JSON.

### semantic-router (SR KNN only)

The **SemanticRouterKNNRouter** loads:

- `knn_model.json` (feature dim 1024 + 14 category one-hot),
- `semantic-router` repo root (for `ml_model_selection/models.py` / `KNNModel.load`),
- **sentence-transformers** embedder (default `Qwen/Qwen3-Embedding-0.6B`; first run downloads weights).

Ensure a checkout exists at `semantic-router/` relative to the monorepo root (or pass absolute `--router-arg` paths).

---

## Command-line interface

Primary entrypoint:

```bash
twinrouterbench static <subcommand> [args]   # static track
twinrouterbench dynamic <subcommand> [args]  # mini-swe-agent harness
twinrouterbench swe <subcommand> [args]     # editor-scaffold harness
```

Compatibility console scripts (same code after the same install):

| Script | Module |
|--------|--------|
| `CommonRouterBench` | `main.cli` |
| `miniswerouterbench` | `miniswerouter.cli` |
| `swerouterbench` | `swerouter.cli` |

For debugging without installing entrypoints:

```bash
export PYTHONPATH=/abs/path/to/TwinRouterBench
python -m miniswerouter.cli run …
```

---

## Static track

### `twinrouterbench static metrics`

Aggregates **Section 11–style** metrics from a JSON file containing an array of **CaseMetrics** objects (see `main.metrics.CaseMetrics` and `case_metrics_from_dict`).

```bash
twinrouterbench static metrics --cases /path/to/cases.json
```

Each element must at least include `case_id`, `task_passed`, and either nominal cost fields or `baseline_steps` / `optimal_steps` / `test_steps` lists with `completion_tokens` (and optional `tier` / `model`). Example skeleton:

```json
[
  {
    "case_id": "example-1",
    "task_passed": true,
    "baseline_cost_nominal": 10.0,
    "optimal_cost_nominal": 4.0,
    "test_cost_nominal": 5.0
  }
]
```

**Question bank:** shipped under **`data/static/`** (`question_bank.jsonl`, `manifest.json`). The `main` package exposes `DATA_DIR` / `STATIC_DATA_DIR` / `QUESTION_BANK_PATH` pointing at that directory (see `main.dataset`). `setuptools` package-data includes those files for wheel installs. Tier-only eval APIs live under `main.eval`.

---

## Dynamic track (`twinrouterbench dynamic …`)

This forwards to **`miniswerouter.cli`** (`run`, `score`, `audit-infra`, `audit-trace-cost`, `render`).

### `run` — main flags

| Flag | Meaning |
|------|---------|
| `--router-import` | Required. `module:factory`, e.g. `swerouter.routers.sr_knn_adapter:SemanticRouterKNNRouter.from_cli_args`. |
| `--router-arg KEY=VALUE` | Repeatable; passed as kwargs to the factory. Values are strings. |
| `--router-label` | Required label stored in `eval_summary.json` and traces. |
| `--output-dir` | Required. Run artifacts root (created if needed). |
| `--base-url` / `--api-key` | Default from `SWEROUTER_*` then `OPENROUTER_*` env after `.env` load. |
| `--instances id1 id2 …` | Optional explicit SWE-bench instance IDs. |
| `--limit N` | Optional cap on how many dataset instances to consider (ordering is harness-defined). |
| `--workers` | Parallel workers (default 2). |
| `--max-steps` | Agent step limit (default 250, matches mini-swe-agent SWE profile). |
| `--budget-usd` | Agent cost limit in USD (default 3). |
| `--run-id` | Stored in summaries; use a new id when you want a logically separate run. |
| `--force-rerun` | Re-run instances even if `results/<instance_id>.json` already exists. |
| `--pool`, `--pricing`, `--ttl`, `--tier-map` | Override locked JSON paths (defaults under `TwinRouterBench/data/dynamic/`). |

**Resume:** without `--force-rerun`, instances that already have `output_dir/results/<instance_id>.json` are skipped.

### `run` output layout

Under `--output-dir`:

- `results/<instance_id>.json` — per-instance outcome (`resolved`, `step_count`, errors, etc.).
- `<instance_id>.trace.jsonl` — per-step router and usage trace.
- `eval_summary.json` — run-level aggregate (`completed`, `resolved_count`, `errors`, …).
- `case_summaries/<instance_id>.summary.json` — condensed per-case view.
- `agent_logs/<instance_id>/agent.log` — mini-swe-agent log.

Long stretches without new console output are normal (Docker pull, repository setup, multi-step LLM calls).

### Example — gold-tier oracle (paths under TwinRouterBench)

From monorepo root, after installing `[dynamic]`:

```bash
twinrouterbench dynamic run \
  --router-import swerouter.routers.gold_tier:GoldTierRouter.from_cli_args \
  --router-arg question_bank_path=TwinRouterBench/data/static/question_bank.jsonl \
  --router-arg tier_to_model_path=TwinRouterBench/data/dynamic/tier_to_model.json \
  --router-arg allowed_instance_ids=django__django-11133 \
  --router-arg label=gold_tier_oracle \
  --router-label gold_tier_oracle \
  --output-dir runs/mini_gt_one \
  --instances django__django-11133 \
  --max-steps 250 --budget-usd 3 --run-id mini_gt_one --force-rerun
```

Adjust `question_bank_path` if your bank lives elsewhere.

### Example — Semantic Router **SR KNN** router

Requires `semantic-router/` at monorepo root and CPU/GPU for embeddings (`embedding_device`).

```bash
twinrouterbench dynamic run \
  --router-import swerouter.routers.sr_knn_adapter:SemanticRouterKNNRouter.from_cli_args \
  --router-arg knn_json_path=semantic-router/src/training/model_selection/ml_model_selection/.cache/ml-models/knn_model.json \
  --router-arg mapping_path=TwinRouterBench/data/dynamic/sr_knn_to_pool.json \
  --router-arg sr_repo_root=semantic-router \
  --router-arg embedding_model=Qwen/Qwen3-Embedding-0.6B \
  --router-arg embedding_device=cpu \
  --router-arg label=sr_knn_smoke \
  --router-arg category=other \
  --router-label sr_knn_smoke \
  --output-dir runs/sr_knn_smoke \
  --instances django__django-11066 django__django-13410 \
  --workers 2 \
  --max-steps 40 \
  --budget-usd 5.0 \
  --run-id sr_knn_smoke \
  --force-rerun
```

| `--router-arg` | Role |
|----------------|------|
| `knn_json_path` | Pretrained `knn_model.json` (e.g. under semantic-router `.cache/ml-models/`). |
| `mapping_path` | `sr_knn_to_pool.json` — maps KNN label strings to `model_id`s in the locked pool. |
| `sr_repo_root` | Root of **semantic-router** checkout (for `KNNModel` loader code path). |
| `embedding_model` | SentenceTransformers model id (training default: Qwen3 embedding). |
| `embedding_device` | `cpu`, `cuda`, or `mps`. |
| `category` | VSR one-hot bucket passed into the feature vector (default `other` for smoke). |

Larger unattended jobs: see `runs/sr_knn_full/resume.sh` in the monorepo (uses `swerouterbench` with the same router args pattern).

### `score`, `audit-*`, `render`

```bash
twinrouterbench dynamic score --run-dir runs/your_run --router-label your_label
twinrouterbench dynamic audit-infra --run-dir runs/your_run
twinrouterbench dynamic audit-trace-cost --run-dir runs/your_run
twinrouterbench dynamic render --score runs/a/score.json runs/b/score.json --out leaderboard.md
```

`score` writes `score.json` (or `--out`) using the same scorer as the editor harness.

---

## Editor scaffold (`twinrouterbench swe …`)

Forwards to **`swerouter.cli`** (full SWE-bench harness in the editor-oriented layout). Requires the same `[dynamic]` extra, Docker, and credentials. See `swerouter/docs/` for protocol details.

---

## Shell helpers

Under `TwinRouterBench/scripts/examples/`:

- `env.inc.sh` — source `TwinRouterBench/.env` and apply the same gateway aliases as Python.
- `example_router_a.sh` / `example_router_b.sh` — wrapped smoke patterns.
- `resume_until_n.sh` — loop `python -m miniswerouter.cli run` until `results/` contains `TARGET_N` JSON files (for long campaigns).

---

## Troubleshooting

| Symptom | Things to check |
|---------|------------------|
| HTTP **401** on chat | `OPENROUTER_BASE_URL` must match the key type (OpenRouter vs CommonStack). Remove or fix `COMMONSTACK_*` if you intend to use raw OpenRouter keys. |
| `missing required connection settings` on `run` | Set `SWEROUTER_API_KEY` / `OPENROUTER_API_KEY` (after `.env`) or pass `--api-key` / `--base-url`. |
| Dynamic import errors for `main.*` | Install editable from `TwinRouterBench/` or set `PYTHONPATH` to the `TwinRouterBench` root. |
| SR KNN `FileNotFoundError` for knn JSON | Ensure paths exist; monorepo root `cwd` is simplest. |
| Very slow first SR KNN step | Embedding model download + CPU encoding; use `embedding_device=cuda` when available. |
| “No output” for many minutes | Docker image pull + SWE environment + agent steps; watch `agent_logs/` or `docker ps`. |

---

## Repository layout (`TwinRouterBench/`)

| Path | Purpose |
|------|---------|
| `main/` | Static-track package (`main.cli`, tokenizer, pricing, eval). |
| `miniswerouter/` | Dynamic track on **mini-swe-agent**. |
| `swerouter/` | Router protocol, pricing, cache simulation, harness, leaderboard; `swerouter/docs/`. |
| `data/static/` | Static track JSONL: `question_bank.jsonl`, `manifest.json`. |
| `data/dynamic/` | Dynamic track locked JSON: `model_pool.json`, `model_pricing.json`, `ttl_policy.json`, `tier_to_model.json`, `sr_knn_to_pool.json`, … |
| `twinrouterbench/` | Meta-CLI dispatcher. |
| `.env.example` | Template for gateway credentials. |

---

## Citation

If you use Twin Router Bench in research, cite the **Twin Router Bench** paper (placeholder—update when DOI is available).

---

## Further reading

- Pricing and cache semantics (Chinese): [`swerouter/docs/pricing_and_cache_zh.md`](swerouter/docs/pricing_and_cache_zh.md)
- Scoring rules: [`swerouter/docs/scoring_zh.md`](swerouter/docs/scoring_zh.md)

---

## Implementation note (CLI forwarding)

`twinrouterbench static|dynamic|swe` dispatches in-process to the existing CLIs. For debugging, you may still invoke `python -m miniswerouter.cli` or `python -m swerouter.cli` with `PYTHONPATH` set to `TwinRouterBench/`.

---

## Appendix: migration

This tree supersedes separate installs of the historical static and dynamic router benchmark packages for **development inside this monorepo**. Use **Twin Router Bench** naming in new scripts and papers; keep legacy pathnames only where required for one-off comparison.
