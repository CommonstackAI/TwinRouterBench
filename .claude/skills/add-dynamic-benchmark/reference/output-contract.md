# Dynamic-track output contract

`swerouter.leaderboard.score.score_run_dir` grades any run directory with this
layout. `RoutedSession` writes it; this file documents it so adapters that
cannot use `RoutedSession` (or reviewers) can check conformance.

```
<run_dir>/
  <instance_id>.trace.jsonl     # one row per routed LLM call + loop_summary marker
  results/<instance_id>.json    # one per instance
```

## Trace step row (one JSON object per line)

| Field | Type | Used by scorer | Meaning |
|---|---|---|---|
| `instance_id` | str | – | instance |
| `step_index` | int | ordering | 0-based call index within the instance |
| `started_at`, `finished_at` | float (unix s) | `--reprice-from-raw-usage` | wall clock of the provider call |
| `model_id` | str | yes | pool model that served the call |
| `provider` | str | – | usage normalizer key from `model_pool.json` |
| `rationale` | str \| null | – | router's free-form explanation |
| `usage` | `{input_tokens, cache_read_tokens, cache_write_tokens, output_tokens}` | yes | normalized buckets |
| `raw_usage` | object | `--reprice-from-raw-usage` | provider `usage` verbatim |
| `step_cost_usd` | float | yes (sum = router cost) | 4-bucket price from `model_pricing.json` |
| `cumulative_cost_usd` | float | – | running total |
| `cache_lookup` | `{hit, reason, cached_prefix_token_count}` | – | harness-side cache hint for the chosen model |

Additional diagnostic fields are allowed. Rows containing `__marker__` are
skipped by the scorer; the final row is:

```json
{"__marker__": "loop_summary", "instance_id": "...", "step_count": 12,
 "finished_by": "submitted", "total_router_cost_usd": 0.31,
 "error": null, "model_distribution": {"deepseek/deepseek-v3.2": 9, "anthropic/claude-opus-4.6": 3}}
```

## Per-instance result (`results/<instance_id>.json`)

| Field | Required | Meaning |
|---|---|---|
| `instance_id` | yes | must match the trace file name |
| `resolved` | yes, bool | official benchmark verdict; never defaulted |
| `trace_path` | yes | absolute, or relative to `run_dir` |
| `agent_error` | yes (null if none) | original exception text from the agent loop |
| `eval_error` | yes (null if none) | original error text from the official evaluator |
| `model_distribution` | yes | calls per model |
| `step_count`, `total_router_cost_usd`, `finished_by` | yes | run summary |
| `pool_fingerprint`, `pricing_schema_version`, `ttl_policy_name` | yes | locked-data provenance |
| `extra` | optional | benchmark-specific fields (report paths, sub-scores) |

SWE-only fields (`patch`, `patch_applied`, FAIL_TO_PASS counts) are not part of
the generic contract; the scorer reads `patch_applied` with a default and it is
only meaningful for SWE-bench.

## Scoring

```
twinrouterbench dynamic score --run-dir <run_dir> --router-label <label> \
  --fixed-failure-penalty-usd <gamma> [--instance-ids-file data/dynamic/<bench>_ids.txt] \
  [--exclude-infra-failures]
```

Bill per instance = router API cost + (unresolved ? γ : 0). γ is chosen per
benchmark by the user; always pass it explicitly.
