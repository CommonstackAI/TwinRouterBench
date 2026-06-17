# SWERouterBench: paying the real bill for per-step LLM routing

The **static track** of Twin Router Bench is offline. It hands routers a curated question bank of 970 pre-recorded conversation prefixes; every router has to return a 0–3 tier id, and we score the prediction against a gold tier. That is useful for training and for offline routing research, but it does not answer the question that gets asked in every production review meeting:

> "If we route through your router on SWE-bench Verified, how many bugs does it actually fix, and how many real dollars does it burn?"

The **dynamic track** is the benchmark that answers that. It is the live SWE-bench counterpart of the static track.

## What it is

SWERouterBench runs the 500 SWE-bench Verified instances end-to-end. At every LLM call the harness hands control to your router; your router picks a concrete `model_id` from an officially locked pool; the harness invokes the chosen vendor, tracks usage, simulates a 5-minute wall-clock prompt cache, and writes a per-step trace.

At the end of the run we ask SWE-bench's own judge whether the final patch resolves the instance (`FAIL_TO_PASS` plus `PASS_TO_PASS` — the same gate the public SWE-bench Verified leaderboard uses). We then convert each step's tokens to USD at published rates and aggregate into a **penalty-inclusive leaderboard bill**, `total_leaderboard_bill_usd` (this is not the same as raw routed API spend; see `total_router_cost_usd`).

No combined score. No synthetic quality metric. Lower dollar amount wins.

## How we score a router

For each instance:

```
passed_instance_bill = Σ router_actual_cost_i
failed_instance_bill = Σ router_actual_cost_i  +  Σ baseline_high_cost_i
                       ↑                          ↑
                       what the vendors actually    one full rerun with the
                       billed the router during     most expensive pool model,
                       the real run                 priced with re-simulated cache
```

`baseline_high_cost_i` is computed by taking the per-step prefix and output token counts the router actually produced, independently simulating a perfect single-model cache chain for the highest-tier model in the pool (5-minute TTL), and pricing the result at that model's published rates. It is a best-effort first-order estimate of "how much would an always-the-strongest baseline have cost on the same trajectory" — the natural billing penalty for a failed run.

Leaderboard sort: `total_leaderboard_bill_usd` ascending. A router that fails every instance ends up paying roughly the always-HIGH baseline anyway (the penalty), so pure "pick the cheapest" strategies cannot cheese the ranking.

## What it is not

SWERouterBench does not try to re-grade patches. The `resolved` flag is whatever `swebench.harness.run_evaluation` says. Our numbers align with the public SWE-bench Verified leaderboard ±3%.

SWERouterBench does not let you bring your own pricing. The official pool and the published prices live in `data/model_pool.json` and `data/model_pricing.json`; both are locked with a `pricing_fingerprint` written into every run. Leaderboard submissions are re-run by maintainers against the official pricing before they are accepted.

SWERouterBench is not a training-data dump. Traces are per-run artifacts and are `.gitignore`d; they belong to whoever ran the router.

## Relationship to the static track

| Dimension | Static track | Dynamic track |
|---|---|---|
| Shape | static 970-row question bank | 500 SWE-bench Verified instances, live-run |
| Router output | tier id 0–3 | concrete `model_id` from locked pool |
| Pass criterion | `pred_tier >= gold_tier` (proxy) | SWE-bench `resolved` (real tests) |
| Pricing | nominal tier prices | published provider prices |
| Cache model | step-distance TTL = 3 | wall-clock TTL = 300s |
| Top-line metric | `scores_v2.combined_score` (4-dim average) | `total_leaderboard_bill_usd` (penalty-inclusive total bill, USD) |
| Deps | lightweight | `swebench`, `docker` |
| PyPI extra | base install | `[dynamic]` optional extra |

The static track tells you *what kind* of model each step needs. The dynamic track tells you *what it actually costs* to be wrong about that.

## Roadmap

The first milestone is the 0.1.0 release. Once the pipeline is cemented, the next beats are:

- Publish baseline runs for the four pool models and a CRB-classifier-powered mixed router.
- Open a submission flow (GitHub issue template + maintainer re-run).
- Quarterly pricing refresh with a `pricing_schema_version` bump.
- Optional extended-cache (1h) variant once community pressure for it is real.

If you maintain a router and want to see it on the leaderboard, open an issue with your run artefacts and we will re-run it against the official pricing.

## See also

- [TwinRouterBench README](../../README.md) — installation and CLI quick start.
- TwinRouterBench paper (withheld for anonymous review).
