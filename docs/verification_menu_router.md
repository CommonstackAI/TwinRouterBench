# VerificationMenuRouter

`VerificationMenuRouter` is a deterministic verification-aware service-menu
baseline for the dynamic TwinRouterBench router protocol.

It maps prefix-level risk signals to four service levels:

| Service menu item | Tier | Meaning |
| --- | --- | --- |
| `cheap_execute` | `low` | Read-only or low-risk inspection. |
| `standard_execute` | `mid` | Ordinary tool/schema use or moderate context. |
| `verify_sensitive` | `mid_high` | File edits, failing tests, tracebacks, or other verification-sensitive evidence. |
| `escalate_frontier` | `high` | Finalization, submit/patch steps with failures, or high-risk long-context edits. |

The router does not call an external verifier and does not change benchmark
labels, scoring, locked pools, pricing, or the dynamic harness. It is
verification-aware in the narrower sense that write actions, failure logs,
strict tool schemas, finalization, long context, late trajectory steps, and
budget pressure change the effective risk score before selecting a tier.

`risk_profile` can be set to `cheap`, `balanced`, or `conservative` through
`--router-arg risk_profile=...`. The default is `balanced`. The rationale
includes a compact feature JSON with `risk_profile` and `hard_floor_reason`
so routing decisions are auditable from traces.

## Static Evaluation

From the checkout root:

```bash
python scripts/evaluate_verification_menu_static.py
```

To compare a different risk profile:

```bash
python scripts/evaluate_verification_menu_static.py --risk-profile conservative
```

The script writes the full summary to:

```text
runs/static_verification_menu_balanced_summary.json
```

`runs/` is gitignored, so the generated summary is for local PR notes rather
than source control.

Local result on the bundled `data/static/question_bank.jsonl`:

| Metric | Value |
| --- | ---: |
| `case_pass_rate_percent` | 97.11 |
| `case_exact_match_percent` | 45.98 |
| `trajectory_pass_rate_percent` | 93.30 |
| `cost_savings_score_percent` | 53.30 |
| `combined_score_percent` | 72.42 |

By benchmark:

| Benchmark | Pass | Exact | Trajectory pass | Cost savings |
| --- | ---: | ---: | ---: | ---: |
| SWE-bench | 98.81 | 51.49 | 91.37 | -7.00 |
| BFCL | 100.00 | 3.23 | 100.00 | 90.68 |
| mtRAG | 94.82 | 80.31 | 94.82 | 88.06 |
| QMSum | 92.41 | 58.62 | 92.41 | 91.30 |
| PinchBench | 93.75 | 54.17 | 68.75 | 27.72 |

## Dynamic Smoke

From the checkout parent directory after installing the dynamic extra:

```bash
twinrouterbench dynamic run \
  --router-import swerouter.routers.verification_menu:VerificationMenuRouter.from_cli_args \
  --router-arg tier_to_model_path=TwinRouterBench/data/dynamic/tier_to_model.json \
  --router-arg label=verification_menu \
  --router-arg risk_profile=balanced \
  --router-label verification_menu \
  --output-dir runs/verification_menu_smoke \
  --instances django__django-11133 \
  --workers 1 \
  --max-steps 60 \
  --budget-usd 3 \
  --run-id verification_menu_smoke \
  --force-rerun
```

Score the smoke run with:

```bash
twinrouterbench dynamic score \
  --run-dir runs/verification_menu_smoke \
  --router-label verification_menu
```

Dynamic smoke requires Docker plus a configured OpenAI-compatible gateway in
`TwinRouterBench/.env`.

Local dynamic smoke status:

| Run | Instance | Result | Router cost | Notes |
| --- | --- | --- | ---: | --- |
| `verification_menu_smoke` | `django__django-11133` | resolved | `$0.17985525` | `FAIL_TO_PASS 1/1`, `PASS_TO_PASS 64/64`; model distribution: Gemini Flash 1, Claude Opus 16. |
| `verification_menu_guard_10554_20260704` | `django__django-10554` | unresolved | `$1.14232555` | Diagnostic rerun after budget-pressure guard; model distribution: Gemini Flash 27, Claude Opus 36. |

The dynamic results are smoke/diagnostic checks only. They do not establish a
leaderboard claim or broad benchmark improvement.
