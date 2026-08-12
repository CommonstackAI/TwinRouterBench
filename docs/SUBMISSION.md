# TwinRouterBench Leaderboard Submission Guide

This document is the **source of truth** for public leaderboard submissions.
Fill the GitHub issue form only after reading this page.

- Public leaderboard: https://commonstackai.github.io/TwinRouterBench/
- Open a submission issue: https://github.com/CommonstackAI/TwinRouterBench/issues/new?template=leaderboard_submission.yml
- Static dataset (download only; **not** a submission inbox): https://huggingface.co/datasets/Amorph/TwinRouterBench

## What we accept

| Item | Requirement |
|------|-------------|
| Track | **Dynamic** SWE-bench Verified **held-out-100** |
| Instance split | Exactly the IDs in [`data/dynamic/dynamic_heldout100_ids.txt`](../data/dynamic/dynamic_heldout100_ids.txt) |
| Pool / pricing / TTL | Locked tables under [`data/dynamic/`](../data/dynamic/) (no custom pricing) |
| Sort key | `total_leaderboard_bill_usd` (lower is better), from official scorer |
| Acceptance | Maintainer review; we may **re-score** or **re-run** before updating the board |

Other tracks or custom splits may be discussed, but are not guaranteed to appear on the public board.

## How to submit (two places)

1. **GitHub Issue** — official request + metadata (router name, fingerprints, reproduction, checklist).
2. **External artifact URL** — downloadable archive of the scoreable run package.

Do **not**:

- upload run logs to the official Hugging Face dataset `Amorph/TwinRouterBench`
- attach the full run archive to the GitHub issue (size limits; packages can be large)
- include API keys, `.env`, or other secrets

Hosting for the archive may be your own Hugging Face dataset, Zenodo, a GitHub Release asset, or any durable URL you control.

## Required evaluation steps

```bash
# 1) Run on the official held-out-100 split
mapfile -t HELDOUT < TwinRouterBench/data/dynamic/dynamic_heldout100_ids.txt
twinrouterbench dynamic run \
  --router-import your.module:YourRouter.from_cli_args \
  --router-arg ... \
  --router-label your_label \
  --output-dir runs/your_run \
  --instances "${HELDOUT[@]}" \
  ...

# 2) Score (subset file keeps headline metrics on held-out-100 only)
twinrouterbench dynamic score \
  --run-dir runs/your_run \
  --router-label your_label \
  --instance-ids-file TwinRouterBench/data/dynamic/dynamic_heldout100_ids.txt

# Optional audits
twinrouterbench dynamic audit-infra --run-dir runs/your_run
twinrouterbench dynamic audit-trace-cost --run-dir runs/your_run
```

Record the TwinRouterBench git commit / tag used for the run; the issue form asks for it.

## Artifact package (what to upload)

### Include (held-out-100 scoreable core)

Typical size: about **5–20 MB**.

- `score.json` from `twinrouterbench dynamic score`
- `results/<instance_id>.json` for all 100 held-out instances
- matching `<instance_id>.trace.jsonl` files
- `eval_summary.json` if present

### Exclude

- `llm_io/` and `*.io.jsonl` (full prompt/response dumps; often **100–500+ MB**)
- secrets (`.env`, API keys, tokens)
- instances outside `dynamic_heldout100_ids.txt`

`agent_logs/` is optional and usually small; omit unless maintainers ask for debug context.

### Integrity (recommended)

Publish `sha256sum` of the archive and paste it into the issue form.

## What to put in the GitHub issue

The [Leaderboard submission](https://github.com/CommonstackAI/TwinRouterBench/issues/new?template=leaderboard_submission.yml) form collects:

- Router display name and `router_label`
- Contact and router code / paper URL (prefer a pinned commit)
- TwinRouterBench commit / tag used for the run
- Headline fields from `score.json`:
  - `total_leaderboard_bill_usd`
  - `total_router_cost_usd`
  - `total_penalty_cost_usd`
  - `resolved_count` / `instance_count` / `resolved_rate`
  - `avg_steps`
  - `pool_fingerprint` / `pricing_fingerprint` / `pricing_schema_version`
  - `failure_penalty_usd`
  - `exclude_infra_failures`
- Artifact URL (+ optional SHA-256)
- Exact reproduction commands
- Checklist confirmation

## Review policy

- Incomplete packages, wrong split, custom pricing, or unreproducible routers may be rejected.
- Maintainers may recompute scores from your traces against the locked `data/dynamic/` tables.
- Accepted entries are written into `leaderboard/data/leaderboard.json` and published via the leaderboard site.

## Questions

Open a normal GitHub issue (not the submission form) for clarification before you run a costly full held-out evaluation.
