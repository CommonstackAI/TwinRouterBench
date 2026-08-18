# TwinRouterBench — `data/`

| Subdirectory | Track | Contents |
|--------------|-------|----------|
| **`static/`** | Static | `question_bank.jsonl`, `manifest.json` — tier routing supervision consumed by `main.dataset` / `main.eval`. |
| **`dynamic/`** | Dynamic | Locked JSON for SWE runs: model pool, pricing, TTL policy, tier map, SR-KNN label mapping, etc. Also `dynamic_heldout100_ids.txt` — the paper’s fixed 100 SWE-bench Verified held-out instance IDs (see top-level README). |

Code defaults: `main.dataset.STATIC_DATA_DIR` → `TwinRouterBench/data/static/`; harness defaults for pool/pricing → `TwinRouterBench/data/dynamic/`.

The construction code that produces static-track candidates lives under
`twinrouterbench/data_generation/`. It writes isolated run directories and does
not overwrite this directory. See [`../docs/DATA_GENERATION.md`](../docs/DATA_GENERATION.md).
