# Changelog

All notable changes to **TwinRouterBench** are documented in this file.

## [Unreleased]

### Added

- Benchmark-agnostic pipeline configuration and the stable
  `run_pipeline(config)` / `twinrouterbench data run --config` interfaces.
- Configurable normalized and single-turn loaders; execution, backend-judge,
  exact-match, and contains evaluators; loader/evaluator/backend plugin contracts.
- Generic OpenAI-compatible executor and a config-only custom QA example.
- Pipeline/suite validation commands and rejection of inline credentials.

- Versioned static data-construction package and `twinrouterbench data` CLI.
- Paper-aligned sequential-locking downgrade search, tier-pool cascade,
  mixed-prefix reconstruction, hardened open-ended judge, and manual audit flow.
- SWE-bench, BFCL, mtRAG, QMSum, and PinchBench adapters with deterministic
  offline fixtures.
- Mock, replay, and optional live-plugin backends; guarded release-candidate
  publication and source/license provenance validation.
- Offline unit, end-to-end, CLI, replay-equivalence, and golden-hash tests.

## [0.1.0] - 2026-04-30

### Changed

- Static track data now lives under **`data/static/`** (alongside **`data/dynamic/`**). `main.dataset.DATA_DIR` points at `data/static/`. The question bank and manifest are copied from `CommonRouterBench/data/` into this tree for a complete checkout.

### Added

- Unified `TwinRouterBench/` source tree with PyPI name **`twinrouterbench`**.
- Meta-CLI `twinrouterbench` with subcommands `static`, `dynamic`, and `swe`.
- Optional extra **`[dynamic]`** for Docker / SWE-bench / mini-swe-agent / LiteLLM dependencies.
- Locked dynamic JSON under `data/dynamic/`.
- Single English `README.md` as the user-facing documentation entrypoint.
