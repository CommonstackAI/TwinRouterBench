# Changelog

All notable changes to **TwinRouterBench** are documented in this file.

## [0.1.0] - 2026-04-30

### Changed

- Static track data now lives under **`data/static/`** (alongside **`data/dynamic/`**). `main.dataset.DATA_DIR` points at `data/static/`. The question bank and manifest ship in-tree for a complete checkout.

### Added

- Unified `TwinRouterBench/` source tree with PyPI name **`twinrouterbench`**.
- Meta-CLI `twinrouterbench` with subcommands `static`, `dynamic`, and `swe`.
- Optional extra **`[dynamic]`** for Docker / SWE-bench / mini-swe-agent / LiteLLM dependencies.
- Locked dynamic JSON under `data/dynamic/`.
- Single English `README.md` as the user-facing documentation entrypoint.
