"""Backward-compatible access to the config-driven built-in registry."""

from __future__ import annotations

from .benchmarking import BenchmarkSpec, builtin_registry


BenchmarkAdapter = BenchmarkSpec
_BUILTINS = builtin_registry()
ADAPTERS: dict[str, BenchmarkSpec] = {
    name: _BUILTINS.get(name) for name in _BUILTINS.names
}
BENCHMARK_ORDER: tuple[str, ...] = _BUILTINS.names


def get_adapter(name: str) -> BenchmarkSpec:
    return _BUILTINS.get(name)
