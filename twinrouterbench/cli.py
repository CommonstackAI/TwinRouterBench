"""Unified CLI for benchmark evaluation and data construction."""

from __future__ import annotations

import sys


def _require_dynamic_track() -> None:
    """Fail fast when dynamic dependencies are not installed."""
    try:
        import docker  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Twin Router Bench dynamic track requires optional dependencies.\n"
            'Install with: pip install "twinrouterbench[dynamic]"\n'
            f"Import error: {exc}"
        ) from exc


def _print_help() -> None:
    text = """Twin Router Bench — unified router benchmark suite.

Usage:
  twinrouterbench static <args>     Static track (question bank, nominal metrics)
  twinrouterbench data <args>       Build/review/publish static supervision
  twinrouterbench dynamic <args>    Dynamic track on mini-swe-agent (requires [dynamic])
  twinrouterbench swe <args>        Editor-scaffold SWE harness (requires [dynamic])

Install:
  pip install twinrouterbench              # static track only
  pip install "twinrouterbench[dynamic]"   # static + dynamic + swe CLI

See README.md in the TwinRouterBench directory for full documentation.
"""
    sys.stdout.write(text)


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        return 0

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "static":
        from main.cli import main as static_main

        static_main(rest)
        return 0

    if cmd == "data":
        from twinrouterbench.data_generation.cli import main as data_main

        return int(data_main(rest))

    if cmd == "dynamic":
        _require_dynamic_track()
        from miniswerouter.cli import main as dynamic_main

        return int(dynamic_main(rest))

    if cmd == "swe":
        _require_dynamic_track()
        from swerouter.cli import main as swe_main

        return int(swe_main(rest))

    sys.stderr.write(f"twinrouterbench: unknown command {cmd!r} (try --help)\n")
    return 2
