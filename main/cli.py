"""CLI for Section 11 aggregate metrics from JSON."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from main.metrics import aggregate_routerbench_metrics, case_metrics_from_dict


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="twinrouterbench static",
        description="TwinRouterBench — static track utilities",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("metrics", help="Aggregate Section 11 metrics from a cases JSON file")
    m.add_argument("--cases", required=True, help="Path to JSON array of CaseMetrics objects")

    args = parser.parse_args(argv)
    if args.cmd == "metrics":
        _cmd_metrics(args.cases)


def _cmd_metrics(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        raw: Any = json.load(f)
    if not isinstance(raw, list):
        raise SystemExit("cases file must be a JSON array")
    cases = [case_metrics_from_dict(item) for item in raw]
    summary = aggregate_routerbench_metrics(cases)
    json.dump(summary, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
