"""Aggregate per-step costs from ``*.trace.jsonl`` for a completed run directory."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any


def audit_trace_cost_metrics(run_dir: Path) -> dict[str, Any]:
    """Sum ``step_cost_usd`` and ``raw_usage.cost`` from trace rows.

    Walks ``run_dir/*.trace.jsonl`` (non-recursive), skips ``__marker__`` rows.
    """

    run_dir = Path(run_dir).resolve()
    sum_step = 0.0
    sum_raw = 0.0
    steps = 0
    traces = 0

    for trace_path in sorted(run_dir.glob("*.trace.jsonl")):
        if not trace_path.is_file():
            continue
        traces += 1
        for raw in trace_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("__marker__"):
                continue
            steps += 1
            sum_step += float(row.get("step_cost_usd") or 0.0)
            ru = row.get("raw_usage")
            if isinstance(ru, Mapping):
                c = ru.get("cost")
                if isinstance(c, (int, float)):
                    sum_raw += float(c)

    out: dict[str, Any] = {
        "run_dir": str(run_dir),
        "trace_files": traces,
        "step_rows": steps,
        "sum_step_cost_usd": sum_step,
        "sum_raw_usage_cost_usd": sum_raw,
        "delta_raw_minus_step_usd": sum_raw - sum_step,
        "ratio_raw_over_step": (sum_raw / sum_step) if sum_step > 0 else None,
    }
    return out


__all__ = ["audit_trace_cost_metrics"]
