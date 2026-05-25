"""Render a Markdown leaderboard from one or more score files produced by
:func:`swerouter.leaderboard.score.score_run_dir`.

The primary sort key is ``total_leaderboard_bill_usd`` (ascending: less money =
better rank). Auxiliary columns are printed for human readability; they do not
influence the rank. Score files with mismatched ``pricing_fingerprint`` are
rendered as separate tables to avoid hidden apples-vs-oranges comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _fmt_usd(x: float) -> str:
    if x == float("inf"):
        return "inf"
    return f"${x:,.4f}"


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _leaderboard_bill_usd(row: dict) -> float:
    """Penalty-inclusive leaderboard total (USD). Supports legacy ``score.json``."""

    if "total_leaderboard_bill_usd" in row:
        return float(row["total_leaderboard_bill_usd"])
    return float(row.get("total_actual_bill_usd", float("inf")))


def _load_scores(score_files: Iterable[Path]) -> list[dict]:
    entries: list[dict] = []
    for p in score_files:
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(f"score file missing: {p}")
        with p.open("r", encoding="utf-8") as fh:
            entries.append(json.load(fh))
    return entries


def _group_by_pricing(entries: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {}
    for e in entries:
        fp = e.get("pricing_fingerprint", "unknown")
        buckets.setdefault(fp, []).append(e)
    return buckets


def _render_one_table(group_label: str, rows: list[dict]) -> str:
    sorted_rows = sorted(rows, key=_leaderboard_bill_usd)
    out: list[str] = []
    out.append(f"### {group_label}")
    out.append("")
    out.append(
        "| Rank | Router | total_leaderboard_bill_usd (sort) | resolved_rate | resolved | "
        "total_router_cost_usd | total_penalty_cost_usd | avg_steps | avg_cost_per_resolved_usd | instances |"
    )
    out.append(
        "|-----:|:-------|-----------------------------:|--------------:|---------:|----------------------:|-----------------------:|----------:|--------------------------:|----------:|"
    )
    for i, e in enumerate(sorted_rows, start=1):
        out.append(
            "| {rank} | {label} | {total_bill} | {resolved_rate} | {resolved}/{instances} | "
            "{router_cost} | {penalty} | {avg_steps:.2f} | {avg_cpr} | {instances} |".format(
                rank=i,
                label=e.get("router_label", "?"),
                total_bill=_fmt_usd(_leaderboard_bill_usd(e)),
                resolved_rate=_fmt_pct(float(e.get("resolved_rate", 0.0))),
                resolved=int(e.get("resolved_count", 0)),
                instances=int(e.get("instance_count", 0)),
                router_cost=_fmt_usd(float(e.get("total_router_cost_usd", 0.0))),
                penalty=_fmt_usd(float(e.get("total_penalty_cost_usd", 0.0))),
                avg_steps=float(e.get("avg_steps", 0.0)),
                avg_cpr=_fmt_usd(float(e.get("avg_cost_per_resolved_usd", float("inf")))),
            )
        )
    out.append("")
    return "\n".join(out)


def render_leaderboard(score_files: Iterable[Path | str]) -> str:
    """Render a markdown string with one table per ``pricing_fingerprint`` group."""
    paths = [Path(p) for p in score_files]
    entries = _load_scores(paths)
    if not entries:
        raise ValueError("no score files provided")

    buckets = _group_by_pricing(entries)
    sections: list[str] = []
    sections.append("# TwinRouterBench Leaderboard")
    sections.append("")
    sections.append(
        "Primary sort key: `total_leaderboard_bill_usd` ascending (lower USD = better). "
        "Auxiliary columns are informational only and do not influence rank."
    )
    sections.append("")

    if len(buckets) > 1:
        sections.append(
            "> Multiple `pricing_fingerprint` groups detected; rendered as separate tables. "
            "Cross-group comparisons are invalid (pricing or pool changed between runs)."
        )
        sections.append("")

    for fp in sorted(buckets):
        sections.append(_render_one_table(f"pricing_fingerprint `{fp}`", buckets[fp]))

    return "\n".join(sections).rstrip() + "\n"


__all__ = ["render_leaderboard"]
