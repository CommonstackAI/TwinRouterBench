"""Score a completed run directory with the single-metric leaderboard rule.

The scorer is pure offline: it reads ``results/<instance>.json`` and the
per-instance ``*.trace.jsonl`` files produced by :mod:`swerouter.harness`` and
applies the formulas documented in ``docs/scoring_zh.md``.

Penalty rule (v2 — fixed opportunity-cost add-on)
-------------------------------------------------
* **Resolved** instance: ``instance_bill = router_actual_cost``.
* **Unresolved** instance: ``instance_bill = router_actual_cost + FAILURE_PENALTY_USD``.

``FAILURE_PENALTY_USD`` is a fixed constant (default \$0.55) representing the
empirical average cost of running the all-Opus baseline on the evaluation set.
This decouples the penalty from step count and pricing tables, avoids
"long trace → exploding penalty" artifacts, and keeps the leaderboard formula
trivial to describe.

Outputs
-------

* ``total_leaderboard_bill_usd`` — sole leaderboard sort key (lower is better).
  Equals ``Σ router_actual + 0.55 × #unresolved``.
  This is **not** raw API spend; for realized routed spend only use
  ``total_router_cost_usd``.
* Auxiliary columns (``total_router_cost_usd``, ``total_penalty_cost_usd``,
  ``resolved_count``, ``resolved_rate``, ``avg_steps``,
  ``avg_cost_per_resolved``) for human readability.
* ``per_instance`` breakdown so human auditors can see where every dollar went.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from swerouter.agent.loop import ModelPoolEntry, load_model_pool
from swerouter.infra_errors import is_excluded_from_fair_metrics
from swerouter.pricing import PricingTable, load_pricing_table, step_real_cost_usd
from swerouter.usage import normalize_usage

FAILURE_PENALTY_USD: float = 0.55
"""Fixed per-instance penalty added to unresolved cases.

This is the empirical average cost of running the all-Opus baseline on the
evaluation set, used as a fixed opportunity-cost proxy.  It intentionally
does NOT scale with step count or token volume so that "long failing traces"
do not cause explosive penalties.
"""

REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCKED_DATA = REPO_ROOT / "data" / "dynamic"
DEFAULT_POOL = _LOCKED_DATA / "model_pool.json"
DEFAULT_PRICING = _LOCKED_DATA / "model_pricing.json"


def _iter_trace_steps(trace_path: Path) -> list[dict]:
    """Return only the step records (no loop_summary marker row)."""
    if not trace_path.is_file():
        raise FileNotFoundError(f"trace file missing: {trace_path}")
    steps: list[dict] = []
    for raw in trace_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as ex:
            raise ValueError(f"corrupt trace line in {trace_path}: {ex}") from ex
        if "__marker__" in row:
            continue
        steps.append(row)
    return steps


def _extract_step_series(
    steps: list[dict],
) -> tuple[list[int], list[int], list[float]]:
    """Return (prefix_tokens, output_tokens, timestamps) aligned by step_index."""
    steps_sorted = sorted(steps, key=lambda s: int(s["step_index"]))
    prefix: list[int] = []
    output: list[int] = []
    ts: list[float] = []
    for s in steps_sorted:
        usage = s.get("usage") or {}
        prefix.append(
            int(usage.get("input_tokens", 0))
            + int(usage.get("cache_read_tokens", 0))
            + int(usage.get("cache_write_tokens", 0))
        )
        output.append(int(usage.get("output_tokens", 0)))
        ts.append(float(s.get("started_at", 0.0)))
    return prefix, output, ts


def _provider_by_model_id(pool: list[ModelPoolEntry]) -> dict[str, str]:
    return {entry.model_id: entry.provider for entry in pool}


def _repriced_router_and_series(
    steps: list[dict],
    *,
    provider_by_model: dict[str, str],
    pricing: PricingTable,
) -> tuple[float, list[int], list[int], list[float]]:
    """Recompute each step's USD cost and prompt/output series from ``raw_usage``.

    Uses current :func:`swerouter.usage.normalize_usage` + ``step_real_cost_usd``
    so historical traces pick up mapping fixes (e.g. OpenRouter
    ``cache_write_tokens``) without re-running Docker.
    """

    steps_sorted = sorted(steps, key=lambda s: int(s["step_index"]))
    router_actual = 0.0
    prefix: list[int] = []
    output: list[int] = []
    ts: list[float] = []
    for s in steps_sorted:
        model_id = s.get("model_id")
        if not isinstance(model_id, str) or model_id not in provider_by_model:
            raise KeyError(
                f"repricing: trace step has missing/unknown model_id {model_id!r} "
                f"(instance trace ordering)"
            )
        raw = s.get("raw_usage")
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"repricing: step missing raw_usage mapping (model_id={model_id!r})"
            )
        provider = provider_by_model[model_id]
        buckets = normalize_usage(provider, dict(raw))
        router_actual += step_real_cost_usd(buckets, pricing.get(model_id))
        prefix.append(buckets.total_prompt_tokens)
        output.append(buckets.output_tokens)
        ts.append(float(s.get("started_at", 0.0)))
    return router_actual, prefix, output, ts




def _load_result_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def score_run_dir(
    *,
    run_dir: Path,
    router_label: str,
    pricing_path: Path | None = None,
    ttl_path: Path | None = None,
    pool_path: Path | None = None,
    exclude_infra_failures: bool = False,
    reprice_from_raw_usage: bool = False,
) -> dict[str, Any]:
    """Compute leaderboard numbers for a run directory produced by :mod:`swerouter.harness`.

    Parameters
    ----------
    run_dir
        The directory passed to ``run_eval`` as ``output_dir``. Must contain
        ``results/<instance_id>.json`` files and the ``*.trace.jsonl`` files.
    router_label
        Human-readable router label written into the score report.
    pricing_path / pool_path
        Optional overrides; defaults to ``data/dynamic/*.json`` under TwinRouterBench
        repo root.
    ttl_path
        Retained for CLI compatibility but **no longer used** by the scorer
        (the fixed-penalty rule does not simulate baseline cache sequences).
    exclude_infra_failures
        When True, instances whose ``agent_error`` or ``eval_error`` matches
        :func:`swerouter.infra_errors.is_transport_or_infra_failure` are omitted
        from headline totals; they remain in ``per_instance`` with
        ``excluded_from_metrics: true``. ``instance_count`` then counts only
        eligible instances; ``raw_instance_count`` / ``infra_excluded_count``
        are also written.
    reprice_from_raw_usage
        When True, recompute each step's router USD from ``raw_usage`` in the
        trace using the current ``normalize_usage`` + ``model_pricing.json``
        (fixes historical traces written under older usage mapping). Raises if
        any step lacks ``raw_usage`` or ``model_id``.
    """

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory missing: {run_dir}")

    pricing = load_pricing_table(pricing_path or DEFAULT_PRICING)
    pool = load_model_pool(pool_path or DEFAULT_POOL)
    high_model_id = next(p.model_id for p in pool if p.is_high_baseline)
    provider_by_model = _provider_by_model_id(pool)

    results_dir = run_dir / "results"
    if not results_dir.is_dir():
        raise FileNotFoundError(f"expected {results_dir} with per-instance result JSONs")
    result_paths = sorted(results_dir.glob("*.json"))
    if not result_paths:
        raise ValueError(f"no per-instance result files under {results_dir}")

    per_instance: list[dict] = []
    total_bill = 0.0
    total_router_cost = 0.0
    total_penalty = 0.0
    resolved_count = 0
    step_counts: list[int] = []

    for rp in result_paths:
        blob = _load_result_file(rp)
        instance_id = blob["instance_id"]
        resolved = bool(blob.get("resolved", False))
        agent_err = blob.get("agent_error")
        eval_err = blob.get("eval_error")
        infra_excluded = is_excluded_from_fair_metrics(
            agent_err if isinstance(agent_err, str) else None,
            eval_err if isinstance(eval_err, str) else None,
        )
        counted = (not exclude_infra_failures) or (not infra_excluded)

        trace_path = Path(blob.get("trace_path") or "")
        if not trace_path.is_absolute():
            trace_path = (run_dir / trace_path).resolve()

        steps = _iter_trace_steps(trace_path)
        if reprice_from_raw_usage:
            router_actual, _prefix, _output, _timestamps = _repriced_router_and_series(
                steps,
                provider_by_model=provider_by_model,
                pricing=pricing,
            )
        else:
            router_actual = sum(float(s.get("step_cost_usd", 0.0)) for s in steps)

        if resolved:
            instance_bill = router_actual
            penalty = 0.0
        else:
            penalty = FAILURE_PENALTY_USD
            instance_bill = router_actual + penalty

        row = {
            "instance_id": instance_id,
            "resolved": resolved,
            "step_count": len(steps),
            "router_actual_cost_usd": router_actual,
            "penalty_usd": penalty,
            "instance_bill_usd": instance_bill,
            "model_distribution": blob.get("model_distribution") or {},
            "patch_applied": bool(blob.get("patch_applied", False)),
            "agent_error": blob.get("agent_error"),
            "eval_error": blob.get("eval_error"),
            "excluded_from_metrics": bool(exclude_infra_failures and infra_excluded),
        }
        per_instance.append(row)

        if counted:
            total_bill += instance_bill
            total_router_cost += router_actual
            total_penalty += penalty
            if resolved:
                resolved_count += 1
            step_counts.append(len(steps))

    raw_n = len(per_instance)
    infra_excluded_n = sum(1 for r in per_instance if r.get("excluded_from_metrics"))
    n = raw_n - infra_excluded_n if exclude_infra_failures else raw_n
    avg_steps = sum(step_counts) / n if n else 0.0
    avg_cost_per_resolved = (
        total_bill / resolved_count if resolved_count > 0 else float("inf")
    )

    pool_fingerprint = "|".join(sorted(p.model_id for p in pool))
    pricing_fp = f"{pricing.schema_version}.{pool_fingerprint}"

    out: dict[str, Any] = {
        "router_label": router_label,
        "run_dir": str(run_dir),
        "pool_fingerprint": pool_fingerprint,
        "pricing_schema_version": pricing.schema_version,
        "pricing_fingerprint": pricing_fp,
        "high_baseline_model_id": high_model_id,
        "failure_penalty_usd": FAILURE_PENALTY_USD,
        "total_leaderboard_bill_usd": total_bill,
        "total_router_cost_usd": total_router_cost,
        "total_penalty_cost_usd": total_penalty,
        "resolved_count": resolved_count,
        "resolved_rate": resolved_count / n if n else 0.0,
        "instance_count": n,
        "avg_steps": avg_steps,
        "avg_cost_per_resolved_usd": avg_cost_per_resolved,
        "per_instance": per_instance,
    }
    if exclude_infra_failures:
        out["exclude_infra_failures"] = True
        out["raw_instance_count"] = raw_n
        out["infra_excluded_count"] = infra_excluded_n
    if reprice_from_raw_usage:
        out["reprice_from_raw_usage"] = True
    return out


__all__ = ["score_run_dir"]
