"""Section 11-style aggregates over per-row eval records (pass rate, nominal cost savings score)."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from main.pricing import step_full_cost_usd, step_nominal_cost_usd
from main.tiers import ID_TO_TIER, TIER_HIGH, public_tier_to_id
from main.tokenizer import (
    estimate_output_tokens_from_delta,
    split_prompt_tokens_for_step,
)

# Baseline trajectory: always route at highest public tier (guide table "高").
_BASELINE_TIER_ID = public_tier_to_id(TIER_HIGH)
_BASELINE_TIER = TIER_HIGH

# Uniform completion tokens per routing step (public bank has no per-step counts).
_ASSUMED_COMPLETION_TOKENS_PER_ROUTING_STEP = 1_000_000

# Fallback output tokens when estimation from message delta is not possible
# (single-turn cases / last step of a trajectory with no internal references).
_FALLBACK_OUTPUT_TOKENS = 500

# Cache TTL: if the same tier was last called more than this many global steps
# ago, the cache is considered expired and a full cache-write is needed.
_CACHE_TTL_STEPS = 3


def _step_nominal_usd(tier_id: int) -> float:
    return step_nominal_cost_usd(
        _ASSUMED_COMPLETION_TOKENS_PER_ROUTING_STEP,
        ID_TO_TIER[tier_id],
    )


def _overall_router_score_percent(a: float, b: float, c: float) -> float:
    """Mean of three 0–100 headline fields; NaN if any argument is NaN."""
    if math.isnan(a) or math.isnan(b) or math.isnan(c):
        return float("nan")
    return (a + b + c) / 3.0


def compute_section11(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Pass rate and cost savings score across rows.

    Pass: ``pred_tier_id >= gold_tier_id``. Rows with ``error`` count as not passed.

    Cost savings (§11.2), one routing step per row, uniform T.
    """
    passed = 0
    total = 0
    sum_save_gt = 0.0
    sum_save_test = 0.0
    cost_score_cases = 0

    baseline_cost = _step_nominal_usd(_BASELINE_TIER_ID)

    for r in rows:
        total += 1
        if "error" in r:
            continue
        pred = r["pred_tier_id"]
        gold = r["gold_tier_id"]
        if pred >= gold:
            passed += 1
            optimal_cost = _step_nominal_usd(gold)
            cost_test = _step_nominal_usd(pred)
            save_gt = baseline_cost - optimal_cost
            save_test = baseline_cost - cost_test
            if save_gt > 0:
                sum_save_gt += save_gt
                sum_save_test += save_test
                cost_score_cases += 1

    pass_rate = passed / total if total else float("nan")
    if sum_save_gt > 0:
        cost_savings_score = 100.0 * sum_save_test / sum_save_gt
    else:
        cost_savings_score = float("nan")

    return {
        "pass_rate": pass_rate,
        "passed": passed,
        "total": total,
        "cost_savings_score": cost_savings_score,
        "cost_score_cases_used": cost_score_cases,
        "sum_save_gt_usd": sum_save_gt,
        "sum_save_test_usd": sum_save_test,
        "assumed_completion_tokens_per_routing_step": _ASSUMED_COMPLETION_TOKENS_PER_ROUTING_STEP,
        "baseline_tier": TIER_HIGH,
        "baseline_nominal_cost_per_step_usd": baseline_cost,
        "cost_formula": "save_gt = baseline.cost - optimal_path.cost per §11.2 (nominal USD)",
        "note": "Sums are nominal USD under uniform T per step. If real completion_tokens v_i "
        "vary by case, §11.2 uses v_i in each cost; ratio equals current only when all v_i are equal.",
    }


def _compute_path_step_cost(
    *,
    tier: str,
    prev_tier: str | None,
    msgs_curr: list[dict[str, Any]],
    msgs_prev: list[dict[str, Any]] | None,
    output_tokens: int,
    cache_expired: bool = False,
) -> float:
    """Full cost (input + cache + output) for one step on a given routing path."""
    inp, cr, cw = split_prompt_tokens_for_step(
        prev_tier=prev_tier,
        curr_tier=tier,
        msgs_prev=msgs_prev,
        msgs_curr=msgs_curr,
        cache_expired=cache_expired,
    )
    return step_full_cost_usd(
        input_tokens=inp,
        cache_read_tokens=cr,
        cache_write_tokens=cw,
        output_tokens=output_tokens,
        tier=tier,
    )


def _estimate_trajectory_output_tokens(
    steps: list[dict[str, Any]],
) -> list[int]:
    """Return estimated output tokens for each step in a trajectory.

    For step N (N < len-1): estimated from the assistant-role delta between
    step N and step N+1.  For the last step (or if no delta available): uses
    the trajectory-internal average when available, else ``_FALLBACK_OUTPUT_TOKENS``.
    """
    n = len(steps)
    out: list[int | None] = [None] * n

    for i in range(n - 1):
        tier = ID_TO_TIER[steps[i]["gold_tier_id"]]
        out[i] = estimate_output_tokens_from_delta(
            steps[i]["messages"],
            steps[i + 1]["messages"],
            tier,
        )

    estimated = [t for t in out if t is not None and t > 0]
    avg = int(sum(estimated) / len(estimated)) if estimated else _FALLBACK_OUTPUT_TOKENS

    return [t if t is not None else avg for t in out]


def _compute_trajectory_pass_exact(
    rows: list[dict[str, Any]],
) -> tuple[int, int, int]:
    """
    Group rows by ``instance_id`` and return ``(passed, exact, total)`` trajectory counts.

    * A trajectory passes iff no step has an error AND every step has ``pred_tier_id >= gold_tier_id``.
    * A trajectory is exact iff no step has an error AND every step has ``pred_tier_id == gold_tier_id``.
    """
    traj_status = _build_trajectory_status(rows)
    total = len(traj_status)
    passed = sum(1 for st in traj_status.values() if not st["has_error"] and st["all_pass"])
    exact = sum(1 for st in traj_status.values() if not st["has_error"] and st["all_exact"])
    return passed, exact, total


def _build_trajectory_status(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Group rows by ``instance_id`` and return per-trajectory state:

        traj_status[iid] = {
            "has_error": bool,
            "all_pass": bool,    # every step has pred >= gold (and no error)
            "all_exact": bool,   # every step has pred == gold (and no error)
            "step_count": int,   # number of rows belonging to this trajectory
        }
    """
    traj_status: dict[str, dict[str, Any]] = {}
    for r in rows:
        iid = r.get("instance_id", r["id"])
        st = traj_status.setdefault(
            iid,
            {"has_error": False, "all_pass": True, "all_exact": True, "step_count": 0},
        )
        st["step_count"] += 1
        if "error" in r:
            st["has_error"] = True
            st["all_pass"] = False
            st["all_exact"] = False
            continue
        pred = r.get("pred_tier_id")
        gold = r.get("gold_tier_id")
        if not isinstance(pred, int) or not isinstance(gold, int):
            raise ValueError(
                "evaluable row must have int pred_tier_id and gold_tier_id "
                f"(id={r.get('id')!r})"
            )
        if pred < gold:
            st["all_pass"] = False
            st["all_exact"] = False
        elif pred != gold:
            st["all_exact"] = False
    return traj_status


def _iter_trajectory_step_costs(
    rows: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """
    Walk the rows trajectory-by-trajectory and yield per-evaluable-step records with
    full-cost numbers for baseline / gold / pred paths.

    Each yielded record:
        {
            "step": <row dict>,
            "baseline_cost": float,   # USD, always-high path
            "gold_cost": float,       # USD, gold tier path
            "pred_cost": float,       # USD, predicted tier path
            "trajectory_passed": bool,
            "trajectory_exact": bool,
            "instance_id": str,
        }

    This is the shared core for ``compute_router_accounting_metrics`` (v1) and
    ``compute_v2_scores`` (v2).  Rows with ``error`` are skipped but still mark the
    trajectory as failed.
    """
    traj_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        iid = r.get("instance_id", r["id"])
        traj_map[iid].append(r)
    for iid in traj_map:
        traj_map[iid].sort(key=lambda r: r.get("step_index", 1))

    for iid, steps in traj_map.items():
        has_error = any("error" in s for s in steps)
        all_step_pass = True
        all_step_exact = True
        evaluable_steps: list[dict[str, Any]] = []
        for s in steps:
            if "error" in s:
                all_step_pass = False
                all_step_exact = False
                continue
            pred = s.get("pred_tier_id")
            gold = s.get("gold_tier_id")
            if not isinstance(pred, int) or not isinstance(gold, int):
                raise ValueError(
                    "evaluable row must have int pred_tier_id and gold_tier_id "
                    f"(id={s.get('id')!r})"
                )
            if pred < gold:
                all_step_pass = False
                all_step_exact = False
            elif pred != gold:
                all_step_exact = False
            evaluable_steps.append(s)

        trajectory_passed = not has_error and all_step_pass
        trajectory_exact = not has_error and all_step_exact

        if not evaluable_steps:
            continue

        output_tokens_list = _estimate_trajectory_output_tokens(steps)
        step_idx_to_pos = {s.get("step_index", 1): i for i, s in enumerate(steps)}

        baseline_last_call: dict[str, int] = {}
        gold_last_call: dict[str, int] = {}
        pred_last_call: dict[str, int] = {}

        for s in evaluable_steps:
            pos = step_idx_to_pos[s.get("step_index", 1)]
            global_step = s.get("step_index", 1)
            pred_tier_id: int = s["pred_tier_id"]
            gold_tier_id: int = s["gold_tier_id"]
            pred_tier = ID_TO_TIER[pred_tier_id]
            gold_tier = ID_TO_TIER[gold_tier_id]
            msgs_curr: list[dict[str, Any]] = s.get("messages", [])
            output_tok = output_tokens_list[pos]

            msgs_prev: list[dict[str, Any]] | None = None
            prev_baseline_tier: str | None = None
            prev_gold_tier: str | None = None
            prev_pred_tier: str | None = None
            if pos > 0:
                prev_s = steps[pos - 1]
                msgs_prev = prev_s.get("messages", [])
                prev_baseline_tier = _BASELINE_TIER
                prev_gold_tier = (
                    ID_TO_TIER[prev_s["gold_tier_id"]]
                    if isinstance(prev_s.get("gold_tier_id"), int)
                    else None
                )
                prev_pred_tier = (
                    ID_TO_TIER[prev_s["pred_tier_id"]]
                    if isinstance(prev_s.get("pred_tier_id"), int)
                    else None
                )

            def _is_cache_expired(last_call: dict[str, int], tier: str) -> bool:
                prev_step = last_call.get(tier)
                if prev_step is None:
                    return False
                return (global_step - prev_step) > _CACHE_TTL_STEPS

            baseline_expired = _is_cache_expired(baseline_last_call, _BASELINE_TIER)
            gold_expired = _is_cache_expired(gold_last_call, gold_tier)
            pred_expired = _is_cache_expired(pred_last_call, pred_tier)

            baseline_cost = _compute_path_step_cost(
                tier=_BASELINE_TIER,
                prev_tier=prev_baseline_tier,
                msgs_curr=msgs_curr,
                msgs_prev=msgs_prev,
                output_tokens=output_tok,
                cache_expired=baseline_expired,
            )
            gold_cost = _compute_path_step_cost(
                tier=gold_tier,
                prev_tier=prev_gold_tier,
                msgs_curr=msgs_curr,
                msgs_prev=msgs_prev,
                output_tokens=output_tok,
                cache_expired=gold_expired,
            )
            pred_cost = _compute_path_step_cost(
                tier=pred_tier,
                prev_tier=prev_pred_tier,
                msgs_curr=msgs_curr,
                msgs_prev=msgs_prev,
                output_tokens=output_tok,
                cache_expired=pred_expired,
            )

            baseline_last_call[_BASELINE_TIER] = global_step
            gold_last_call[gold_tier] = global_step
            pred_last_call[pred_tier] = global_step

            yield {
                "step": s,
                "baseline_cost": baseline_cost,
                "gold_cost": gold_cost,
                "pred_cost": pred_cost,
                "trajectory_passed": trajectory_passed,
                "trajectory_exact": trajectory_exact,
                "instance_id": iid,
            }


def compute_router_accounting_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Trajectory-level accounting metrics with full cost model (input + cache + output).

    **Trajectory grouping**: rows are grouped by ``instance_id``.  Single-turn
    rows (``total_steps == 1``) each form their own trajectory.

    **Pass/fail at trajectory level**: a trajectory *fails* if any step has an
    ``error`` or any step's ``pred_tier_id < gold_tier_id``.

    **Cost model**: each step's cost = cache_read + cache_write + output,
    computed separately for baseline (always ``high``), gold, and pred paths.
    Cold starts (first step, tier switch, prefix mismatch, or cache TTL
    exceeded) bill all prompt tokens as cache_write; same-tier continuation
    within TTL uses cache_read for the prefix and cache_write for the delta.
    Cache TTL: if the same tier was last called more than 3 global steps ago,
    the cache is considered expired.

    * ``pass_rate_percent``: trajectory-level pass rate (100 * passed_trajectories / total)
    * ``exact_match_rate_percent``: 100 * (all steps exact) / total trajectories
    * ``accounting_savings_score_percent``: 100 * N / D
      - D = sum over all evaluable steps of (baseline_cost - gold_cost)
      - N: pass trajectory steps contribute (baseline_cost - pred_cost);
            fail trajectory steps contribute -pred_cost
    * ``overall_score_percent``: mean of the three above; NaN if any is NaN.

    Returned keys include ``total_trajectories``, ``D_usd``, ``N_usd``, and
    ``fallback_output_tokens`` (see package README for semantics).
    """
    passed_trajectories, exact_trajectories, total_trajectories = _compute_trajectory_pass_exact(rows)
    skipped_steps = sum(1 for r in rows if "error" in r)
    n_evaluable_steps = 0
    d_sum = 0.0
    n_sum = 0.0

    for rec in _iter_trajectory_step_costs(rows):
        n_evaluable_steps += 1
        baseline_cost = rec["baseline_cost"]
        gold_cost = rec["gold_cost"]
        pred_cost = rec["pred_cost"]
        d_sum += baseline_cost - gold_cost
        if rec["trajectory_passed"]:
            n_sum += baseline_cost - pred_cost
        else:
            n_sum -= pred_cost

    pass_rate_percent = (
        100.0 * passed_trajectories / total_trajectories
        if total_trajectories
        else float("nan")
    )
    exact_match_rate_percent = (
        100.0 * exact_trajectories / total_trajectories
        if total_trajectories
        else float("nan")
    )

    if d_sum > 0:
        accounting_savings_score_percent = 100.0 * n_sum / d_sum
        ratio_note = (
            "100 * N / D; trajectory-level pass/fail. "
            "Pass: N += baseline_cost - pred_cost; Fail: N -= pred_cost."
        )
    elif total_trajectories == 0:
        accounting_savings_score_percent = float("nan")
        ratio_note = "No trajectories."
    else:
        accounting_savings_score_percent = float("nan")
        ratio_note = "D == 0: no nominal savings vs always-high (e.g. all gold high)."

    overall_score_percent = _overall_router_score_percent(
        pass_rate_percent,
        exact_match_rate_percent,
        accounting_savings_score_percent,
    )

    return {
        "total_trajectories": total_trajectories,
        "passed_trajectories": passed_trajectories,
        "exact_match_trajectories": exact_trajectories,
        "evaluable_step_count": n_evaluable_steps,
        "skipped_step_count": skipped_steps,
        "D_usd": d_sum,
        "N_usd": n_sum,
        "pass_rate_percent": pass_rate_percent,
        "exact_match_rate_percent": exact_match_rate_percent,
        "accounting_savings_score_percent": accounting_savings_score_percent,
        "overall_score_percent": overall_score_percent,
        "fallback_output_tokens": _FALLBACK_OUTPUT_TOKENS,
        "note": ratio_note,
    }


def _compute_cost_savings_per_benchmark(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Full-cost savings aggregation per benchmark under the **natural-accounting**
    user-bill model (trajectory-level pass/fail).  All gold tiers are included
    (no row is excluded; gold=high steps are accounted for by every path's real
    cost under the full-cost model).

    User-bill model:
        passed trajectory  -> user_bill = Σ pred_cost                    (router ran successfully)
        failed  trajectory -> user_bill = Σ pred_cost + Σ baseline_cost  (router's whole chain
                                                                          wasted + one always-high
                                                                          re-run of every step)

    Trajectory-level accumulation (same grouping as
    ``compute_router_accounting_metrics``):

        D_b += baseline_cost                                   # per evaluable step
        if trajectory_passed:                                  # no error AND every step pred>=gold
            N_b += baseline_cost - pred_cost                   # credit step-level savings
        else:                                                  # any error or any step pred<gold
            N_b -= pred_cost                                   # router's step cost is wasted;
                                                               # the Σ baseline retry is already
                                                               # covered by D - (user_bill - pred)

    Verification (all-failed trajectory, all steps pred<gold):
        Σ N = -Σ pred;  savings = Σ baseline - user_bill = Σ baseline - (Σ pred + Σ baseline)
                              = -Σ pred.  ✓ matches.

    Verification (mixed failed trajectory, some steps passed):
        Every step contributes -pred regardless of step-level pass/fail —
        because the whole chain has to be re-run once the trajectory failed,
        the step-level "savings" on individually-passed steps inside a failed
        trajectory are not credited.  This matches the user's intent that a
        failed multi-turn trajectory costs "router's whole chain pred + one
        full-high re-run of the whole chain".

    Interpretation: ``cost_savings_score = 100 * N / D`` reads as "fraction of
    the always-high total bill that was actually saved, after accounting for
    one always-high retry on every failed trajectory".  Score lies in
    ``(-∞, 100]``; stays in ``[0, 100]`` as long as total waste + implicit
    retry stays below the baseline total.

    Returns per-benchmark dict:
        {
            "D_usd", "N_usd",
            "step_count",                  # evaluable step count (no error)
            "failed_trajectory_count",
            "failed_retry_baseline_usd",   # Σ baseline_cost over all evaluable
                                           # steps of every failed trajectory —
                                           # informational only; already implicit
                                           # in the N = -Σ pred formula.
        }

    Note: ``input/cache_read/cache_write/output`` are all combined inside
    ``baseline_cost`` / ``gold_cost`` / ``pred_cost`` via ``step_full_cost_usd``
    (see ``main/pricing.py``).  The current token splitter
    (``split_prompt_tokens_for_step``) assigns ``input_tokens = 0`` under a
    "prompt cache always on" modelling assumption, so every prompt token is
    billed as either cache_read (prefix hit) or cache_write (cold start / tier
    switch / cache expired / delta).
    """
    per_bench: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "D_usd": 0.0,
            "N_usd": 0.0,
            "step_count": 0,
            "failed_trajectory_count": 0,
            "failed_retry_baseline_usd": 0.0,
        }
    )

    traj_status = _build_trajectory_status(rows)

    failed_traj_seen_by_bench: dict[str, set[str]] = defaultdict(set)

    for rec in _iter_trajectory_step_costs(rows):
        s = rec["step"]
        b = s["benchmark"]
        iid = rec["instance_id"]
        st = traj_status.get(iid)
        if st is None:
            raise ValueError(
                f"trajectory status missing for evaluable row (iid={iid!r}); "
                "this indicates _build_trajectory_status / _iter_trajectory_step_costs "
                "are out of sync."
            )
        trajectory_passed = not st["has_error"] and st["all_pass"]

        pb = per_bench[b]
        pb["step_count"] += 1
        pb["D_usd"] += rec["baseline_cost"]
        if trajectory_passed:
            pb["N_usd"] += rec["baseline_cost"] - rec["pred_cost"]
        else:
            pb["N_usd"] -= rec["pred_cost"]
            pb["failed_retry_baseline_usd"] += rec["baseline_cost"]
            if iid not in failed_traj_seen_by_bench[b]:
                failed_traj_seen_by_bench[b].add(iid)
                pb["failed_trajectory_count"] += 1

    # Count failed trajectories that consist entirely of error rows (no
    # evaluable steps emitted from _iter_trajectory_step_costs).  Their
    # contribution to D/N is zero, but we still want them visible in the
    # failed_trajectory_count for reporting.
    for iid, st in traj_status.items():
        if not st["has_error"] and st["all_pass"]:
            continue
        if any(iid in seen for seen in failed_traj_seen_by_bench.values()):
            continue
        bench: str | None = None
        for r in rows:
            riid = r.get("instance_id", r["id"])
            if riid == iid:
                bench = r["benchmark"]
                break
        if bench is None:
            continue
        failed_traj_seen_by_bench[bench].add(iid)
        per_bench[bench]["failed_trajectory_count"] += 1

    return dict(per_bench)


def compute_v2_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Headline v2 scoring.  Produces exactly FOUR component scores plus their mean.

    1. ``case_pass_rate_percent`` — per-row ``pred_tier_id >= gold_tier_id``
       (rows with errors count as failures; denominator = total row count).
    2. ``case_exact_match_percent`` — per-row ``pred_tier_id == gold_tier_id``
       (same denominator; errors count as misses).
    3. ``trajectory_pass_rate_percent`` — fraction of **rows** whose entire
       trajectory passes (every step has ``pred_tier_id >= gold_tier_id`` and
       no error).  Denominator is total rows (same as metric 1) so each
       evaluation step is weighted equally; trajectory-level all-or-nothing
       gating is still applied. Guarantees ``trajectory_pass ≤ case_pass``.
    4. ``cost_savings_score_percent`` — full-cost savings ratio under the
       natural-accounting user-bill model, aggregated over **all rows (every
       gold tier, including gold=high)**:
         (a) Denominator: ``D_b += baseline_cost`` per evaluable step (the
             "always-high" total bill);
         (b) Trajectory-level pass/fail drives the numerator:
               - passed trajectory (no error AND every step pred>=gold):
                     ``N_b += baseline_cost - pred_cost`` per step;
               - failed trajectory (any step has an error OR pred<gold):
                     ``N_b -= pred_cost`` per step.
             The failed-trajectory user bill is thereby modelled as
             ``Σ pred_cost`` (router's wasted chain) + ``Σ baseline_cost``
             (one full always-high re-run of every step in the chain);
             savings relative to always-high reduce to exactly ``-Σ pred``.
       Across benchmarks the score is **macro-weighted by total row count**
       per benchmark (same denominator scope as metric 1).  Score lies in
       ``(-∞, 100]``; stays in ``[0, 100]`` whenever user bill stays below
       the always-high bill.
    5. ``combined_score_percent`` — arithmetic mean of the four scores above.

    Rationale for the cost_savings scope (``D = Σ baseline`` over all rows):
    ``gold=high`` rows inherently have no cost-savings headroom (baseline
    already equals gold).  Using ``D = Σ (baseline - gold)`` (v1 accounting)
    under-weights ``gold=high``-dominated benchmarks because their D collapses
    to ~0.  Using ``D = Σ baseline`` keeps each benchmark's weight proportional
    to its row count, and makes the score read as "what fraction of the
    always-high total bill was actually saved".
    """
    total_rows = len(rows)
    error_rows = sum(1 for r in rows if "error" in r)

    case_pass = 0
    case_exact = 0
    for r in rows:
        if "error" in r:
            continue
        pred = r.get("pred_tier_id")
        gold = r.get("gold_tier_id")
        if not isinstance(pred, int) or not isinstance(gold, int):
            raise ValueError(
                "evaluable row must have int pred_tier_id and gold_tier_id "
                f"(id={r.get('id')!r})"
            )
        if pred >= gold:
            case_pass += 1
        if pred == gold:
            case_exact += 1

    case_pass_rate_percent = 100.0 * case_pass / total_rows if total_rows else float("nan")
    case_exact_match_percent = 100.0 * case_exact / total_rows if total_rows else float("nan")

    traj_status = _build_trajectory_status(rows)
    total_traj = len(traj_status)
    passed_traj = sum(
        1 for st in traj_status.values() if not st["has_error"] and st["all_pass"]
    )
    # Case-weighted trajectory pass: a row contributes to the numerator iff the
    # trajectory it belongs to passes (all steps pred>=gold, no error).
    # Denominator is total_rows so metric 3 and metric 1 share the same scale and
    # trajectory_pass <= case_pass is guaranteed by construction.
    rows_in_passed_trajectories = sum(
        st["step_count"] for st in traj_status.values() if not st["has_error"] and st["all_pass"]
    )
    trajectory_pass_rate_percent = (
        100.0 * rows_in_passed_trajectories / total_rows if total_rows else float("nan")
    )

    per_bench_rowcount: dict[str, int] = defaultdict(int)
    for r in rows:
        per_bench_rowcount[r["benchmark"]] += 1

    per_bench_raw = _compute_cost_savings_per_benchmark(rows)
    per_bench_scores: dict[str, dict[str, Any]] = {}
    for b, rowcount in per_bench_rowcount.items():
        raw = per_bench_raw.get(
            b,
            {
                "D_usd": 0.0,
                "N_usd": 0.0,
                "step_count": 0,
                "failed_trajectory_count": 0,
                "failed_retry_baseline_usd": 0.0,
            },
        )
        d_b = raw["D_usd"]
        n_b = raw["N_usd"]
        cost_b = 100.0 * n_b / d_b if d_b > 0 else float("nan")
        per_bench_scores[b] = {
            "row_count": rowcount,
            "step_count": raw["step_count"],
            "failed_trajectory_count": raw["failed_trajectory_count"],
            "failed_retry_baseline_usd": raw["failed_retry_baseline_usd"],
            "D_usd": d_b,
            "N_usd": n_b,
            "cost_savings_score_percent": cost_b,
            "weight_in_global_cost_savings": 0.0,
        }

    total_row_for_weight = sum(per_bench_rowcount.values())
    if total_row_for_weight > 0:
        weighted_sum = 0.0
        have_any = False
        for b, sc in per_bench_scores.items():
            w = sc["row_count"] / total_row_for_weight
            sc["weight_in_global_cost_savings"] = w
            s = sc["cost_savings_score_percent"]
            if not math.isnan(s):
                weighted_sum += w * s
                have_any = True
        cost_savings_score_percent = weighted_sum if have_any else float("nan")
    else:
        cost_savings_score_percent = float("nan")

    components = [
        case_pass_rate_percent,
        case_exact_match_percent,
        trajectory_pass_rate_percent,
        cost_savings_score_percent,
    ]
    if any(math.isnan(c) for c in components):
        combined_score_percent = float("nan")
    else:
        combined_score_percent = sum(components) / len(components)

    return {
        "case_pass_rate_percent": case_pass_rate_percent,
        "case_exact_match_percent": case_exact_match_percent,
        "trajectory_pass_rate_percent": trajectory_pass_rate_percent,
        "cost_savings_score_percent": cost_savings_score_percent,
        "combined_score_percent": combined_score_percent,
        "total_rows": total_rows,
        "error_rows": error_rows,
        "case_pass_count": case_pass,
        "case_exact_count": case_exact,
        "total_trajectories": total_traj,
        "passed_trajectories": passed_traj,
        "rows_in_passed_trajectories": rows_in_passed_trajectories,
        "by_benchmark": per_bench_scores,
        "note": (
            "cost_savings_score_percent is macro-averaged across benchmarks by each "
            "benchmark's total row count (same scope as case_pass_rate). All gold tiers "
            "are included; D_b = Σ baseline_cost over every evaluable step. Numerator is "
            "trajectory-level: passed trajectories (no error AND every step pred>=gold) "
            "contribute N_b += baseline_cost - pred_cost per step; failed trajectories "
            "contribute N_b -= pred_cost per step, which under D = Σ baseline encodes the "
            "failed-trajectory user bill = Σ pred_cost + Σ baseline_cost (router's chain "
            "wasted + one full always-high re-run of every step). No additional retry "
            "penalty is applied on top. trajectory_pass_rate_percent uses row-weighted "
            "denominator: a row counts toward the numerator iff its entire trajectory "
            "passes."
        ),
    }


def aggregate_by_benchmark(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | Any]]:
    """Per-benchmark buckets with tier accuracy, valid rate, and nested Section 11 fields."""
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["benchmark"]].append(r)

    out: dict[str, dict[str, Any]] = {}
    for b in sorted(by.keys()):
        brows = by[b]
        s = len(brows)
        err = sum(1 for r in brows if "error" in r)
        ok_parse = s - err
        exact = sum(1 for r in brows if r.get("match"))
        s11 = compute_section11(brows)
        acct = compute_router_accounting_metrics(brows)
        out[b] = {
            "sampled": s,
            "exact_match": exact,
            "api_errors": err,
            "tier_match_accuracy": exact / ok_parse if ok_parse else float("nan"),
            "valid_response_rate": ok_parse / s if s else float("nan"),
            "pass_rate": s11["pass_rate"],
            "passed": s11["passed"],
            "cost_savings_score": s11["cost_savings_score"],
            "cost_score_cases_used": s11["cost_score_cases_used"],
            "router_accounting": acct,
        }
    return out
