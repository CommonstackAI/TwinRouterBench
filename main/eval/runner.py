"""Run eval over selected question-bank rows and build the standard summary JSON object."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from main.dataset import QUESTION_BANK_PATH
from main.eval.predictors import QuestionBankRouterPredictor
from main.eval.section11 import (
    aggregate_by_benchmark,
    compute_router_accounting_metrics,
    compute_section11,
    compute_v2_scores,
)
from main.eval.sampling import rows_per_benchmark, select_question_bank_rows


def evaluate_question_bank_rows(
    predictor: QuestionBankRouterPredictor,
    rows: list[dict[str, Any]],
    *,
    predictor_label: str,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """
    Return ``(per_row_records, errors, exact_match_count)``.

    Each successful record includes ``id``, ``benchmark``, ``gold_tier_id``, ``pred_tier_id``,
    ``match``, ``passed``, and optional ``usage``. Failed predictions add ``error`` instead of pred fields.
    """
    per_row: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    correct = 0
    n_total = len(rows)

    for idx, row in enumerate(rows, start=1):
        rid = row["id"]
        bench = row["benchmark"]
        if not isinstance(bench, str):
            raise ValueError(f"row {rid!r} missing string benchmark")
        gold = row["target_tier_id"]
        if not isinstance(gold, int):
            raise ValueError(f"row {rid!r} target_tier_id must be int")

        try:
            pred_obj = predictor.predict(row)
            pred = pred_obj.tier_id
            ok = pred == gold
            if ok:
                correct += 1
            passed = pred >= gold
            rec: dict[str, Any] = {
                "id": rid,
                "benchmark": bench,
                "gold_tier_id": gold,
                "pred_tier_id": pred,
                "match": ok,
                "passed": passed,
                "instance_id": row.get("instance_id", rid),
                "step_index": row.get("step_index", 1),
                "total_steps": row.get("total_steps", 1),
                "messages": row.get("messages", []),
            }
            if pred_obj.usage is not None:
                rec["usage"] = pred_obj.usage
            per_row.append(rec)
            if progress:
                progress(
                    f"[{idx}/{n_total}] id={rid} benchmark={bench} "
                    f"pred_tier_id={pred} gold_tier_id={gold} "
                    f"exact_match={ok} pass_pred_ge_gold={passed}"
                )
        except Exception as ex:  # noqa: BLE001 — record per-row failure for batch evals
            err_s = str(ex)
            errors.append({"id": rid, "error": err_s})
            per_row.append(
                {
                    "id": rid,
                    "benchmark": bench,
                    "gold_tier_id": gold,
                    "error": err_s,
                    "instance_id": row.get("instance_id", rid),
                    "step_index": row.get("step_index", 1),
                    "total_steps": row.get("total_steps", 1),
                    "messages": row.get("messages", []),
                }
            )
            if progress:
                progress(
                    f"[{idx}/{n_total}] id={rid} benchmark={bench} "
                    f"pred_tier_id=<error> gold_tier_id={gold} "
                    f"exact_match=False pass_pred_ge_gold=False err={ex!r}"
                )

    return per_row, errors, correct


def build_eval_summary(
    *,
    per_row: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    correct: int,
    predictor_label: str,
    shard: Path | str,
    sample_mode: str,
    seed: int,
    proportional_quotas: dict[str, int] | None,
    benchmark_counts: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the same top-level JSON shape as the live eval CLI."""
    n = len(per_row)
    err_n = len(errors)
    graded = n - err_n
    counts = benchmark_counts if benchmark_counts is not None else rows_per_benchmark(per_row)
    s11_global = compute_section11(per_row)
    router_acct = compute_router_accounting_metrics(per_row)
    scores_v2 = compute_v2_scores(per_row)
    tier_acc_evaluable = correct / graded if graded else float("nan")
    summary: dict[str, Any] = {
        "classifier": predictor_label,
        "shard": str(shard),
        "sample_mode": sample_mode,
        "sampled": n,
        "seed": seed,
        "proportional_quotas": proportional_quotas,
        "benchmark_counts": counts,
        "exact_match": correct,
        "tier_match_accuracy": tier_acc_evaluable,
        "accuracy_excluding_errors": tier_acc_evaluable,
        "api_errors": err_n,
        "valid_response_rate": graded / n if n else float("nan"),
        "scores_v2": scores_v2,
        "section_11": s11_global,
        "router_accounting": router_acct,
        "by_benchmark": aggregate_by_benchmark(per_row),
        "errors": errors,
        "rows": per_row,
    }
    if extra:
        # caller keys first so core fields win on conflict
        summary = {**extra, **summary}
    return summary


def run_question_bank_eval(
    predictor: QuestionBankRouterPredictor,
    *,
    predictor_label: str,
    n: int | None = None,
    seed: int = 42,
    shard: Path | None = None,
    progress: Callable[[str], None] | None = None,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Select rows (full bank or proportional), run predictor, return summary dict.

    ``extra_summary`` is merged into the output (e.g. ``{"model": "..."}``); core metric keys win.
    """
    path = shard if shard is not None else QUESTION_BANK_PATH
    rows, sample_mode, quotas = select_question_bank_rows(n=n, seed=seed, shard=path)
    per_row, errors, correct = evaluate_question_bank_rows(
        predictor,
        rows,
        predictor_label=predictor_label,
        progress=progress,
    )
    bc = rows_per_benchmark(rows)
    return build_eval_summary(
        per_row=per_row,
        errors=errors,
        correct=correct,
        predictor_label=predictor_label,
        shard=path,
        sample_mode=sample_mode,
        seed=seed,
        proportional_quotas=quotas,
        benchmark_counts=bc,
        extra=extra_summary,
    )
