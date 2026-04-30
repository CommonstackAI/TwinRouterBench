"""Section 11 aggregate metrics and tier-only routing supervision accuracy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from main.pricing import StepCost, path_nominal_cost_usd
from main.tiers import TIER_HIGH, TIER_LOW, TIER_MID, TIER_MID_HIGH, public_tier_to_id


def compute_case_savings(
    baseline_cost: float,
    optimal_cost: float,
    test_cost: float,
) -> tuple[float, float]:
    save_gt = baseline_cost - optimal_cost
    save_test = baseline_cost - test_cost
    return save_gt, save_test


@dataclass
class CaseMetrics:
    """Per-case inputs for Section 11 aggregation (from your eval harness)."""

    case_id: str
    task_passed: bool
    baseline_cost_nominal: float | None = None
    optimal_cost_nominal: float | None = None
    test_cost_nominal: float | None = None
    baseline_steps: list[StepCost] | None = None
    optimal_steps: list[StepCost] | None = None
    test_steps: list[StepCost] | None = None

    def resolved_costs(self) -> tuple[float, float, float]:
        b = self.baseline_cost_nominal
        o = self.optimal_cost_nominal
        t = self.test_cost_nominal
        if self.baseline_steps is not None:
            b = path_nominal_cost_usd(self.baseline_steps)
        if self.optimal_steps is not None:
            o = path_nominal_cost_usd(self.optimal_steps)
        if self.test_steps is not None:
            t = path_nominal_cost_usd(self.test_steps)
        if b is None or o is None or t is None:
            raise ValueError(
                "CaseMetrics needs baseline/optimal/test costs or *_steps to compute path_nominal_cost_usd"
            )
        return b, o, t


def aggregate_routerbench_metrics(
    cases: list[CaseMetrics],
    *,
    cap_cost_score_at_100: bool = False,
) -> dict[str, Any]:
    """
    Section 11-style summary. Cost score uses only task_passed cases with save_gt > 0.
    cost_savings_score = 100 * sum(save_test) / sum(save_gt) on that subset.
    """
    if not cases:
        raise ValueError("cases must be non-empty")

    valid = len(cases)
    passed = sum(1 for c in cases if c.task_passed)
    pass_rate = passed / valid if valid else 0.0

    sum_save_gt = 0.0
    sum_save_test = 0.0
    cost_score_cases = 0
    money_saved_test_vals: list[float] = []

    for c in cases:
        if not c.task_passed:
            continue
        b, o, test_c = c.resolved_costs()
        save_gt, save_test = compute_case_savings(b, o, test_c)
        money_saved_test_vals.append(save_test)
        if save_gt <= 0:
            continue
        sum_save_gt += save_gt
        sum_save_test += save_test
        cost_score_cases += 1

    if sum_save_gt > 0:
        cost_savings_score = 100.0 * sum_save_test / sum_save_gt
        if cap_cost_score_at_100:
            cost_savings_score = min(cost_savings_score, 100.0)
    else:
        cost_savings_score = float("nan")

    mean_saved = (
        sum(money_saved_test_vals) / len(money_saved_test_vals) if money_saved_test_vals else float("nan")
    )
    total_saved = sum(money_saved_test_vals)

    return {
        "valid_cases": valid,
        "passed_cases": passed,
        "pass_rate": pass_rate,
        "cost_score_cases_used": cost_score_cases,
        "cost_score_excludes_nonpositive_save_gt": True,
        "pricing": {
            "unit": "USD_per_1M_output_tokens",
            TIER_LOW: 0.5,
            TIER_MID: 1.2,
            TIER_MID_HIGH: 3.0,
            TIER_HIGH: 20.0,
        },
        "cost_score_rule": (
            "100 * sum(save_test) / sum(save_gt) over passed cases with save_gt > 0; "
            "costs from fixed tier rates * completion tokens"
        ),
        "sum_save_gt_usd": sum_save_gt,
        "sum_save_test_usd": sum_save_test,
        "cost_savings_score": cost_savings_score,
        "money_saved_test": {
            "currency": "USD",
            "definition": "baseline_nominal - test_nominal per case (passed cases only in mean/total below)",
            "mean_per_case_over_passed": mean_saved,
            "total_over_passed": total_saved,
        },
    }


def routing_supervision_accuracy(
    gold_rows: list[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Step-level accuracy: prediction must match gold target_tier or target_tier_id.
    Each prediction dict may include predicted_tier (str) and/or predicted_tier_id (int).
    """
    correct = 0
    total = 0
    missing_pred = 0
    for row in gold_rows:
        rid = row.get("id")
        if rid is None:
            raise ValueError("gold row missing id")
        pred = predictions_by_id.get(rid)
        if pred is None:
            missing_pred += 1
            total += 1
            continue
        gold_tier = row.get("target_tier")
        gold_id = row.get("target_tier_id")
        if gold_tier is None and gold_id is None:
            raise ValueError(f"gold row {rid!r} missing target_tier and target_tier_id")
        if gold_tier is not None and gold_id is not None:
            if public_tier_to_id(gold_tier) != gold_id:
                raise ValueError(f"gold row {rid!r} has inconsistent target_tier / target_tier_id")
        ok = False
        pt = pred.get("predicted_tier")
        pid = pred.get("predicted_tier_id")
        if gold_tier is not None and pt == gold_tier:
            ok = True
        if gold_id is not None and pid is not None and int(pid) == int(gold_id):
            ok = True
        if gold_tier is not None and pt is None and pid is not None:
            if public_tier_to_id(gold_tier) == int(pid):
                ok = True
        if gold_id is not None and pid is None and pt is not None:
            if public_tier_to_id(pt) == int(gold_id):
                ok = True
        total += 1
        if ok:
            correct += 1
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else float("nan"),
        "missing_predictions": missing_pred,
    }


def case_metrics_from_dict(d: dict[str, Any]) -> CaseMetrics:
    """Build CaseMetrics from JSON CLI payload (nested steps as dicts)."""

    def steps_from_json(raw: Any) -> list[StepCost] | None:
        if raw is None:
            return None
        if not isinstance(raw, list):
            raise ValueError("baseline_steps / optimal_steps / test_steps must be a list")
        out: list[StepCost] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("each step must be an object")
            out.append(
                StepCost(
                    completion_tokens=int(item["completion_tokens"]),
                    model=item.get("model"),
                    tier=item.get("tier"),
                )
            )
        return out

    return CaseMetrics(
        case_id=str(d["case_id"]),
        task_passed=bool(d["task_passed"]),
        baseline_cost_nominal=d.get("baseline_cost_nominal"),
        optimal_cost_nominal=d.get("optimal_cost_nominal"),
        test_cost_nominal=d.get("test_cost_nominal"),
        baseline_steps=steps_from_json(d.get("baseline_steps")),
        optimal_steps=steps_from_json(d.get("optimal_steps")),
        test_steps=steps_from_json(d.get("test_steps")),
    )
