#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.eval import FunctionPredictor, run_question_bank_eval
from main.tiers import TIER_TO_ID
from swerouter.routers.verification_menu import score_prefix


def _row_tools(row: dict[str, Any]) -> tuple[Any, ...]:
    raw = row.get("tools")
    if raw is None:
        raw = row.get("functions")
    if raw is None:
        return ()
    if isinstance(raw, tuple):
        return raw
    if isinstance(raw, list):
        return tuple(raw)
    return (raw,)


def _make_predictor(risk_profile: str):
    def predict_row(row: dict[str, Any]) -> int:
        tier, _, _ = score_prefix(
            messages=row.get("messages") or [],
            step_index=max(int(row.get("step_index") or 1) - 1, 0),
            max_steps=int(row.get("total_steps") or 0),
            tools=_row_tools(row),
            budget_so_far_usd=0.0,
            budget_usd=None,
            risk_profile=risk_profile,
        )
        return TIER_TO_ID[tier]

    return predict_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VerificationMenuRouter on the static question bank."
    )
    parser.add_argument(
        "--risk-profile",
        choices=("cheap", "balanced", "conservative"),
        default="balanced",
        help="Router risk profile to evaluate (default: balanced).",
    )
    args = parser.parse_args()

    summary = run_question_bank_eval(
        FunctionPredictor(_make_predictor(args.risk_profile)),
        predictor_label=f"verification_menu_static_{args.risk_profile}",
        shard=Path("data/static/question_bank.jsonl"),
        n=None,
        seed=42,
        extra_summary={"risk_profile": args.risk_profile},
    )
    out = Path(f"runs/static_verification_menu_{args.risk_profile}_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["scores_v2"], indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
