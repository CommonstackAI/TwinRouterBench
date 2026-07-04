#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.eval import FunctionPredictor, run_question_bank_eval


def always_high(_: dict) -> int:
    return 3


def main() -> None:
    summary = run_question_bank_eval(
        FunctionPredictor(always_high),
        predictor_label="always_high_repro",
        shard=Path("data/static/question_bank.jsonl"),
        n=None,
        seed=42,
    )
    print(json.dumps(summary["scores_v2"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
