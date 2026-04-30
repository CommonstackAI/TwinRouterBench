"""Question-bank routing eval: sampling, Section 11 aggregates, and pluggable predictors (LLM or callable)."""

from main.eval.predictors import (
    FunctionPredictor,
    LlmDigitClassifierPredictor,
    QuestionBankRouterPredictor,
    TierPrediction,
)
from main.eval.runner import (
    build_eval_summary,
    evaluate_question_bank_rows,
    run_question_bank_eval,
)
from main.eval.sampling import (
    default_manifest_path,
    load_all_question_bank_rows,
    manifest_proportional_quotas,
    proportional_reservoir_sample,
    rows_per_benchmark,
    select_question_bank_rows,
)
from main.eval.section11 import (
    aggregate_by_benchmark,
    compute_router_accounting_metrics,
    compute_section11,
    compute_v2_scores,
)

__all__ = [
    "FunctionPredictor",
    "LlmDigitClassifierPredictor",
    "QuestionBankRouterPredictor",
    "TierPrediction",
    "aggregate_by_benchmark",
    "build_eval_summary",
    "compute_router_accounting_metrics",
    "compute_section11",
    "compute_v2_scores",
    "default_manifest_path",
    "evaluate_question_bank_rows",
    "load_all_question_bank_rows",
    "manifest_proportional_quotas",
    "proportional_reservoir_sample",
    "rows_per_benchmark",
    "run_question_bank_eval",
    "select_question_bank_rows",
]
