"""Pluggable predictors: LLM digit classifier or arbitrary callables (rules, sklearn, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from main.chat_messages import question_bank_messages_to_classifier_prompt
from main.router_llm import OpenAICompatRouterClassifier


@dataclass(frozen=True)
class TierPrediction:
    """Single routing tier prediction for one question-bank row."""

    tier_id: int
    usage: dict[str, Any] | None = None


class QuestionBankRouterPredictor(Protocol):
    """Predict ``tier_id`` (0–3) from a raw ``question_bank.jsonl`` row dict."""

    def predict(self, row: dict[str, Any]) -> TierPrediction:
        """Must validate ``row`` shape or raise; network/model errors may propagate."""
        ...


class LlmDigitClassifierPredictor:
    """
    OpenAI-compat chat model that emits a single digit; uses the package classifier prompt.
    """

    def __init__(self, classifier: OpenAICompatRouterClassifier) -> None:
        self._clf = classifier

    def predict(self, row: dict[str, Any]) -> TierPrediction:
        rid = row["id"]
        messages = row["messages"]
        if not isinstance(messages, list):
            raise ValueError(f"row {rid!r} messages must be a list")
        functions = row.get("functions")
        prompt = question_bank_messages_to_classifier_prompt(messages, functions=functions)
        result = self._clf.predict_tier_id(prompt)
        return TierPrediction(tier_id=result.tier_id, usage=result.usage)


class FunctionPredictor:
    """
    Wrap ``fn(row) -> int`` for heuristics, trained classifiers, or other non-chat backends.

    ``fn`` receives the full JSON row (``id``, ``benchmark``, ``messages``, ``target_tier_id``, ...).
    """

    def __init__(self, fn: Callable[[dict[str, Any]], int]) -> None:
        self._fn = fn

    def predict(self, row: dict[str, Any]) -> TierPrediction:
        tier_id = self._fn(row)
        if not isinstance(tier_id, int):
            raise TypeError(
                f"predictor function must return int tier_id, got {type(tier_id).__name__}"
            )
        return TierPrediction(tier_id=tier_id, usage=None)
