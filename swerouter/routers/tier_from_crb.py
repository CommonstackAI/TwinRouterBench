"""``TierFromCRBRouter``: use CommonRouterBench's tier classifier as the
upstream brain and translate its 0..3 tier id into a concrete pool model.

This demonstrates how an existing CRB-compatible router drops straight into
SWERouterBench by mapping through ``data/tier_to_model.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from main.router_llm import OpenAICompatRouterClassifier
from main.tiers import PUBLIC_TIERS

from swerouter.router import RouterContext, RouterDecision


@dataclass
class TierFromCRBRouter:
    """Use an LLM-backed tier classifier (CRB) and map tier -> concrete model.

    Parameters
    ----------
    classifier
        Pre-configured :class:`main.router_llm.OpenAICompatRouterClassifier`
        instance. The user constructs it with their own base_url / api_key /
        classifier model so SWERouterBench does not hardcode any endpoint.
    tier_to_model_path
        Path to ``data/tier_to_model.json``.
    label
        Human-readable label for the trace / leaderboard.
    """

    classifier: OpenAICompatRouterClassifier
    tier_to_model_path: Path
    label: str

    def __post_init__(self) -> None:
        path = Path(self.tier_to_model_path)
        if not path.is_file():
            raise FileNotFoundError(f"tier_to_model file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        raw = doc.get("map")
        if not isinstance(raw, dict):
            raise ValueError("tier_to_model.map must be an object")
        missing = [t for t in PUBLIC_TIERS if t not in raw]
        if missing:
            raise ValueError(
                f"tier_to_model.map missing tiers: {missing}. Required: {sorted(PUBLIC_TIERS)}"
            )
        resolved: dict[int, str] = {}
        for i, tier_name in enumerate(PUBLIC_TIERS):
            model = raw[tier_name]
            if not isinstance(model, str) or not model:
                raise ValueError(f"tier_to_model.map[{tier_name!r}] must be non-empty string")
            resolved[i] = model
        # Store on the instance under a name that can't collide with user fields.
        object.__setattr__(self, "_tier_id_to_model", resolved)

    def _flatten_messages_to_prompt(
        self, messages: tuple[dict, ...]
    ) -> str:
        """Flatten the OpenAI-style messages list into a classifier prompt.

        Mirrors CRB's ``question_bank_messages_to_classifier_prompt`` formatting
        style (role: content, blank line between turns) so the classifier sees
        familiar text.
        """
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            if isinstance(content, list):
                text = "\n".join(
                    str(b.get("text", "")) if isinstance(b, dict) else str(b)
                    for b in content
                )
            elif content is None:
                text = ""
            else:
                text = str(content)
            parts.append(f"{role}:\n{text}")
        return "\n\n".join(parts)

    def select(self, ctx: RouterContext) -> RouterDecision:
        prompt = self._flatten_messages_to_prompt(ctx.messages)
        pred = self.classifier.predict_tier_id(prompt)
        tier_id = pred.tier_id
        mapping = getattr(self, "_tier_id_to_model")
        if tier_id not in mapping:
            raise ValueError(
                f"TierFromCRBRouter got tier_id={tier_id!r}, not in map {sorted(mapping)}"
            )
        model_id = mapping[tier_id]
        if model_id not in ctx.available_models:
            raise ValueError(
                f"TierFromCRBRouter mapped tier={tier_id} -> {model_id!r}, "
                f"but that model is not in the locked pool"
            )
        return RouterDecision(
            model_id=model_id,
            rationale=f"tier_from_crb tier_id={tier_id}",
        )
