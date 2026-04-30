"""``AlwaysModelRouter``: pick the same concrete ``model_id`` every step.

Useful for baseline leaderboard rows (always-deepseek / always-opus etc.) and
for smoke testing the pipeline end to end.
"""

from __future__ import annotations

from dataclasses import dataclass

from swerouter.router import RouterContext, RouterDecision


@dataclass
class AlwaysModelRouter:
    """Return the same ``model_id`` for every step regardless of context."""

    model_id: str
    label: str

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("AlwaysModelRouter.model_id must be non-empty")

    def select(self, ctx: RouterContext) -> RouterDecision:
        if self.model_id not in ctx.available_models:
            raise ValueError(
                f"AlwaysModelRouter[{self.label!r}] configured with model_id="
                f"{self.model_id!r} which is not in the locked pool "
                f"({list(ctx.available_models)})"
            )
        return RouterDecision(model_id=self.model_id, rationale="always_model_baseline")
