"""``RoundRobinRouter``: cycle through a list of ``model_id``s by step index."""

from __future__ import annotations

from dataclasses import dataclass

from swerouter.router import RouterContext, RouterDecision


@dataclass
class RoundRobinRouter:
    """Deterministic round-robin over a fixed list of pool model_ids."""

    model_ids: tuple[str, ...]
    label: str

    def __post_init__(self) -> None:
        if not self.model_ids:
            raise ValueError("RoundRobinRouter requires at least one model_id")

    def select(self, ctx: RouterContext) -> RouterDecision:
        for mid in self.model_ids:
            if mid not in ctx.available_models:
                raise ValueError(
                    f"RoundRobinRouter[{self.label!r}] model_id {mid!r} not in pool"
                )
        chosen = self.model_ids[ctx.step_index % len(self.model_ids)]
        return RouterDecision(model_id=chosen, rationale=f"round_robin step={ctx.step_index}")
