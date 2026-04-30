"""``GoldTierRouter`` (oracle): route each step using the static track
gold tier labels from ``data/static/question_bank.jsonl``, then map tier -> concrete
``model_id`` via ``data/dynamic/tier_to_model.json``.

Use-cases:

* Smoke test the dynamic pipeline with a deterministic router.
* Validate that the per-step scoring (actual cost + failed-run penalty) is
  computed correctly.

Behaviour:

* If the live agent loop runs more steps than the CRB bank recorded for an
  instance, the router **saturates at the last recorded tier** rather than
  silently defaulting to HIGH: this reflects "use whatever the last known
  difficulty was" without leaking any extra ground truth.
* If the instance has no rows in CRB's bank at all, the router raises
  :class:`ValueError` at construction time (fail fast).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from main.tiers import PUBLIC_TIERS

from swerouter.router import RouterContext, RouterDecision


@dataclass
class GoldTierRouter:
    """Per-step oracle using CRB's published gold tier labels.

    Parameters
    ----------
    question_bank_path
        Path to ``data/static/question_bank.jsonl`` (under ``TwinRouterBench/``).
    tier_to_model_path
        Path to ``data/dynamic/tier_to_model.json`` (under ``TwinRouterBench/``).
    allowed_instance_ids
        Instance ids this router will serve. Construction fails if any of them
        has no ``swebench`` rows in the question bank.
    label
        Human-readable router label for traces and leaderboard rows.
    """

    question_bank_path: Path
    tier_to_model_path: Path
    allowed_instance_ids: tuple[str, ...]
    label: str

    def __post_init__(self) -> None:
        qpath = Path(self.question_bank_path)
        if not qpath.is_file():
            raise FileNotFoundError(f"question_bank not found: {qpath}")
        tpath = Path(self.tier_to_model_path)
        if not tpath.is_file():
            raise FileNotFoundError(f"tier_to_model not found: {tpath}")

        # Tier -> model id
        with tpath.open("r", encoding="utf-8") as fh:
            raw_map = (json.load(fh) or {}).get("map") or {}
        missing = [t for t in PUBLIC_TIERS if t not in raw_map]
        if missing:
            raise ValueError(f"tier_to_model.map missing tiers: {missing}")
        tier_to_model: dict[str, str] = {t: str(raw_map[t]) for t in PUBLIC_TIERS}

        # Load per-instance step -> tier sequence from the CRB question bank.
        # The bank is small (~970 rows) so loading once in memory is fine.
        by_instance: dict[str, list[tuple[int, str]]] = {}
        with qpath.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("benchmark") != "swebench":
                    continue
                iid = row.get("instance_id")
                step_index = int(row.get("step_index", -1))
                tier = row.get("target_tier")
                if not isinstance(iid, str) or tier not in tier_to_model:
                    continue
                by_instance.setdefault(iid, []).append((step_index, tier))

        for items in by_instance.values():
            items.sort(key=lambda x: x[0])

        # Validate each allowed instance has bank coverage.
        missing_instances = [
            iid for iid in self.allowed_instance_ids if iid not in by_instance
        ]
        if missing_instances:
            raise ValueError(
                f"GoldTierRouter[{self.label!r}]: CRB question_bank has no swebench rows for "
                f"instance_ids: {missing_instances}"
            )
        for iid in self.allowed_instance_ids:
            if not by_instance[iid]:
                raise ValueError(
                    f"GoldTierRouter[{self.label!r}]: empty tier sequence for instance_id={iid!r}"
                )

        object.__setattr__(self, "_tier_by_step", {
            iid: [t for _, t in by_instance[iid]] for iid in self.allowed_instance_ids
        })
        object.__setattr__(self, "_tier_to_model", tier_to_model)

    def select(self, ctx: RouterContext) -> RouterDecision:
        tier_seq = getattr(self, "_tier_by_step").get(ctx.instance_id)
        if tier_seq is None:
            raise ValueError(
                f"GoldTierRouter[{self.label!r}]: instance_id={ctx.instance_id!r} not in allowed list"
            )
        if ctx.step_index < len(tier_seq):
            tier = tier_seq[ctx.step_index]
            source = "crb_gold_exact"
        else:
            tier = tier_seq[-1]
            source = f"crb_gold_saturated_last_of_{len(tier_seq)}"
        model_id = getattr(self, "_tier_to_model")[tier]
        if model_id not in ctx.available_models:
            raise ValueError(
                f"GoldTierRouter[{self.label!r}]: mapped tier={tier!r} -> {model_id!r}, "
                f"not in pool {list(ctx.available_models)}"
            )
        return RouterDecision(
            model_id=model_id,
            rationale=f"gold_tier step={ctx.step_index} tier={tier} source={source}",
        )

    @classmethod
    def from_cli_args(
        cls,
        *,
        question_bank_path: str,
        tier_to_model_path: str,
        allowed_instance_ids: str,
        label: str,
    ) -> "GoldTierRouter":
        """CLI-friendly factory.

        Accepts everything as plain strings so it is compatible with
        ``swerouterbench run --router-arg key=value`` (which only forwards
        ``str`` values). ``allowed_instance_ids`` is a comma-separated list of
        SWE-bench instance ids; whitespace around entries is ignored and empty
        entries are rejected (fail fast).
        """

        iids = tuple(
            x.strip() for x in allowed_instance_ids.split(",") if x.strip()
        )
        if not iids:
            raise ValueError(
                "GoldTierRouter.from_cli_args: allowed_instance_ids is empty "
                "after parsing; expected a comma-separated list of instance ids"
            )
        return cls(
            question_bank_path=Path(question_bank_path),
            tier_to_model_path=Path(tier_to_model_path),
            allowed_instance_ids=iids,
            label=label,
        )
