"""Typed records shared by the data-generation stages."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any


TIER_ORDER: tuple[str, ...] = ("low", "mid", "mid_high", "high")
TIER_TO_ID = {tier: index for index, tier in enumerate(TIER_ORDER)}


def validate_tier(value: str) -> str:
    if value not in TIER_TO_ID:
        raise ValueError(f"unknown tier {value!r}; expected one of {TIER_ORDER}")
    return value


def lower_tier(value: str) -> str:
    validate_tier(value)
    index = TIER_TO_ID[value]
    if index == 0:
        raise ValueError("low is already the cheapest tier")
    return TIER_ORDER[index - 1]


@dataclass(frozen=True)
class SourceMetadata:
    uri: str
    license: str = "unknown"
    version: str = "unknown"


@dataclass(frozen=True)
class StepSpec:
    hint: str
    minimum_tier: str
    observation: str = "step completed"
    fail_models: tuple[str, ...] = ()
    force_pass_models: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.hint != "not_downgradeable":
            validate_tier(self.hint)
        validate_tier(self.minimum_tier)


@dataclass(frozen=True)
class TaskSpec:
    benchmark: str
    benchmark_display: str
    scenario: str
    instance_id: str
    initial_messages: tuple[dict[str, Any], ...]
    steps: tuple[StepSpec, ...]
    source: SourceMetadata
    benchmark_version: str
    benchmark_subset: str = ""
    functions: tuple[dict[str, Any], ...] = ()
    seed_passed: bool = True
    judge_passed: bool = True
    judge_reason: str = "fixture verdict"
    blocked_tier_combinations: tuple[dict[str, str], ...] = ()
    fixture_reviews: dict[str, dict[str, str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TaskSpec":
        source = raw.get("source") or {}
        return cls(
            benchmark=str(raw["benchmark"]),
            benchmark_display=str(raw.get("benchmark_display") or raw["benchmark"]),
            scenario=str(raw["scenario"]),
            instance_id=str(raw["instance_id"]),
            initial_messages=tuple(copy.deepcopy(raw.get("initial_messages") or [])),
            steps=tuple(
                StepSpec(
                    hint=str(step.get("hint", "low")),
                    minimum_tier=str(step.get("minimum_tier", "high")),
                    observation=str(step.get("observation", "step completed")),
                    fail_models=tuple(str(x) for x in step.get("fail_models", [])),
                    force_pass_models=tuple(
                        str(x) for x in step.get("force_pass_models", [])
                    ),
                )
                for step in raw.get("steps", [])
            ),
            source=SourceMetadata(
                uri=str(source.get("uri", "unknown")),
                license=str(source.get("license", "unknown")),
                version=str(source.get("version", "unknown")),
            ),
            benchmark_version=str(raw.get("benchmark_version", "unknown")),
            benchmark_subset=str(raw.get("benchmark_subset", "")),
            functions=tuple(copy.deepcopy(raw.get("functions") or [])),
            seed_passed=bool(raw.get("seed_passed", True)),
            judge_passed=bool(raw.get("judge_passed", True)),
            judge_reason=str(raw.get("judge_reason", "fixture verdict")),
            blocked_tier_combinations=tuple(
                {str(k): str(v) for k, v in item.items()}
                for item in raw.get("blocked_tier_combinations", [])
            ),
            fixture_reviews={
                str(k): {str(rk): str(rv) for rk, rv in value.items()}
                for k, value in (raw.get("fixture_reviews") or {}).items()
            },
            metadata=copy.deepcopy(raw.get("metadata") or {}),
        )
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Assignment:
    tier: str
    model: str

    def __post_init__(self) -> None:
        validate_tier(self.tier)

    def to_dict(self) -> dict[str, str]:
        return {"tier": self.tier, "model": self.model}


@dataclass(frozen=True)
class ExecutionResult:
    passed: bool
    step_count: int
    prefixes: tuple[tuple[dict[str, Any], ...], ...]
    responses: tuple[dict[str, Any], ...]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "step_count": self.step_count,
            "prefixes": [list(copy.deepcopy(prefix)) for prefix in self.prefixes],
            "responses": list(copy.deepcopy(self.responses)),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExecutionResult":
        return cls(
            passed=bool(raw["passed"]),
            step_count=int(raw["step_count"]),
            prefixes=tuple(
                tuple(copy.deepcopy(prefix)) for prefix in raw.get("prefixes", [])
            ),
            responses=tuple(copy.deepcopy(raw.get("responses", []))),
            reason=str(raw.get("reason", "")),
        )


@dataclass(frozen=True)
class DowngradeHint:
    step_index: int
    start_tier: str
    reason: str

    def __post_init__(self) -> None:
        if self.start_tier != "not_downgradeable":
            validate_tier(self.start_tier)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DowngradeHint":
        return cls(
            step_index=int(raw["step_index"]),
            start_tier=str(raw["start_tier"]),
            reason=str(raw.get("reason", "")),
        )


@dataclass(frozen=True)
class JudgeResult:
    passed: bool
    faithfulness: bool
    appropriateness: bool
    completeness: bool
    evidence_conflict: bool = False
    uncertain: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "JudgeResult":
        return cls(
            passed=bool(raw["passed"]),
            faithfulness=bool(raw.get("faithfulness", raw["passed"])),
            appropriateness=bool(raw.get("appropriateness", raw["passed"])),
            completeness=bool(raw.get("completeness", raw["passed"])),
            evidence_conflict=bool(raw.get("evidence_conflict", False)),
            uncertain=bool(raw.get("uncertain", False)),
            reason=str(raw.get("reason", "")),
        )
