"""Paper-aligned downgrade-and-cascade construction engine."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Sequence

from .backends import ExecutionBackend, RecordingBackend
from .benchmarking import BenchmarkRegistry, BenchmarkSpec, builtin_registry
from .cache import TrialCache
from .source import SourceSpec
from .types import (
    TIER_ORDER,
    TIER_TO_ID,
    Assignment,
    DowngradeHint,
    ExecutionResult,
    TaskSpec,
)


def _json_text(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_text(row) + "\n")


def _prefix_hash(prefix: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256(_json_text(list(prefix)).encode("utf-8")).hexdigest()


@dataclass
class GenerationConfig:
    run_id: str = "reference-mock-v1"
    backend_name: str = "mock"
    seed: int = 20260508
    review_rate: float = 0.10
    cascade_size: int = 3
    max_cases_per_benchmark: int = 0
    collector: str = "twinrouterbench-reference-pipeline"
    collected_at: str = "2026-05-08"
    model_pool_path: str = ""
    source_uris: dict[str, str] = field(default_factory=dict)
    generation_parameters: dict[str, Any] = field(
        default_factory=lambda: {"temperature": 0, "top_p": 1}
    )
    prompt_versions: dict[str, str] = field(
        default_factory=lambda: {
            "tier_proposer": "tier_proposer_v1",
            "open_ended_judge": "open_ended_judge_v1",
        }
    )

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id cannot be empty")
        if not 0 <= self.review_rate <= 1:
            raise ValueError("review_rate must be between 0 and 1")
        if self.cascade_size <= 0:
            raise ValueError("cascade_size must be positive")
        if self.max_cases_per_benchmark < 0:
            raise ValueError("max_cases_per_benchmark cannot be negative")

    @classmethod
    def from_file(cls, path: str | Path) -> "GenerationConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("generation config must be a JSON object")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_model_pool(path: str | Path | None = None) -> tuple[str, dict[str, list[str]]]:
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        raw = json.loads(
            resources.files(__package__)
            .joinpath("model_pool_v2.json")
            .read_text(encoding="utf-8")
        )
    tiers = raw.get("tiers")
    if not isinstance(tiers, dict) or tuple(tiers) != TIER_ORDER:
        raise ValueError(f"model pool tiers must appear in order {TIER_ORDER}")
    normalized: dict[str, list[str]] = {}
    seen: set[str] = set()
    for tier in TIER_ORDER:
        models = tiers.get(tier)
        if not isinstance(models, list) or not models:
            raise ValueError(f"model pool tier {tier!r} must be a non-empty list")
        normalized[tier] = [str(model) for model in models]
        duplicate = seen.intersection(normalized[tier])
        if duplicate:
            raise ValueError(f"models occur in multiple tiers: {sorted(duplicate)}")
        seen.update(normalized[tier])
    return str(raw.get("version", "unknown")), normalized


class GenerationPipeline:
    """Construct step-level labels under Algorithm A1 and tier-pool cascade."""

    ARTIFACTS = (
        "config.lock.json",
        "backend_events.jsonl",
        "seed_trajectories.jsonl",
        "downgrade_hints.jsonl",
        "execution_trials.jsonl",
        "rejections.jsonl",
        "labels.pre_review.jsonl",
        "review_queue.jsonl",
        "manifest.json",
    )

    def __init__(
        self,
        *,
        output_dir: str | Path,
        backend: ExecutionBackend,
        config: GenerationConfig | None = None,
        registry: BenchmarkRegistry | None = None,
        adapters: dict[str, BenchmarkSpec] | None = None,
    ) -> None:
        if registry is not None and adapters is not None:
            raise ValueError("pass registry or adapters, not both")
        self.output_dir = Path(output_dir)
        self.config = config or GenerationConfig()
        self.registry = registry or (
            BenchmarkRegistry(list(adapters.values()), version="legacy-adapters")
            if adapters is not None
            else builtin_registry()
        )
        pool_path = self.config.model_pool_path or None
        self.model_pool_version, self.model_pool = load_model_pool(pool_path)
        self.backend = backend
        self.cache = TrialCache()

    def generate(self, benchmark: str | Sequence[str] = "all") -> dict[str, Any]:
        benchmarks = self._normalize_benchmarks(benchmark)
        self._prepare_output_dir()
        config_lock = {
            **self.config.to_dict(),
            "benchmarks": benchmarks,
            "model_pool_version": self.model_pool_version,
            "model_pool": self.model_pool,
            "benchmark_suite_version": self.registry.version,
            "benchmark_suite_fingerprint": self.registry.fingerprint,
            "benchmark_definitions": {
                name: self.registry.get(name).to_dict() for name in benchmarks
            },
        }
        _write_json(self.output_dir / "config.lock.json", config_lock)

        backend = RecordingBackend(
            self.backend, self.output_dir / "backend_events.jsonl"
        )
        seeds: list[dict[str, Any]] = []
        hints: list[dict[str, Any]] = []
        trials: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        labels: list[dict[str, Any]] = []
        sources: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for benchmark_name in benchmarks:
            adapter = self.registry.get(benchmark_name)
            source = SourceSpec.parse(
                self.config.source_uris.get(
                    benchmark_name, adapter.source_uri
                )
            )
            tasks = adapter.load_tasks(source)
            if self.config.max_cases_per_benchmark:
                tasks = tasks[: self.config.max_cases_per_benchmark]
            for task in tasks:
                provenance = SourceSpec.parse(task.source.uri)
                source_row = {
                    "uri": task.source.uri,
                    "kind": provenance.kind,
                    "locator": (
                        provenance.locator
                        if provenance.kind != "local"
                        else "<materialized-local-source>"
                    ),
                    "revision": provenance.revision,
                    "path": provenance.path,
                    "is_pinned": provenance.is_pinned,
                    "materialized_as": source.kind,
                    "license": task.source.license,
                    "version": task.source.version,
                }
                if source_row not in sources[benchmark_name]:
                    sources[benchmark_name].append(source_row)
                result = self._generate_task(
                    task=task,
                    adapter=adapter,
                    backend=backend,
                    seeds=seeds,
                    hints=hints,
                    trials=trials,
                    rejections=rejections,
                )
                labels.extend(result)

        review_queue = self._build_review_queue(labels, benchmarks)
        pending_ids = {row["id"] for row in review_queue}
        for row in labels:
            adapter = self.registry.get(str(row["benchmark"]))
            if row["id"] in pending_ids:
                row["review_status"] = "pending"
            elif adapter.manual_review:
                row["review_status"] = "not_sampled"
            else:
                row["review_status"] = "not_required"

        _write_jsonl(self.output_dir / "seed_trajectories.jsonl", seeds)
        _write_jsonl(self.output_dir / "downgrade_hints.jsonl", hints)
        _write_jsonl(self.output_dir / "execution_trials.jsonl", trials)
        _write_jsonl(self.output_dir / "rejections.jsonl", rejections)
        _write_jsonl(self.output_dir / "labels.pre_review.jsonl", labels)
        _write_jsonl(self.output_dir / "review_queue.jsonl", review_queue)

        ready_without_review = not review_queue
        if ready_without_review:
            final_rows = []
            for row in labels:
                final = copy.deepcopy(row)
                final["pipeline_stage"] = "ground_truth_ready"
                final_rows.append(final)
            _write_jsonl(self.output_dir / "labels.final.jsonl", final_rows)

        manifest = self._build_manifest(
            benchmarks=benchmarks,
            labels=labels,
            seeds=seeds,
            trials=trials,
            rejections=rejections,
            review_queue=review_queue,
            sources=sources,
            status="ready" if ready_without_review else "awaiting_review",
        )
        _write_json(self.output_dir / "manifest.json", manifest)
        return manifest

    def _generate_task(
        self,
        *,
        task: TaskSpec,
        adapter: BenchmarkSpec,
        backend: RecordingBackend,
        seeds: list[dict[str, Any]],
        hints: list[dict[str, Any]],
        trials: list[dict[str, Any]],
        rejections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        high_models = self.model_pool["high"]
        strong_model = high_models[0]
        seed = backend.run_seed(task, strong_model)
        seeds.append(
            {
                "benchmark": task.benchmark,
                "instance_id": task.instance_id,
                "strong_model": strong_model,
                **seed.to_dict(),
            }
        )
        if seed.step_count != len(task.steps) or len(seed.prefixes) != len(task.steps):
            raise ValueError(
                f"seed result for {task.instance_id} has inconsistent step count/prefixes"
            )
        seed_verdict = adapter.trial_evaluator.evaluate(task, seed, backend)
        if not self._verdict_passed(seed_verdict):
            rejections.append(
                {
                    "benchmark": task.benchmark,
                    "instance_id": task.instance_id,
                    "stage": "seed",
                    "reason": seed_verdict.reason or "strong seed trajectory failed",
                }
            )
            return []

        proposed = list(backend.propose_hints(task, seed))
        self._validate_hints(task, proposed)
        hints.extend(
            {
                "benchmark": task.benchmark,
                "instance_id": task.instance_id,
                **hint.to_dict(),
                "prompt_version": self.config.prompt_versions["tier_proposer"],
            }
            for hint in proposed
        )

        assignments = [Assignment("high", strong_model) for _ in task.steps]
        canonical = seed
        for hint in proposed:
            step_index = hint.step_index
            if hint.start_tier == "not_downgradeable":
                continue
            accepted = False
            for tier in TIER_ORDER[TIER_TO_ID[hint.start_tier] : TIER_TO_ID["high"]]:
                tier_models = self.model_pool[tier][: self.config.cascade_size]
                for cascade_index, model in enumerate(tier_models, 1):
                    candidate = list(assignments)
                    candidate[step_index - 1] = Assignment(tier, model)
                    key = self.cache.make_key(
                        task.instance_id,
                        step_index,
                        candidate,
                        self.config.generation_parameters,
                    )
                    execution = self.cache.get(key)
                    cache_hit = execution is not None
                    if execution is None:
                        execution = backend.run_mixed(
                            task,
                            candidate,
                            target_step=step_index,
                            generation_parameters=self.config.generation_parameters,
                        )
                        self.cache.put(key, execution)
                    verdict = adapter.trial_evaluator.evaluate(task, execution, backend)
                    same_length = execution.step_count == seed.step_count
                    accepted_attempt = bool(
                        self._verdict_passed(verdict) and same_length
                    )
                    trials.append(
                        {
                            "benchmark": task.benchmark,
                            "instance_id": task.instance_id,
                            "target_step": step_index,
                            "candidate_tier": tier,
                            "candidate_model": model,
                            "cascade_index": cascade_index,
                            "assignments": [item.to_dict() for item in candidate],
                            "passed": verdict.passed,
                            "same_step_count": same_length,
                            "accepted": accepted_attempt,
                            "cache_hit": cache_hit,
                            "reason": verdict.reason or execution.reason,
                            "prefix_hashes": [
                                _prefix_hash(prefix) for prefix in execution.prefixes
                            ],
                        }
                    )
                    if accepted_attempt:
                        assignments = candidate
                        canonical = execution
                        self.cache.invalidate_downstream(task.instance_id, step_index)
                        accepted = True
                        break
                if accepted:
                    break

        final_verdict = adapter.final_evaluator.evaluate(task, canonical, backend)
        if not self._verdict_passed(final_verdict):
            rejection = {
                "benchmark": task.benchmark,
                "instance_id": task.instance_id,
                "stage": adapter.final_evaluator.stage,
                "reason": final_verdict.reason or "final evaluator rejected the case",
            }
            if adapter.final_evaluator.stage != "execution":
                rejection["judge"] = final_verdict.to_dict()
            rejections.append(rejection)
            return []

        if canonical.step_count != len(task.steps):
            raise AssertionError(
                f"canonical mixed trajectory for {task.instance_id} is not valid"
            )

        rows: list[dict[str, Any]] = []
        for index, assignment in enumerate(assignments, 1):
            row: dict[str, Any] = {
                "id": f"{task.benchmark}_{task.instance_id}_step_{index}",
                "benchmark": task.benchmark,
                "benchmark_display": task.benchmark_display,
                "scenario": task.scenario,
                "instance_id": task.instance_id,
                "step_index": index,
                "total_steps": len(assignments),
                "messages": list(copy.deepcopy(canonical.prefixes[index - 1])),
                "target_tier": assignment.tier,
                "target_tier_id": TIER_TO_ID[assignment.tier],
                "selected_model": assignment.model,
                "benchmark_subset": task.benchmark_subset,
                "benchmark_version": task.benchmark_version,
                "pipeline_stage": "degradation_search_done",
                "collector": self.config.collector,
                "collected_at": self.config.collected_at,
                "source": asdict(task.source),
                "search_hint": proposed[index - 1].start_tier,
                "prompt_versions": copy.deepcopy(self.config.prompt_versions),
            }
            if task.functions:
                row["functions"] = list(copy.deepcopy(task.functions))
            rows.append(row)
        return rows

    def _build_review_queue(
        self, labels: list[dict[str, Any]], benchmarks: Sequence[str]
    ) -> list[dict[str, Any]]:
        rows_by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in labels:
            rows_by_benchmark[str(row["benchmark"])].append(row)

        queue: list[dict[str, Any]] = []
        for benchmark in benchmarks:
            adapter = self.registry.get(benchmark)
            candidates = sorted(rows_by_benchmark[benchmark], key=lambda row: row["id"])
            if not adapter.manual_review or not candidates or self.config.review_rate == 0:
                continue
            count = min(len(candidates), math.ceil(len(candidates) * self.config.review_rate))
            rng = random.Random(f"{self.config.seed}:{benchmark}")
            selected_ids = {row["id"] for row in rng.sample(candidates, count)}
            for row in candidates:
                if row["id"] not in selected_ids:
                    continue
                queue.append(
                    {
                        "id": row["id"],
                        "benchmark": benchmark,
                        "instance_id": row["instance_id"],
                        "step_index": row["step_index"],
                        "messages": copy.deepcopy(row["messages"]),
                        "target_tier": row["target_tier"],
                        "verdict": "",
                        "reviewer": "",
                        "notes": "",
                    }
                )
        return sorted(queue, key=lambda row: row["id"])

    def _build_manifest(
        self,
        *,
        benchmarks: Sequence[str],
        labels: list[dict[str, Any]],
        seeds: list[dict[str, Any]],
        trials: list[dict[str, Any]],
        rejections: list[dict[str, Any]],
        review_queue: list[dict[str, Any]],
        sources: dict[str, list[dict[str, Any]]],
        status: str,
    ) -> dict[str, Any]:
        label_counts = Counter(str(row["benchmark"]) for row in labels)
        seed_counts = Counter(str(row["benchmark"]) for row in seeds)
        reject_counts = Counter(str(row["benchmark"]) for row in rejections)
        return {
            "schema": "twinrouterbench.data_generation_run.v1",
            "run_id": self.config.run_id,
            "status": status,
            "benchmarks": list(benchmarks),
            "backend": self.config.backend_name,
            "model_pool_version": self.model_pool_version,
            "benchmark_suite_version": self.registry.version,
            "benchmark_suite_fingerprint": self.registry.fingerprint,
            "prompt_versions": copy.deepcopy(self.config.prompt_versions),
            "source_registry": {name: rows for name, rows in sorted(sources.items())},
            "counts": {
                "seed_instances": dict(sorted(seed_counts.items())),
                "emitted_rows": dict(sorted(label_counts.items())),
                "rejected_instances": dict(sorted(reject_counts.items())),
                "execution_trials": len(trials),
                "review_queue": len(review_queue),
            },
            "cache": self.cache.stats(),
            "artifacts": {
                "config": "config.lock.json",
                "backend_events": "backend_events.jsonl",
                "seeds": "seed_trajectories.jsonl",
                "hints": "downgrade_hints.jsonl",
                "trials": "execution_trials.jsonl",
                "rejections": "rejections.jsonl",
                "pre_review_labels": "labels.pre_review.jsonl",
                "review_queue": "review_queue.jsonl",
                "final_labels": (
                    "labels.final.jsonl" if status == "ready" else None
                ),
            },
        }

    def _normalize_benchmarks(self, benchmark: str | Sequence[str]) -> list[str]:
        if benchmark == "all":
            return list(self.registry.names)
        values = [benchmark] if isinstance(benchmark, str) else list(benchmark)
        if not values:
            raise ValueError("at least one benchmark is required")
        result: list[str] = []
        for value in values:
            self.registry.get(value)
            if value not in result:
                result.append(value)
        return result

    @staticmethod
    def _verdict_passed(verdict: Any) -> bool:
        return bool(
            verdict.passed
            and not verdict.uncertain
            and not verdict.evidence_conflict
            and verdict.faithfulness
            and verdict.appropriateness
            and verdict.completeness
        )

    def _prepare_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        conflicts = [name for name in self.ARTIFACTS if (self.output_dir / name).exists()]
        if (self.output_dir / "labels.final.jsonl").exists():
            conflicts.append("labels.final.jsonl")
        if conflicts:
            raise FileExistsError(
                f"generation run directory already contains artifacts: {sorted(conflicts)}"
            )

    @staticmethod
    def _validate_hints(task: TaskSpec, hints: Sequence[DowngradeHint]) -> None:
        if len(hints) != len(task.steps):
            raise ValueError(
                f"proposer returned {len(hints)} hints for {len(task.steps)} steps "
                f"in {task.instance_id}"
            )
        indices = [hint.step_index for hint in hints]
        expected = list(range(1, len(task.steps) + 1))
        if indices != expected:
            raise ValueError(
                f"proposer step indices for {task.instance_id} are {indices}; "
                f"expected {expected}"
            )
