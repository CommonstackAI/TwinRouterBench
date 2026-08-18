"""Config-driven benchmark registry, loaders, and evaluators.

The construction engine consumes only normalized :class:`TaskSpec` records.
This module turns benchmark configuration into three narrow extension points:

* a task loader normalizes source records;
* a trial evaluator decides whether a candidate execution is sufficient;
* a final evaluator applies any stricter publication-time judgment.

Built-in components cover normalized multi-step records and common single-turn
datasets. Complex agent environments can provide ``module:factory`` plugins
without changing the shared downgrade/search/review/publish pipeline.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from .backends import ExecutionBackend
from .source import SourceSpec
from .types import ExecutionResult, JudgeResult, TaskSpec


SUITE_SCHEMA = "twinrouterbench.benchmark_suite.v1"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_MISSING = object()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_document(path: Path) -> Any:
    if path.is_dir():
        candidate = path / "tasks.jsonl"
        if not candidate.is_file():
            candidate = path / "tasks.json"
        path = candidate
    if not path.is_file():
        raise FileNotFoundError(
            f"benchmark source must be a JSON/JSONL file or contain tasks.jsonl: {path}"
        )
    if path.suffix == ".jsonl":
        rows = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            rows.append(row)
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def _records(raw: Any, *, records_field: str = "tasks") -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw = raw.get(records_field, [])
    if not isinstance(raw, list):
        raise ValueError("benchmark source must contain a list of records")
    if not all(isinstance(item, dict) for item in raw):
        raise ValueError("every benchmark source record must be an object")
    return [copy.deepcopy(item) for item in raw]


def _lookup(value: Any, path: str, default: Any = _MISSING) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            if default is _MISSING:
                raise ValueError(f"record is missing configured field {path!r}")
            return default
    return current


def _task_source_uri(benchmark: "BenchmarkSpec", source: SourceSpec) -> str:
    if benchmark.provenance_uri:
        return benchmark.provenance_uri
    if source.kind == "local":
        return "local://<materialized-source>"
    return source.uri


def _component(raw: Any, default_type: str) -> dict[str, Any]:
    if raw is None:
        return {"type": default_type}
    if isinstance(raw, str):
        return {"type": raw}
    if not isinstance(raw, dict):
        raise ValueError("component configuration must be a string or object")
    result = copy.deepcopy(raw)
    result.setdefault("type", default_type)
    return result


def _load_factory(path: str) -> Any:
    if ":" not in path:
        raise ValueError("plugin factory must use module:attribute syntax")
    module_name, attribute = path.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(f"plugin attribute does not exist: {path}") from exc


@runtime_checkable
class TaskLoader(Protocol):
    def load_tasks(
        self, benchmark: "BenchmarkSpec", source: SourceSpec
    ) -> list[TaskSpec]: ...


@runtime_checkable
class ExecutionEvaluator(Protocol):
    stage: str

    def evaluate(
        self,
        task: TaskSpec,
        execution: ExecutionResult,
        backend: ExecutionBackend,
    ) -> JudgeResult: ...


class NormalizedTaskLoader:
    """Load native ``TaskSpec`` JSON/JSONL records."""

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options = options or {}

    def load_tasks(
        self, benchmark: "BenchmarkSpec", source: SourceSpec
    ) -> list[TaskSpec]:
        if source.kind == "fixture":
            fixture_name = benchmark.fixture_name or source.locator
            if source.locator != fixture_name:
                raise ValueError(
                    f"benchmark {benchmark.name} expects fixture {fixture_name!r}, "
                    f"got {source.locator!r}"
                )
            fixture = resources.files(__package__).joinpath(
                "fixtures", f"{fixture_name}.json"
            )
            raw = json.loads(fixture.read_text(encoding="utf-8"))
        else:
            raw = _read_document(source.require_materialized_path())

        records_field = str(self.options.get("records_field", "tasks"))
        tasks = []
        for item in _records(raw, records_field=records_field):
            item.setdefault("benchmark", benchmark.name)
            item.setdefault("benchmark_display", benchmark.display_name)
            item.setdefault("scenario", benchmark.default_scenario)
            item.setdefault("benchmark_version", benchmark.benchmark_version)
            item.setdefault("benchmark_subset", benchmark.benchmark_subset)
            item.setdefault(
                "source",
                {
                    "uri": _task_source_uri(benchmark, source),
                    "license": benchmark.source_license,
                    "version": benchmark.source_version,
                },
            )
            tasks.append(TaskSpec.from_dict(item))
        _validate_tasks(benchmark, tasks)
        return tasks


class SingleTurnTaskLoader:
    """Normalize common prompt/answer table formats entirely from configuration."""

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options = options or {}
        self.fields = {
            "instance_id": "id",
            "messages": "messages",
            "prompt": "question",
            "reference": "answer",
            "target_tier": "target_tier",
            "hint": "hint",
            "functions": "functions",
            "scenario": "scenario",
            "metadata": "metadata",
            **self.options.get("field_map", {}),
        }

    def load_tasks(
        self, benchmark: "BenchmarkSpec", source: SourceSpec
    ) -> list[TaskSpec]:
        if source.kind == "fixture":
            raise ValueError("single_turn loader requires a materialized local source")
        raw = _read_document(source.require_materialized_path())
        records_field = str(self.options.get("records_field", "tasks"))
        tasks: list[TaskSpec] = []
        for index, row in enumerate(_records(raw, records_field=records_field), 1):
            instance_id = str(_lookup(row, self.fields["instance_id"], index))
            messages = _lookup(row, self.fields["messages"], None)
            if messages is None:
                prompt = _lookup(row, self.fields["prompt"])
                messages = []
                system_prompt = str(self.options.get("system_prompt", "")).strip()
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": str(prompt)})
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"single-turn record {instance_id!r} has no messages")

            target_tier = str(
                _lookup(
                    row,
                    self.fields["target_tier"],
                    self.options.get("default_minimum_tier", "high"),
                )
            )
            hint = str(
                _lookup(
                    row,
                    self.fields["hint"],
                    self.options.get("default_hint", "low"),
                )
            )
            reference = _lookup(row, self.fields["reference"], None)
            metadata = _lookup(row, self.fields["metadata"], {})
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata for {instance_id!r} must be an object")
            metadata = copy.deepcopy(metadata)
            if reference is not None:
                metadata.setdefault("reference", copy.deepcopy(reference))
            metadata.setdefault("raw_record_index", index)

            functions = _lookup(row, self.fields["functions"], [])
            scenario = str(
                _lookup(row, self.fields["scenario"], benchmark.default_scenario)
            )
            tasks.append(
                TaskSpec.from_dict(
                    {
                        "benchmark": benchmark.name,
                        "benchmark_display": benchmark.display_name,
                        "scenario": scenario,
                        "instance_id": instance_id,
                        "initial_messages": messages,
                        "steps": [
                            {
                                "hint": hint,
                                "minimum_tier": target_tier,
                                "observation": str(
                                    self.options.get(
                                        "observation", "single-turn response completed"
                                    )
                                ),
                            }
                        ],
                        "source": {
                            "uri": _task_source_uri(benchmark, source),
                            "license": benchmark.source_license,
                            "version": benchmark.source_version,
                        },
                        "benchmark_version": benchmark.benchmark_version,
                        "benchmark_subset": benchmark.benchmark_subset,
                        "functions": functions or [],
                        "metadata": metadata,
                    }
                )
            )
        _validate_tasks(benchmark, tasks)
        return tasks


class PassThroughEvaluator:
    stage = "execution"

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        del options

    def evaluate(
        self,
        task: TaskSpec,
        execution: ExecutionResult,
        backend: ExecutionBackend,
    ) -> JudgeResult:
        del task, backend
        passed = bool(execution.passed)
        return JudgeResult(
            passed=passed,
            faithfulness=passed,
            appropriateness=passed,
            completeness=passed,
            evidence_conflict=False,
            uncertain=False,
            reason=execution.reason,
        )


class BackendJudgeEvaluator:
    stage = "open_ended_judge"

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        del options

    def evaluate(
        self,
        task: TaskSpec,
        execution: ExecutionResult,
        backend: ExecutionBackend,
    ) -> JudgeResult:
        return backend.judge_open_ended(task, execution)


class TextMatchEvaluator:
    stage = "text_match"

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        self.mode = str(options.get("mode", "exact"))
        if self.mode not in {"exact", "contains"}:
            raise ValueError("text_match evaluator mode must be exact or contains")
        self.reference_key = str(options.get("reference_key", "reference"))
        self.case_sensitive = bool(options.get("case_sensitive", False))
        self.collapse_whitespace = bool(options.get("collapse_whitespace", True))

    def _normalize(self, value: Any) -> str:
        if isinstance(value, (dict, list)):
            result = _canonical(value)
        else:
            result = str(value)
        result = result.strip()
        if self.collapse_whitespace:
            result = " ".join(result.split())
        if not self.case_sensitive:
            result = result.casefold()
        return result

    def evaluate(
        self,
        task: TaskSpec,
        execution: ExecutionResult,
        backend: ExecutionBackend,
    ) -> JudgeResult:
        del backend
        if not execution.passed:
            return JudgeResult(
                passed=False,
                faithfulness=False,
                appropriateness=False,
                completeness=False,
                reason=execution.reason or "backend execution failed",
            )
        expected = _lookup(task.metadata, self.reference_key)
        references = expected if isinstance(expected, list) else [expected]
        content: Any = ""
        if execution.responses:
            content = execution.responses[-1].get("content", "")
        actual = self._normalize(content)
        normalized = [self._normalize(item) for item in references]
        if self.mode == "exact":
            passed = actual in normalized
        else:
            passed = any(item in actual for item in normalized)
        reason = (
            f"{self.mode} text match passed"
            if passed
            else f"{self.mode} text match failed"
        )
        return JudgeResult(
            passed=passed,
            faithfulness=passed,
            appropriateness=passed,
            completeness=passed,
            reason=reason,
        )


def _validate_tasks(benchmark: "BenchmarkSpec", tasks: Sequence[TaskSpec]) -> None:
    for task in tasks:
        if task.benchmark != benchmark.name:
            raise ValueError(
                f"task {task.instance_id!r} has benchmark {task.benchmark!r}; "
                f"expected {benchmark.name!r}"
            )
        if not task.steps:
            raise ValueError(f"task {task.instance_id!r} has no routed steps")
        if not benchmark.multi_step and len(task.steps) != 1:
            raise ValueError(
                f"single-turn benchmark {benchmark.name!r} received "
                f"{len(task.steps)} steps for {task.instance_id!r}"
            )


def _build_loader(spec: dict[str, Any]) -> TaskLoader:
    kind = str(spec.get("type", "normalized"))
    options = copy.deepcopy(spec.get("options") or {})
    if kind == "normalized":
        return NormalizedTaskLoader(options)
    if kind == "single_turn":
        return SingleTurnTaskLoader(options)
    if kind == "plugin":
        factory_path = str(spec.get("factory", ""))
        factory = _load_factory(factory_path)
        loader = factory(options)
        if not callable(getattr(loader, "load_tasks", None)):
            raise TypeError(f"loader plugin {factory_path!r} does not implement TaskLoader")
        return loader
    raise ValueError(f"unknown task loader type {kind!r}")


def _build_evaluator(spec: dict[str, Any]) -> ExecutionEvaluator:
    kind = str(spec.get("type", "execution"))
    options = copy.deepcopy(spec.get("options") or {})
    if kind == "execution":
        return PassThroughEvaluator(options)
    if kind == "backend_judge":
        return BackendJudgeEvaluator(options)
    if kind in {"exact_match", "contains"}:
        options.setdefault("mode", "exact" if kind == "exact_match" else "contains")
        return TextMatchEvaluator(options)
    if kind == "plugin":
        factory_path = str(spec.get("factory", ""))
        factory = _load_factory(factory_path)
        evaluator = factory(options)
        if not callable(getattr(evaluator, "evaluate", None)) or not isinstance(
            getattr(evaluator, "stage", None), str
        ):
            raise TypeError(
                f"evaluator plugin {factory_path!r} does not implement ExecutionEvaluator"
            )
        return evaluator
    raise ValueError(f"unknown evaluator type {kind!r}")


@dataclass
class BenchmarkSpec:
    name: str
    display_name: str
    default_scenario: str
    multi_step: bool
    manual_review: bool
    source_uri: str
    source_license: str = "unknown"
    source_version: str = "unknown"
    provenance_uri: str = ""
    benchmark_version: str = "unknown"
    benchmark_subset: str = ""
    fixture_name: str = ""
    loader_config: dict[str, Any] = field(default_factory=lambda: {"type": "normalized"})
    trial_evaluator_config: dict[str, Any] = field(
        default_factory=lambda: {"type": "execution"}
    )
    final_evaluator_config: dict[str, Any] = field(
        default_factory=lambda: {"type": "execution"}
    )
    executor_config: dict[str, Any] = field(default_factory=dict)
    loader: TaskLoader = field(init=False, repr=False)
    trial_evaluator: ExecutionEvaluator = field(init=False, repr=False)
    final_evaluator: ExecutionEvaluator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not _NAME_RE.fullmatch(self.name):
            raise ValueError(
                f"benchmark name {self.name!r} must match {_NAME_RE.pattern}"
            )
        if not self.display_name.strip() or not self.default_scenario.strip():
            raise ValueError("benchmark display_name and scenario cannot be empty")
        SourceSpec.parse(self.source_uri)
        self.loader = _build_loader(self.loader_config)
        self.trial_evaluator = _build_evaluator(self.trial_evaluator_config)
        self.final_evaluator = _build_evaluator(self.final_evaluator_config)

    @property
    def open_ended(self) -> bool:
        return self.final_evaluator_config.get("type") == "backend_judge"

    def load_tasks(self, source: SourceSpec | None = None) -> list[TaskSpec]:
        return self.loader.load_tasks(self, source or SourceSpec.parse(self.source_uri))

    def to_dict(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "scenario": self.default_scenario,
            "multi_step": self.multi_step,
            "manual_review": self.manual_review,
            "source": {
                "uri": self.source_uri,
                "license": self.source_license,
                "version": self.source_version,
                **({"provenance_uri": self.provenance_uri} if self.provenance_uri else {}),
            },
            "benchmark_version": self.benchmark_version,
            "benchmark_subset": self.benchmark_subset,
            **({"fixture_name": self.fixture_name} if self.fixture_name else {}),
            "loader": copy.deepcopy(self.loader_config),
            "evaluation": {
                "trial": copy.deepcopy(self.trial_evaluator_config),
                "final": copy.deepcopy(self.final_evaluator_config),
            },
            **({"executor": copy.deepcopy(self.executor_config)} if self.executor_config else {}),
        }

    @classmethod
    def from_dict(
        cls, name: str, raw: dict[str, Any], *, base_dir: Path | None = None
    ) -> "BenchmarkSpec":
        if not isinstance(raw, dict):
            raise ValueError(f"benchmark {name!r} configuration must be an object")
        source_raw = raw.get("source", f"fixture://{name}")
        if isinstance(source_raw, str):
            source_uri = source_raw
            source_license = str(raw.get("license", "unknown"))
            source_version = str(raw.get("source_version", "unknown"))
            provenance_uri = str(raw.get("provenance_uri", ""))
        elif isinstance(source_raw, dict):
            source_uri = str(source_raw.get("uri", ""))
            source_license = str(source_raw.get("license", "unknown"))
            source_version = str(source_raw.get("version", "unknown"))
            provenance_uri = str(source_raw.get("provenance_uri", ""))
        else:
            raise ValueError(f"benchmark {name!r} source must be a string or object")
        if not source_uri:
            raise ValueError(f"benchmark {name!r} source URI cannot be empty")
        source_uri = _resolve_local_uri(source_uri, base_dir)
        evaluation = raw.get("evaluation") or {}
        if not isinstance(evaluation, dict):
            raise ValueError(f"benchmark {name!r} evaluation must be an object")
        trial = _component(evaluation.get("trial"), "execution")
        final = _component(evaluation.get("final", evaluation.get("trial")), "execution")
        return cls(
            name=name,
            display_name=str(raw.get("display_name", name)),
            default_scenario=str(raw.get("scenario", "generic")),
            multi_step=bool(raw.get("multi_step", False)),
            manual_review=bool(raw.get("manual_review", False)),
            source_uri=source_uri,
            source_license=source_license,
            source_version=source_version,
            provenance_uri=provenance_uri,
            benchmark_version=str(raw.get("benchmark_version", source_version)),
            benchmark_subset=str(raw.get("benchmark_subset", "")),
            fixture_name=str(raw.get("fixture_name", "")),
            loader_config=_component(raw.get("loader"), "normalized"),
            trial_evaluator_config=trial,
            final_evaluator_config=final,
            executor_config=copy.deepcopy(raw.get("executor") or {}),
        )


def _resolve_local_uri(uri: str, base_dir: Path | None) -> str:
    if base_dir is None:
        return uri
    parsed = SourceSpec.parse(uri)
    if parsed.kind != "local":
        return uri
    path = Path(parsed.locator).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return f"local://{path}"


class BenchmarkRegistry:
    def __init__(self, specs: Sequence[BenchmarkSpec], *, version: str = "custom") -> None:
        if not specs:
            raise ValueError("benchmark suite must define at least one benchmark")
        self.version = version
        self._specs: dict[str, BenchmarkSpec] = {}
        for spec in specs:
            if spec.name in self._specs:
                raise ValueError(f"duplicate benchmark name: {spec.name}")
            self._specs[spec.name] = spec

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def get(self, name: str) -> BenchmarkSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ValueError(
                f"unknown benchmark {name!r}; expected one of {self.names}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SUITE_SCHEMA,
            "version": self.version,
            "benchmarks": {name: spec.to_dict() for name, spec in self._specs.items()},
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict()).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, base_dir: Path | None = None
    ) -> "BenchmarkRegistry":
        if not isinstance(raw, dict):
            raise ValueError("benchmark suite configuration must be an object")
        schema = str(raw.get("schema", SUITE_SCHEMA))
        if schema != SUITE_SCHEMA:
            raise ValueError(f"unsupported benchmark suite schema {schema!r}")
        benchmarks = raw.get("benchmarks")
        if not isinstance(benchmarks, dict) or not benchmarks:
            raise ValueError("benchmark suite requires a non-empty benchmarks object")
        specs = [
            BenchmarkSpec.from_dict(str(name), value, base_dir=base_dir)
            for name, value in benchmarks.items()
        ]
        return cls(specs, version=str(raw.get("version", "custom")))

    @classmethod
    def from_file(cls, path: str | Path) -> "BenchmarkRegistry":
        path = Path(path).expanduser().resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw, base_dir=path.parent)


def builtin_registry() -> BenchmarkRegistry:
    raw = json.loads(
        resources.files(__package__)
        .joinpath("benchmark_suite_v2.json")
        .read_text(encoding="utf-8")
    )
    return BenchmarkRegistry.from_dict(raw)
