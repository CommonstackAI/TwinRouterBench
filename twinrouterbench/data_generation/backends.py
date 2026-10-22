"""Execution backends for construction runs.

``MockBackend`` is deterministic and exercises the complete algorithm without
network access. ``ReplayBackend`` rehydrates calls captured by a prior run.
Live execution is a plugin contract so benchmark owners can connect their
actual harness without coupling the public package to private infrastructure.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from .types import (
    TIER_TO_ID,
    Assignment,
    DowngradeHint,
    ExecutionResult,
    JudgeResult,
    TaskSpec,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_key(operation: str, request: dict[str, Any]) -> str:
    payload = f"{operation}\n{_canonical(request)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def locked_prefix_key(execution: ExecutionResult, target_step: int) -> str:
    raw = execution.to_dict()
    return request_key("locked_prefix", {
        "prefixes": raw["prefixes"][:target_step],
        "responses": raw["responses"][:target_step - 1],
        "functions_by_step": execution.verification.get("functions_by_step", [])[:target_step],
    })


@runtime_checkable
class ExecutionBackend(Protocol):
    def run_seed(self, task: TaskSpec, strong_model: str) -> ExecutionResult: ...

    def propose_hints(
        self, task: TaskSpec, seed: ExecutionResult
    ) -> Sequence[DowngradeHint]: ...

    def run_mixed(
        self,
        task: TaskSpec,
        assignments: Sequence[Assignment],
        *,
        target_step: int,
        generation_parameters: dict[str, Any],
        locked_execution: ExecutionResult | None = None,
    ) -> ExecutionResult: ...

    def judge_open_ended(
        self, task: TaskSpec, execution: ExecutionResult
    ) -> JudgeResult: ...


class MockBackend:
    """Deterministic paper-protocol simulator driven by normalized fixtures."""

    def __init__(self, model_pool: dict[str, list[str]]) -> None:
        self.model_pool = model_pool
        self.model_to_tier = {
            model: tier for tier, models in model_pool.items() for model in models
        }

    def run_seed(self, task: TaskSpec, strong_model: str) -> ExecutionResult:
        assignments = [Assignment("high", strong_model) for _ in task.steps]
        result = self._execute(task, assignments)
        if task.seed_passed:
            return result
        return ExecutionResult(
            passed=False,
            step_count=result.step_count,
            prefixes=result.prefixes,
            responses=result.responses,
            reason="fixture seed failure",
        )

    def propose_hints(
        self, task: TaskSpec, seed: ExecutionResult
    ) -> Sequence[DowngradeHint]:
        del seed
        return [
            DowngradeHint(
                step_index=index,
                start_tier=step.hint,
                reason=(
                    "fixture marks the step as non-downgradeable"
                    if step.hint == "not_downgradeable"
                    else f"probe from {step.hint}"
                ),
            )
            for index, step in enumerate(task.steps, 1)
        ]

    def run_mixed(
        self,
        task: TaskSpec,
        assignments: Sequence[Assignment],
        *,
        target_step: int,
        generation_parameters: dict[str, Any],
        locked_execution: ExecutionResult | None = None,
    ) -> ExecutionResult:
        del generation_parameters
        return self._execute(
            task, assignments, target_step=target_step,
            locked_execution=locked_execution,
        )

    def judge_open_ended(
        self, task: TaskSpec, execution: ExecutionResult
    ) -> JudgeResult:
        passed = bool(task.judge_passed and execution.passed)
        return JudgeResult(
            passed=passed,
            faithfulness=passed,
            appropriateness=passed,
            completeness=passed,
            evidence_conflict=not task.judge_passed,
            uncertain=False,
            reason=task.judge_reason,
        )

    def _execute(
        self, task: TaskSpec, assignments: Sequence[Assignment], *,
        target_step: int = 1,
        locked_execution: ExecutionResult | None = None,
    ) -> ExecutionResult:
        if len(assignments) != len(task.steps):
            raise ValueError(
                f"assignment count {len(assignments)} does not match "
                f"{len(task.steps)} steps for {task.instance_id}"
            )

        passed = True
        reason = "all assigned steps satisfy fixture constraints"
        for index, (step, assignment) in enumerate(zip(task.steps, assignments), 1):
            actual_tier = self.model_to_tier.get(assignment.model)
            if actual_tier != assignment.tier:
                passed = False
                reason = f"step {index}: model is not in assigned tier"
                break
            if assignment.model in step.force_pass_models:
                continue
            if assignment.model in step.fail_models:
                passed = False
                reason = f"step {index}: fixture forces model failure"
                break
            if TIER_TO_ID[assignment.tier] < TIER_TO_ID[step.minimum_tier]:
                passed = False
                reason = f"step {index}: assigned tier is below minimum"
                break

        if passed:
            tiers_by_step = {
                str(index): assignment.tier
                for index, assignment in enumerate(assignments, 1)
            }
            for blocked in task.blocked_tier_combinations:
                if all(tiers_by_step.get(step) == tier for step, tier in blocked.items()):
                    passed = False
                    reason = "fixture blocks this cross-step tier combination"
                    break

        messages = [copy.deepcopy(item) for item in task.initial_messages]
        prefixes: list[tuple[dict[str, Any], ...]] = []
        responses: list[dict[str, Any]] = []
        if locked_execution is not None:
            # Synthetic fixtures have no external tool state. Live plugins
            # must also restore/replay the corresponding environment state.
            messages = list(copy.deepcopy(locked_execution.prefixes[target_step - 1]))
            prefixes = list(copy.deepcopy(locked_execution.prefixes[:target_step - 1]))
            responses = list(copy.deepcopy(locked_execution.responses[:target_step - 1]))
        for index, (step, assignment) in enumerate(zip(task.steps, assignments), 1):
            if locked_execution is not None and index < target_step:
                continue
            prefixes.append(tuple(copy.deepcopy(messages)))
            response = {
                "role": "assistant",
                "content": f"synthetic response for step {index} at tier {assignment.tier}",
            }
            responses.append(copy.deepcopy(response))
            messages.append(response)
            messages.append(
                {
                    "role": "tool",
                    "name": "fixture_environment",
                    "content": step.observation,
                }
            )

        return ExecutionResult(
            passed=passed,
            step_count=len(task.steps),
            prefixes=tuple(prefixes),
            responses=tuple(responses),
            reason=reason,
        )


class RecordingBackend:
    """Append every backend request/result pair to a replayable JSONL log."""

    def __init__(
        self, backend: ExecutionBackend, path: Path, *,
        generation_context: dict[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self.path = path
        self.generation_context = copy.deepcopy(generation_context or {})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def _record(
        self, operation: str, request: dict[str, Any], result: Any
    ) -> None:
        if self.generation_context:
            request = {**request, "generation_context": self.generation_context}
        if hasattr(result, "to_dict"):
            encoded = result.to_dict()
        elif isinstance(result, Sequence):
            encoded = [item.to_dict() for item in result]
        else:
            raise TypeError(f"unsupported backend result for {operation}: {type(result)}")
        row = {
            "operation": operation,
            "request_key": request_key(operation, request),
            "request": request,
            "result": encoded,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical(row) + "\n")

    def validate_task(self, task: TaskSpec) -> None:
        validate = getattr(self.backend, "validate_task", None)
        if validate is not None:
            validate(task)
        if len(task.steps) > 1:
            parameters = inspect.signature(self.backend.run_mixed).parameters
            if "locked_execution" not in parameters and not any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            ):
                raise ValueError(
                    "multi-step backend must accept locked_execution and restore "
                    "its locked responses and environment state before target_step"
                )

    def run_seed(self, task: TaskSpec, strong_model: str) -> ExecutionResult:
        request = {
            "benchmark": task.benchmark,
            "instance_id": task.instance_id,
            "strong_model": strong_model,
        }
        result = self.backend.run_seed(task, strong_model)
        self._record("seed", request, result)
        return result

    def propose_hints(
        self, task: TaskSpec, seed: ExecutionResult
    ) -> Sequence[DowngradeHint]:
        request = {
            "benchmark": task.benchmark,
            "instance_id": task.instance_id,
            "seed_passed": seed.passed,
            "step_count": seed.step_count,
            "seed_key": request_key("execution", seed.to_dict()),
        }
        result = list(self.backend.propose_hints(task, seed))
        self._record("hints", request, result)
        return result

    def run_mixed(
        self,
        task: TaskSpec,
        assignments: Sequence[Assignment],
        *,
        target_step: int,
        generation_parameters: dict[str, Any],
        locked_execution: ExecutionResult | None = None,
    ) -> ExecutionResult:
        request = {
            "benchmark": task.benchmark,
            "instance_id": task.instance_id,
            "target_step": target_step,
            "assignments": [item.to_dict() for item in assignments],
            "generation_parameters": generation_parameters,
        }
        extra: dict[str, Any] = {}
        if locked_execution is not None:
            request["locked_prefix_key"] = locked_prefix_key(locked_execution, target_step)
            extra["locked_execution"] = copy.deepcopy(locked_execution)
        result = self.backend.run_mixed(
            task,
            assignments,
            target_step=target_step,
            generation_parameters=generation_parameters,
            **extra,
        )
        self._record("mixed", request, result)
        return result

    def judge_open_ended(
        self, task: TaskSpec, execution: ExecutionResult
    ) -> JudgeResult:
        request = {
            "benchmark": task.benchmark,
            "instance_id": task.instance_id,
            "execution_passed": execution.passed,
            "step_count": execution.step_count,
            "execution_key": request_key("execution", execution.to_dict()),
        }
        result = self.backend.judge_open_ended(task, execution)
        self._record("judge", request, result)
        return result


class ReplayBackend:
    """Serve deterministic backend results from ``backend_events.jsonl``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.generation_context: dict[str, Any] = {}
        if not self.path.is_file():
            raise FileNotFoundError(f"replay log does not exist: {self.path}")
        self._events: dict[tuple[str, str], deque[Any]] = defaultdict(deque)
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["operation"]), str(row["request_key"]))
            self._events[key].append(row["result"])

    def configure_generation(
        self, *, generation_parameters: dict[str, Any],
        prompt_versions: dict[str, str],
    ) -> None:
        self.generation_context = copy.deepcopy({
            "generation_parameters": generation_parameters,
            "prompt_versions": prompt_versions,
        })

    def _get(self, operation: str, request: dict[str, Any]) -> Any:
        if self.generation_context:
            request = {**request, "generation_context": self.generation_context}
        key = (operation, request_key(operation, request))
        if not self._events.get(key):
            raise KeyError(
                f"replay log has no {operation} event for request {_canonical(request)}"
            )
        return copy.deepcopy(self._events[key].popleft())

    def run_seed(self, task: TaskSpec, strong_model: str) -> ExecutionResult:
        raw = self._get(
            "seed",
            {
                "benchmark": task.benchmark,
                "instance_id": task.instance_id,
                "strong_model": strong_model,
            },
        )
        return ExecutionResult.from_dict(raw)

    def propose_hints(
        self, task: TaskSpec, seed: ExecutionResult
    ) -> Sequence[DowngradeHint]:
        raw = self._get(
            "hints",
            {
                "benchmark": task.benchmark,
                "instance_id": task.instance_id,
                "seed_passed": seed.passed,
                "step_count": seed.step_count,
                "seed_key": request_key("execution", seed.to_dict()),
            },
        )
        return [DowngradeHint.from_dict(item) for item in raw]

    def run_mixed(
        self,
        task: TaskSpec,
        assignments: Sequence[Assignment],
        *,
        target_step: int,
        generation_parameters: dict[str, Any],
        locked_execution: ExecutionResult | None = None,
    ) -> ExecutionResult:
        request = {
            "benchmark": task.benchmark,
            "instance_id": task.instance_id,
            "target_step": target_step,
            "assignments": [item.to_dict() for item in assignments],
            "generation_parameters": generation_parameters,
        }
        if locked_execution is not None:
            request["locked_prefix_key"] = locked_prefix_key(locked_execution, target_step)
        raw = self._get("mixed", request)
        return ExecutionResult.from_dict(raw)

    def judge_open_ended(
        self, task: TaskSpec, execution: ExecutionResult
    ) -> JudgeResult:
        raw = self._get(
            "judge",
            {
                "benchmark": task.benchmark,
                "instance_id": task.instance_id,
                "execution_passed": execution.passed,
                "step_count": execution.step_count,
                "execution_key": request_key("execution", execution.to_dict()),
            },
        )
        return JudgeResult.from_dict(raw)


def load_live_backend(spec: str, config: dict[str, Any]) -> ExecutionBackend:
    """Load ``module:factory`` and validate the returned backend protocol."""

    if ":" not in spec:
        raise ValueError("live backend must use module:factory syntax")
    module_name, attribute = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    backend = factory(config) if callable(factory) else factory
    if not isinstance(backend, ExecutionBackend):
        raise TypeError(
            f"live backend {spec!r} does not implement the ExecutionBackend protocol"
        )
    return backend
