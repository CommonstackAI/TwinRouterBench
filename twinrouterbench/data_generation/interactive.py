"""Execution-backed search over resettable benchmark environments."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from .types import Assignment, DowngradeHint, ExecutionResult, JudgeResult, TaskSpec


@dataclass(frozen=True)
class Observation:
    messages: tuple[dict[str, Any], ...] = ()
    done: bool = False
    discard_response: bool = False


@dataclass(frozen=True)
class OfficialVerdict:
    passed: bool
    reason: str
    evidence: dict[str, Any]

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("official verdict must be a boolean")


class Environment(Protocol):
    def step(self, response: dict[str, Any]) -> Observation: ...
    def evaluate(self) -> OfficialVerdict: ...
    def close(self) -> None: ...


class InteractiveBackend:
    """Replay locked actions in a fresh environment, then generate the suffix.

    Environment factories must reset state on construction and use official
    evaluators. Observations are executed again and compared, never copied from
    the recorded trajectory. A mismatch rejects the candidate without a paid
    suffix call. This class does not assume that task completion implies success.
    """

    def __init__(self, client: Any, environment_factory: Callable[[TaskSpec], Environment]) -> None:
        self.client = client
        self.environment_factory = environment_factory

    def configure_generation(self, **kwargs: Any) -> None:
        self.client.configure_generation(**kwargs)

    @property
    def prompt_artifacts(self) -> Any:
        return self.client.prompt_artifacts

    @property
    def proposer_model(self) -> str:
        return self.client.proposer_model

    @property
    def judge_model(self) -> str:
        return self.client.judge_model

    def close(self) -> None:
        close = getattr(self.environment_factory, "close", None)
        if close is not None and not isinstance(self.environment_factory, type):
            close()

    def run_seed(self, task: TaskSpec, strong_model: str) -> ExecutionResult:
        maximum = len(task.steps)
        if task.metadata.get("infer_steps_from_seed"):
            maximum = int(task.metadata["max_steps"])
        if maximum <= 0:
            raise ValueError("max_steps must be positive")
        return self._execute(task, [Assignment("high", strong_model)] * maximum,
            generation_parameters=self.client.generation_parameters)

    def propose_hints(self, task: TaskSpec, seed: ExecutionResult) -> Sequence[DowngradeHint]:
        return self.client.propose_hints(task, seed)

    def judge_open_ended(self, task: TaskSpec, execution: ExecutionResult) -> JudgeResult:
        return self.client.judge_open_ended(task, execution)

    def run_mixed(self, task: TaskSpec, assignments: Sequence[Assignment], *,
                  target_step: int, generation_parameters: dict[str, Any],
                  locked_execution: ExecutionResult | None = None) -> ExecutionResult:
        if not 1 <= target_step <= len(assignments):
            raise ValueError("target_step is out of range")
        if target_step > 1 and locked_execution is None:
            raise ValueError("multi-step search requires a locked execution")
        return self._execute(task, assignments, target_step=target_step,
            generation_parameters=generation_parameters, locked_execution=locked_execution)

    def _execute(self, task: TaskSpec, assignments: Sequence[Assignment], *,
                 generation_parameters: dict[str, Any], target_step: int = 1,
                 locked_execution: ExecutionResult | None = None) -> ExecutionResult:
        env = self.environment_factory(task)
        messages = list(copy.deepcopy(task.initial_messages))
        prefixes: list[tuple[dict[str, Any], ...]] = []
        responses: list[dict[str, Any]] = []
        checkpoints: list[str] = []
        done = False
        try:
            start = 0
            restore = getattr(env, "restore", None)
            if locked_execution is not None and target_step > 1 and restore is not None:
                restore(locked_execution, target_step)
                start = target_step - 1
                messages = list(copy.deepcopy(locked_execution.prefixes[start]))
                prefixes = list(copy.deepcopy(locked_execution.prefixes[:start]))
                responses = list(copy.deepcopy(locked_execution.responses[:start]))
                checkpoints = list(locked_execution.verification["checkpoints"][:start])
            for index, assignment in enumerate(assignments):
                if index < start:
                    continue
                prefixes.append(tuple(copy.deepcopy(messages)))
                if locked_execution is not None and index < target_step:
                    if tuple(messages) != locked_execution.prefixes[index]:
                        return ExecutionResult(False, len(responses), tuple(prefixes[:-1]),
                            tuple(responses), "environment replay changed the locked prefix",
                            verification={"prefix_replay_matches": False})
                if index < target_step - 1:
                    response = copy.deepcopy(locked_execution.responses[index])
                else:
                    response = self.client._chat(model=assignment.model,
                        messages=messages, functions=task.functions,
                        generation_parameters=generation_parameters)
                responses.append(copy.deepcopy(response))
                messages.append(copy.deepcopy(response))
                observation = env.step(response)
                if observation.discard_response:
                    messages.pop()
                messages.extend(copy.deepcopy(observation.messages))
                done = observation.done
                checkpoint = getattr(env, "checkpoint", None)
                checkpoints.append(checkpoint() if checkpoint is not None and not done else "")
                if done:
                    break
            if not done:
                return ExecutionResult(False, len(responses), tuple(prefixes), tuple(responses),
                    "environment did not terminate within the trajectory step limit",
                    verification={"terminated": False})
            verdict = env.evaluate()
            return ExecutionResult(verdict.passed, len(responses), tuple(prefixes),
                tuple(responses), verdict.reason,
                verification={**verdict.evidence, "checkpoints": checkpoints})
        finally:
            env.close()
