"""One live backend for all five workloads, sharing one provider spend cap."""

from __future__ import annotations

from typing import Any, Sequence

from .openai_backend import create_backend as create_client
from .types import Assignment, ExecutionResult, TaskSpec


class OfficialBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.client = create_client(config)
        self.backends: dict[str, Any] = {}

    def _backend(self, benchmark: str) -> Any:
        if benchmark not in self.backends:
            from .interactive import InteractiveBackend
            options = self.config.get("backend_options") or {}
            if benchmark == "swebench":
                from .swebench_backend import SwebenchEnvironments
                factory = SwebenchEnvironments(options.get("swebench_work_dir", "runs/swebench-execution"),
                    eval_timeout=int(options.get("eval_timeout_seconds", 1800)))
                backend = InteractiveBackend(self.client, factory)
            elif benchmark == "bfcl":
                from .bfcl_backend import BFCLEnvironment
                backend = InteractiveBackend(self.client, BFCLEnvironment)
            elif benchmark == "pinchbench":
                from .harness_proxy import HarnessProxyBackend
                from .pinchbench_backend import PinchBenchRunner
                backend = HarnessProxyBackend(self.client, PinchBenchRunner({**options,
                    "work_dir": options.get("pinchbench_work_dir", "runs/pinchbench-execution")}, self.client.judge_model))
            elif benchmark in {"mtrag", "qmsum"}:
                backend = self.client
            else:
                raise ValueError(f"no official execution integration for {benchmark!r}")
            self.backends[benchmark] = backend
        return self.backends[benchmark]

    def configure_generation(self, **kwargs: Any) -> None:
        self.client.configure_generation(**kwargs)

    @property
    def prompt_artifacts(self):
        return self.client.prompt_artifacts

    @property
    def proposer_model(self):
        return self.client.proposer_model

    @property
    def judge_model(self):
        return self.client.judge_model

    @property
    def model_aliases(self):
        return self.client.model_aliases

    def validate_benchmark(self, benchmark: Any) -> None:
        backend = self._backend(benchmark.name)
        if hasattr(backend, "validate_benchmark"):
            backend.validate_benchmark(benchmark)

    def run_seed(self, task: TaskSpec, strong_model: str) -> ExecutionResult:
        return self._backend(task.benchmark).run_seed(task, strong_model)

    def propose_hints(self, task, seed):
        return self.client.propose_hints(task, seed)

    def judge_open_ended(self, task, execution):
        return self.client.judge_open_ended(task, execution)

    def run_mixed(self, task: TaskSpec, assignments: Sequence[Assignment], *, target_step: int,
                  generation_parameters: dict[str, Any], locked_execution: ExecutionResult | None = None) -> ExecutionResult:
        return self._backend(task.benchmark).run_mixed(task, assignments, target_step=target_step,
            generation_parameters=generation_parameters, locked_execution=locked_execution)

    def cost_summary(self) -> dict[str, Any]:
        return self.client.budget.summary() if self.client.budget else {"unmetered": True}

    def close(self) -> None:
        for backend in self.backends.values():
            if hasattr(backend, "close"):
                backend.close()


def create_backend(config: dict[str, Any]) -> OfficialBackend:
    return OfficialBackend(config)
