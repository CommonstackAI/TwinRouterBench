"""Live SWE-bench execution using the public mini scaffold and official grader."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .interactive import InteractiveBackend, Observation, OfficialVerdict
from .openai_backend import create_backend as create_client
from .types import ExecutionResult, TaskSpec


def task_from_instance(instance: dict[str, Any], *, max_steps: int = 250,
                       source_uri: str = "local://swebench-materialized") -> TaskSpec:
    from jinja2 import StrictUndefined, Template
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL
    from miniswerouter.agent.prompts import INSTANCE_TEMPLATE, SYSTEM_TEMPLATE

    return TaskSpec.from_dict({
        "benchmark": "swebench", "scenario": "code_swe",
        "instance_id": instance["instance_id"],
        "initial_messages": [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user", "content": Template(INSTANCE_TEMPLATE, undefined=StrictUndefined).render(task=instance["problem_statement"])},
        ],
        "functions": [BASH_TOOL],
        "steps": [{"hint": "low", "minimum_tier": "high"}],
        "source": {"uri": source_uri, "license": "Apache-2.0", "version": "SWE-bench-Verified"},
        "benchmark_version": "SWE-bench-Verified-mini-2.2.8",
        "metadata": {"instance": copy.deepcopy(instance), "infer_steps_from_seed": True, "max_steps": max_steps},
    })


class SwebenchLoader:
    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options

    def load_tasks(self, benchmark: Any, source: Any) -> list[TaskSpec]:
        path = source.require_materialized_path()
        text = path.read_text()
        rows = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(text)
        if not isinstance(rows, list):
            raise ValueError("SWE-bench source must contain materialized instance records")
        tasks = [task_from_instance(row, max_steps=int(self.options.get("max_steps", 250)),
            source_uri=benchmark.provenance_uri or "local://swebench-materialized") for row in rows]
        return tasks


def create_loader(options: dict[str, Any]) -> SwebenchLoader:
    return SwebenchLoader(options)


class SwebenchEnvironment:
    def __init__(self, factory: "SwebenchEnvironments", task: TaskSpec) -> None:
        from minisweagent.models.litellm_model import LitellmModel
        from miniswerouter.harness.env import SwebenchContainerEnv, SwebenchContainerEnvConfig
        from miniswerouter.agent.prompts import (
            OBSERVATION_TEMPLATE, FORMAT_ERROR_TEMPLATE, MINI_SWEBENCH_ENV_DEFAULTS,
        )
        from swerouter.harness.container_runner import SwebenchContainerHandle, make_test_spec_for_instance

        self.factory = factory
        self.task = task
        self.run_id = "trb-static-" + uuid.uuid4().hex
        self.test_spec = make_test_spec_for_instance(task.metadata["instance"], image_namespace=factory.image_namespace)
        self.handle = SwebenchContainerHandle(self.test_spec, self.run_id,
            factory.work_dir / self.run_id / "agent.log")
        try:
            self.handle.start()
        except BaseException:
            self.handle.stop()
            raise
        self.env = SwebenchContainerEnv(self.handle,
            config=SwebenchContainerEnvConfig(**MINI_SWEBENCH_ENV_DEFAULTS))
        self.formatter = LitellmModel(model_name="openai/unused-local-formatter",
            observation_template=OBSERVATION_TEMPLATE, format_error_template=FORMAT_ERROR_TEMPLATE,
            cost_tracking="ignore_errors")
        self.patch = ""
        self.submitted = False

    def step(self, response: dict[str, Any]) -> Observation:
        from minisweagent.exceptions import Submitted, FormatError
        from minisweagent.models.utils.actions_toolcall import parse_toolcall_actions
        try:
            calls = [SimpleNamespace(id=call["id"], function=SimpleNamespace(**call["function"]))
                     for call in response.get("tool_calls", [])]
            actions = parse_toolcall_actions(calls,
                format_error_template=self.formatter.config.format_error_template)
        except FormatError as exc:
            return Observation(tuple(exc.messages), discard_response=True)
        message = {**response, "extra": {"actions": actions}}
        try:
            outputs = [self.env.execute(action) for action in actions]
        except Submitted as exc:
            self.submitted = True
            self.patch = exc.messages[0].get("extra", {}).get("submission", "")
            return Observation(done=True)
        observations = self.formatter.format_observation_messages(message, outputs, self.env.get_template_vars())
        return Observation(tuple(observations))

    def checkpoint(self) -> str:
        image = self.handle.container.commit(
            repository="twinrouter-static-checkpoint", tag=uuid.uuid4().hex)
        self.factory.images[image.id] = self.task.instance_id
        return image.id

    def restore(self, execution: ExecutionResult, target_step: int) -> None:
        image_id = execution.verification["checkpoints"][target_step - 2]
        if self.factory.images.get(image_id) != self.task.instance_id:
            raise ValueError("SWE-bench checkpoint is not owned by this task/run")
        self.handle.container.remove(force=True)
        self.handle.container = self.handle.client.containers.create(
            image_id, entrypoint="/bin/bash", command=["-c", "sleep infinity"],
            name=self.run_id + "-restored", working_dir="/testbed",
        )
        self.handle.container.start()

    def evaluate(self) -> OfficialVerdict:
        from swerouter.harness.container_runner import run_upstream_eval
        report = run_upstream_eval(test_spec=self.test_spec, instance_id=self.task.instance_id,
            patch_text=self.patch if self.submitted else "", run_id=self.run_id + "-grade",
            timeout_sec=self.factory.eval_timeout, rm_image=False,
            model_name="twinrouter-static-validation")
        if report.error and self.patch.strip():
            raise RuntimeError(f"official SWE-bench evaluation failed: {report.error}")
        return OfficialVerdict(bool(report.resolved),
            "official SWE-bench resolved=" + str(bool(report.resolved)), {
                "harness": "swebench", "harness_version": importlib.metadata.version("swebench"),
                "scaffold": "mini-swe-agent==2.2.8", "instance_id": self.task.instance_id,
                "resolved": bool(report.resolved), "patch_applied": report.patch_applied,
                "report_path": str(report.report_path or ""),
            })

    def close(self) -> None:
        self.handle.stop()


class SwebenchEnvironments:
    def __init__(self, work_dir: str | Path, *, image_namespace: str = "swebench", eval_timeout: int = 1800) -> None:
        if importlib.metadata.version("mini-swe-agent") != "2.2.8":
            raise RuntimeError("static SWE-bench backend requires mini-swe-agent==2.2.8")
        if importlib.metadata.version("swebench") != "3.0.15":
            raise RuntimeError("static SWE-bench backend requires swebench==3.0.15")
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.image_namespace = image_namespace
        self.eval_timeout = eval_timeout
        self.images: dict[str, str] = {}

    def __call__(self, task: TaskSpec) -> SwebenchEnvironment:
        return SwebenchEnvironment(self, task)

    def close(self) -> None:
        if not self.images:
            return
        import docker
        client = docker.from_env()
        for image_id in reversed(list(self.images)):
            client.images.remove(image_id, force=False)
            del self.images[image_id]


def create_backend(config: dict[str, Any]) -> InteractiveBackend:
    options = config.get("backend_options") or {}
    factory = SwebenchEnvironments(options.get("work_dir", "runs/swebench-execution"),
        image_namespace=options.get("image_namespace", "swebench"),
        eval_timeout=int(options.get("eval_timeout_seconds", 1800)))
    backend = InteractiveBackend(create_client(config), factory)
    return backend
