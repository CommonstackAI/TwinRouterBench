"""PinchBench's own OpenClaw runner and grader, with a capped model proxy."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from .harness_proxy import HarnessProxyBackend
from .interactive import OfficialVerdict
from .openai_backend import create_backend as create_client
from .types import TaskSpec


def _load_tasks_module(skill_dir: Path) -> Any:
    spec = importlib.util.spec_from_file_location("twinrouter_pinch_tasks", skill_dir / "scripts/lib_tasks.py")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses with postponed annotations resolve their defining module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PinchBenchLoader:
    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options

    def load_tasks(self, benchmark: Any, source: Any) -> list[TaskSpec]:
        # Source is a pinned/materialized official skill directory. No agent
        # execution occurs during loading and no task is silently skipped.
        root = source.require_materialized_path()
        loader = _load_tasks_module(root).TaskLoader(root / "tasks")
        ids = self.options.get("task_ids")
        paths = [root / "tasks" / f"{name}.md" for name in ids] if ids else sorted((root / "tasks").glob("task_*.md"))
        tasks = []
        for path in paths:
            task = loader.load_task(path)
            tasks.append(TaskSpec.from_dict({
                "benchmark": benchmark.name, "benchmark_display": benchmark.display_name,
                "scenario": benchmark.default_scenario, "instance_id": task.task_id,
                "initial_messages": [{"role": "user", "content": task.prompt}],
                "steps": [{"hint": "low", "minimum_tier": "high"}],
                "source": {"uri": benchmark.provenance_uri or "local://pinchbench-official-skill",
                           "license": "MIT", "version": benchmark.source_version},
                "benchmark_version": benchmark.benchmark_version,
                "metadata": {"skill_dir": str(root), "task_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "infer_steps_from_seed": True, "capture_initial_prefix": True,
                    "freeze_initial_prefix": True, "max_steps": int(self.options.get("max_steps", 200))},
            }))
        return tasks


def create_loader(options: dict[str, Any]) -> PinchBenchLoader:
    return PinchBenchLoader(options)


class PinchBenchRunner:
    def __init__(self, options: dict[str, Any], judge_model: str) -> None:
        self.options = options
        self.judge_model = judge_model
        self.work_dir = Path(options.get("work_dir", "runs/pinchbench-execution")).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, task: TaskSpec, base_url: str, nonce: str) -> OfficialVerdict:
        run_dir = self.work_dir / ("trial-" + uuid.uuid4().hex)
        run_dir.mkdir()
        request = {
            "instance_id": task.instance_id, "skill_dir": task.metadata["skill_dir"],
            "task_sha256": task.metadata["task_sha256"],
            "work_dir": str(self.work_dir), "run_dir": str(run_dir),
            "proxy_base": base_url, "nonce": nonce,
            "node_path": self.options["node_path"], "openclaw_path": self.options["openclaw_path"],
            "judge_model": self.judge_model,
            "thresholds": self.options.get("thresholds", {"automated": 0.85, "llm_judge": 0.65, "hybrid": 0.75}),
        }
        (run_dir / "request.json").write_text(json.dumps(request))
        env = {key: value for key, value in os.environ.items()
               if not key.endswith("API_KEY") and key not in {"ANTHROPIC_AUTH_TOKEN", "OPENAI_ACCESS_TOKEN"}}
        with (run_dir / "runner.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "twinrouterbench.data_generation.pinchbench_runner",
                str(run_dir / "request.json")], stdout=log, stderr=subprocess.STDOUT,
                env=env, start_new_session=True)
            try:
                code = process.wait(timeout=int(self.options.get("timeout_seconds", 1800)))
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        if code:
            raise RuntimeError(f"PinchBench runner exited {code}; see {run_dir / 'runner.log'}")
        report = json.loads((run_dir / "result.json").read_text())
        return OfficialVerdict(report["passed"], "official PinchBench grade=" + str(report["score"]), report)


def create_backend(config: dict[str, Any]) -> HarnessProxyBackend:
    client = create_client(config)
    return HarnessProxyBackend(client, PinchBenchRunner(config.get("backend_options") or {}, client.judge_model))
