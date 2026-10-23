"""Isolated subprocess bridge to a materialized PinchBench skill checkout."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path


def run(path: Path) -> None:
    request = json.loads(path.read_text())
    root = Path(request["work_dir"]).resolve()
    task_id = request["instance_id"]
    if not task_id or Path(task_id).name != task_id:
        raise ValueError("invalid PinchBench task id")
    fixed = root / ("task-" + task_id)
    fixed.mkdir(exist_ok=True)
    marker = fixed / ".twinrouter-owned"
    if any(fixed.iterdir()) and not marker.exists():
        raise ValueError("refusing to reset an unowned task workspace")
    marker.touch()
    state = fixed / "state"
    if state.is_symlink():
        raise ValueError("state directory must not be a symlink")
    if state.exists():
        shutil.rmtree(state)
    state.mkdir()
    skill_dir = Path(request["skill_dir"])
    task_path = skill_dir / "tasks" / (task_id + ".md")
    if hashlib.sha256(task_path.read_bytes()).hexdigest() != request["task_sha256"]:
        raise ValueError("PinchBench task changed after materialization")
    os.environ["OPENCLAW_STATE_DIR"] = str(state)
    os.environ["OPENCLAW_CONFIG_PATH"] = str(state / "openclaw.json")
    os.environ["NO_PROXY"] = "127.0.0.1,localhost," + os.environ.get("NO_PROXY", "")
    wrapper = fixed / "bin"
    wrapper.mkdir(exist_ok=True)
    executable = wrapper / "openclaw"
    # The upstream helper translates workspace options to its gateway's
    # /home/node path. This bridge runs locally, so translate only CLI path
    # options back to this run's state directory; never rewrite prompt text.
    executable.write_text("#!" + sys.executable + "\nimport os, sys\n" +
        "args = sys.argv[1:]\n" +
        "for i in range(1, len(args)):\n" +
        " if args[i-1] in {'--workspace', '--agent-dir'} and args[i].startswith('/home/node/.openclaw/'):\n" +
        "  args[i] = " + repr(str(state)) + " + args[i][len('/home/node/.openclaw'):]\n" +
        "if args and args[0] == 'agent': args.insert(1, '--local')\n" +
        "os.execv(" + repr(request["node_path"]) + ", " + repr([request["node_path"], request["openclaw_path"]]) + " + args)\n")
    executable.chmod(0o755)
    os.environ["PATH"] = str(wrapper) + os.pathsep + str(Path(request["node_path"]).parent) + os.pathsep + os.environ["PATH"]
    virtual_model = "anthropic/claude-opus-4.6"
    workspace = state / "workspace" / task_id
    config = {
        "models": {"providers": {"openrouter": {
            "baseUrl": request["proxy_base"] + "/v1", "api": "openai-completions", "apiKey": "local-proxy",
            "models": [{"id": virtual_model, "name": "TwinRouter capped proxy", "reasoning": False,
                        "input": ["text"], "contextWindow": 200000, "maxTokens": 8192,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}],
        }}},
        # Match the upstream task preparation, which removes onboarding files.
        # Local CLI mode must not recreate them or ask a shared gateway to
        # approve the benchmark's own shell commands.
        "agents": {"defaults": {"workspace": str(workspace), "skipBootstrap": True,
                                 "model": {"primary": "openrouter/" + virtual_model}}},
        "tools": {"exec": {"host": "gateway", "security": "full", "ask": "off"}},
    }
    (state / "openclaw.json").write_text(json.dumps(config))
    # These are upstream modules, run without editing their tool/grade logic.
    sys.path.insert(0, str(skill_dir / "scripts"))
    import lib_agent
    import lib_grading
    import lib_tasks
    task = lib_tasks.TaskLoader(skill_dir / "tasks").load_task(task_path)
    agent_id = "trb-" + task_id
    if not lib_agent.ensure_agent_exists(agent_id, "openrouter/" + virtual_model, workspace):
        raise RuntimeError("official OpenClaw agent setup failed")
    execution = lib_agent.execute_openclaw_task(task=task, agent_id=agent_id,
        model_id="openrouter/" + virtual_model, run_id="fixed-prefix", timeout_multiplier=1,
        skill_dir=skill_dir)
    stage = urllib.request.Request(request["proxy_base"] + "/__stage",
        data=b'{"stage":"judge"}', headers={"Content-Type": "application/json", "X-TwinRouter-Nonce": request["nonce"]})
    with urllib.request.urlopen(stage, timeout=10):
        pass
    # The official judge also goes through the same capped proxy. Give it the
    # existing virtual endpoint instead of configuring an unmetered provider.
    grade = lib_grading.grade_task(task=task, execution_result=execution, skill_dir=skill_dir,
        judge_model="openrouter/" + virtual_model)
    threshold = float(request["thresholds"][task.grading_type])
    report = {"harness": "pinchbench", "task_id": task_id, "passed": grade.score >= threshold,
              "score": grade.score, "threshold": threshold, "grade": grade.to_dict(),
              "execution_status": execution.get("status"), "workspace": execution.get("workspace"),
              "local_execution_policy": config["tools"]["exec"], "skip_bootstrap": True,
              "source_hashes": {name: hashlib.sha256((skill_dir / "scripts" / name).read_bytes()).hexdigest()
                                for name in ("lib_agent.py", "lib_tasks.py", "lib_grading.py")}}
    result_dir = Path(request["run_dir"])
    (result_dir / "execution.json").write_text(json.dumps(execution, default=str))
    (result_dir / "result.json").write_text(json.dumps(report, indent=2))
    shutil.copytree(state, result_dir / "state", dirs_exist_ok=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
