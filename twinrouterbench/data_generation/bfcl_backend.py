"""Stateful BFCL multi-turn tool execution and the official BFCL checkers."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import re
import uuid
from pathlib import Path
from typing import Any

from .interactive import InteractiveBackend, Observation, OfficialVerdict
from .openai_backend import create_backend as create_client
from .types import TaskSpec


def task_from_entry(entry: dict[str, Any], ground_truth: list[list[str]], *,
                    max_steps: int = 200) -> TaskSpec:
    if not entry.get("function"):
        raise ValueError("BFCL entry must include its materialized official function documentation")
    from bfcl_eval.model_handler.utils import convert_to_tool
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.constants.enums import ModelStyle
    functions = convert_to_tool(entry["function"], GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
    questions = entry["question"]
    return TaskSpec.from_dict({
        "benchmark": "bfcl", "scenario": "bfcl_tool_use_multi_turn",
        "instance_id": entry["id"],
        # Match the official OpenAI function-calling handler: it starts with
        # the dataset's first turn, without an invented system instruction.
        "initial_messages": copy.deepcopy(questions[0]),
        "functions": functions,
        "steps": [{"hint": "low", "minimum_tier": "high"}],
        "source": {"uri": "local://bfcl-eval-2026.3.23-materialized", "license": "Apache-2.0", "version": "BFCL-v4"},
        "benchmark_version": "BFCL-v4-multi-turn",
        "metadata": {"bfcl_entry": copy.deepcopy(entry), "ground_truth": copy.deepcopy(ground_truth),
                     "infer_steps_from_seed": True, "max_steps": max_steps},
    })


class BFCLEnvironment:
    def __init__(self, task: TaskSpec) -> None:
        self.task = task
        self.entry = task.metadata["bfcl_entry"]
        self.ground_truth = task.metadata["ground_truth"]
        self.model_namespace = "trb_static_" + uuid.uuid4().hex
        self.turn = 0
        self.results: list[list[list[str]]] = [[] for _ in self.entry["question"]]
        self.allowed = {
            (item.get("function", item))["name"] for item in task.functions
        }

    def step(self, response: dict[str, Any]) -> Observation:
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
        calls = response.get("tool_calls") or []
        if calls:
            decoded = []
            for call in calls:
                function = call["function"]
                name = function["name"]
                if name not in self.allowed or not re.fullmatch(r"[a-zA-Z]\w*", name):
                    raise ValueError(f"BFCL function is not in the task schema: {name!r}")
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict) or any(not re.fullmatch(r"[a-zA-Z_]\w*", key) for key in arguments):
                    raise ValueError("BFCL function arguments must be a JSON object with named parameters")
                decoded.append(name + "(" + ", ".join(key + "=" + repr(value) for key, value in arguments.items()) + ")")
            outputs, _ = execute_multi_turn_func_call(decoded,
                copy.deepcopy(self.entry["initial_config"]), self.entry["involved_classes"],
                self.model_namespace, self.entry["id"],
                long_context=any(word in self.entry["id"] for word in ("long_context", "composite")))
            self.results[self.turn].append(decoded)
            return Observation(tuple({"role": "tool", "tool_call_id": call["id"], "content": output}
                                     for call, output in zip(calls, outputs)))
        if not self.results[self.turn]:
            self.results[self.turn].append([])
        self.turn += 1
        if self.turn == len(self.entry["question"]):
            return Observation(done=True)
        return Observation(tuple(copy.deepcopy(self.entry["question"][self.turn])))

    def evaluate(self) -> OfficialVerdict:
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
            multi_turn_checker, multi_turn_irrelevance_checker,
        )
        if len(self.ground_truth) != len(self.results):
            raise ValueError("BFCL ground-truth turn count does not match the task")
        model = self.model_namespace + "_score"
        state = multi_turn_checker(self.results, self.ground_truth, copy.deepcopy(self.entry),
            self.entry["id"].rsplit("_", 1)[0], model)
        irrelevant = multi_turn_irrelevance_checker(self.results, self.ground_truth)
        passed = state.get("valid") is True and irrelevant.get("valid") is True
        return OfficialVerdict(passed, "official BFCL state/response/irrelevance check: " + str(passed), {
            "harness": "bfcl-eval", "harness_version": importlib.metadata.version("bfcl-eval"),
            "state_and_response": state, "irrelevance": irrelevant,
            "decoded_calls": copy.deepcopy(self.results),
        })

    def close(self) -> None:
        from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
        for key in list(vars(multi_turn_utils)):
            if key.startswith(self.model_namespace + "_"):
                delattr(multi_turn_utils, key)


class BFCLLoader:
    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options

    def load_tasks(self, benchmark: Any, source: Any) -> list[TaskSpec]:
        path = source.require_materialized_path()
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(path.read_text())
        # Records intentionally retain the original entry and ground truth,
        # rather than inventing state from a released per-step prompt.
        return [task_from_entry(row["entry"], row["ground_truth"],
            max_steps=int(self.options.get("max_steps", 200))) for row in rows]


def create_loader(options: dict[str, Any]) -> BFCLLoader:
    return BFCLLoader(options)


def create_backend(config: dict[str, Any]) -> InteractiveBackend:
    return InteractiveBackend(create_client(config), BFCLEnvironment)
