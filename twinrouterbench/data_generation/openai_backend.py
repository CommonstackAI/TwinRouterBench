"""Generic OpenAI-compatible execution backend for config-only benchmarks."""

from __future__ import annotations

import copy
import json
import os
import urllib.error
import urllib.request
from typing import Any, Sequence

from .types import Assignment, DowngradeHint, ExecutionResult, JudgeResult, TaskSpec


class OpenAICompatibleBackend:
    """Execute normalized tasks against any OpenAI-compatible chat endpoint.

    This backend is intended for prompt/response benchmarks whose correctness
    can be expressed by a configured evaluator (exact match, contains, or a
    plugin). Agentic benchmarks with sandboxes or real tools should provide a
    benchmark-owned ``ExecutionBackend`` plugin instead.
    """

    def __init__(
        self,
        model_pool: dict[str, list[str]],
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 120,
        max_tokens: int = 1024,
        judge_model: str = "",
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI-compatible backend API key is empty")
        if not base_url.strip():
            raise ValueError("OpenAI-compatible backend base URL is empty")
        self.model_pool = model_pool
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.judge_model = judge_model or model_pool["high"][0]

    def _chat(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        functions: Sequence[dict[str, Any]] = (),
        generation_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        parameters = copy.deepcopy(generation_parameters or {})
        body: dict[str, Any] = {
            "model": model,
            "messages": list(copy.deepcopy(messages)),
            "temperature": parameters.pop("temperature", 0),
            "top_p": parameters.pop("top_p", 1),
            "max_tokens": parameters.pop("max_tokens", self.max_tokens),
            **parameters,
        }
        if functions:
            body["tools"] = [
                item
                if item.get("type") == "function"
                else {"type": "function", "function": copy.deepcopy(item)}
                for item in functions
            ]
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "TwinRouterBench-generic-backend",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(
                f"OpenAI-compatible request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
        try:
            message = raw["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("provider returned an invalid chat-completion envelope") from exc
        result = {
            "role": str(message.get("role") or "assistant"),
            "content": message.get("content", ""),
        }
        if message.get("tool_calls"):
            result["tool_calls"] = copy.deepcopy(message["tool_calls"])
        return result

    def _execute(
        self,
        task: TaskSpec,
        assignments: Sequence[Assignment],
        generation_parameters: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        if len(assignments) != len(task.steps):
            raise ValueError("assignment count does not match normalized task steps")
        messages = [copy.deepcopy(item) for item in task.initial_messages]
        prefixes: list[tuple[dict[str, Any], ...]] = []
        responses: list[dict[str, Any]] = []
        for index, (step, assignment) in enumerate(zip(task.steps, assignments), 1):
            prefixes.append(tuple(copy.deepcopy(messages)))
            response = self._chat(
                model=assignment.model,
                messages=messages,
                functions=task.functions,
                generation_parameters=generation_parameters,
            )
            responses.append(copy.deepcopy(response))
            messages.append(response)
            if index < len(task.steps):
                messages.append(
                    {
                        "role": "user",
                        "content": f"[Environment observation] {step.observation}",
                    }
                )
        return ExecutionResult(
            passed=True,
            step_count=len(task.steps),
            prefixes=tuple(prefixes),
            responses=tuple(responses),
            reason="provider execution completed",
        )

    def run_seed(self, task: TaskSpec, strong_model: str) -> ExecutionResult:
        return self._execute(
            task, [Assignment("high", strong_model) for _ in task.steps]
        )

    def propose_hints(
        self, task: TaskSpec, seed: ExecutionResult
    ) -> Sequence[DowngradeHint]:
        del seed
        return [
            DowngradeHint(index, step.hint, "configured normalized-task hint")
            for index, step in enumerate(task.steps, 1)
        ]

    def run_mixed(
        self,
        task: TaskSpec,
        assignments: Sequence[Assignment],
        *,
        target_step: int,
        generation_parameters: dict[str, Any],
    ) -> ExecutionResult:
        del target_step
        return self._execute(task, assignments, generation_parameters)

    def judge_open_ended(
        self, task: TaskSpec, execution: ExecutionResult
    ) -> JudgeResult:
        prompt = {
            "task_messages": list(task.initial_messages),
            "candidate_responses": list(execution.responses),
            "reference": task.metadata.get("reference"),
            "criteria": [
                "faithfulness",
                "appropriateness",
                "completeness",
                "evidence_conflict",
                "uncertain",
            ],
            "output": "Return only a JSON object with boolean fields passed, faithfulness, appropriateness, completeness, evidence_conflict, uncertain, plus a short reason.",
        }
        message = self._chat(
            model=self.judge_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict benchmark evaluator. Return valid JSON only.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            generation_parameters={"temperature": 0, "max_tokens": 512},
        )
        content = str(message.get("content") or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].lstrip()
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("judge returned invalid JSON") from exc
        return JudgeResult.from_dict(raw)


def create_backend(config: dict[str, Any]) -> OpenAICompatibleBackend:
    """Plugin factory driven entirely by non-secret config plus environment."""

    from .pipeline import load_model_pool

    options = copy.deepcopy(config.get("backend_options") or {})
    key_env = str(options.pop("api_key_env", "TWINROUTERBENCH_API_KEY"))
    base_env = str(options.pop("base_url_env", "TWINROUTERBENCH_API_BASE"))
    default_base = str(options.pop("default_base_url", ""))
    api_key = os.environ.get(key_env) or os.environ.get("COMMONSTACK_API_KEY", "")
    base_url = (
        os.environ.get(base_env)
        or os.environ.get("COMMONSTACK_API_BASE")
        or default_base
    )
    _, pool = load_model_pool(config.get("model_pool_path") or None)
    return OpenAICompatibleBackend(
        pool,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=float(options.pop("timeout_seconds", 120)),
        max_tokens=int(options.pop("max_tokens", 1024)),
        judge_model=str(options.pop("judge_model", "")),
    )
