"""CommonStack live transport smoke backend.

This plugin performs a real OpenAI-compatible API request for every backend
operation while retaining the deterministic fixture oracle for pass/fail
decisions. It validates credentials, provider transport, plugin loading, and
the complete construction state machine. It does not produce official
benchmark scores or replace benchmark-owned execution harnesses.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Sequence

from .backends import MockBackend
from .types import Assignment, DowngradeHint, ExecutionResult, JudgeResult, TaskSpec


DEFAULT_BASE_URL = "https://api.commonstack.ai/v1"
DEFAULT_PROBE_MODEL = "google/gemini-2.5-flash"


class CommonStackSmokeBackend(MockBackend):
    """Exercise every live backend method through a real provider request."""

    def __init__(
        self,
        model_pool: dict[str, list[str]],
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        probe_model: str = DEFAULT_PROBE_MODEL,
        timeout_seconds: float = 90,
    ) -> None:
        super().__init__(model_pool)
        if not api_key.strip():
            raise ValueError("COMMONSTACK_API_KEY is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.probe_model = probe_model
        self.timeout_seconds = timeout_seconds

    def _probe(self, operation: str, task: TaskSpec, payload: dict[str, Any]) -> None:
        task_preview = json.dumps(
            list(task.initial_messages), ensure_ascii=False, sort_keys=True
        )[:4000]
        request_body = {
            "model": self.probe_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a transport health check. Reply with exactly OK.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "benchmark": task.benchmark,
                            "instance_id": task.instance_id,
                            "operation": operation,
                            "task_preview": task_preview,
                            "function_count": len(task.functions),
                            **payload,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 8,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "TwinRouterBench-live-smoke",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(
                f"CommonStack {operation} probe failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"CommonStack {operation} probe failed: {exc}") from exc
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"CommonStack {operation} probe returned an invalid response envelope"
            ) from exc
        if not str(content or "").strip():
            raise RuntimeError(f"CommonStack {operation} probe returned empty content")

    def run_seed(self, task: TaskSpec, strong_model: str) -> ExecutionResult:
        self._probe("seed", task, {"strong_model": strong_model})
        return super().run_seed(task, strong_model)

    def propose_hints(
        self, task: TaskSpec, seed: ExecutionResult
    ) -> Sequence[DowngradeHint]:
        self._probe("hints", task, {"step_count": seed.step_count})
        return super().propose_hints(task, seed)

    def run_mixed(
        self,
        task: TaskSpec,
        assignments: Sequence[Assignment],
        *,
        target_step: int,
        generation_parameters: dict[str, Any],
    ) -> ExecutionResult:
        self._probe(
            "mixed",
            task,
            {
                "target_step": target_step,
                "candidate_tier": assignments[target_step - 1].tier,
                "candidate_model": assignments[target_step - 1].model,
            },
        )
        return super().run_mixed(
            task,
            assignments,
            target_step=target_step,
            generation_parameters=generation_parameters,
        )

    def judge_open_ended(
        self, task: TaskSpec, execution: ExecutionResult
    ) -> JudgeResult:
        self._probe("judge", task, {"execution_passed": execution.passed})
        return super().judge_open_ended(task, execution)


def create_backend(config: dict[str, Any]) -> CommonStackSmokeBackend:
    """Factory used by ``--live-backend module:factory``."""

    from .pipeline import load_model_pool

    _, model_pool = load_model_pool(config.get("model_pool_path") or None)
    return CommonStackSmokeBackend(
        model_pool,
        api_key=os.environ.get("COMMONSTACK_API_KEY", ""),
        base_url=os.environ.get("COMMONSTACK_API_BASE", DEFAULT_BASE_URL),
        probe_model=os.environ.get("COMMONSTACK_SMOKE_MODEL", DEFAULT_PROBE_MODEL),
        timeout_seconds=float(os.environ.get("COMMONSTACK_TIMEOUT_SECONDS", "90")),
    )
