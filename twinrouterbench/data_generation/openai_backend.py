"""Generic OpenAI-compatible execution backend for config-only benchmarks."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from importlib import resources
from pathlib import Path
from typing import Any, Sequence

from .types import Assignment, DowngradeHint, ExecutionResult, JudgeResult, TaskSpec
from .budget import ApiBudget, request_reservation


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
        proposer_model: str = "",
        generation_parameters: dict[str, Any] | None = None,
        prompt_versions: dict[str, str] | None = None,
        model_aliases: dict[str, str] | None = None,
        pricing_catalog: dict[str, dict[str, Any]] | None = None,
        budget: ApiBudget | None = None,
        json_response_format: dict[str, Any] | None = None,
        judge_max_tokens: int = 512,
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
        self.proposer_model = proposer_model or model_pool["high"][0]
        self.model_aliases = dict(model_aliases or {})
        self.pricing_catalog = pricing_catalog or {}
        self.budget = budget
        self.json_response_format = copy.deepcopy(json_response_format)
        if type(judge_max_tokens) is not int or judge_max_tokens <= 0:
            raise ValueError("judge_max_tokens must be a positive integer")
        self.judge_max_tokens = judge_max_tokens
        self.configure_generation(
            generation_parameters=generation_parameters or {},
            prompt_versions=prompt_versions,
        )

    def configure_generation(
        self, *, generation_parameters: dict[str, Any],
        prompt_versions: dict[str, str] | None,
    ) -> None:
        self.generation_parameters = copy.deepcopy(generation_parameters)
        versions = prompt_versions if prompt_versions is not None else {
            "tier_proposer": "tier_proposer_v1",
            "open_ended_judge": "open_ended_judge_v1",
        }
        self.prompts: dict[str, str] = {}
        self.prompt_artifacts: dict[str, dict[str, Any]] = {}
        for stage in ("tier_proposer", "open_ended_judge"):
            version = versions.get(stage, "")
            if not re.fullmatch(r"[a-zA-Z0-9_\-]+", version):
                raise ValueError(f"invalid or missing prompt version for {stage}")
            content = resources.files(__package__).joinpath(
                "prompts", f"{version}.txt"
            ).read_bytes()
            self.prompts[stage] = content.decode("utf-8")
            self.prompt_artifacts[stage] = {
                "version": version,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            if self.json_response_format is not None:
                self.prompt_artifacts[stage]["response_format"] = copy.deepcopy(self.json_response_format)
            if stage == "open_ended_judge":
                self.prompt_artifacts[stage]["max_tokens"] = self.judge_max_tokens

    def validate_benchmark(self, benchmark: Any) -> None:
        if benchmark.multi_step:
            raise ValueError(
                "generic OpenAI backend does not execute multi-step environments; "
                "provide a benchmark-owned execution backend with an official verdict"
            )
        for stage in ("trial_evaluator_config", "final_evaluator_config"):
            if getattr(benchmark, stage).get("type") == "execution":
                raise ValueError(
                    "generic OpenAI backend requires an independent trial and final "
                    "evaluator (exact_match, contains, backend_judge, or plugin); "
                    "execution completion is not a correctness verdict"
                )

    def validate_task(self, task: TaskSpec) -> None:
        if len(task.steps) != 1 or task.functions:
            raise ValueError(
                "generic OpenAI backend supports text-only single-turn tasks; "
                "multi-step/tool execution requires a benchmark-owned backend"
            )

    @staticmethod
    def _json_response(message: dict[str, Any]) -> dict[str, Any]:
        content = str(message.get("content") or "").strip()
        if content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()
            if content.startswith("json"):
                content = content[4:].lstrip()
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            # Some compatible providers prefix the one structured verdict
            # with prose even when JSON mode was requested. Accept only the
            # first object if it consumes the entire remainder. Multiple or
            # malformed objects and trailing commentary stay invalid; no
            # verdict is inferred from the prose.
            start = content.find("{")
            try:
                if start <= 0:
                    raise ValueError("no single trailing JSON object")
                raw, end = json.JSONDecoder().raw_decode(content[start:])
                suffix = content[start + end:].strip()
                prefix = content[:start].rstrip()
                fenced = (suffix == "```" and prefix.count("```") == 1
                          and re.search(r"(?:^|\n)```(?:json)?\s*$", prefix) is not None)
                if suffix and not fenced:
                    raise ValueError("content follows the JSON object")
            except ValueError:
                raise RuntimeError("evaluator/proposer returned invalid JSON") from exc
        if not isinstance(raw, dict):
            raise RuntimeError("evaluator/proposer must return a JSON object")
        return raw

    def _chat(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        functions: Sequence[dict[str, Any]] = (),
        generation_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        parameters = copy.deepcopy(generation_parameters or {})
        if {"model", "messages", "tools", "functions"}.intersection(parameters):
            raise ValueError("generation parameters cannot override the selected model or task")
        if "max_tokens" in parameters and "max_completion_tokens" in parameters:
            raise ValueError("specify only one output token limit")
        if "max_completion_tokens" not in parameters:
            parameters["max_tokens"] = parameters.get("max_tokens", self.max_tokens)
        body: dict[str, Any] = {
            "model": self.model_aliases.get(model, model),
            "messages": list(copy.deepcopy(messages)),
            "temperature": parameters.pop("temperature", 0),
            "top_p": parameters.pop("top_p", 1),
            **parameters,
        }
        if functions:
            body["tools"] = [
                item
                if item.get("type") == "function"
                else {"type": "function", "function": copy.deepcopy(item)}
                for item in functions
            ]
        reservation = None
        pricing = None
        if self.budget is not None:
            pricing = self.pricing_catalog.get(body["model"])
            if pricing is None:
                raise ValueError(f"no provider price for {body['model']!r}; refusing unbudgeted call")
            reservation = self.budget.reserve(
                body["model"], request_reservation(body, pricing)
            )
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
            detail = exc.read().decode("utf-8", "replace").replace(self.api_key, "[REDACTED]")[:500]
            raise RuntimeError(
                f"OpenAI-compatible request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
        if reservation is not None:
            self.budget.finish(
                reservation, usage=raw.get("usage") or {}, pricing=pricing,
                provider=str(raw.get("provider") or raw.get("model") or ""),
            )
        self.last_completion = copy.deepcopy(raw)
        try:
            message = raw["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("provider returned an invalid chat-completion envelope") from exc
        if not isinstance(message, dict):
            raise RuntimeError("provider returned an invalid assistant message")
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
        self.validate_task(task)
        if len(assignments) != len(task.steps):
            raise ValueError("assignment count does not match normalized task steps")
        messages = [copy.deepcopy(item) for item in task.initial_messages]
        prefixes: list[tuple[dict[str, Any], ...]] = []
        responses: list[dict[str, Any]] = []
        for assignment in assignments:
            prefixes.append(tuple(copy.deepcopy(messages)))
            response = self._chat(
                model=assignment.model,
                messages=messages,
                functions=task.functions,
                generation_parameters=generation_parameters,
            )
            responses.append(copy.deepcopy(response))
            if response.get("tool_calls"):
                raise ValueError(
                    "generic OpenAI backend cannot execute tool calls; "
                    "provide a benchmark-owned backend"
                )
            messages.append(response)
        return ExecutionResult(
            passed=True,
            step_count=len(task.steps),
            prefixes=tuple(prefixes),
            responses=tuple(responses),
            reason="provider response received; correctness requires independent evaluation",
            verified=False,
        )

    def run_seed(self, task: TaskSpec, strong_model: str) -> ExecutionResult:
        return self._execute(
            task, [Assignment("high", strong_model) for _ in task.steps],
            self.generation_parameters,
        )

    def propose_hints(
        self, task: TaskSpec, seed: ExecutionResult
    ) -> Sequence[DowngradeHint]:
        # Proposal/judgment can be reused by an execution-backed multi-step
        # backend; only this class's transport-only execution is single-turn.
        message = self._chat(
            model=self.proposer_model,
            messages=[
                {"role": "system", "content": self.prompts["tier_proposer"]},
                {"role": "user", "content": json.dumps({
                    "task_messages": list(task.initial_messages),
                    "successful_trajectory": seed.to_dict(),
                }, ensure_ascii=False)},
            ],
            generation_parameters={"temperature": 0, "max_tokens": self.max_tokens,
                **({"response_format": self.json_response_format} if self.json_response_format is not None else {})},
        )
        raw = self._json_response(message)
        steps = raw.get("steps")
        if not isinstance(steps, list) or len(steps) != len(task.steps):
            raise ValueError("tier proposer must return exactly one hint per step")
        if any(not isinstance(item, dict) or type(item.get("step_index")) is not int for item in steps):
            raise ValueError("tier proposer step indices must be integers")
        hints = [DowngradeHint.from_dict(item) for item in steps]
        if [hint.step_index for hint in hints] != list(range(1, len(task.steps) + 1)):
            raise ValueError("tier proposer returned invalid step indices")
        if any(hint.start_tier == "high" for hint in hints):
            raise ValueError("tier proposer must use a lower tier or not_downgradeable")
        return hints

    def run_mixed(
        self,
        task: TaskSpec,
        assignments: Sequence[Assignment],
        *,
        target_step: int,
        generation_parameters: dict[str, Any],
        locked_execution: ExecutionResult | None = None,
    ) -> ExecutionResult:
        del locked_execution
        if target_step != 1:
            raise ValueError("generic OpenAI backend only supports target_step=1")
        return self._execute(task, assignments, generation_parameters)

    def judge_open_ended(
        self, task: TaskSpec, execution: ExecutionResult
    ) -> JudgeResult:
        prompt = {
            "task_messages": list(task.initial_messages),
            "candidate_responses": list(execution.responses),
            "reference": task.metadata.get("reference"),
            "conversation_prefixes": [list(prefix) for prefix in execution.prefixes],
        }
        message = self._chat(
            model=self.judge_model,
            messages=[
                {
                    "role": "system",
                    "content": self.prompts["open_ended_judge"],
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            generation_parameters={"temperature": 0, "max_tokens": self.judge_max_tokens,
                **({"response_format": self.json_response_format} if self.json_response_format is not None else {})},
        )
        raw = self._json_response(message)
        for field in (
            "passed", "faithfulness", "appropriateness", "completeness",
            "evidence_conflict", "uncertain",
        ):
            if field not in raw or type(raw[field]) is not bool:
                raise ValueError(f"judge field {field!r} must be a JSON boolean")
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
    catalog_path = str(options.pop("pricing_catalog_path", ""))
    prices = {}
    if catalog_path:
        catalog = json.loads(Path(catalog_path).expanduser().read_text())
        for item in catalog.get("data", []):
            prices[item["id"]] = item.get("pricing") or {}
    limit = options.pop("budget_usd", os.environ.get("TWINROUTERBENCH_API_BUDGET_USD", ""))
    budget = None
    if limit:
        ledger = options.pop("ledger_path", os.environ.get(
            "TWINROUTERBENCH_API_LEDGER",
            str(Path.home() / ".local/state/twinrouterbench/api-spend.jsonl"),
        ))
        budget = ApiBudget(ledger, float(limit))
    return OpenAICompatibleBackend(
        pool,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=float(options.pop("timeout_seconds", 120)),
        max_tokens=int(options.pop("max_tokens", 1024)),
        judge_model=str(options.pop("judge_model", "")),
        proposer_model=str(options.pop("proposer_model", "")),
        generation_parameters=config.get("generation_parameters"),
        prompt_versions=config.get("prompt_versions"),
        model_aliases=options.pop("model_aliases", {}),
        pricing_catalog=prices,
        budget=budget,
        json_response_format=options.pop("json_response_format", None),
        judge_max_tokens=options.pop("judge_max_tokens", 512),
    )
