"""Per-step agent loop.

Given one :class:`AgentConfig` and one :class:`Router`, run the tool-use loop
against one SWE-bench Docker container until the model calls ``finish``, the
step budget is reached, or the USD budget is reached.

Every step writes a JSON line to ``trace_path`` so the leaderboard scorer can
reconstruct per-step costs and cache behaviour without holding everything in
memory.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from swerouter.agent.prompts import SYSTEM_PROMPT, render_user_prompt
from swerouter.agent.tools import (
    FINISH_SENTINEL,
    TOOL_REGISTRY,
    ToolResult,
    default_tool_schemas,
    execute_tool_call,
)
from swerouter.cache import PromptCacheModel, TTLPolicy
from swerouter.llm_client import ChatCallResult, LLMClient
from swerouter.pricing import PricingTable, step_real_cost_usd
from swerouter.router import (
    RouterContext,
    RunConfig,
    Router,
    validate_decision,
)


@dataclass(frozen=True)
class ModelPoolEntry:
    """One entry from ``data/model_pool.json`` resolved at run start."""

    model_id: str
    provider: str
    is_high_baseline: bool
    cache_control_style: str | None = None


@dataclass(frozen=True)
class AgentConfig:
    """Everything the agent loop needs to run one instance."""

    max_steps: int = 40
    budget_usd: float = 10.0
    max_response_tokens: int = 4096
    temperature: float = 0.0
    select_timeout_sec: float = 30.0


@dataclass
class AgentRunResult:
    """Summary produced by :func:`run_agent_loop` for a single instance."""

    instance_id: str
    step_count: int
    finished_by: str
    total_router_cost_usd: float
    trace_path: Path
    error: str | None = None
    model_distribution: dict[str, int] = field(default_factory=dict)


def _pool_to_available_models(pool: Iterable[ModelPoolEntry]) -> tuple[str, ...]:
    return tuple(p.model_id for p in pool)


def _router_entry_for(
    pool: Iterable[ModelPoolEntry], model_id: str
) -> ModelPoolEntry:
    for p in pool:
        if p.model_id == model_id:
            return p
    raise ValueError(
        f"model_id {model_id!r} returned by router is not in the pool (validation slipped)"
    )


def _trace_write(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSONL row to the trace file."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        fh.write("\n")


def _io_write(path: Path | None, row: Mapping[str, Any]) -> None:
    """Append one JSONL row to the full-I/O debug log, if configured.

    Unlike ``trace.jsonl``, this file stores the *verbatim* request/response
    payloads (full ``messages`` history, full assistant message, full tool
    stdout). It is intentionally verbose; consumers should stream-parse it.
    """
    if path is None:
        return
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        fh.write("\n")


def _extract_tool_calls(assistant_message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = assistant_message.get("tool_calls") or []
    if not isinstance(raw, list):
        raise ValueError(f"assistant.tool_calls is not a list: {type(raw).__name__}")
    return raw


def _normalize_tool_result_content(result: ToolResult) -> str:
    """Cap tool stdout so a single tool response cannot explode into the next
    prompt. Hard limit matches OpenAI's tool-result truncation rule of thumb."""
    limit = 64_000
    if len(result.content) <= limit:
        return result.content
    head = result.content[: limit // 2]
    tail = result.content[-limit // 2 :]
    return f"{head}\n\n... (truncated from {len(result.content)} chars) ...\n\n{tail}"


def run_agent_loop(
    *,
    instance_id: str,
    repo: str,
    base_commit: str,
    problem_statement: str,
    container: Any,
    router: Router,
    pool: list[ModelPoolEntry],
    pricing: PricingTable,
    ttl: TTLPolicy,
    llm: LLMClient,
    config: AgentConfig,
    trace_path: Path,
    io_path: Path | None = None,
) -> AgentRunResult:
    """Drive one SWE-bench instance through the tool-use loop.

    The function is synchronous. It expects ``container`` to already be a
    running ``docker.models.containers.Container`` with the repository checked
    out at ``/testbed``. Patch extraction (``git diff``) is performed by the
    caller after this function returns.
    """

    available_models = _pool_to_available_models(pool)
    cache = PromptCacheModel(ttl)
    user_prompt = render_user_prompt(
        repo=repo,
        base_commit=base_commit,
        instance_id=instance_id,
        problem_statement=problem_statement,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    tools = default_tool_schemas()

    run_config = RunConfig(
        max_steps=config.max_steps,
        budget_usd=config.budget_usd,
        wallclock_ttl_sec=ttl.wallclock_ttl_sec,
        select_timeout_sec=config.select_timeout_sec,
    )

    model_dist: dict[str, int] = {}
    total_cost = 0.0
    step_index = 0
    finished_by = "max_steps_reached"
    error: str | None = None

    # Clear any pre-existing trace file for this instance.
    if trace_path.exists():
        trace_path.unlink()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if io_path is not None:
        if io_path.exists():
            io_path.unlink()
        io_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        while step_index < config.max_steps:
            now_ts = time.time()
            ctx = RouterContext(
                instance_id=instance_id,
                step_index=step_index,
                messages=tuple(messages),
                tools=tuple(tools),
                available_models=available_models,
                cache_state=cache.snapshot(
                    now_ts=now_ts, available_models=available_models
                ),
                budget_so_far_usd=total_cost,
                run_config=run_config,
            )
            raw_decision = router.select(ctx)
            decision = validate_decision(
                raw_decision, available_models=available_models
            )
            pool_entry = _router_entry_for(pool, decision.model_id)

            cache_lookup = cache.lookup(
                model_id=decision.model_id,
                messages=tuple(messages),
                now_ts=now_ts,
            )

            # Snapshot the exact prompt we are about to send BEFORE llm.chat
            # mutates nothing (defensive copy for the full-I/O log).
            request_messages_snapshot = (
                [dict(m) for m in messages] if io_path is not None else None
            )

            chat_result: ChatCallResult = llm.chat(
                model_id=decision.model_id,
                provider=pool_entry.provider,
                messages=messages,
                tools=tools,
                max_tokens=config.max_response_tokens,
                temperature=config.temperature,
                cache_control_style=pool_entry.cache_control_style,
            )

            step_cost = step_real_cost_usd(
                chat_result.normalized_usage, pricing.get(decision.model_id)
            )
            total_cost += step_cost
            model_dist[decision.model_id] = model_dist.get(decision.model_id, 0) + 1

            prefix_total = chat_result.normalized_usage.total_prompt_tokens
            cache.update(
                model_id=decision.model_id,
                messages=tuple(messages),
                prefix_token_count=prefix_total,
                ts=now_ts,
            )

            # Record a short preview of the last message the LLM saw going
            # into this step. Gives post-hoc debuggers enough context to see
            # why the model picked this action (i.e. what the last tool
            # result looked like).
            last_msg = messages[-1] if messages else {}
            last_msg_content = last_msg.get("content")
            if isinstance(last_msg_content, list):
                last_msg_content = " ".join(
                    str(b.get("text", "") or b) for b in last_msg_content
                )
            elif last_msg_content is None:
                last_msg_content = ""
            last_msg_preview = str(last_msg_content)[:500]

            trace_row: dict[str, Any] = {
                "instance_id": instance_id,
                "step_index": step_index,
                "started_at": now_ts,
                "finished_at": time.time(),
                "model_id": decision.model_id,
                "provider": pool_entry.provider,
                "rationale": decision.rationale,
                "latency_ms": chat_result.latency_ms,
                "prompt_messages_count": len(messages),
                "prompt_tail_role": last_msg.get("role"),
                "prompt_tail_preview": last_msg_preview,
                "usage": asdict(chat_result.normalized_usage),
                "raw_usage": dict(chat_result.raw_usage),
                "cache_lookup": {
                    "hit": cache_lookup.hit,
                    "reason": cache_lookup.reason,
                    "cached_prefix_token_count": cache_lookup.cached_prefix_token_count,
                },
                "step_cost_usd": step_cost,
                "cumulative_cost_usd": total_cost,
            }

            assistant = chat_result.assistant_message
            tool_calls = _extract_tool_calls(assistant)
            content_text = assistant.get("content")
            trace_row["assistant_content_len"] = (
                len(content_text) if isinstance(content_text, str) else 0
            )
            trace_row["tool_call_count"] = len(tool_calls)
            # Record a concise preview of each tool_call so offline tools
            # (e.g. scripts/analyze_gt_vs_trace.py) can diagnose divergence
            # from CRB's gold trajectory without re-running the whole loop.
            tc_preview: list[dict[str, Any]] = []
            for tc in tool_calls:
                if not isinstance(tc, Mapping):
                    continue
                fn = tc.get("function") or {}
                name = fn.get("name") if isinstance(fn, Mapping) else None
                raw_args = fn.get("arguments") if isinstance(fn, Mapping) else None
                if isinstance(raw_args, (dict, list)):
                    args_str = json.dumps(raw_args, ensure_ascii=False)
                else:
                    args_str = str(raw_args) if raw_args is not None else ""
                max_args_chars = 400
                truncated = len(args_str) > max_args_chars
                if truncated:
                    args_str = args_str[:max_args_chars]
                tc_preview.append(
                    {
                        "tool_name": name,
                        "args_preview": args_str,
                        "args_truncated": truncated,
                    }
                )
            trace_row["tool_calls_preview"] = tc_preview
            if isinstance(content_text, str) and content_text:
                trace_row["assistant_content_preview"] = content_text[:300]
            _trace_write(trace_path, trace_row)

            # Budget gate AFTER writing the step (so the expensive step is
            # recorded; the NEXT step does not fire).
            if total_cost > config.budget_usd:
                finished_by = "budget_exhausted"
                break

            # Append the assistant message exactly as the model produced it.
            messages.append(dict(assistant))

            if not tool_calls:
                # Model returned prose with no tool call -> either it's done
                # talking without finishing, or it's going to continue on the
                # next turn. We stop if content contains an unambiguous
                # give-up signal; otherwise we prod it to use finish.
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Please either continue making progress using the tools, "
                            "or call the finish tool if you are done."
                        ),
                    }
                )
                _io_write(
                    io_path,
                    {
                        "kind": "step",
                        "instance_id": instance_id,
                        "step_index": step_index,
                        "started_at": now_ts,
                        "finished_at": time.time(),
                        "model_id": decision.model_id,
                        "provider": pool_entry.provider,
                        "rationale": decision.rationale,
                        "request": {
                            "messages": request_messages_snapshot,
                            "tools": tools,
                            "max_tokens": config.max_response_tokens,
                            "temperature": config.temperature,
                            "cache_control_style": pool_entry.cache_control_style,
                        },
                        "response": {
                            "assistant_message": dict(assistant),
                            "raw_usage": dict(chat_result.raw_usage),
                            "normalized_usage": asdict(chat_result.normalized_usage),
                            "latency_ms": chat_result.latency_ms,
                        },
                        "tool_results": [],
                    },
                )
                step_index += 1
                continue

            finish_triggered = False
            tool_result_previews: list[dict[str, Any]] = []
            tool_results_full: list[dict[str, Any]] = []
            for tc in tool_calls:
                tc_id = tc.get("id")
                fn = tc.get("function") or {}
                name = fn.get("name")
                arg_json = fn.get("arguments") or "{}"
                if not isinstance(name, str) or name not in TOOL_REGISTRY:
                    raise ValueError(
                        f"model emitted unknown tool_call: name={name!r}"
                    )
                if not isinstance(arg_json, str):
                    raise TypeError(
                        f"tool_call.function.arguments must be string, got {type(arg_json).__name__}"
                    )
                result = execute_tool_call(container, name, arg_json)
                tool_result_previews.append(
                    {
                        "tool_name": result.tool_name,
                        "ok": result.ok,
                        "content_preview": (result.content or "")[:500],
                        "content_length": len(result.content or ""),
                    }
                )
                normalized_tool_content = _normalize_tool_result_content(result)
                if io_path is not None:
                    tool_results_full.append(
                        {
                            "tool_call_id": tc_id,
                            "tool_name": result.tool_name,
                            "ok": result.ok,
                            "arguments": arg_json,
                            "content_raw": result.content,
                            "content_raw_length": len(result.content or ""),
                            "content_sent_to_model": normalized_tool_content,
                            "content_truncated_for_model": (
                                len(result.content or "") > len(normalized_tool_content)
                            ),
                        }
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": normalized_tool_content,
                    }
                )
                if result.content == FINISH_SENTINEL:
                    finish_triggered = True
            # Append tool result previews to the trace row we already wrote
            # by writing a lightweight follow-up line tagged with the same
            # step_index. Simpler than seeking+rewriting the JSONL line.
            _trace_write(
                trace_path,
                {
                    "__marker__": "tool_results",
                    "instance_id": instance_id,
                    "step_index": step_index,
                    "tool_results": tool_result_previews,
                },
            )
            _io_write(
                io_path,
                {
                    "kind": "step",
                    "instance_id": instance_id,
                    "step_index": step_index,
                    "started_at": now_ts,
                    "finished_at": time.time(),
                    "model_id": decision.model_id,
                    "provider": pool_entry.provider,
                    "rationale": decision.rationale,
                    "request": {
                        "messages": request_messages_snapshot,
                        "tools": tools,
                        "max_tokens": config.max_response_tokens,
                        "temperature": config.temperature,
                        "cache_control_style": pool_entry.cache_control_style,
                    },
                    "response": {
                        "assistant_message": dict(assistant),
                        "raw_usage": dict(chat_result.raw_usage),
                        "normalized_usage": asdict(chat_result.normalized_usage),
                        "latency_ms": chat_result.latency_ms,
                    },
                    "tool_results": tool_results_full,
                },
            )

            step_index += 1
            if finish_triggered:
                finished_by = "finish_tool"
                break

        else:
            finished_by = "max_steps_reached"

    except Exception as ex:  # noqa: BLE001 — we intentionally surface every failure
        error = f"{type(ex).__name__}: {ex}"
        finished_by = "error"
        raise
    finally:
        # Ensure we always record a loop-summary line so downstream can detect
        # partial runs even if an exception escaped.
        _trace_write(
            trace_path,
            {
                "__marker__": "loop_summary",
                "instance_id": instance_id,
                "step_count": step_index,
                "finished_by": finished_by,
                "total_router_cost_usd": total_cost,
                "error": error,
                "model_distribution": dict(model_dist),
            },
        )
        _io_write(
            io_path,
            {
                "kind": "loop_summary",
                "instance_id": instance_id,
                "step_count": step_index,
                "finished_by": finished_by,
                "total_router_cost_usd": total_cost,
                "error": error,
                "model_distribution": dict(model_dist),
            },
        )

    return AgentRunResult(
        instance_id=instance_id,
        step_count=step_index,
        finished_by=finished_by,
        total_router_cost_usd=total_cost,
        trace_path=trace_path,
        error=error,
        model_distribution=dict(model_dist),
    )


def load_model_pool(path: str | Path) -> list[ModelPoolEntry]:
    """Parse ``data/model_pool.json`` into a list of :class:`ModelPoolEntry`."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"model pool file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        raise ValueError("model pool root must be object")
    pool_raw = doc.get("pool")
    if not isinstance(pool_raw, list) or not pool_raw:
        raise ValueError("model pool .pool must be a non-empty array")
    out: list[ModelPoolEntry] = []
    high_count = 0
    for item in pool_raw:
        if not isinstance(item, dict):
            raise ValueError("model pool entries must be objects")
        model_id = item.get("model_id")
        provider = item.get("provider")
        is_high = bool(item.get("is_high_baseline", False))
        cc_style = item.get("cache_control_style")
        if cc_style is not None and not isinstance(cc_style, str):
            raise ValueError(
                f"model pool entry cache_control_style must be string or null: {item!r}"
            )
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"model pool entry missing model_id: {item!r}")
        if not isinstance(provider, str) or not provider:
            raise ValueError(f"model pool entry missing provider: {item!r}")
        out.append(
            ModelPoolEntry(
                model_id=model_id,
                provider=provider,
                is_high_baseline=is_high,
                cache_control_style=cc_style,
            )
        )
        if is_high:
            high_count += 1
    if high_count != 1:
        raise ValueError(
            f"model pool must designate exactly one is_high_baseline=true model, got {high_count}"
        )
    return out


def high_baseline_model_id(pool: Iterable[ModelPoolEntry]) -> str:
    """Return the single pool entry marked as ``is_high_baseline``."""
    for p in pool:
        if p.is_high_baseline:
            return p.model_id
    raise ValueError("no is_high_baseline model in pool")
