"""``MiniRouterAgent``: subclass of ``minisweagent.agents.default.DefaultAgent``
that wires SWERouterBench's per-step :class:`swerouter.router.Router`
protocol into mini's scaffold.

The override is deliberately narrow:

* We reuse mini's message/history management (``self.messages``,
  ``add_messages``, ``run``, ``save``) and its limit checks
  (``step_limit`` / ``cost_limit`` via ``LimitsExceeded``).
* We override :meth:`query` -- the exact hook point mini's own docstring
  advertises ("Override to add hooks") -- to (a) build a
  :class:`swerouter.router.RouterContext` from the live conversation, (b)
  call the router and validate the decision against the pool, and (c)
  dispatch to :meth:`RouterAwareModel.query_as` so the right pool member
  runs the step.
* After the model call we also append a SWERouterBench-compatible row to
  ``trace_path``; :mod:`swerouter.leaderboard.score` consumes that file
  verbatim, so the same scorer used by SWERouterBench grades
  MiniSWERouterBench runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import LimitsExceeded

from swerouter.cache import PromptCacheModel, TTLPolicy
from swerouter.router import (
    CacheStateSnapshot,
    Router,
    RouterContext,
    RunConfig,
    validate_decision,
)
from miniswerouter.agent.model import RouterAwareModel, StepRecord


class MiniRouterAgent(DefaultAgent):
    """``DefaultAgent`` with per-step router dispatch and SWERouterBench trace.

    The caller constructs a :class:`RouterAwareModel` and a live
    :class:`SwebenchContainerEnv`, then this agent:

    1. Runs mini's scaffold (system prompt -> task prompt -> linear
       bash-only step loop -> ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``).
    2. At every step, consults ``self.router`` to pick one ``model_id`` from
       the locked pool and dispatches the LLM call to that pool member.
    3. Writes one trace row per step to ``self.trace_path`` in the same
       JSONL shape that ``swerouter/leaderboard/score.py`` expects. After
       the run we append the ``__marker__: loop_summary`` row so existing
       SWERouterBench scorer / analyzer tooling treats the file identically
       to a SWERouterBench-emitted trace.
    """

    def __init__(
        self,
        model: RouterAwareModel,
        env,
        *,
        router: Router,
        instance_id: str,
        trace_path: Path,
        ttl: TTLPolicy,
        budget_usd: float,
        select_timeout_sec: float = 30.0,
        config_class: type | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            env=env,
            config_class=config_class or AgentConfig,
            **kwargs,
        )
        if not isinstance(model, RouterAwareModel):
            raise TypeError(
                "MiniRouterAgent requires a RouterAwareModel; got "
                f"{type(model).__name__}"
            )
        self._router = router
        self._instance_id = instance_id
        self._trace_path = Path(trace_path)
        self._ttl = ttl
        self._budget_usd = float(budget_usd)
        self._select_timeout_sec = float(select_timeout_sec)
        self._cache = PromptCacheModel(ttl)

        # Start the trace file empty so `score_run_dir` reads our rows only.
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        if self._trace_path.exists():
            self._trace_path.unlink()

    # ------------------------------------------------------------------
    # main override
    # ------------------------------------------------------------------

    def query(self) -> dict:
        """Single-step router dispatch -- replaces ``DefaultAgent.query``."""

        if (
            0 < self.config.step_limit <= self.n_calls
            or 0 < self.config.cost_limit <= self.cost
        ):
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        step_index = self.n_calls
        self.n_calls += 1

        ctx = self._build_router_context(step_index)
        raw_decision = self._router.select(ctx)
        decision = validate_decision(
            raw_decision, available_models=ctx.available_models
        )
        model_id = decision.model_id

        message = self.model.query_as(
            model_id, self.messages, step_index=step_index
        )
        self.cost += float(message["extra"].get("cost", 0.0))
        self.add_messages(message)

        self._update_cache_after_call(model_id, step_index)
        self._write_trace_row(ctx, decision, message)
        return message

    # ------------------------------------------------------------------
    # router context / cache bookkeeping
    # ------------------------------------------------------------------

    def _build_router_context(self, step_index: int) -> RouterContext:
        now_ts = time.time()
        available = self.model.available_models
        cache_snapshot: CacheStateSnapshot = self._cache.snapshot(
            now_ts=now_ts, available_models=available
        )
        run_cfg = RunConfig(
            max_steps=int(self.config.step_limit or 0),
            budget_usd=self._budget_usd,
            wallclock_ttl_sec=self._ttl.wallclock_ttl_sec,
            select_timeout_sec=self._select_timeout_sec,
        )
        return RouterContext(
            instance_id=self._instance_id,
            step_index=step_index,
            messages=tuple(self.messages),
            tools=(),  # mini uses a single BASH_TOOL internally; routers don't branch on tools.
            available_models=available,
            cache_state=cache_snapshot,
            budget_so_far_usd=float(self.cost),
            run_config=run_cfg,
        )

    def _update_cache_after_call(self, model_id: str, step_index: int) -> None:
        step_log = self.model.step_log
        if not step_log:
            return
        last: StepRecord = step_log[-1]
        if last.step_index != step_index or last.model_id != model_id:
            raise RuntimeError(
                "RouterAwareModel step log out of sync with MiniRouterAgent "
                f"(expected step={step_index} model={model_id}, got "
                f"step={last.step_index} model={last.model_id})"
            )
        self._cache.update(
            model_id=model_id,
            messages=tuple(self.messages),
            prefix_token_count=last.usage.total_prompt_tokens,
            ts=last.finished_at,
        )

    # ------------------------------------------------------------------
    # SWERouterBench trace emission
    # ------------------------------------------------------------------

    def _write_trace_row(
        self,
        ctx: RouterContext,
        decision: Any,
        message: Mapping[str, Any],
    ) -> None:
        extra = message.get("extra") or {}
        sw = dict(extra.get("swerouter") or {})
        step_index = int(sw.get("step_index", ctx.step_index))

        # Prompt-tail preview: last message before the assistant turn.
        prompt_tail_role = ""
        prompt_tail_preview = ""
        if ctx.messages:
            tail = ctx.messages[-1]
            prompt_tail_role = str(tail.get("role", ""))
            content = tail.get("content")
            if isinstance(content, str):
                prompt_tail_preview = content[:500]
            elif content is not None:
                prompt_tail_preview = json.dumps(content, ensure_ascii=False)[:500]

        # mini's LitellmModel stores parsed actions under extra.actions.
        actions = extra.get("actions") or []
        tool_calls_preview = []
        for act in actions:
            cmd = act.get("command") if isinstance(act, Mapping) else None
            tool_calls_preview.append(
                {
                    "tool_name": "bash",
                    "args_preview": (cmd or "")[:240],
                    "args_truncated": bool(cmd and len(cmd) > 240),
                }
            )

        assistant_content_preview = ""
        content = message.get("content")
        if isinstance(content, str):
            assistant_content_preview = content[:300]

        row = {
            "instance_id": self._instance_id,
            "step_index": step_index,
            "started_at": sw.get("started_at"),
            "finished_at": sw.get("finished_at"),
            "model_id": sw.get("model_id"),
            "provider": sw.get("provider"),
            "rationale": getattr(decision, "rationale", None),
            "latency_ms": sw.get("latency_ms"),
            "prompt_messages_count": len(ctx.messages),
            "prompt_tail_role": prompt_tail_role,
            "prompt_tail_preview": prompt_tail_preview,
            "usage": sw.get("usage") or {},
            "raw_usage": sw.get("raw_usage") or {},
            "step_cost_usd": float(sw.get("step_cost_usd", 0.0)),
            "cumulative_cost_usd": float(self.cost),
            "assistant_content_len": len(content) if isinstance(content, str) else 0,
            "tool_call_count": len(tool_calls_preview),
            "tool_calls_preview": tool_calls_preview,
            "assistant_content_preview": assistant_content_preview,
            "cache_lookup": _snapshot_cache_hint(
                ctx.cache_state, sw.get("model_id", "")
            ),
        }
        with self._trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # lifecycle hooks
    # ------------------------------------------------------------------

    def run(self, task: str = "", **kwargs) -> dict:
        try:
            result = super().run(task, **kwargs)
        finally:
            self._write_loop_summary()
        return result

    def _write_loop_summary(self) -> None:
        """Append SWERouterBench's ``__marker__: loop_summary`` row."""

        step_log = self.model.step_log
        total_cost = sum(s.step_cost_usd for s in step_log)
        model_dist: dict[str, int] = {}
        for s in step_log:
            model_dist[s.model_id] = model_dist.get(s.model_id, 0) + 1
        last_message = self.messages[-1] if self.messages else {}
        exit_status = last_message.get("extra", {}).get("exit_status") if isinstance(last_message, Mapping) else None
        summary = {
            "__marker__": "loop_summary",
            "instance_id": self._instance_id,
            "step_count": len(step_log),
            "finished_by": exit_status or "mini_default",
            "total_router_cost_usd": total_cost,
            "error": None,
            "model_distribution": model_dist,
        }
        with self._trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, ensure_ascii=False) + "\n")


def _snapshot_cache_hint(cache_state: CacheStateSnapshot, model_id: str) -> dict:
    """Expose the harness-side cache view for the chosen model."""

    view = cache_state.get(model_id)
    if view is None:
        return {
            "hit": False,
            "reason": "cold_start_first_call",
            "cached_prefix_token_count": 0,
        }
    has_prior = view.last_call_ts is not None
    return {
        "hit": has_prior,
        "reason": "prefix_match" if has_prior else "cold_start_first_call",
        "cached_prefix_token_count": int(view.prefix_token_count),
    }


__all__ = ["MiniRouterAgent"]
