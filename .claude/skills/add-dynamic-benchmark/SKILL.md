---
name: add-dynamic-benchmark
description: Add a new benchmark to the TwinRouterBench dynamic track by inserting the per-call router interface into that benchmark's own agent harness, so every LLM call is routed to a model from the locked pool, priced from provider usage, and graded by the existing leaderboard scorer. Use when asked to make any benchmark (Terminal-Bench, a QA suite, a custom agent task, ...) "router-enabled", "dynamic", or scorable with `twinrouterbench dynamic score`.
---

# Add a benchmark to the dynamic routing track

The goal is always the same: **the benchmark keeps running exactly as its
official harness runs it; the only change is that each LLM call goes through
the router.** Success comes from the benchmark's own official verdict; cost
comes from the provider's `usage` priced with the locked pricing table; the
existing scorer turns both into the leaderboard bill.

The SWE-bench Verified track is the reference implementation. Read it before
writing code:

| Concern | SWE-bench reference | Generic building block |
|---|---|---|
| Router protocol and validation | `swerouter/router.py` | same (reuse, never copy) |
| Hook into the agent's LLM call | `miniswerouter/agent/agent.py` (`MiniRouterAgent.query`) | `RoutedSession.call` |
| Dispatch to the chosen pool model | `miniswerouter/agent/model.py` (`RouterAwareModel.query_as`) | your `call_model` callable |
| Usage → 4 buckets → USD | `swerouter/usage.py`, `swerouter/pricing.py` | done inside `RoutedSession` |
| Prompt-cache hints for routers | `swerouter/cache.py` | done inside `RoutedSession` |
| Official verdict + per-instance result | `miniswerouter/harness/run_instance.py` | `RoutedSession.finish` |
| Multi-instance runner, resume | `miniswerouter/harness/run_eval.py` | copy the pattern |
| CLI (`--router-import module:Factory`) | `miniswerouter/cli.py` | copy the pattern |
| Scoring | `swerouter/leaderboard/score.py` | reuse unchanged |

`twinrouterbench/dynamic_adapter/session.py` (`RoutedSession`) is the
benchmark-independent half of the SWE wiring. Use it; do not re-implement
context building, pricing, cache tracking, or trace writing.
`tests/test_dynamic_adapter.py` shows the full call → finish → score cycle.

## Hard rules

1. **Parity first.** Do not change the benchmark's prompts, tools, step or
   cost limits, environment, or evaluator. Pin the official harness version
   and record it. (The SWE track had to be re-synced byte-for-byte to
   mini-swe-agent 2.2.8 prompts in PR #3; avoid that by never drifting.)
2. **Official verdict only.** `resolved` comes from the benchmark's own pass
   predicate. No fallbacks that rescue a run the official harness would fail
   (PR #3 removed a `git diff` patch fallback for exactly this reason).
3. **Every billed LLM call is routed and recorded.** Planner, sub-agent,
   summarizer, title, or retry calls all cost money. If the harness makes calls
   you think should not be routed, stop and ask the user; do not silently
   leave them unrouted or unbilled.
4. **Fail fast.** No default model, no assumed zero cost, no swallowed
   exceptions. Missing `usage` is an error. Errors reach `finish(...)` with the
   original message so `swerouter/infra_errors.py` can classify transport
   failures.
5. **Locked data.** Use `data/dynamic/model_pool.json`, `model_pricing.json`,
   and `ttl_policy.json`. A different pool or price snapshot is a new
   benchmark version, not a local tweak.
6. **No benchmark-name branches in shared code.** Benchmark specifics live in
   the new adapter package only.

## Step 0: decisions to get from the user before coding

Stop and ask if any of these are unknown; do not pick values yourself:

- Benchmark name, official harness repo and pinned version/commit.
- The official success predicate and where it is produced.
- The instance split to report (and whether it must be disjoint from anything).
- The unresolved-instance penalty γ for this benchmark. The SWE value
  (`$0.60`) was set slightly above the unrouted Opus average cost **on
  SWE-bench**; it does not transfer. A common choice is to run the unrouted
  high model first and set γ from its average per-instance cost, but the user
  decides.
- Whether the benchmark's own evaluator uses an LLM judge (then: which model,
  pinned how, and judge cost is reported separately from router cost).

## Step 1: find the LLM chokepoint

Read the harness and locate where an assistant message is produced from a
message list. Classify:

- **A. In-process Python harness** (a model/client object the agent calls,
  e.g. mini-swe-agent `Model.query`, a LiteLLM/OpenAI client wrapper).
  Replace that call with `RoutedSession.call`. Preferred.
- **B. External agent process speaking an OpenAI-compatible HTTP API**
  (CLI agents, non-Python harnesses). Point its base URL at a local routing
  proxy whose request handler calls `RoutedSession.call`; the transport layer
  can follow `twinrouterbench/opencode_router/proxy.py` (PR #9). Map each
  request to an instance id explicitly (one proxy per instance, or a session
  header); never guess the instance from content. Request `usage` on streams
  (`stream_options.include_usage`).
- **C. LLM calls scattered across many code paths, concurrent calls inside one
  instance, or a hosted harness you cannot intercept.** Stop and report to the
  user with what you found; `RoutedSession` is not thread-safe and assumes a
  single ordered call sequence per instance.

Write down: the exact function/line, what the message list contains at that
point (system prompt, tool results, retrieved context), and where tool schemas
live.

## Step 2: wire the session (pattern A)

```python
from twinrouterbench.dynamic_adapter import ModelCallResult, RoutedSession, SessionLimits

def make_call_model(client):            # client = the harness's own LLM client
    def call_model(model_id, messages, tools):
        resp = client.chat(model=model_id, messages=messages, tools=list(tools) or None)
        return ModelCallResult(message=resp.message_dict, raw_usage=resp.usage_dict)
    return call_model

session = RoutedSession(
    router=router, instance_id=iid, run_dir=output_dir,
    call_model=make_call_model(client),
    limits=SessionLimits(max_steps=OFFICIAL_STEP_LIMIT, budget_usd=OFFICIAL_COST_LIMIT),
)
# inside the agent loop, replacing the original LLM call:
assistant_message = session.call(history, tool_schemas)
```

- `call_model` must send the request unchanged except for the model id.
  Keep the harness's own generation parameters, retry policy, and
  provider-specific options (e.g. Anthropic `cache_control` as
  `RouterAwareModel._build_submodel` does for `cache_control_style == "anthropic"`).
- Pass the full router-visible prefix (`history`) and the real tool schemas.
  The SWE track passes `tools=()` only because mini has one fixed bash tool;
  new benchmarks should pass their schemas.
- `raw_usage` is the provider's `usage` object from that same response. If the
  client hides it, extend the client; do not estimate tokens.
- Cross-provider message formats: all pool models are reached through the
  OpenAI-compatible endpoint. If the harness stores provider-native message
  shapes, convert at the chokepoint and add a test with a tool-call turn.

## Step 3: verdict and per-instance result

After the official evaluator runs:

```python
try:
    run_agent(...)                       # unchanged official agent loop
    agent_error = None
except Exception as exc:                 # record, do not hide
    agent_error = f"{type(exc).__name__}: {exc}"
verdict = official_evaluate(...)         # benchmark's own predicate
session.finish(resolved=verdict.passed, finished_by=exit_status,
               agent_error=agent_error, eval_error=eval_error,
               extra={"official_report_path": str(report_path)})
```

If the evaluator itself fails, set `resolved=False` and put the original error
in `eval_error`; never mark it resolved. Benchmark-specific fields go in
`extra`. `finish` writes `results/<iid>.json` and the trace summary in the
layout described in `reference/output-contract.md`.

## Step 4: package and CLI

Mirror the SWE layout in a new top-level package `<bench>router/`:

```
<bench>router/
  agent/          # chokepoint integration (pattern A) or proxy (pattern B)
  harness/
    run_instance.py   # env setup → agent → official eval → session.finish
    run_eval.py       # multi-instance runner with resume-from-disk
  cli.py          # run / score, same flags as miniswerouter/cli.py
```

- `run` takes `--router-import module:Factory`, repeated `--router-arg k=v`,
  `--router-label`, `--output-dir`, `--instances`/`--instance-ids-file`, and
  the official limits. Reuse `_instantiate_router` semantics from
  `miniswerouter/cli.py`.
- `score` forwards to `swerouter.leaderboard.score.score_run_dir` and always
  passes `--fixed-failure-penalty-usd` with the γ the user chose.
- Register the package in `pyproject.toml` (`packages.find.include`, optional
  dependency extra for the benchmark's harness) and add a subcommand in
  `twinrouterbench/cli.py` next to `dynamic`.
- Commit the reported instance list as `data/dynamic/<bench>_ids.txt`.

## Step 5: verification ladder (do each, report results honestly)

1. **Unit tests** with a fake client, modeled on `tests/test_dynamic_adapter.py`:
   routed calls see the full prefix and tools; an out-of-pool decision fails
   before any model call; missing usage fails; the run directory scores.
2. **Parity check of inputs**: diff prompts/templates/limits against the
   pinned official harness; they must be identical.
3. **One-instance live smoke** with
   `swerouter.routers.always_model:AlwaysModelRouter` on the high model, then
   `swerouter.routers.round_robin:RoundRobinRouter` to prove model switching mid-trajectory
   works (tool calls survive the switch).
4. **Unrouted parity run**: always-high through the adapter on a small subset,
   compared with the official harness running the same model directly.
   Resolve rate and step counts should match within run-to-run noise; if
   they do not, find the cause before running any router.
5. **Score** with `score --fixed-failure-penalty-usd <γ>` and check
   `total_router_cost_usd` against the provider's own billing for the smoke
   runs.

Live runs spend money. Confirm budget with the user before steps 3–4.

## Step 6: documentation

Add `docs/dynamic/<bench>.md` (setup, pinned versions, split, γ and how it was
chosen, commands, known deviations) and a row in the README dynamic-track
section. State explicitly anything that differs from the official harness.

## Done when

- [ ] All billed LLM calls go through `RoutedSession.call`.
- [ ] Official harness inputs unchanged (diff attached).
- [ ] `resolved` comes only from the official predicate.
- [ ] Unit tests pass; live smoke and unrouted parity results reported.
- [ ] `score` works on the run directory with the user-chosen γ.
- [ ] Docs and instance list committed.
