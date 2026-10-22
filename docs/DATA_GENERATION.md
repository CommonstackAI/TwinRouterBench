# Static data-generation pipeline

TwinRouterBench exposes the construction stages behind the static routing
supervision as a versioned Python package and CLI. This is a reference
construction engine, not the historical infrastructure that generated the
released 970 rows. Optional native execution bridges and original-data loaders
are available below. They do not establish exact reproduction of those rows;
the historical selection, sampling and infrastructure still need an audit.

The default path is completely offline. It uses five deterministic synthetic
fixtures to exercise the same state transitions as a live run; no API key,
network connection, Docker daemon, or original benchmark download is required.

## One interface for any benchmark

The public entry point is a complete, credential-free JSON pipeline config:

```bash
twinrouterbench data run \
  --config configs/data_generation/custom_qa_pipeline.json \
  --output-dir runs/custom-qa
```

```python
from twinrouterbench.data_generation import run_pipeline

manifest = run_pipeline("pipeline.json", output_dir="runs/my-benchmark")
```

The shared engine never switches on benchmark names. A suite config declares:

```json
{
  "schema": "twinrouterbench.data_pipeline.v1",
  "backend": "mock",
  "benchmarks": {
    "my_benchmark": {
      "display_name": "My Benchmark",
      "scenario": "single_turn_qa",
      "source": {
        "uri": "local://tasks.jsonl",
        "license": "CC-BY-4.0",
        "version": "v1"
      },
      "loader": {
        "type": "single_turn",
        "options": {
          "field_map": {
            "instance_id": "id",
            "prompt": "question",
            "reference": "answer",
            "target_tier": "target_tier"
          }
        }
      },
      "evaluation": {
        "trial": "exact_match",
        "final": "exact_match"
      }
    }
  },
  "run": {"benchmarks": "all", "output_dir": "runs/my-benchmark"}
}
```

Built-in loaders are `normalized` and `single_turn`. Built-in evaluators are
`execution`, `backend_judge`, `exact_match`, and `contains`. The generic
OpenAI-compatible executor is exposed as
`twinrouterbench.data_generation.openai_backend:create_backend`; its API key is
read only from an environment variable.

Complex agent benchmarks cannot be made declarative at the sandbox boundary.
They implement the narrow `TaskLoader`, `ExecutionEvaluator`, or
`ExecutionBackend` protocols through `module:factory`; search, caching, review,
provenance, and publication remain unchanged. Thus adding a normal QA benchmark
requires only data plus config, while adding SWE-bench-like environments
requires one thin harness plugin rather than a new pipeline.

Validate configs before spending API budget:

```bash
twinrouterbench data config validate --config pipeline.json
twinrouterbench data suite validate --config benchmark-suite.json
```

## Construction stages

For each normalized task, the pipeline performs:

1. Run or replay a successful strong-model seed trajectory. Failed seeds are
   rejected.
2. Produce one conservative downgrade-search hint per routed step. A hint is a
   pruning device and never becomes a label without execution.
3. Apply Algorithm A1 sequential locking. Previously accepted steps keep their
   exact responses and corresponding environment state, the current step uses
   a candidate tier/model, and all future steps remain at `high`.
4. Execute the remaining mixed-model trajectory. A candidate is accepted only
   when the configured evaluator passes, the trajectory has the same number
   of steps as the seed, all response/prefix records are present, and all
   prefixes through the target step and responses before it match the lock.
5. Probe up to three models inside each candidate tier. The tier passes when at
   least one pool model passes.
6. Rebuild later router-visible prefixes from the successful mixed trajectory.
   Cached trials are keyed by benchmark, instance, step, assignments, generation
   parameters, and the actual locked prefix. Downstream entries in the same
   benchmark are invalidated when an upstream lock changes.
7. Apply the hardened Faithfulness/Appropriateness/Completeness judge to mtRAG
   and QMSum to seeds, every candidate, and the final trajectory. Evidence
   conflicts, incomplete answers, and uncertainty fail. Applying the judge only
   at the end would select cheap incorrect candidates without trying higher tiers.
8. Export approximately 10% of BFCL, SWE-bench, and PinchBench routed steps for
   manual review.
9. Apply `tight` or `further_downgradeable` verdicts. `uncertain` blocks
   publication.
10. Validate and merge finalized runs into a separate release-candidate
    directory.

Invalid or missing trajectory records and a changed locked prefix reject the
whole case. They are construction failures, not evidence for a higher tier.

## Native execution bridges

Use the plugin backend
`twinrouterbench.data_generation.official_backend:create_backend` to share one
OpenAI-compatible client and one budget ledger across the five workloads.
Materialize original benchmark inputs explicitly; downloads and credentials
are never part of a pipeline configuration.

For the SWE bridge, install `pip install -e '.[generation-swe]'` in a dedicated
environment with Docker available. This extra pins the tested mini-swe-agent
and SWE-bench versions; incompatible versions fail before environment allocation.
Other native runtimes are installed separately at the versions listed below.
The shared spend ledger requires POSIX file locking (Linux/macOS); importing
other backends does not require that platform capability.

| Workload | Loader plugin | Execution and scoring |
| --- | --- | --- |
| SWE-bench Verified | `twinrouterbench.data_generation.swebench_backend:create_loader` | mini-swe-agent 2.2.8 tool formatting and submission; SWE-bench 3.0.15 official Docker evaluator |
| BFCL multi-turn | `twinrouterbench.data_generation.bfcl_backend:create_loader` | bfcl-eval 2026.3.23 stateful tools and official state/response/irrelevance checks |
| PinchBench | `twinrouterbench.data_generation.pinchbench_backend:create_loader` | materialized official skill runner and grader; isolated OpenClaw state with a local capped model proxy |
| mtRAG | `twinrouterbench.data_generation.text_loaders:create_mtrag_loader` | original conversation/passages plus the hardened answer judge |
| QMSum | `twinrouterbench.data_generation.text_loaders:create_qmsum_loader` | original general/specific query prompts plus the hardened answer judge |

Loaders are configured as `{"type":"plugin","factory":"module:factory",
"options":{}}`. SWE inputs are raw instance records. BFCL records contain
`entry` (including the official materialized function documentation) and
`ground_truth`; use `bfcl_eval.utils.load_dataset_entry` and
`load_ground_truth_entry` to materialize them. PinchBench's source is the skill
directory, with optional `task_ids`. The text loaders consume original JSONL.

The mtRAG loader records input/accepted/rejected counts, filter policy and the
source SHA256. Its default requires human relevance feedback for every passage,
as well as `ANSWERABLE` and nonempty contexts. Historical scripts accepted
missing feedback vacuously; `require_human_feedback:false` explicitly selects
that older behavior. Neither policy alone establishes the historical 970-row
selection. QMSum specific-query spans are inclusive and bounds checked.

SWE locks use owned Docker image checkpoints and preserve the recorded visible
prefix; BFCL resets and reexecutes prior calls and compares their observations.
PinchBench reexecutes native tools, checks observations and tool schemas through
the target, and explicitly freezes the seed's initial scaffold messages when
present. It rejects initial assistant/tool history; auxiliary calls and compacted
inputs retain their actual messages and must match the locked trace through the
target. It never replaces new tool feedback. Native prompt/tool/grade behavior and runtime versions belong
in the run evidence. These bridges are opt-in; offline fixtures remain default.

Set `COMMONSTACK_API_KEY` and `COMMONSTACK_API_BASE` in a private environment
file outside Git. A live capped run also needs
`TWINROUTERBENCH_API_BUDGET_USD` and a saved provider catalog via backend option
`pricing_catalog_path`. Optional `model_aliases` map an identifier only to the
same underlying model, and are recorded in the config lock. Missing models
must not be silently replaced.

The default ledger is `~/.local/state/twinrouterbench/api-spend.jsonl`; override
it with `TWINROUTERBENCH_API_LEDGER` or backend option `ledger_path`. Share the
same ledger for the whole authorized budget. Each textual request reserves a
conservative maximum before network access. Failures retain the reservation,
no requests retry automatically, and successful requests settle from token
usage and saved prices. `reported_usd` is the sum of provider-reported costs
when present; zero does not imply free calls. `accounted_usd` includes retained
reservations and conservative token estimates and is not an invoice.

For providers that support JSON mode, backend option
`json_response_format:{"type":"json_object"}` requests structured proposer and
judge responses and records that choice alongside prompt hashes. The parser
still validates every boolean and fails on malformed output; it never guesses a
verdict from explanatory prose. A provider preface followed by one complete JSON
object is accepted; multiple objects or trailing prose are rejected. Native `max_completion_tokens` is forwarded
without injecting a second `max_tokens` limit, and both limits together are
rejected before spending.

`judge_max_tokens` explicitly controls the judge's output budget (default 512),
is recorded with its prompt artifact, and can be raised for providers that emit
long explanations. Truncated JSON is rejected; increasing this limit never
changes the boolean checks or acceptance criteria.

Backend options include `swebench_work_dir`, `pinchbench_work_dir`,
`eval_timeout_seconds`, and explicit `node_path`/`openclaw_path` for PinchBench.
PinchBench's configured pass thresholds default to the historical predicates:
automated 0.85, LLM judge 0.65, hybrid 0.75. Its runner writes its request,
execution, grade and logs to an owned trial directory, with a bounded timeout.
No shared gateway, global Docker cleanup or other project's process is used.
The disposable local OpenClaw config disables automatic bootstrap files, matching
the upstream fixture preparation, and configures local benchmark shell execution
without requests to another gateway. The execution policy is recorded in the
native report; global OpenClaw settings are not modified.

Offline regression tests do not prove live benchmark availability. Preserve a
separate validation report with actual task IDs, runtime versions, native
positive/negative controls, model calls, failures and costs. A small smoke test
using available original models is not a full original-pool cascade or a label
stability experiment.

## Quick start

Generate all five fixture datasets:

```bash
twinrouterbench data generate \
  --benchmark all \
  --backend mock \
  --config configs/data_generation/mock_all.json \
  --output-dir runs/data-generation/mock-all
```

Generate one dataset:

```bash
twinrouterbench data generate \
  --benchmark swebench \
  --backend mock \
  --output-dir runs/data-generation/mock-swebench
```

Limit a smoke run to the first normalized case from each selected benchmark:

```bash
twinrouterbench data generate \
  --benchmark all \
  --backend mock \
  --max-cases 1 \
  --output-dir runs/data-generation/mock-one-each
```

The same APIs are importable:

```python
from twinrouterbench.data_generation import GenerationConfig, GenerationPipeline
from twinrouterbench.data_generation.backends import MockBackend
from twinrouterbench.data_generation.pipeline import load_model_pool

_, pool = load_model_pool()
pipeline = GenerationPipeline(
    output_dir="runs/data-generation/example",
    backend=MockBackend(pool),
    config=GenerationConfig(run_id="example"),
)
pipeline.generate("bfcl")
```

## Manual review

Export a writable review template:

```bash
twinrouterbench data review export \
  --run-dir runs/data-generation/mock-all \
  --output reviews/mock-all.jsonl
```

Each queued row accepts one verdict:

- `tight`: keep the generated tier;
- `further_downgradeable`: lower the tier by exactly one level;
- `uncertain`: leave the run blocked for another review pass.

Apply completed reviews:

```bash
twinrouterbench data review apply \
  --run-dir runs/data-generation/mock-all \
  --reviews reviews/mock-all.jsonl
```

`labels.final.jsonl` is emitted only after every queued row is resolved.

## Publish a release candidate

Generation never writes directly to `data/static/`. Publish one or more ready
runs into an isolated candidate directory:

```bash
twinrouterbench data publish \
  --runs runs/data-generation/mock-all \
  --output-dir runs/release-candidate/static-v2

twinrouterbench data validate \
  --path runs/release-candidate/static-v2
```

Publication checks required fields, tier/id consistency, duplicate IDs,
contiguous steps, source/license metadata, unresolved reviews, and internal
model/reviewer field leakage.

## Backends

### Mock

`--backend mock` uses deterministic fixtures for SWE-bench, BFCL, mtRAG,
QMSum, and PinchBench. It is the CI path.

### Replay

Every run writes `backend_events.jsonl`. Re-run the complete algorithm without
calling a model or harness:

```bash
twinrouterbench data generate \
  --benchmark all \
  --backend replay \
  --replay-log runs/data-generation/mock-all/backend_events.jsonl \
  --output-dir runs/data-generation/replayed
```

For the supplied fixtures, mock and replay produce byte-identical labels and
trial logs.

New logs identify the generation parameters, seed trajectory, locked prefix,
and judged execution. Repeated identical requests are replayed in recorded
order, including differing judge verdicts. Logs created before the
`verified-sequential-locking-v2` protocol lack these identifiers and cannot be
silently reused as new verified runs. Existing historical files are not changed.

### Live plugin

Live infrastructure is supplied as a `module:factory` plugin implementing
`ExecutionBackend`:

```bash
twinrouterbench data generate \
  --benchmark swebench \
  --backend live \
  --live-backend my_harness.backend:create_backend \
  --source swebench=local:///data/normalized/swebench/tasks.jsonl \
  --output-dir runs/data-generation/swe-live
```

The plugin owns provider and benchmark execution. Credentials must be read from
the environment or a secret manager; they are never part of the generation
config or run manifest.

Multi-step plugins must implement the updated method:

```python
def run_mixed(self, task, assignments, *, target_step,
              generation_parameters, locked_execution=None):
    ...
```

`target_step` is one-based. `locked_execution` is a copy of the last accepted
trajectory. Restore its environment checkpoint, or reset the environment and
replay the previously accepted actions without regenerating their LLM responses.
Then execute the candidate step and the remaining high-tier suffix with real
tool/environment feedback. Return the complete prefixes/responses, including
the frozen part, and set `passed` from the official task predicate. A matching
message prefix alone cannot prove that an external environment was restored;
the harness plugin is responsible for that state contract. Legacy multi-step
plugins without the new argument fail before seed execution.

`ExecutionResult.verified=True` attests that `passed` comes from the execution
backend's official predicate (or the explicitly synthetic fixture oracle).
Transport-only backends must set `verified=False`; the `execution` evaluator
rejects those results. A real independent text/plugin/judge evaluator can still
evaluate the returned answer. Records without a `verified` field are read as
unverified.

### Generic OpenAI-compatible text backend

This backend supports text-only single-turn tasks. Both trial and final
evaluation must use `exact_match`, `contains`, `backend_judge`, or a genuine
independent evaluator plugin. Multi-step environments and tool execution are
rejected, rather than replaying prerecorded observations and treating an API
response as a successful task. The built-in mtRAG and QMSum trial evaluators
now use `backend_judge` as well as their final evaluator; this change is
identified by built-in suite version `twinrouterbench-builtins-v3-2026-10-22`.

The backend actually loads the packaged `tier_proposer` and `open_ended_judge`
prompt versions, sends them to the proposer/judge, and records their SHA-256
hashes and model IDs in `config.lock.json`. Both models default to the first
`high` pool model; backend options `proposer_model` and `judge_model` can select
them explicitly. Proposed hints come from that model, not the input fixture's
configured hint. Malformed proposals or judge output stop the run. Judge
fields must be JSON booleans; strings such as `"false"` are rejected.

`generation.generation_parameters` applies to seed and candidate answer calls,
including a provider seed if supplied there. `generation.seed` controls audit
sampling, not provider randomness. The proposer/judge use temperature zero.
The public sources and original benchmark preprocessing must still be supplied
and validated by the benchmark owner; successfully running a schema fixture
does not establish reproduction of the original release.

### CommonStack live transport smoke

The repository includes a provider smoke plugin that makes a real CommonStack
chat-completion request for every seed, hint, mixed-execution, and judge backend
operation while retaining the deterministic fixture oracle:

```bash
export COMMONSTACK_API_KEY=...  # never commit this value

twinrouterbench data generate \
  --benchmark all \
  --backend live \
  --live-backend twinrouterbench.data_generation.commonstack_smoke:create_backend \
  --config configs/data_generation/commonstack_smoke.json \
  --output-dir runs/data-generation/commonstack-smoke
```

This verifies credentials, provider transport, plugin loading, all five
adapters, downgrade search, judging, and artifact generation. Pass/fail remains
fixture-oracle based, so these smoke outputs are not official benchmark scores
and must not be published as supervision data. Each provider probe includes a
bounded preview of the normalized task messages so message serialization and
real task content are exercised without recording provider responses.

## Source registry

Accepted source URI forms are:

```text
local:///absolute/path/tasks.jsonl
hf://organization/dataset@revision/path
github://owner/repository@commit/path
fixture://swebench
```

Remote URIs record provenance only. The pipeline does not download them
implicitly. Materialize remote data explicitly and pass a `local://` path for a
real run. GitHub/Hugging Face revisions should be pinned to immutable commits
for release construction.

Normalized inputs are JSON or JSONL documents containing `TaskSpec` fields.
The five adapter fixtures under `twinrouterbench/data_generation/fixtures/`
serve as minimal schema examples.

## Run artifacts

Each run is isolated and contains:

```text
config.lock.json
backend_events.jsonl
seed_trajectories.jsonl
downgrade_hints.jsonl
execution_trials.jsonl
rejections.jsonl
labels.pre_review.jsonl
review_queue.jsonl
reviews.applied.jsonl       # after review
labels.final.jsonl          # only when ready
manifest.json
```

The public `question_bank.jsonl` is produced only by `data publish`. Internal
fields such as selected model IDs, reviewer identity, and search traces are not
copied into public rows.

## Tests

The full suite is offline:

```bash
python -m pytest -q
```

It covers all five built-ins, arbitrary config-only benchmarks, generic
OpenAI-compatible execution, loader/evaluator plugins, single-dataset generation, cascade fallback,
sequential locking, mixed-prefix reconstruction, cache invalidation, fixed-step
acceptance, hardened judging, review correction, publication gates, source URI
validation, replay equivalence, CLI behavior, and golden output hashes. Additional
regressions cover unverified execution rejection, unsupported tool/environment
execution, strict judge booleans, prefix drift and mutation, incomplete traces,
cache isolation, and generation-parameter mismatches on replay. A local HTTP
server exercises seed → model hint → rejected low candidate → accepted mid
candidate → final judge → replay, using the packaged prompt bytes with no
commercial API calls.
