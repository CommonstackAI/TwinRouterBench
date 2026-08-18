# Static data-generation pipeline

TwinRouterBench exposes the construction stages behind the static routing
supervision as a versioned Python package and CLI. The implementation follows
the v2 paper protocol for both multi-step and single-turn collection paths.

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
   locked assignments, the current step uses a candidate tier/model, and all
   future steps remain at `high`.
4. Execute the complete mixed-model trajectory. A candidate is accepted only
   when the task passes **and the trajectory has the same number of steps** as
   the seed.
5. Probe up to three models inside each candidate tier. The tier passes when at
   least one pool model passes.
6. Rebuild later router-visible prefixes from the successful mixed trajectory.
   Cached trials are keyed by instance, step, assignments, and generation
   parameters; downstream entries are invalidated when an upstream lock changes.
7. Apply the hardened Faithfulness/Appropriateness/Completeness judge to mtRAG
   and QMSum. Evidence conflicts, incomplete answers, and uncertainty fail.
8. Export approximately 10% of BFCL, SWE-bench, and PinchBench routed steps for
   manual review.
9. Apply `tight` or `further_downgradeable` verdicts. `uncertain` blocks
   publication.
10. Validate and merge finalized runs into a separate release-candidate
    directory.

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
validation, replay equivalence, CLI behavior, and golden output hashes.
