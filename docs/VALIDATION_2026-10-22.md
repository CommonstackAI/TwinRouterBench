# Pipeline repair validation — 2026-10-22

This report covers the native data-generation repairs appended to PR #9.
Historical supervision, original remote workspaces, paper files and other
projects were not modified.

This is software acceptance on small native cases. It does not establish full
model-pool coverage, reproduce the historical 970 rows, or measure label stability.

For the current repair PR, acceptance covers source loading, actual execution,
independent grading, prompt use and frozen-prefix behavior on selected cases.
Random original-source smoke tests supplement the initial cases below. Exact
historical release reconstruction and label-stability experiments are separate
research work and are not prerequisites for this repair PR.

| Workload | Native check | Result and limit |
| --- | --- | --- |
| SWE-bench Verified, `psf__requests-1142` | Official gold vs nonempty unrelated patch; three-step native scaffold with an owned Docker checkpoint restored before the target | Gold resolves; unrelated patch applies but fails. Scripted gold seed and restored candidate pass with identical locked inputs. This is an unpaid gold control, not a model solving the task. Two real Opus seed attempts returned upstream HTTP 429. |
| BFCL, `multi_turn_base_0` | Official tool state/response/irrelevance checks; real Opus seed and MiniMax last-step candidate | Official gold passes and no-tools fails. Opus seed passes in 16 steps. Candidate preserves the prefix but does not terminate within that length and is rejected. Other BFCL categories are not certified by this check. |
| PinchBench, `task_01_calendar` | Official grader and original OpenClaw runner | Gold ICS scores 1.0, empty workspace 0.0. An initial real MiniMax run completes seven calls and scores 1.0. The final run after token-limit cleanup completes eight calls and scores 0.833333, below the unchanged 0.85 threshold, and is rejected. Both outcomes are preserved; a plumbing check does not require a weak model to solve every task. This is native execution validation, not an original high-model labeling seed or full cascade. |
| mtRAG, `c1e53df98a28f19448f4af244304b89a<::>1` | Original input loader and hardened judge on a deterministically selected shortest-input case | Real seed and MiniMax answer accepted; reference accepted and irrelevant answer rejected. Strict loader accepts 705 source records; this count is not the historical release selection. Full cascade stops at unavailable low-tier provider. |
| QMSum, `meeting_30_specific_5` | Original specific-span loader, real generation, hardened judge, structured-result parsing | Seed/candidate answers were rejected for incompleteness. Reference verdict is true and irrelevant-answer verdict false. A saved response with explanatory prose followed by one complete JSON verdict is reused to validate the parser repair. No outcome is forced to pass. |

The native inputs were selected for plumbing validation, not accuracy estimation.
Text cases use shortest input length with ID as the tie breaker. Earlier failures,
partial results and original errors remain preserved.

Repairs found during live checks include distinct SWE work/grader container names,
Pinch host workspace mapping, retention of actual auxiliary model-call inputs,
mutually exclusive output-token limits, and parsing one explicit final JSON object
without inferring a verdict from preceding prose. Malformed/multiple JSON objects,
non-boolean verdict fields and changed locked prefixes remain rejected.

The local regression command is:

```bash
python -m pytest -q
```

Initial acceptance: **79 passed locally and remotely**. It covers offline construction/replay, actual local HTTP,
state replay, prefix rejection, native bridge regressions, original-data filtering
and spend reservation. The remote final test log is stored separately below.

Native versions exercised: mini-swe-agent 2.2.8, SWE-bench 3.0.15,
bfcl-eval 2026.3.23, Docker 29.1.3, OpenClaw 2026.3.30 (`94279d0`),
Node 22.16.0 and PinchBench skill commit
`1e2ba6b5a3f527dee2f4fab0220c4b9ed429dd00`.

Remote evidence root: `/data/cwy/twinrouter-pipeline-20261022/.validation`:

- `swe-official-controls.json`, `swe-native-controls-fixed/RESULT.json`
- `bfcl-official-controls.json`, `bfcl-job/{seed,candidate,RESULT}.json`
- `pinch-controls/RESULT.json`, `pinch-job-final/{seed,RESULT}.json`, `pinch-verified/{seed,RESULT}.json`
- `mtrag-live-pipeline/backend_events.jsonl`, `text-controls-job/` original calls
- `qmsum-json-control/{RESULT.json,call-001-response.json,call-002-response.json}`
- `jobs/` independent bounded scripts, PIDs and raw logs
- `pytest-final.log`, `pytest-final.xml`, `pytest-final-result.json`
- `summary-current.json`

All calls share one account-wide ledger and a $20 cap. Final accounting,
including the random-source follow-up, is reported below. The provider did not
report billed dollar costs; `reported_usd=0` does not mean the calls were free.
No further paid retry is queued.

Full original-pool construction remains unavailable with this endpoint:
DeepSeek v3.2 and GLM 4.5 Air returned HTTP 503/no provider, Qwen 3.5 9B/27B were
absent from its catalog, and two SWE Opus requests returned upstream quota HTTP
429. Models were not silently substituted to claim a completed original cascade.
Full-pool runs and repeated-sampling stability checks remain separate work.

## Follow-up: random original-source acceptance for this PR

Original-source sampling validates this repair PR's integration scope. Each
benchmark's ID-sorted source frame was sampled once
with `random.Random('20261022:' + benchmark).sample(frame, 1)`. Draws were
recorded before execution, and failed cases were not replaced.

| Source frame | Random case | Execution outcome |
| --- | --- | --- |
| SWE-bench Verified, 500 instances | `django__django-15098` | Official gold patch resolves; native three-step control and frozen-checkpoint candidate both pass. Unpaid scripted control, not a model accuracy result. |
| BFCL multi-turn base, 200 instances | `multi_turn_base_163` | Seven real model calls complete. Official checker rejects the TravelAPI state: model budget 8000 vs expected 2000. |
| mtRAG, 705 eligible instances | `941445ba11ba7ba2c92c5184c9d798d6<::>2` | Seed passes, strong-model hints generated; cheaper candidate rejected for unsupported additions. |
| QMSum, 281 queries | `meeting_3_specific_5` | Seed, strong-model hints, candidate execution and independent judging complete; seed and candidate pass. |
| PinchBench, 23 tasks | `task_12_skill_search` | Six native model calls; official grader scores all six fixture checks correctly, total 1.0. |

The sampling manifest includes source hashes, Python/package versions and
population sizes. Evidence root:
`/data/cwy/twinrouter-pipeline-20261022/.validation/random-source-20261022`.
`SUMMARY.json` contains all outcomes; `RANDOM_CHECKS_DONE.json` marks completion.
Original failed attempts and responses are retained.

The random checks identified two additional integration issues. Pinch's local
runner required unavailable gateway approval and recreated bootstrap files
removed by the official fixture preparation. Its disposable config now supports
benchmark shell execution locally and skips bootstrap creation; the policy is
recorded. The mtRAG judge exhausted its original 512-token output budget and
subsequently wrapped its complete verdict in Markdown. The parser accepts one
complete final fenced JSON object, still rejecting ambiguity and truncation.
`judge_max_tokens` is configurable and recorded; this retry explicitly used
2048, while the default remains 512. The saved seed and complete judge response
were reused rather than regenerated.

SWE's selected cold image initially exceeded a 300-second download timeout.
One bounded retry downloaded the same selected instance; no cached replacement
was selected. Image digest:
`sha256:5e4e354d124aef08bbad1d086bc043571187382d0382a69cb0c1df1e0baf8be8`.

Updated regressions: **79 passed locally and remotely**. Updated conservative
account-wide cost: **$3.025566**, including five retained uncertain reservations
(116 requests, 111 settled). The random-source follow-up added approximately
$0.49125 to conservative accounting. No labels were published and no stability claim was added.

## PR integration checks

The SWE bridge now rejects incompatible harness versions before allocating
resources. Install its pinned dependencies with `pip install -e '.[generation-swe]'`. Optional POSIX budgeting fails
explicitly on unsupported platforms without breaking imports of other backends.
Both failure paths have offline regressions.

On the existing PR #9 head (`958db4a`), the complete combined suite reports
**94 passed** in 9.96 seconds, including OpenCode routing regressions and the
two additional compatibility checks above. `git diff --check` passes. This
integration check used no paid API calls.
