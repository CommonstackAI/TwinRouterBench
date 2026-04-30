"""System and user prompt templates for the SWERouterBench agent loop.

Wording is aligned with the **official SWE-bench reference inference prompt**
(``swebench.inference.make_datasets.create_instance.prompt_style_3``): the
task is framed as "produce a single git-applicable patch" and the issue is
wrapped inside an ``<issue>`` block so the model recognizes the task shape
used by upstream baselines.

Unlike upstream ``prompt_style_3`` (which bakes full file contents into the
prompt for a one-shot completion), our agent loop leaves the code base on the
SWE-bench Docker container and expects the model to use the provided tools
(bash / str_replace_editor / finish) to explore and edit. The per-instance
step budget is externally capped by the caller (see
``scripts/smoke_1_case.py``) at ``len(CRB_gold_tier_sequence)``; when the
budget is exhausted the harness extracts whatever ``git diff`` shows and
hands it to ``swebench.harness.run_evaluation`` for the official
``resolved`` verdict.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are an autonomous software engineer solving a real GitHub issue. The target repository is already checked out inside a Linux container at /testbed, at the commit containing the bug.

Your job: produce a single git-applicable patch that resolves the issue. The harness will capture your patch by running `git diff` at the end and feed it to the official SWE-bench evaluation (FAIL_TO_PASS + PASS_TO_PASS).

Every turn you MUST call exactly one tool. Never reply with plain prose. Available tools:
- bash: run any shell command inside the container. Use this to explore (grep/find/ls), run tests, inspect data.
- str_replace_editor: view / create / str_replace / insert / undo_edit on files under /testbed.
- finish: signal that your patch is ready; the harness will stop and capture the diff.

Constraints
- Only edit files inside /testbed.
- The patch must be minimal - change only what the issue requires.
- You have a fixed turn budget enforced by the harness; if you run out before calling finish, the current diff is captured as-is and evaluated.
- Call finish as soon as your edits pass the relevant tests (or as soon as you are confident); do not spin.
"""


USER_PROMPT_TEMPLATE = """You will be provided with an issue statement from repository {repo} at commit {base_commit}. Solve it by editing files under /testbed and producing a minimal git-applicable patch.

<issue>
{problem_statement}
</issue>

Workflow guidance
1. Use bash and str_replace_editor to locate the file(s) responsible for the described behaviour.
2. Reproduce the bug in the container if it helps you confirm the fault.
3. Make the smallest correct edit.
4. Optionally run the repository's test suite (pytest / tox / project-specific scripts) to gain confidence.
5. Call the finish tool when the patch is ready.

Your first tool call should usually be a bash or str_replace_editor exploration step, not finish.
"""


def render_user_prompt(
    *,
    repo: str,
    base_commit: str,
    instance_id: str,
    problem_statement: str,
) -> str:
    """Render the per-instance user prompt. ``instance_id`` is accepted for
    signature compatibility with callers even though the SWE-bench style-3
    aligned template does not currently inject it directly (the instance
    identity is implicit in ``repo`` + ``base_commit`` + ``problem_statement``).
    """
    if not all(
        isinstance(v, str) for v in (repo, base_commit, instance_id, problem_statement)
    ):
        raise TypeError("render_user_prompt: all arguments must be strings")
    return USER_PROMPT_TEMPLATE.format(
        repo=repo,
        base_commit=base_commit,
        problem_statement=problem_statement,
    )
