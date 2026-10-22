"""Materialize the historical mtRAG/QMSum prompt formats from original data."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .types import TaskSpec


MTRAG_SYSTEM = (
    "You are a helpful, accurate assistant. You are given a multi-turn conversation "
    "and reference passages retrieved from a knowledge base. Answer the user's latest "
    "question based on the reference passages and conversation history. Be concise and "
    "factual. If the passages do not contain enough information, say so honestly."
)
QMSUM_SYSTEM = (
    "You are a helpful assistant for query-based meeting summarization (QMSum). "
    "Answer the user's query using the meeting transcript below. "
    "Be accurate, concise, and faithful to what was discussed in the transcript."
)


def _records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, list) else [raw]


def mtrag_filter_reason(task: dict[str, Any], *, require_human_feedback: bool = True) -> str:
    tags = task.get("Answerability", task.get("answerability", []))
    if isinstance(tags, str):
        tags = [tags]
    if "ANSWERABLE" not in tags:
        return "not_answerable"
    contexts = task.get("contexts") or []
    if not contexts or any(not str(context.get("text", "")).strip() for context in contexts):
        return "no_contexts"
    for context in contexts:
        feedback = context.get("feedback", {}).get("relevant", {})
        reviews = [review for review in feedback.values() if isinstance(review, dict)]
        if require_human_feedback and not reviews:
            return "missing_human_relevance"
        if any(review.get("value") != "yes" for review in reviews):
            return "irrelevant_context"
    return ""


def _task(benchmark: Any, instance_id: str, messages: list[dict], reference: Any,
          metadata: dict[str, Any]) -> TaskSpec:
    if not reference:
        raise ValueError(f"{instance_id} is missing its reference answer")
    return TaskSpec.from_dict({
        "benchmark": benchmark.name, "benchmark_display": benchmark.display_name,
        "scenario": benchmark.default_scenario, "instance_id": instance_id,
        "initial_messages": messages,
        "steps": [{"hint": "low", "minimum_tier": "high"}],
        "source": {"uri": benchmark.provenance_uri or "local://materialized-original-source",
                   "license": benchmark.source_license, "version": benchmark.source_version},
        "benchmark_version": benchmark.benchmark_version,
        "benchmark_subset": benchmark.benchmark_subset,
        "metadata": {"reference": reference, **metadata},
    })


class MTRAGLoader:
    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options
        self.filter_report: dict[str, Any] = {}

    def load_tasks(self, benchmark: Any, source: Any) -> list[TaskSpec]:
        path = source.require_materialized_path()
        rows = _records(path)
        tasks = []
        rejected: Counter[str] = Counter()
        for row in rows:
            reason = mtrag_filter_reason(row,
                require_human_feedback=bool(self.options.get("require_human_feedback", True)))
            if reason:
                rejected[reason] += 1
                continue
            passages = []
            for index, context in enumerate(row["contexts"], 1):
                header = f"[Passage {index}]" + (" " + context["title"] if context.get("title") else "")
                passages.append(header + "\n" + context["text"] + "\n\n")
            messages = [{"role": "system", "content": MTRAG_SYSTEM},
                {"role": "user", "content": "Here are reference passages from the knowledge base:\n\n" + "".join(passages).strip()},
                {"role": "assistant", "content": "Thank you, I will use these passages to help answer your questions."}]
            messages.extend({"role": "user" if turn["speaker"] == "user" else "assistant", "content": turn["text"]}
                            for turn in row["input"])
            if messages[-1]["role"] != "user":
                raise ValueError("mtRAG generation task must end at the current user question")
            reference = [target["text"] for target in row["targets"]]
            tasks.append(_task(benchmark, row["task_id"], messages, reference,
                {"preprocessing": "mtrag-reference-v2", "conversation_id": row.get("conversation_id")}))
        self.filter_report = {
            "protocol": "mtrag-reference-v2", "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "input_records": len(rows), "accepted_records": len(tasks), "rejected": dict(rejected),
            "require_human_feedback": bool(self.options.get("require_human_feedback", True)),
        }
        return tasks


class QMSumLoader:
    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options

    def load_tasks(self, benchmark: Any, source: Any) -> list[TaskSpec]:
        tasks = []
        for meeting_index, meeting in enumerate(_records(source.require_materialized_path())):
            meeting_id = str(meeting.get("meeting_id", f"meeting_{meeting_index}"))
            transcript = meeting["meeting_transcripts"]
            for kind in ("general", "specific"):
                for query_index, query in enumerate(meeting.get(kind + "_query_list", [])):
                    indices = range(len(transcript))
                    if kind == "specific" and query.get("relevant_text_span"):
                        selected: set[int] = set()
                        for span in query["relevant_text_span"]:
                            first, last = map(int, span)
                            if not 0 <= first <= last < len(transcript):
                                raise ValueError("QMSum relevant span is outside the transcript")
                            selected.update(range(first, last + 1))
                        indices = sorted(selected)
                    text = "\n".join(transcript[i]["speaker"] + ": " + transcript[i]["content"] for i in indices)
                    messages = [{"role": "system", "content": QMSUM_SYSTEM},
                        {"role": "user", "content": f"## Meeting transcript\n{text}\n\n## Query\n{query['query']}\n\nAnswer this query based only on the transcript above."}]
                    tasks.append(_task(benchmark, f"{meeting_id}_{kind}_{query_index}", messages, query["answer"],
                        {"meeting_id": meeting_id, "query_kind": kind, "query_index": query_index}))
        return tasks


def create_mtrag_loader(options: dict[str, Any]) -> MTRAGLoader:
    return MTRAGLoader(options)


def create_qmsum_loader(options: dict[str, Any]) -> QMSumLoader:
    return QMSumLoader(options)
