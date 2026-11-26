"""Verify report provenance and recompute scale metrics from raw query rows."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = {
    "dev_retrieval.json": "evals/retrieval_dev.py",
    "dev_answers.json": "evals/answer_dev.py",
    "vector_dev.json": "evals/vector_dev.py",
    "vector_answers_dev.json": "evals/answer_dev.py",
    "dev_facts.json": "evals/extract_dev.py",
    "facts_retrieval_dev.json": "evals/facts_retrieval_dev.py",
    "fact_answers_dev.json": "evals/answer_facts_dev.py",
    "frozen_config.json": "evals/freeze.py",
    "frozen_retrieval.json": "evals/frozen_retrieval.py",
    "frozen_answers.json": "evals/frozen_answers.py",
    "second_judge.json": "evals/second_judge.py",
    "all_500_summary.json": "evals/all_500.py",
    "paired_significance.json": "evals/paired_significance.py",
    "scale_1k.json": "experiments/scale_memory.py",
    "scale_10k.json": "experiments/scale_memory.py",
    "scale_100k.json": "experiments/scale_memory.py",
}
REQUIRED = {"dev_retrieval.json", "scale_1k.json", "scale_10k.json", "scale_100k.json"}


def _check_freeze_hash(saved: str, frozen_bytes: bytes) -> None:
    """Accept the committed LF bytes or their original Windows CRLF encoding."""
    lf = frozen_bytes.replace(b"\r\n", b"\n")
    if b"\r" in lf:
        raise ValueError("frozen config has unexpected line endings")
    hashes = {
        hashlib.sha256(lf).hexdigest(),
        hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest(),
    }
    if saved not in hashes:
        raise ValueError("freeze_sha256 differs from both LF and CRLF frozen config")


def _close(actual: float, saved: float) -> None:
    if not math.isclose(actual, saved, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"saved metric differs from raw rows: {saved} versus {actual}")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    lower = math.floor(at)
    return ordered[lower] + (
        ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
    ) * (at - lower)


def _check_scale(report: dict[str, Any]) -> None:
    rows = report["per_query"]
    if len(rows) != report["queries"] or report["status"] != "COMPLETE":
        raise ValueError("scale query count or status differs")
    for label, source in (
        ("exact", "exact_search_ms"),
        ("indexed", "indexed_search_ms"),
    ):
        times = [row[source] for row in rows]
        _close(statistics.median(times), report[f"{label}_p50_search_ms"])
        _close(_percentile(times, 0.95), report[f"{label}_p95_search_ms"])
    recall = sum(
        len(set(row["exact_top_10"]) & set(row["indexed_top_10"]))
        / len(row["exact_top_10"])
        for row in rows
    ) / len(rows)
    agreement = sum(
        row["exact_top_10"][0] == row["indexed_top_10"][0] for row in rows
    ) / len(rows)
    _close(recall, report["indexed_recall_at_10"])
    _close(agreement, report["indexed_top_1_agreement"])


def _check_rows(report: dict[str, Any]) -> None:
    if report.get("phase") == "frozen_test_second_judge":
        rows = report["per_question"]
        if len(rows) != report["sample_count"] or len(
            {(row["question_id"], row["system"]) for row in rows}
        ) != len(rows):
            raise ValueError("second-judge sample has missing or duplicate rows")
        agreement = sum(
            row["primary_correct"] == row["secondary_correct"] for row in rows
        )
        if agreement != report["agreement_count"]:
            raise ValueError("second-judge agreement count differs from raw rows")
        _close(agreement / len(rows), report["agreement"])
        return
    if "per_question" not in report or "summary" not in report:
        return
    rows = report["per_question"]
    expected = report["question_count"] * len(report["summary"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} per-question rows, got {len(rows)}")
    if len({row["question_id"] for row in rows}) != report["question_count"]:
        raise ValueError("per-question IDs do not match question_count")
    if report["phase"] in (
        "dev_answers",
        "frozen_test_answers",
        "combined_dev_and_frozen_test_answers",
    ):
        for arm, measured in report["summary"].items():
            selected = [row for row in rows if f"{row['system']}:{row['k']}" == arm]
            if len(selected) != report["question_count"] or len(
                {row["question_id"] for row in selected}
            ) != len(selected):
                raise ValueError(f"answer arm {arm} has missing or duplicate rows")
            correct = sum(row["correct"] for row in selected)
            if correct != measured["correct"] or len(selected) != measured["total"]:
                raise ValueError(f"answer arm {arm} counts differ from raw rows")
            _close(correct / len(selected), measured["accuracy"])
            if "p50_incremental_prompt_tokens" in measured:
                deltas = [row["incremental_prompt_tokens"] for row in selected]
                if all(delta is not None for delta in deltas):
                    _close(
                        statistics.median(deltas),
                        measured["p50_incremental_prompt_tokens"],
                    )
                elif measured["p50_incremental_prompt_tokens"] is not None:
                    raise ValueError("incremental prompt tokens are incomplete")
            for kind, counts in measured["by_type"].items():
                typed = [row for row in selected if row["question_type"] == kind]
                if counts != {
                    "correct": sum(row["correct"] for row in typed),
                    "total": len(typed),
                }:
                    raise ValueError(
                        f"answer arm {arm} type {kind} differs from raw rows"
                    )
            if {row["question_type"] for row in selected} != set(measured["by_type"]):
                raise ValueError(f"answer arm {arm} omits a question type")
    elif report["phase"] in ("dev_retrieval_only", "frozen_test_retrieval"):
        for system, measured in report["summary"].items():
            selected = [row for row in rows if row["system"] == system]
            if len(selected) != report["question_count"] or len(
                {row["question_id"] for row in selected}
            ) != len(selected):
                raise ValueError(
                    f"retrieval system {system} has missing or duplicate rows"
                )
            for k in (3, 5, 10):
                hit = sum(
                    row["first_evidence_rank"] is not None
                    and row["first_evidence_rank"] <= k
                    for row in selected
                ) / len(selected)
                _close(hit, measured[f"hit_at_{k}"])
            times = sorted(row["search_latency_ms"] for row in selected)
            _close(statistics.median(times), measured["p50_search_ms"])
            _close(times[math.ceil(len(times) * 0.95) - 1], measured["p95_search_ms"])


def main() -> None:
    checked = 0
    for filename, source in RUNNERS.items():
        path = ROOT / "results" / filename
        if not path.exists():
            if filename in REQUIRED:
                raise FileNotFoundError(path)
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        actual = hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
        if report["runner_sha256"] != actual:
            raise ValueError(f"runner hash differs for {filename}; rerun its benchmark")
        if "freeze_sha256" in report:
            _check_freeze_hash(
                report["freeze_sha256"],
                (ROOT / "results/frozen_config.json").read_bytes(),
            )
        if filename == "paired_significance.json":
            source = (ROOT / "results/frozen_answers.json").read_bytes()
            if report["frozen_answers_sha256"] != hashlib.sha256(source).hexdigest():
                raise ValueError("paired significance source hash differs")
        _check_rows(report)
        if filename.startswith("scale_"):
            _check_scale(report)
        checked += 1
    print(f"Verified {checked} benchmark artifacts and their runner hashes.")


if __name__ == "__main__":
    main()
