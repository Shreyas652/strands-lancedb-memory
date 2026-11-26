"""Exact paired test for the frozen LanceDB FTS and file-store answers."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path
from typing import Any

ARMS = {"lancedb_fts:3", "file:3"}


def exact_mcnemar_p(fts_only: int, file_only: int) -> float:
    """Two-sided exact binomial p-value for discordant paired outcomes."""
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in (fts_only, file_only)
    ):
        raise ValueError("discordant counts must be nonnegative integers")
    discordant = fts_only + file_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, k) for k in range(min(fts_only, file_only) + 1))
    return min(1.0, 2 * tail / 2**discordant)


def analyze(report: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    """Pair raw correctness rows by question ID, rejecting missing or duplicate arms."""
    if report.get("phase") != "frozen_test_answers":
        raise ValueError("expected a frozen test answer report")
    paired: dict[str, dict[str, bool]] = {}
    for row in report["per_question"]:
        arm = f"{row['system']}:{row['k']}"
        if arm not in ARMS:
            continue
        question_id = row["question_id"]
        if not isinstance(row["correct"], bool) or arm in paired.setdefault(
            question_id, {}
        ):
            raise ValueError(f"invalid or duplicate paired answer for {question_id}")
        paired[question_id][arm] = row["correct"]
    if len(paired) != report["question_count"] or any(
        set(outcomes) != ARMS for outcomes in paired.values()
    ):
        raise ValueError("paired answer arms are missing questions")
    fts_only = sum(
        outcomes["lancedb_fts:3"] and not outcomes["file:3"]
        for outcomes in paired.values()
    )
    file_only = sum(
        outcomes["file:3"] and not outcomes["lancedb_fts:3"]
        for outcomes in paired.values()
    )
    both = sum(
        outcomes["lancedb_fts:3"] and outcomes["file:3"] for outcomes in paired.values()
    )
    neither = len(paired) - fts_only - file_only - both
    difference = (fts_only - file_only) / len(paired)
    return {
        "phase": "frozen_test_paired_significance",
        "frozen_answers_sha256": source_sha256,
        "question_count": len(paired),
        "fts_only_correct": fts_only,
        "file_only_correct": file_only,
        "both_correct": both,
        "neither_correct": neither,
        "exact_mcnemar_two_sided_p": exact_mcnemar_p(fts_only, file_only),
        "paired_accuracy_difference": difference,
        "paired_accuracy_difference_points": difference * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("results/frozen_answers.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/paired_significance.json")
    )
    args = parser.parse_args()
    source = args.input.read_bytes()
    report = analyze(json.loads(source), hashlib.sha256(source).hexdigest())
    report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "fts_only_correct",
                    "file_only_correct",
                    "exact_mcnemar_two_sided_p",
                    "paired_accuracy_difference_points",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
