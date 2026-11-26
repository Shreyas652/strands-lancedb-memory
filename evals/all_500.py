"""Combine frozen test and matching dev arms without changing selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.split import sha256_file


def combine(
    manifest: dict[str, Any],
    frozen: dict[str, Any],
    dev: dict[str, Any],
    vector: dict[str, Any],
    test: dict[str, Any],
) -> dict[str, Any]:
    if (
        frozen["status"] != "FROZEN"
        or dev["phase"] != "dev_answers"
        or vector["phase"] != "dev_answers"
        or test["phase"] != "frozen_test_answers"
        or frozen["dataset_sha256"] != manifest["dataset_sha256"]
        or any(
            report["dataset_sha256"] != manifest["dataset_sha256"]
            for report in (dev, vector, test)
        )
        or dev["question_count"] != manifest["dev_count"]
        or test["question_count"] != manifest["test_count"]
        or test["frozen_system"] != frozen["system"]
        or test["frozen_k"] != frozen["k"]
        or len({report["official_prompt_sha256"] for report in (dev, vector, test)})
        != 1
        or len({report["reader_model"] for report in (dev, vector, test)}) != 1
        or len({report["judge_model"] for report in (dev, vector, test)}) != 1
    ):
        raise ValueError("answer artifacts do not match the frozen experiment")
    k = frozen["k"]
    selected_dev = dev if frozen["system"] == "lancedb_fts" else vector
    arms = {
        ("no_memory", 0): dev,
        ("file", k): dev,
        ("test", k): dev,
        (frozen["system"], k): selected_dev,
    }
    combined = []
    for (system, limit), report in arms.items():
        chosen = [
            row
            for row in report["per_question"]
            if row["system"] == system and row["k"] == limit
        ]
        if len(chosen) != manifest["dev_count"] or {
            row["question_id"] for row in chosen
        } != set(manifest["dev_question_ids"]):
            raise ValueError("development arm is missing a question")
        combined.extend(dict(row, split="dev") for row in chosen)
        held_out = [
            row
            for row in test["per_question"]
            if row["system"] == system and row["k"] == limit
        ]
        if len(held_out) != manifest["test_count"] or {
            row["question_id"] for row in held_out
        } & set(manifest["dev_question_ids"]):
            raise ValueError("test arm is incomplete or overlaps development")
        combined.extend(dict(row, split="test") for row in held_out)
    summary = {}
    for system, limit in arms:
        rows = [
            row for row in combined if row["system"] == system and row["k"] == limit
        ]
        by_type: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            by_type[row["question_type"]].append(row["correct"])
        summary[f"{system}:{limit}"] = {
            "correct": sum(row["correct"] for row in rows),
            "total": len(rows),
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "by_type": {
                kind: {"correct": sum(values), "total": len(values)}
                for kind, values in sorted(by_type.items())
            },
        }
    return {
        "phase": "combined_dev_and_frozen_test_answers",
        "dataset_sha256": manifest["dataset_sha256"],
        "question_count": manifest["case_count"],
        "dev_count": manifest["dev_count"],
        "test_count": manifest["test_count"],
        "frozen_system": frozen["system"],
        "frozen_k": k,
        "reader_model": test["reader_model"],
        "judge_model": test["judge_model"],
        "official_prompt_sha256": test["official_prompt_sha256"],
        "summary": summary,
        "per_question": combined,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--dev-answers", type=Path, required=True)
    parser.add_argument("--vector-answers", type=Path, required=True)
    parser.add_argument("--test-answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("combined report already exists")
    paths = (
        args.manifest,
        args.frozen,
        args.dev_answers,
        args.vector_answers,
        args.test_answers,
    )
    manifest, frozen, dev, vector, test = (
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    )
    if (
        frozen["manifest_sha256"] != sha256_file(args.manifest)
        or frozen["source_report_sha256"]["baseline_answers"]
        != sha256_file(args.dev_answers)
        or frozen["source_report_sha256"]["vector_answers"]
        != sha256_file(args.vector_answers)
        or test["freeze_sha256"] != sha256_file(args.frozen)
    ):
        raise ValueError("source report changed after configuration freeze")
    report = combine(manifest, frozen, dev, vector, test)
    report["source_sha256"] = {path.name: sha256_file(path) for path in paths}
    report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
