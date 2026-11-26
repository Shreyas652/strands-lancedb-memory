"""Freeze the raw-session package configuration using development answers only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.split import sha256_file

TIE_RULE = "maximum correct answers; then FTS before hybrid before vector; then smaller k; then lexical name"


def _valid_answer(report: dict[str, Any], manifest: dict[str, Any], name: str) -> None:
    if (
        report["phase"] != "dev_answers"
        or report["dataset_sha256"] != manifest["dataset_sha256"]
        or report["question_count"] != manifest["dev_count"]
        or report["reader_model"] != "gpt-4o-mini-2024-07-18"
        or report["judge_model"] != "gpt-4o-2024-08-06"
        or report["temperature"] != 0
    ):
        raise ValueError(f"{name} does not match the dev benchmark contract")
    expected_ids = set(manifest["dev_question_ids"])
    if {row["question_id"] for row in report["per_question"]} != expected_ids:
        raise ValueError(f"{name} has an unexpected question set")
    if any(
        item["total"] != manifest["dev_count"] for item in report["summary"].values()
    ):
        raise ValueError(f"{name} has an incomplete arm")
    for arm, measured in report["summary"].items():
        rows = [
            row
            for row in report["per_question"]
            if f"{row['system']}:{row['k']}" == arm
        ]
        if (
            len(rows) != manifest["dev_count"]
            or {row["question_id"] for row in rows} != expected_ids
            or sum(row["correct"] for row in rows) != measured["correct"]
        ):
            raise ValueError(f"{name} arm {arm} differs from its raw rows")


def choose(
    manifest: dict[str, Any],
    baseline: dict[str, Any],
    vector: dict[str, Any],
    baseline_retrieval: dict[str, Any],
    vector_retrieval: dict[str, Any],
) -> dict[str, Any]:
    _valid_answer(baseline, manifest, "baseline answers")
    _valid_answer(vector, manifest, "vector answers")
    if baseline["official_prompt_sha256"] != vector["official_prompt_sha256"]:
        raise ValueError("answer reports used different judge prompts")
    if (
        baseline["reader_instructions_sha256"] != vector["reader_instructions_sha256"]
        or baseline["max_context_characters"] != vector["max_context_characters"]
    ):
        raise ValueError("answer reports used different reader settings")
    expected_baseline = {f"lancedb_fts:{k}" for k in (3, 5, 10)}
    expected_vector = {
        f"{system}:{k}" for system in vector_retrieval["summary"] for k in (3, 5, 10)
    }
    if not expected_baseline.issubset(
        baseline["summary"]
    ) or not expected_vector.issubset(vector["summary"]):
        raise ValueError("predeclared answer ablation grid is incomplete")
    for report in (baseline, vector):
        for measured in report["summary"].values():
            if measured["accuracy"] != measured["correct"] / measured["total"]:
                raise ValueError("answer summary accuracy differs from its count")
    candidates = []
    for report, retrieval, package_only in (
        (baseline, baseline_retrieval, "lancedb_fts"),
        (vector, vector_retrieval, None),
    ):
        if (
            retrieval["phase"] != "dev_retrieval_only"
            or retrieval["dataset_sha256"] != manifest["dataset_sha256"]
            or retrieval["question_count"] != manifest["dev_count"]
        ):
            raise ValueError("retrieval report differs from the development split")
        for arm, measured in report["summary"].items():
            system, k_text = arm.rsplit(":", 1)
            if package_only is not None and system != package_only:
                continue
            if package_only is None and system not in retrieval["summary"]:
                continue
            k = int(k_text)
            if k not in (3, 5, 10):
                raise ValueError(
                    "candidate result limit is outside the predeclared grid"
                )
            candidates.append(
                {
                    "arm": arm,
                    "system": system,
                    "k": k,
                    "correct": measured["correct"],
                    "total": measured["total"],
                    "accuracy": measured["accuracy"],
                    "retrieval_hit_at_k": retrieval["summary"][system][f"hit_at_{k}"],
                }
            )
    if not candidates or len({item["arm"] for item in candidates}) != len(candidates):
        raise ValueError("missing or duplicate package candidate")

    # The tie rule is deliberately fixed before viewing the held-out test set.
    def rank(item: dict[str, Any]) -> tuple[int, int, int, str]:
        complexity = (
            0
            if item["system"] == "lancedb_fts"
            else 1
            if item["system"].startswith("hybrid_rrf:")
            else 2
        )
        return (-item["correct"], complexity, item["k"], item["arm"])

    winner = min(candidates, key=rank)
    system = winner["system"]
    return {
        "status": "FROZEN",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "selection_rule": TIE_RULE,
        "system": system,
        "k": winner["k"],
        "retrieval": "fts"
        if system == "lancedb_fts"
        else "vector"
        if system == "vector:1.00"
        else "hybrid_rrf",
        "vector_weight": None
        if system == "lancedb_fts"
        else float(system.split(":", 1)[1]),
        "embedding_model_id": None
        if system == "lancedb_fts"
        else vector_retrieval["embedding_model_id"],
        "dataset_sha256": manifest["dataset_sha256"],
        "dev_count": manifest["dev_count"],
        "sealed_test_ids_sha256": manifest["sealed_test_ids_sha256"],
        "reader_model": baseline["reader_model"],
        "judge_model": baseline["judge_model"],
        "official_prompt_sha256": baseline["official_prompt_sha256"],
        "max_context_characters": baseline["max_context_characters"],
        "winning_dev_correct": winner["correct"],
        "candidate_results": sorted(candidates, key=rank),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-answers", type=Path, required=True)
    parser.add_argument("--vector-answers", type=Path, required=True)
    parser.add_argument("--baseline-retrieval", type=Path, required=True)
    parser.add_argument("--vector-retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "frozen configuration already exists; do not retune on test"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = {
        "baseline_answers": args.baseline_answers,
        "vector_answers": args.vector_answers,
        "baseline_retrieval": args.baseline_retrieval,
        "vector_retrieval": args.vector_retrieval,
    }
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    split_digest = sha256_file(args.manifest)
    if any(report["split_sha256"] != split_digest for report in reports.values()):
        raise ValueError("development report differs from the sealed split manifest")
    frozen = choose(
        manifest,
        reports["baseline_answers"],
        reports["vector_answers"],
        reports["baseline_retrieval"],
        reports["vector_retrieval"],
    )
    frozen["manifest_sha256"] = split_digest
    frozen["source_report_sha256"] = {
        name: sha256_file(path) for name, path in paths.items()
    }
    frozen["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "system": frozen["system"],
                "k": frozen["k"],
                "dev_correct": frozen["winning_dev_correct"],
            }
        )
    )


if __name__ == "__main__":
    main()
