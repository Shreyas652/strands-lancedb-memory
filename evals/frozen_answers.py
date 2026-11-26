"""Evaluate one frozen package configuration and fixed baselines on test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evals.answer_dev import _planned_cases, estimate_cost, evaluate
from evals.frozen_retrieval import sealed_ids
from evals.model_api import ModelClient, load_key
from evals.split import sha256_file


def load_inputs(
    dataset: Path, manifest: Path, frozen: Path, retrieval_report: Path
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, str],
    dict[str, Any],
]:
    ids, config = sealed_ids(dataset, manifest, frozen)
    retrieval = json.loads(retrieval_report.read_text(encoding="utf-8"))
    if (
        retrieval["phase"] != "frozen_test_retrieval"
        or retrieval["dataset_sha256"] != sha256_file(dataset)
        or retrieval["split_sha256"] != sha256_file(manifest)
        or retrieval["freeze_sha256"] != sha256_file(frozen)
        or retrieval["question_count"] != len(ids)
    ):
        raise ValueError("frozen retrieval differs from the sealed test")
    expected_systems = {config["system"], "file", "test"}
    by_retrieval: dict[tuple[str, str], dict[str, Any]] = {}
    for row in retrieval["per_question"]:
        key = (row["question_id"], row["system"])
        if (
            row["question_id"] not in ids
            or row["system"] not in expected_systems
            or key in by_retrieval
        ):
            raise ValueError("frozen retrieval contains an unexpected row")
        by_retrieval[key] = row
    if len(by_retrieval) != len(ids) * len(expected_systems):
        raise ValueError("frozen retrieval is incomplete")
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    by_id = {item["question_id"]: item for item in raw}
    references = {question_id: by_id[question_id]["answer"] for question_id in ids}
    cases = [
        {
            "question_id": question_id,
            "question_type": by_id[question_id]["question_type"],
            "question": by_id[question_id]["question"],
            "question_date": by_id[question_id]["question_date"],
            "haystack_sessions": by_id[question_id]["haystack_sessions"],
            "haystack_session_ids": by_id[question_id]["haystack_session_ids"],
            "haystack_dates": by_id[question_id]["haystack_dates"],
        }
        for question_id in ids
    ]
    return cases, by_retrieval, references, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=Path("data/model_calls.jsonl"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--budget-usd", type=float, default=20.0)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "frozen test answers already exist; do not rerun or retune"
        )
    cases, by_retrieval, references, config = load_inputs(
        args.dataset, args.manifest, args.frozen, args.retrieval_report
    )
    if (
        config["reader_model"] != "gpt-4o-mini-2024-07-18"
        or config["judge_model"] != "gpt-4o-2024-08-06"
    ):
        raise ValueError("frozen model configuration differs from the evaluator")
    systems = [config["system"], "file", "test"]
    plans = _planned_cases(
        cases,
        by_retrieval,
        systems,
        [config["k"]],
        config["max_context_characters"],
    )
    estimate = estimate_cost(plans, references)
    print(f"planned answers: {len(plans)}; conservative cost estimate: ${estimate:.4f}")
    if args.estimate_only:
        return
    client = ModelClient(
        key=load_key(args.env_file), ledger=args.ledger, budget_usd=args.budget_usd
    )
    if estimate + client.spent_usd > args.budget_usd:
        raise ValueError("frozen answer estimate exceeds remaining budget")
    result = evaluate(
        plans,
        references,
        client,
        dataset=args.dataset,
        manifest=args.manifest,
        retrieval_report=args.retrieval_report,
        max_chars=config["max_context_characters"],
    )
    result["phase"] = "frozen_test_answers"
    result["shared_answer_evaluator_sha256"] = result["runner_sha256"]
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["freeze_sha256"] = sha256_file(args.frozen)
    result["frozen_system"] = config["system"]
    result["frozen_k"] = config["k"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "cost_usd": result["api_cost_usd_attributable"],
            }
        )
    )


if __name__ == "__main__":
    main()
