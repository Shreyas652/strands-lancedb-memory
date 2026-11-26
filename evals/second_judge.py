"""Independently regrade 50 seeded frozen-test answers with GPT-4o mini."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from evals.model_api import ModelClient, billed_cost, load_key
from evals.official_prompt import get_anscheck_prompt
from evals.split import sha256_file

SECOND_JUDGE_MODEL = "gpt-4o-mini-2024-07-18"
SAMPLE_SEED = 20261125
SAMPLE_SIZE = 50


def sample_plans(dataset: Path, frozen_answers: Path) -> list[dict[str, Any]]:
    report = json.loads(frozen_answers.read_text(encoding="utf-8"))
    if (
        report["phase"] != "frozen_test_answers"
        or report["dataset_sha256"] != sha256_file(dataset)
        or report["judge_model"] != "gpt-4o-2024-08-06"
    ):
        raise ValueError("the source must be the complete frozen test answers")
    rows = report["per_question"]
    if len(rows) != report["question_count"] * 4 or len(
        {(row["question_id"], row["system"]) for row in rows}
    ) != len(rows):
        raise ValueError("frozen answer rows are incomplete or duplicated")
    selected = random.Random(SAMPLE_SEED).sample(
        sorted(rows, key=lambda row: (row["question_id"], row["system"])), SAMPLE_SIZE
    )
    source = json.loads(dataset.read_text(encoding="utf-8"))
    by_id = {item["question_id"]: item for item in source}
    plans = []
    for row in selected:
        prompt = get_anscheck_prompt(
            row["question_type"],
            by_id[row["question_id"]]["question"],
            by_id[row["question_id"]]["answer"],
            row["answer"],
            abstention=row["question_id"].endswith("_abs"),
        )
        plans.append(
            {
                "question_id": row["question_id"],
                "system": row["system"],
                "k": row["k"],
                "primary_correct": row["correct"],
                "prompt": prompt,
            }
        )
    return plans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frozen-answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=Path("data/model_calls.jsonl"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--budget-usd", type=float, default=20.0)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("second judge sample already exists")
    plans = sample_plans(args.dataset, args.frozen_answers)
    estimate = sum(
        billed_cost(SECOND_JUDGE_MODEL, len(plan["prompt"].encode("utf-8")) + 32, 10)
        for plan in plans
    )
    print(
        f"planned second-judge calls: {len(plans)}; conservative estimate: ${estimate:.4f}"
    )
    if args.estimate_only:
        return
    client = ModelClient(
        key=load_key(args.env_file), ledger=args.ledger, budget_usd=args.budget_usd
    )
    if estimate + client.spent_usd > args.budget_usd:
        raise ValueError("second judge estimate exceeds remaining budget")
    rows = []
    used = set()
    for index, plan in enumerate(plans, 1):
        result = client.call(SECOND_JUDGE_MODEL, plan["prompt"], max_tokens=10)
        used.add(result.request_sha256)
        secondary_correct = "yes" in result.text.casefold()
        rows.append(
            {
                "question_id": plan["question_id"],
                "system": plan["system"],
                "k": plan["k"],
                "primary_correct": plan["primary_correct"],
                "secondary_correct": secondary_correct,
                "secondary_response": result.text,
                "prompt_sha256": result.prompt_sha256,
                "response_model": result.response_model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }
        )
        print(f"regraded {index}/{len(plans)}", flush=True)
    report = {
        "phase": "frozen_test_second_judge",
        "dataset_sha256": sha256_file(args.dataset),
        "frozen_answers_sha256": sha256_file(args.frozen_answers),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "official_prompt_sha256": hashlib.sha256(
            (Path(__file__).parent / "official_prompt.py").read_bytes()
        ).hexdigest(),
        "sample_seed": SAMPLE_SEED,
        "sample_count": SAMPLE_SIZE,
        "primary_model": "gpt-4o-2024-08-06",
        "second_model": SECOND_JUDGE_MODEL,
        "agreement_count": sum(
            row["primary_correct"] == row["secondary_correct"] for row in rows
        ),
        "api_cost_usd_attributable": sum(
            client.records[digest]["cost_usd"] for digest in used
        ),
        "api_cost_usd_ledger_total": client.spent_usd,
        "per_question": rows,
    }
    report["agreement"] = report["agreement_count"] / SAMPLE_SIZE
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "agreement": report["agreement"],
                "cost_usd": report["api_cost_usd_attributable"],
            }
        )
    )


if __name__ == "__main__":
    main()
