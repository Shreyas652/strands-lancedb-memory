"""Grade fact-shaped dev memories with the same reader and official judge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evals.answer_dev import _load_inputs, _reader_prompt, estimate_cost, evaluate
from evals.model_api import ModelClient, load_key
from evals.split import sha256_file


def _plans(
    cases: list[dict[str, Any]],
    retrieval: dict[tuple[str, str], dict[str, Any]],
    extraction: dict[str, Any],
    systems: list[str],
    k_values: list[int],
    max_chars: int,
) -> list[dict[str, Any]]:
    plans = []
    facts_by_question = extraction["facts_by_question"]
    for case in cases:
        question_id = case["question_id"]
        facts = facts_by_question[question_id]
        for system, k in [
            ("no_memory", 0),
            *((name, value) for name in systems for value in k_values),
        ]:
            row = None if system == "no_memory" else retrieval[(question_id, system)]
            contents = []
            included = []
            if row is not None:
                if (
                    not len(row["retrieved_fact_indices"])
                    == len(row["retrieved_contents"])
                    == len(row["retrieved_session_ordinals"])
                ):
                    raise ValueError("fact retrieval references are misaligned")
                for index, (fact_index, content, ordinal) in enumerate(
                    zip(
                        row["retrieved_fact_indices"][:k],
                        row["retrieved_contents"][:k],
                        row["retrieved_session_ordinals"][:k],
                        strict=True,
                    ),
                    start=1,
                ):
                    if type(fact_index) is not int or not 0 <= fact_index < len(facts):
                        raise ValueError("fact retrieval index is invalid")
                    if (
                        facts[fact_index]["fact"] != content
                        or facts[fact_index]["session_ordinal"] != ordinal
                    ):
                        raise ValueError(
                            "fact retrieval content differs from extraction"
                        )
                    text = f"# Memory {index}\n{content}"
                    if sum(map(len, contents)) + len(text) > max_chars:
                        continue
                    contents.append(text)
                    included.append(ordinal)
            context = "\n\n".join(contents)
            plans.append(
                {
                    "question_id": question_id,
                    "question_type": case["question_type"],
                    "system": system,
                    "k": k,
                    "reader_prompt": _reader_prompt(
                        case["question"], case["question_date"], context
                    ),
                    "context_characters": len(context),
                    "included_session_ordinals": included,
                    "question": case["question"],
                    "evidence_rank": None
                    if row is None
                    else row["first_evidence_rank"],
                }
            )
    return plans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=Path("data/model_calls.jsonl"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--systems", nargs="+", default=["fact_append_fts", "fact_key_fts"]
    )
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--max-context-chars", type=int, default=30000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--budget-usd", type=float, default=20.0)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    if args.max_context_chars < 1 or any(k < 1 or k > 10 for k in args.k):
        parser.error("context limit must be positive and k must be between 1 and 10")
    cases, retrieval, _, references = _load_inputs(
        args.dataset, args.manifest, args.retrieval_report, args.limit
    )
    extraction = json.loads(args.extraction.read_text(encoding="utf-8"))
    if (
        extraction["dataset_sha256"] != sha256_file(args.dataset)
        or extraction["split_sha256"] != sha256_file(args.manifest)
        or extraction["question_count"] != len(cases)
    ):
        raise ValueError("fact extraction report differs from selected dev cases")
    plans = _plans(
        cases, retrieval, extraction, args.systems, args.k, args.max_context_chars
    )
    planned = estimate_cost(plans, references)
    print(f"planned answers: {len(plans)}; conservative cost estimate: ${planned:.4f}")
    if args.estimate_only:
        return
    client = ModelClient(
        key=load_key(args.env_file), ledger=args.ledger, budget_usd=args.budget_usd
    )
    if planned + client.spent_usd > args.budget_usd:
        raise ValueError("fact answer estimate exceeds remaining budget")
    report = evaluate(
        plans,
        references,
        client,
        dataset=args.dataset,
        manifest=args.manifest,
        retrieval_report=args.retrieval_report,
        max_chars=args.max_context_chars,
    )
    report["shared_answer_evaluator_sha256"] = report["runner_sha256"]
    report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["extraction_sha256"] = sha256_file(args.extraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "cost_usd": report["api_cost_usd_attributable"],
            }
        )
    )


if __name__ == "__main__":
    main()
