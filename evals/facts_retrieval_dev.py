"""Compare append-only and keyed fact memories on the sealed dev split."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from evals.extract_dev import _load_cases
from evals.retrieval_dev import _evidence_labels
from evals.split import sha256_file
from strands_lancedb_memory import LanceDBMemoryStore


def _micros(value: datetime) -> int:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    difference = value - epoch
    return (
        difference.days * 86_400 + difference.seconds
    ) * 1_000_000 + difference.microseconds


def _rows(
    case: dict[str, Any], facts: list[dict[str, Any]], *, keyed: bool
) -> list[dict[str, Any]]:
    session_ids = case["haystack_session_ids"]
    dates = case["haystack_dates"]
    now = _micros(datetime.now(UTC))
    rows = []
    for index, fact in enumerate(facts):
        ordinal = fact["session_ordinal"]
        if type(ordinal) is not int or not 0 <= ordinal < len(session_ids):
            raise ValueError("fact refers to a nonexistent session")
        date = datetime.strptime(dates[ordinal], "%Y/%m/%d (%a) %H:%M").replace(
            tzinfo=UTC
        )
        entry_id = hashlib.sha256(
            json.dumps(
                [case["question_id"], index, ordinal, fact["fact"]], ensure_ascii=False
            ).encode()
        ).hexdigest()
        rows.append(
            {
                "entry_id": entry_id,
                "tenant_id": case["question_id"],
                "content": fact["fact"],
                "role": None,
                "run_id": None,
                "sequence_number": None,
                "fact_key": fact["key"] if keyed else None,
                "valid_from": _micros(date),
                "recorded_at": now,
                "source": "fact",
                "metadata_json": json.dumps(
                    {
                        "session_id": session_ids[ordinal],
                        "session_ordinal": ordinal,
                        "fact_index": index,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def _seed(store: LanceDBMemoryStore, rows: list[dict[str, Any]]) -> None:
    # Direct bulk add keeps this *retrieval* experiment tractable; no public
    # add throughput claim is inferred from it.
    store._initialize_sync()
    if rows:
        table = store._open_table()
        table.add(pa.Table.from_pylist(rows, schema=store._schema()))


async def evaluate(
    dataset: Path, manifest: Path, extraction: Path, *, limit: int | None = None
) -> dict[str, Any]:
    cases = _load_cases(dataset, manifest, limit)
    saved = json.loads(extraction.read_text(encoding="utf-8"))
    if (
        saved["phase"] != "dev_query_independent_fact_extraction"
        or saved["dataset_sha256"] != sha256_file(dataset)
        or saved["split_sha256"] != sha256_file(manifest)
        or saved["question_count"] != len(cases)
    ):
        raise ValueError("fact extraction report does not match this dev selection")
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    selected_ids = {case["question_id"] for case in cases}
    questions = {
        item["question_id"]: item["question"]
        for item in raw
        if item["question_id"] in selected_ids
    }
    del raw
    labels = _evidence_labels(dataset, {case["question_id"] for case in cases})
    records = []
    seeded_facts = 0
    with tempfile.TemporaryDirectory(prefix="strands-lmem-facts-") as temporary:
        root = Path(temporary)
        for position, case in enumerate(cases, start=1):
            question_id = case["question_id"]
            facts = saved["facts_by_question"][question_id]
            for name, keyed in (("fact_append_fts", False), ("fact_key_fts", True)):
                store = LanceDBMemoryStore(
                    uri=root / name,
                    tenant_id=question_id,
                    retrieval="fts",
                    update_strategy="fact_key" if keyed else "append",
                )
                rows = _rows(case, facts, keyed=keyed)
                await asyncio.to_thread(_seed, store, rows)
                seeded_facts += len(rows)
                started = time.perf_counter()
                entries = await store.search(
                    questions[question_id], {"max_search_results": 10}
                )
                duration = (time.perf_counter() - started) * 1000
                session_ids = [str(entry.metadata["session_id"]) for entry in entries]
                rank = next(
                    (
                        index
                        for index, session_id in enumerate(session_ids, 1)
                        if session_id in labels[question_id]
                    ),
                    None,
                )
                records.append(
                    {
                        "question_id": question_id,
                        "system": name,
                        "retrieved_session_ids": session_ids,
                        "retrieved_session_ordinals": [
                            int(entry.metadata["session_ordinal"]) for entry in entries
                        ],
                        "retrieved_fact_indices": [
                            int(entry.metadata["fact_index"]) for entry in entries
                        ],
                        "retrieved_contents": [entry.content for entry in entries],
                        "first_evidence_rank": rank,
                        "search_latency_ms": duration,
                        "injected_characters_at_3": sum(
                            len(entry.content) for entry in entries[:3]
                        ),
                        "injected_characters_at_5": sum(
                            len(entry.content) for entry in entries[:5]
                        ),
                        "injected_characters_at_10": sum(
                            len(entry.content) for entry in entries[:10]
                        ),
                    }
                )
            print(f"completed fact retrieval {position}/{len(cases)}", flush=True)
    summary = {}
    for system in sorted({row["system"] for row in records}):
        rows = [row for row in records if row["system"] == system]
        timings = sorted(row["search_latency_ms"] for row in rows)
        summary[system] = {
            f"hit_at_{k}": sum(
                row["first_evidence_rank"] is not None
                and row["first_evidence_rank"] <= k
                for row in rows
            )
            / len(rows)
            for k in (3, 5, 10)
        }
        summary[system]["p50_search_ms"] = statistics.median(timings)
        summary[system]["p95_search_ms"] = timings[math.ceil(len(timings) * 0.95) - 1]
    return {
        "phase": "dev_retrieval_only",
        "dataset_sha256": sha256_file(dataset),
        "split_sha256": sha256_file(manifest),
        "extraction_sha256": sha256_file(extraction),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "question_count": len(cases),
        "seeded_facts_across_two_modes": seeded_facts,
        "api_cost_usd": 0,
        "api_calls": 0,
        "answer_accuracy": None,
        "summary": summary,
        "per_question": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = asyncio.run(
        evaluate(args.dataset, args.manifest, args.extraction, limit=args.limit)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {"question_count": report["question_count"], "summary": report["summary"]}
        )
    )


if __name__ == "__main__":
    main()
