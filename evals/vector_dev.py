"""Local-vector and weighted-hybrid retrieval grid on the sealed dev split."""

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

from evals.local_embedder import FastEmbedAdapter
from evals.retrieval_dev import _evidence_labels, _session_text
from evals.split import make_split, sha256_file
from strands_lancedb_memory import Embedder, LanceDBMemoryStore


def _utc_micros(value: datetime) -> int:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = value - epoch
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _seed_case(
    store: LanceDBMemoryStore, case: dict[str, Any], embedder: Embedder
) -> int:
    sessions = case["haystack_sessions"]
    session_ids = case["haystack_session_ids"]
    dates = case["haystack_dates"]
    if not len(sessions) == len(session_ids) == len(dates):
        raise ValueError("session arrays differ in length")
    texts = [
        _session_text(session_id, date, turns, ordinal)
        for ordinal, (session_id, date, turns) in enumerate(
            zip(session_ids, dates, sessions, strict=True)
        )
    ]
    vectors = embedder.embed_documents(texts)
    if len(vectors) != len(texts):
        raise ValueError("embedder omitted a session vector")
    now = _utc_micros(datetime.now(UTC))
    rows = []
    for ordinal, (session_id, date, text, vector) in enumerate(
        zip(session_ids, dates, texts, vectors, strict=True)
    ):
        valid_from = datetime.strptime(date, "%Y/%m/%d (%a) %H:%M").replace(tzinfo=UTC)
        entry_id = hashlib.sha256(
            json.dumps(
                [case["question_id"], ordinal, session_id], separators=(",", ":")
            ).encode()
        ).hexdigest()
        rows.append(
            {
                "entry_id": entry_id,
                "tenant_id": case["question_id"],
                "content": text,
                "role": None,
                "run_id": None,
                "sequence_number": None,
                "fact_key": None,
                "valid_from": _utc_micros(valid_from),
                "recorded_at": now,
                "source": "fact",
                "metadata_json": json.dumps(
                    {"session_id": session_id, "session_ordinal": ordinal},
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                "vector": vector,
            }
        )
    # One batch per question keeps this retrieval ablation tractable. Search
    # still uses the public store API; insertion throughput is not claimed.
    store._write_rows(rows)
    return len(rows)


async def evaluate(
    dataset: Path,
    manifest: Path,
    cache_dir: Path,
    *,
    limit: int | None,
    weights: list[float],
) -> dict[str, Any]:
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    if saved != make_split(
        dataset, seed=saved["seed"], dev_fraction=saved["dev_fraction"]
    ):
        raise ValueError("dataset or sealed split changed")
    ids = (
        saved["dev_question_ids"]
        if limit is None
        else saved["dev_question_ids"][:limit]
    )
    if not ids:
        raise ValueError("no development cases selected")
    if any(not 0 <= weight <= 1 or not math.isfinite(weight) for weight in weights):
        raise ValueError("weights must be finite and between 0 and 1")
    embedder = FastEmbedAdapter(cache_dir)
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    cases = {
        item["question_id"]: {
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "question": item["question"],
            "haystack_sessions": item["haystack_sessions"],
            "haystack_session_ids": item["haystack_session_ids"],
            "haystack_dates": item["haystack_dates"],
        }
        for item in raw
        if item["question_id"] in ids
    }
    del raw
    labels = _evidence_labels(dataset, set(ids))
    records = []
    indexed_sessions = 0
    with tempfile.TemporaryDirectory(prefix="strands-lmem-vector-") as scratch:
        uri = Path(scratch) / "db"
        for position, question_id in enumerate(ids, start=1):
            case = cases[question_id]
            seed = LanceDBMemoryStore(
                uri=uri,
                tenant_id=question_id,
                retrieval="hybrid_rrf",
                embedder=embedder,
            )
            indexed_sessions += await asyncio.to_thread(
                _seed_case, seed, case, embedder
            )
            configurations = [
                ("vector", 1.0),
                *(("hybrid_rrf", value) for value in weights),
            ]
            for mode, weight in configurations:
                store = LanceDBMemoryStore(
                    uri=uri,
                    tenant_id=question_id,
                    retrieval=mode,
                    embedder=embedder,
                    vector_weight=weight,
                )
                started = time.perf_counter()
                entries = await store.search(
                    case["question"], {"max_search_results": 10}
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                retrieved_ids = [str(entry.metadata["session_id"]) for entry in entries]
                retrieved_ordinals = [
                    int(entry.metadata["session_ordinal"]) for entry in entries
                ]
                evidence = labels[question_id]
                rank = next(
                    (
                        index
                        for index, session_id in enumerate(retrieved_ids, 1)
                        if session_id in evidence
                    ),
                    None,
                )
                records.append(
                    {
                        "question_id": question_id,
                        "question_type": case["question_type"],
                        "system": f"{mode}:{weight:.2f}",
                        "retrieved_session_ids": retrieved_ids,
                        "retrieved_session_ordinals": retrieved_ordinals,
                        "first_evidence_rank": rank,
                        "search_latency_ms": elapsed_ms,
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
            print(f"completed vector case {position}/{len(ids)}", flush=True)
    summary = {}
    for system in sorted({row["system"] for row in records}):
        rows = [row for row in records if row["system"] == system]
        latencies = sorted(row["search_latency_ms"] for row in rows)
        summary[system] = {
            f"hit_at_{k}": sum(
                row["first_evidence_rank"] is not None
                and row["first_evidence_rank"] <= k
                for row in rows
            )
            / len(rows)
            for k in (3, 5, 10)
        }
        summary[system]["p50_search_ms"] = statistics.median(latencies)
        summary[system]["p95_search_ms"] = latencies[
            math.ceil(len(latencies) * 0.95) - 1
        ]
    return {
        "phase": "dev_retrieval_only",
        "dataset_sha256": sha256_file(dataset),
        "split_sha256": sha256_file(manifest),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": saved["seed"],
        "question_count": len(ids),
        "indexed_sessions": indexed_sessions,
        "embedding_model_id": embedder.model_id,
        "embedding_weights_sha256": embedder.model_weights_sha256,
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
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--weights", nargs="+", type=float, default=[0.3, 0.5, 0.7, 0.9]
    )
    args = parser.parse_args()
    result = asyncio.run(
        evaluate(
            args.dataset,
            args.manifest,
            args.model_cache,
            limit=args.limit,
            weights=args.weights,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"question_count": result["question_count"], "summary": result["summary"]}
        )
    )


if __name__ == "__main__":
    main()
