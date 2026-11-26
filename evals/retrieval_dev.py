"""Offline retrieval evaluation on sealed LongMemEval development cases."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from strands.storage import LocalFileStorage
from strands.vended_memory_stores.file_memory_store import FileMemoryStore
from strands.vended_memory_stores.test_memory_store import TestMemoryStore

from evals.split import make_split, sha256_file
from strands_lancedb_memory import LanceDBMemoryStore


def _session_text(
    session_id: str, date: str, turns: list[dict[str, Any]], ordinal: int
) -> str:
    body = "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)
    if not body.strip():
        raise ValueError(f"empty session {session_id}")
    return f"# Session {ordinal}: {session_id}\nDate: {date}\n{body}"


def _session_ref_from_file(content: str) -> tuple[int, str]:
    heading = content.splitlines()[0]
    if not heading.startswith("# Session "):
        raise ValueError("FileMemoryStore returned a malformed session heading")
    ordinal_text, separator, session_id = heading.removeprefix("# Session ").partition(
        ": "
    )
    if not separator or not session_id:
        raise ValueError("FileMemoryStore returned a malformed session ID")
    return int(ordinal_text), session_id


def _session_id_from_file(content: str) -> str:
    return _session_ref_from_file(content)[1]


def _evidence_labels(dataset: Path, dev_ids: set[str]) -> dict[str, set[str]]:
    # This is the scoring boundary. The store, reader and seeding code never
    # receive gold answers or evidence IDs.
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    return {
        item["question_id"]: set(item["answer_session_ids"])
        for item in raw
        if item["question_id"] in dev_ids
    }


async def evaluate(
    dataset: Path, manifest: Path, *, limit: int | None = None
) -> dict[str, Any]:
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    rebuilt = make_split(
        dataset, seed=saved["seed"], dev_fraction=saved["dev_fraction"]
    )
    if saved != rebuilt:
        raise ValueError("dataset or split manifest changed; test partition is sealed")
    dev_ids = saved["dev_question_ids"][:limit]
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not dev_ids:
        raise ValueError("no development cases selected")
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    selected = {
        item["question_id"]: {
            "question": item["question"],
            "question_type": item["question_type"],
            "haystack_sessions": item["haystack_sessions"],
            "haystack_session_ids": item["haystack_session_ids"],
            "haystack_dates": item["haystack_dates"],
        }
        for item in raw
        if item["question_id"] in dev_ids
    }
    del raw
    if len(selected) != len(dev_ids):
        raise ValueError("development case missing from dataset")
    labels = _evidence_labels(dataset, set(dev_ids))
    records: list[dict[str, Any]] = []
    indexed_sessions = 0
    with tempfile.TemporaryDirectory(prefix="strands-lmem-dev-") as scratch:
        root = Path(scratch)
        for position, question_id in enumerate(dev_ids, start=1):
            source = selected[question_id]
            sessions = source["haystack_sessions"]
            session_ids = source["haystack_session_ids"]
            dates = source["haystack_dates"]
            if not len(sessions) == len(session_ids) == len(dates):
                raise ValueError(f"misaligned sessions for {question_id}")
            # Only these fields reach the retrieval systems. Gold answer and
            # evidence labels remain in the scorer above.
            question = source["question"]
            kind = source["question_type"]
            ours = LanceDBMemoryStore(uri=root / "lancedb", tenant_id=question_id)
            file_store = FileMemoryStore(
                name="file",
                storage=LocalFileStorage(str(root / "file" / question_id)),
            )
            test_store = TestMemoryStore(name="test", persist=False)
            for ordinal, (session_id, date, turns) in enumerate(
                zip(session_ids, dates, sessions, strict=True)
            ):
                text = _session_text(session_id, date, turns, ordinal)
                await ours.add(
                    text,
                    {
                        "session_id": session_id,
                        "session_ordinal": ordinal,
                        "idempotency_key": f"{ordinal}:{session_id}",
                    },
                )
                await file_store.add(text)
                await test_store.add(
                    text, {"session_id": session_id, "session_ordinal": ordinal}
                )
                indexed_sessions += 1
            for name, store in (
                ("lancedb_fts", ours),
                ("file", file_store),
                ("test", test_store),
            ):
                start = time.perf_counter()
                entries = await store.search(question, {"max_search_results": 10})
                elapsed_ms = (time.perf_counter() - start) * 1000
                if name == "file":
                    refs = [_session_ref_from_file(entry.content) for entry in entries]
                    retrieved_ordinals = [ordinal for ordinal, _ in refs]
                    retrieved_ids = [session_id for _, session_id in refs]
                else:
                    retrieved_ids = [
                        str(entry.metadata["session_id"]) for entry in entries
                    ]
                    retrieved_ordinals = [
                        int(entry.metadata["session_ordinal"]) for entry in entries
                    ]
                evidence = labels[question_id]
                rank = next(
                    (
                        index
                        for index, value in enumerate(retrieved_ids, 1)
                        if value in evidence
                    ),
                    None,
                )
                records.append(
                    {
                        "question_id": question_id,
                        "question_type": kind,
                        "system": name,
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
            print(f"completed {position}/{len(dev_ids)}", flush=True)
    summary: dict[str, Any] = {}
    for name in ("lancedb_fts", "file", "test"):
        rows = [row for row in records if row["system"] == name]
        latencies = sorted(row["search_latency_ms"] for row in rows)
        summary[name] = {
            f"hit_at_{k}": sum(
                row["first_evidence_rank"] is not None
                and row["first_evidence_rank"] <= k
                for row in rows
            )
            / len(rows)
            for k in (3, 5, 10)
        }
        summary[name]["p50_search_ms"] = statistics.median(latencies)
        summary[name]["p95_search_ms"] = latencies[math.ceil(len(latencies) * 0.95) - 1]
    return {
        "phase": "dev_retrieval_only",
        "dataset_sha256": sha256_file(dataset),
        "split_sha256": sha256_file(manifest),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": saved["seed"],
        "question_count": len(dev_ids),
        "indexed_sessions": indexed_sessions,
        "model_id": None,
        "prompt_sha256": None,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.dataset, args.manifest, limit=args.limit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"question_count": report["question_count"], "summary": report["summary"]}
        )
    )


if __name__ == "__main__":
    main()
