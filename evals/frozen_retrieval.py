"""Run the sealed test retrieval exactly after a dev-only configuration freeze."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from strands.storage import LocalFileStorage
from strands.vended_memory_stores.file_memory_store import FileMemoryStore
from strands.vended_memory_stores.test_memory_store import TestMemoryStore

from evals.local_embedder import FastEmbedAdapter
from evals.retrieval_dev import _session_ref_from_file, _session_text
from evals.split import make_split, sha256_file
from evals.vector_dev import _seed_case
from strands_lancedb_memory import LanceDBMemoryStore


def sealed_ids(
    dataset: Path, manifest: Path, frozen: Path
) -> tuple[list[str], dict[str, Any]]:
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    if saved != make_split(
        dataset, seed=saved["seed"], dev_fraction=saved["dev_fraction"]
    ):
        raise ValueError("dataset or split manifest changed")
    config = json.loads(frozen.read_text(encoding="utf-8"))
    if (
        config["status"] != "FROZEN"
        or config["dataset_sha256"] != saved["dataset_sha256"]
        or config["manifest_sha256"] != sha256_file(manifest)
        or config["sealed_test_ids_sha256"] != saved["sealed_test_ids_sha256"]
        or config["runner_sha256"]
        != hashlib.sha256(
            (Path(__file__).parent / "freeze.py").read_bytes()
        ).hexdigest()
    ):
        raise ValueError(
            "frozen configuration is missing or differs from the sealed split"
        )
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    dev_ids = set(saved["dev_question_ids"])
    test_ids = sorted(
        item["question_id"] for item in raw if item["question_id"] not in dev_ids
    )
    digest = hashlib.sha256(
        json.dumps(test_ids, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        digest != saved["sealed_test_ids_sha256"]
        or len(test_ids) != saved["test_count"]
    ):
        raise ValueError("held-out test IDs differ from their sealed hash")
    return test_ids, config


async def _one_case(
    source: dict[str, Any],
    question_id: str,
    config: dict[str, Any],
    root: Path,
    embedder: FastEmbedAdapter | None,
    evidence: set[str],
) -> list[dict[str, Any]]:
    session_ids = source["haystack_session_ids"]
    dates = source["haystack_dates"]
    sessions = source["haystack_sessions"]
    if not len(session_ids) == len(dates) == len(sessions):
        raise ValueError("test case session arrays differ in length")
    mode = config["retrieval"]
    ours = LanceDBMemoryStore(
        uri=root / "lancedb",
        tenant_id=question_id,
        retrieval=mode,
        embedder=embedder,
        vector_weight=config["vector_weight"]
        if config["vector_weight"] is not None
        else 0.5,
    )
    if mode == "fts":
        for ordinal, (session_id, date, turns) in enumerate(
            zip(session_ids, dates, sessions, strict=True)
        ):
            await ours.add(
                _session_text(session_id, date, turns, ordinal),
                {
                    "session_id": session_id,
                    "session_ordinal": ordinal,
                    "idempotency_key": f"{ordinal}:{session_id}",
                },
            )
    else:
        if embedder is None:
            raise ValueError("frozen vector configuration needs cached model weights")
        await asyncio.to_thread(
            _seed_case, ours, dict(source, question_id=question_id), embedder
        )
    file_store = FileMemoryStore(
        name="file", storage=LocalFileStorage(str(root / "file" / question_id))
    )
    test_store = TestMemoryStore(name="test", persist=False)
    for ordinal, (session_id, date, turns) in enumerate(
        zip(session_ids, dates, sessions, strict=True)
    ):
        text = _session_text(session_id, date, turns, ordinal)
        await file_store.add(text)
        await test_store.add(
            text, {"session_id": session_id, "session_ordinal": ordinal}
        )
    records = []
    for name, store in (
        (config["system"], ours),
        ("file", file_store),
        ("test", test_store),
    ):
        started = time.perf_counter()
        entries = await store.search(source["question"], {"max_search_results": 10})
        elapsed = (time.perf_counter() - started) * 1000
        if name == "file":
            refs = [_session_ref_from_file(entry.content) for entry in entries]
            ordinals = [ordinal for ordinal, _ in refs]
            ids = [session_id for _, session_id in refs]
        else:
            ids = [str(entry.metadata["session_id"]) for entry in entries]
            ordinals = [int(entry.metadata["session_ordinal"]) for entry in entries]
        rank = next(
            (
                index
                for index, session_id in enumerate(ids, 1)
                if session_id in evidence
            ),
            None,
        )
        records.append(
            {
                "question_id": question_id,
                "question_type": source["question_type"],
                "system": name,
                "retrieved_session_ids": ids,
                "retrieved_session_ordinals": ordinals,
                "first_evidence_rank": rank,
                "search_latency_ms": elapsed,
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
    return records


async def evaluate(
    dataset: Path,
    manifest: Path,
    frozen: Path,
    cache_dir: Path | None,
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(
            "frozen test result already exists; do not rerun or retune"
        )
    test_ids, config = sealed_ids(dataset, manifest, frozen)
    model = None
    if config["retrieval"] != "fts":
        if cache_dir is None:
            raise ValueError("frozen vector configuration needs --model-cache")
        model = FastEmbedAdapter(cache_dir)
        if model.model_id != config["embedding_model_id"]:
            raise ValueError("cached embedding model differs from frozen configuration")
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
        if item["question_id"] in test_ids
    }
    # Evidence labels are retained only by this scorer, never passed to a store.
    labels = {
        item["question_id"]: set(item["answer_session_ids"])
        for item in raw
        if item["question_id"] in test_ids
    }
    del raw
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    freeze_hash = sha256_file(frozen)
    finished: dict[str, list[dict[str, Any]]] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            question_id = item["question_id"]
            if (
                item["runner_sha256"] != runner_hash
                or item["freeze_sha256"] != freeze_hash
                or question_id not in selected
                or question_id in finished
                or len(item["rows"]) != 3
                or {row["system"] for row in item["rows"]}
                != {config["system"], "file", "test"}
            ):
                raise ValueError("frozen retrieval checkpoint is stale or inconsistent")
            finished[question_id] = item["rows"]
    with tempfile.TemporaryDirectory(prefix="strands-lmem-frozen-") as temporary:
        root = Path(temporary)
        for position, question_id in enumerate(test_ids, start=1):
            if question_id in finished:
                continue
            rows = await _one_case(
                selected[question_id],
                question_id,
                config,
                root,
                model,
                labels[question_id],
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "question_id": question_id,
                            "runner_sha256": runner_hash,
                            "freeze_sha256": freeze_hash,
                            "rows": rows,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            finished[question_id] = rows
            print(f"completed frozen retrieval {position}/{len(test_ids)}", flush=True)
    records = [row for question_id in test_ids for row in finished[question_id]]
    summary = {}
    for system in (config["system"], "file", "test"):
        rows = [row for row in records if row["system"] == system]
        times = sorted(row["search_latency_ms"] for row in rows)
        summary[system] = {
            f"hit_at_{k}": sum(
                row["first_evidence_rank"] is not None
                and row["first_evidence_rank"] <= k
                for row in rows
            )
            / len(rows)
            for k in (3, 5, 10)
        }
        summary[system]["p50_search_ms"] = statistics.median(times)
        summary[system]["p95_search_ms"] = times[math.ceil(len(times) * 0.95) - 1]
    return {
        "phase": "frozen_test_retrieval",
        "question_count": len(test_ids),
        "dataset_sha256": sha256_file(dataset),
        "split_sha256": sha256_file(manifest),
        "freeze_sha256": freeze_hash,
        "runner_sha256": runner_hash,
        "api_cost_usd": 0,
        "api_calls": 0,
        "summary": summary,
        "per_question": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/frozen_retrieval_checkpoint.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(
        evaluate(
            args.dataset,
            args.manifest,
            args.frozen,
            args.model_cache,
            args.checkpoint,
            args.output,
        )
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
