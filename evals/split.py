"""Create a fixed, stratified LongMemEval dev split without using gold answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_digest(ids: list[str]) -> str:
    canonical = json.dumps(sorted(ids), separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def make_split(
    dataset: Path, *, seed: int, dev_fraction: float = 0.2
) -> dict[str, Any]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("dataset must be a nonempty list")
    by_type: dict[str, list[str]] = defaultdict(list)
    all_ids: set[str] = set()
    for item in raw:
        question_id = item["question_id"]
        question_type = item["question_type"]
        if not isinstance(question_id, str) or not isinstance(question_type, str):
            raise TypeError("question IDs and types must be strings")
        if question_id in all_ids:
            raise ValueError(f"duplicate question ID: {question_id}")
        all_ids.add(question_id)
        by_type[question_type].append(question_id)
    target = round(len(raw) * dev_fraction)
    quotas = {kind: len(ids) * dev_fraction for kind, ids in by_type.items()}
    counts = {kind: math.floor(quota) for kind, quota in quotas.items()}
    remaining = target - sum(counts.values())
    for kind in sorted(by_type, key=lambda value: (-(quotas[value] % 1), value))[
        :remaining
    ]:
        counts[kind] += 1
    rng = random.Random(seed)
    dev: list[str] = []
    test: list[str] = []
    for kind in sorted(by_type):
        ids = sorted(by_type[kind])
        rng.shuffle(ids)
        dev.extend(ids[: counts[kind]])
        test.extend(ids[counts[kind] :])
    return {
        "dataset_sha256": sha256_file(dataset),
        "seed": seed,
        "dev_fraction": dev_fraction,
        "case_count": len(raw),
        "dev_count": len(dev),
        "test_count": len(test),
        "dev_by_type": dict(sorted(counts.items())),
        "dev_question_ids": sorted(dev),
        "sealed_test_ids_sha256": _ids_digest(test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    manifest = make_split(args.dataset, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"saved {args.output}: {manifest['dev_count']} dev, {manifest['test_count']} sealed"
    )


if __name__ == "__main__":
    main()
