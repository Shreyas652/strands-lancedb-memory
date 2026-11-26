"""Fetch the exact public LongMemEval-S cleaned file used by the reports."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_s_cleaned.json"
)
EXPECTED_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(output: Path) -> None:
    if output.exists():
        if sha256_file(output) != EXPECTED_SHA256:
            raise ValueError("existing dataset differs from the benchmark fingerprint")
        print(f"Verified {output} ({EXPECTED_SHA256})")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError("unfinished dataset download exists; inspect it first")
    try:
        with (
            urllib.request.urlopen(DATASET_URL, timeout=60) as response,
            temporary.open("xb") as target,
        ):
            digest = hashlib.sha256()
            for block in iter(lambda: response.read(1024 * 1024), b""):
                target.write(block)
                digest.update(block)
        if digest.hexdigest() != EXPECTED_SHA256:
            raise ValueError(
                "downloaded dataset differs from the benchmark fingerprint"
            )
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Saved {output} ({EXPECTED_SHA256})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/longmemeval_s_cleaned.json")
    )
    args = parser.parse_args()
    fetch(args.output)


if __name__ == "__main__":
    main()
