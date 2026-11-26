"""Verify the vendored judge prompt against a pinned LongMemEval source checkout."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from evals.official_prompt import get_anscheck_prompt

EXPECTED_SOURCE_SHA256 = (
    "5085eb9ae08b91c6c4f97c6bea9b8c2d6e55f9ec72c27734cafa0a04f00a430a"
)


def verify(source: Path) -> dict[str, Any]:
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError("official source differs from the pinned checkout")
    original = source.read_text(encoding="utf-8")
    node = next(
        item
        for item in ast.parse(original).body
        if isinstance(item, ast.FunctionDef) and item.name == "get_anscheck_prompt"
    )
    only_function = ast.Module(body=[node], type_ignores=[])
    namespace: dict[str, Any] = {}
    exec(compile(only_function, str(source), "exec"), namespace)  # noqa: S102
    original_function = namespace["get_anscheck_prompt"]
    types = (
        "single-session-user",
        "single-session-assistant",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
        "single-session-preference",
    )
    digests = {}
    for kind in types:
        prompt = get_anscheck_prompt(kind, "question", "reference", "candidate")
        if prompt != original_function(kind, "question", "reference", "candidate"):
            raise AssertionError(f"vendored prompt differs for {kind}")
        digests[kind] = hashlib.sha256(prompt.encode()).hexdigest()
    abstention = get_anscheck_prompt(
        "single-session-user", "question", "reference", "candidate", abstention=True
    )
    if abstention != original_function(
        "single-session-user", "question", "reference", "candidate", abstention=True
    ):
        raise AssertionError("vendored abstention prompt differs")
    return {
        "source_commit": "9e0b455f4ef0e2ab8f2e582289761153549043fc",
        "source_sha256": source_sha256,
        "prompt_sha256_by_type": digests,
        "abstention_prompt_sha256": hashlib.sha256(abstention.encode()).hexdigest(),
        "all_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("official prompt variants matched:", len(result["prompt_sha256_by_type"]) + 1)


if __name__ == "__main__":
    main()
