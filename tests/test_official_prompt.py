"""Pin the copied official judge prompt text without calling a model."""

import hashlib
import json
from pathlib import Path

from evals.official_prompt import get_anscheck_prompt


def test_official_prompt_matches_verified_fingerprints():
    path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "official_prompt_verification.json"
    )
    verified = json.loads(path.read_text(encoding="utf-8"))
    assert verified["all_match"] is True
    for kind, expected in verified["prompt_sha256_by_type"].items():
        prompt = get_anscheck_prompt(kind, "question", "reference", "candidate")
        assert hashlib.sha256(prompt.encode()).hexdigest() == expected
    abstention = get_anscheck_prompt(
        "single-session-user", "question", "reference", "candidate", abstention=True
    )
    assert (
        hashlib.sha256(abstention.encode()).hexdigest()
        == verified["abstention_prompt_sha256"]
    )
