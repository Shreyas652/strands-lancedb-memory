"""Reject a GitHub release tag that disagrees with the package version."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    expected = f"v{metadata['project']['version']}"
    actual = os.environ.get("RELEASE_TAG")
    if actual != expected:
        raise SystemExit(f"release tag {actual!r} does not match {expected!r}")
    print(f"Release tag {actual} matches pyproject.toml.")


if __name__ == "__main__":
    main()
