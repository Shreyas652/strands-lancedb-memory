"""Fail if source or report files that Git could add contain credential shapes.

This intentionally prints only the path, never the matched value. Ignored
local files such as `.env` and `data/` are outside the PR surface.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

SECRET = re.compile(
    rb"(?:sk-proj-|sk-)[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}"
)


def main() -> None:
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    )
    suspicious = []
    for raw in paths.split(b"\0"):
        if not raw:
            continue
        path = Path(os.fsdecode(raw))
        if path.is_file() and SECRET.search(path.read_bytes()):
            suspicious.append(str(path))
    if suspicious:
        raise SystemExit("Possible credential in: " + ", ".join(sorted(suspicious)))
    print("No credential patterns found in PR-visible files.")


if __name__ == "__main__":
    main()
