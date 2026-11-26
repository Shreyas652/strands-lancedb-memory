"""Check every README metric row against its committed JSON source."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = re.compile(r"^results/[A-Za-z0-9_./-]+\.json#/[-A-Za-z0-9_./:]+$")
RESULT_LINK = re.compile(
    r"^\[`(?P<reference>results/[A-Za-z0-9_./-]+\.json#/[-A-Za-z0-9_./:]+)`\]"
    r"\(https://github\.com/Shreyas652/strands-lancedb-memory/blob/main/"
    r"(?P<path>results/[A-Za-z0-9_./-]+\.json)\)$"
)
PAIRED_REFERENCES = {
    "/fts_only_correct",
    "/file_only_correct",
    "/exact_mcnemar_two_sided_p",
}


def _get(report: Any, pointer: str) -> float:
    value = report
    for key in pointer.strip("/").split("/"):
        value = value[key.replace("~1", "/").replace("~0", "~")]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("report pointer must resolve to a number")
    return value


def _check(display: str, actual: float) -> None:
    is_percent = display.endswith("%")
    shown = display.removesuffix("%")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", shown):
        raise ValueError(f"invalid displayed number: {display}")
    decimals = len(shown.partition(".")[2]) if "." in shown else 0
    multiplier = 100 if is_percent else 1
    expected = f"{Decimal(str(actual)) * multiplier:.{decimals}f}"
    if shown != expected:
        raise ValueError(
            f"README says {display}, report rounds to {expected}{'%' if is_percent else ''}"
        )


def _reference_from_link(cell: str) -> str:
    match = RESULT_LINK.fullmatch(cell)
    if match is None:
        raise ValueError(f"invalid result link: {cell}")
    reference = match["reference"]
    if not SOURCE.fullmatch(reference) or match["path"] != reference.split("#", 1)[0]:
        raise ValueError(f"result link does not match its source: {cell}")
    return reference


def main() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    checked = 0
    paired_references: set[str] = set()
    for line in readme.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or "results/" not in cells[2]:
            continue
        metric, display, link = cells
        del metric
        reference = _reference_from_link(link)
        file_name, pointer = reference.split("#", 1)
        if file_name == "results/paired_significance.json":
            paired_references.add(pointer)
        result_file = (ROOT / file_name).resolve()
        if not result_file.is_relative_to(ROOT / "results"):
            raise ValueError("result reference escapes results/")
        report = json.loads(result_file.read_text(encoding="utf-8"))
        _check(display, _get(report, pointer))
        checked += 1
    if missing := PAIRED_REFERENCES - paired_references:
        raise ValueError(
            f"README is missing paired significance metrics: {sorted(missing)}"
        )
    if not checked:
        raise ValueError("README has no result-linked metric rows")
    print(f"Verified {checked} README metrics against committed JSON results.")


if __name__ == "__main__":
    main()
