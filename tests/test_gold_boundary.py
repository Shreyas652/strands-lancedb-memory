"""Prevent production store code from importing benchmark scoring modules."""

import ast
from pathlib import Path


def test_package_does_not_import_evaluation_or_gold_code():
    package = Path(__file__).resolve().parents[1] / "src" / "strands_lancedb_memory"
    for source in package.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        modules = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        modules.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            name == "evals" or name.startswith("evals.") or "scoring" in name
            for name in modules
        ), source
