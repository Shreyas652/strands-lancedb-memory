"""Exercise an installed wheel without importing the repository source tree."""

from __future__ import annotations

import asyncio
import inspect
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

from strands_lancedb_memory import LanceDBMemoryStore


def main() -> None:
    if version("strands-lancedb-memory") != "0.1.0":
        raise RuntimeError("installed package version is not 0.1.0")
    module_path = Path(inspect.getfile(LanceDBMemoryStore)).resolve()
    if not module_path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(
            f"package was imported outside the fresh venv: {module_path}"
        )
    with tempfile.TemporaryDirectory(prefix="strands-lmem-smoke-") as temporary:
        store = LanceDBMemoryStore(
            uri=Path(temporary) / "memory",
            tenant_id="smoke-user",
        )
        asyncio.run(store.add("The user prefers green tea."))
        matches = asyncio.run(store.search("green tea"))
        if not any("green tea" in entry.content for entry in matches):
            raise RuntimeError("installed store did not retrieve the added fact")
    print("Installed package imported and retrieved one fact offline.")


if __name__ == "__main__":
    main()
