"""Exercise simultaneous local LanceDB writers from independent Python processes."""

import asyncio
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import lancedb

from strands_lancedb_memory import LanceDBMemoryStore
from strands_lancedb_memory.store import MemoryStorageError


def worker(uri: str, prefix: str) -> None:
    async def write() -> None:
        store = LanceDBMemoryStore(uri=uri, tenant_id="probe")
        for number in range(10):
            await store.add(f"shared word {prefix} {number}")

    asyncio.run(write())


def main(output: Path | None = None) -> None:
    with tempfile.TemporaryDirectory() as temp:
        uri = str(Path(temp) / "db")
        processes = [
            subprocess.Popen(
                [sys.executable, __file__, "worker", uri, prefix],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for prefix in ("first", "second")
        ]
        outcomes = []
        for process in processes:
            _, stderr = process.communicate(timeout=90)
            outcomes.append(
                {"exit_code": process.returncode, "error_tail": stderr[-600:]}
            )
        store = LanceDBMemoryStore(uri=uri, tenant_id="probe", writable=False)
        try:
            rows = asyncio.run(store.search("shared", {"max_search_results": 30}))
            count = len(rows)
        except MemoryStorageError as exc:
            count = None
            outcomes.append({"reader_error": repr(exc)})
        report = {
            "expected_count": 20,
            "observed_count": count,
            "writers": outcomes,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "lancedb": lancedb.__version__,
        }
        if output is not None:
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        main(
            Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--save" else None
        )
