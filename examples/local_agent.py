"""Run a real Strands Agent with local LanceDB memory and an offline stub model.

This example needs no cloud credentials. Replace ``StubModel`` with your model
to make the Agent generate answers from recalled memories.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from strands import Agent
from strands.memory import MemoryManager
from strands.models import Model

from strands_lancedb_memory import LanceDBMemoryStore


class StubModel(Model):
    def update_config(self, **model_config: Any) -> None:
        del model_config

    def get_config(self) -> dict[str, str]:
        return {"model_id": "offline-example-stub"}

    async def structured_output(self, *args: Any, **kwargs: Any):
        del args, kwargs
        if False:
            yield {}

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        del messages, tool_specs, system_prompt, kwargs
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": "Offline stub response."}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="strands-lmem-example-") as temporary:
        store = LanceDBMemoryStore(
            uri=Path(temporary) / "memory",
            tenant_id="example-user",
            ingestion="raw",
            agent_run_id="this-example-run",
        )
        asyncio.run(store.add("The user prefers green tea."))
        agent = Agent(
            model=StubModel(),
            callback_handler=None,
            memory_manager=MemoryManager(stores=[store]),
        )
        agent("What does the user prefer?")
        recalled = asyncio.run(store.search("green tea"))
        assert recalled and "green tea" in recalled[0].content
        print("Agent and local LanceDB memory worked offline.")


if __name__ == "__main__":
    main()
