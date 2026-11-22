"""Trace installed Strands 1.65.0 MemoryManager routing without network/model calls."""

import asyncio
import copy
import json
import tempfile
from pathlib import Path
from typing import Any

from strands import Agent
from strands.memory import MemoryManager
from strands.memory.extraction.triggers import InvocationTrigger
from strands.memory.extraction.types import ExtractionResult
from strands.models import Model
from strands.vended_memory_stores.test_memory_store import TestMemoryStore

CALLS: list[dict[str, Any]] = []


class StubModel(Model):
    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[list[dict]] = []

    def update_config(self, **model_config: Any) -> None:
        del model_config

    def get_config(self) -> dict[str, str]:
        return {"model_id": "phase0-stub"}

    async def structured_output(self, *args: Any, **kwargs: Any):
        del args, kwargs
        if False:
            yield {}

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        del tool_specs, system_prompt, kwargs
        self.seen.append(copy.deepcopy(messages))
        self.calls += 1
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": f"stub answer {self.calls}"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


class FixedExtractor:
    async def extract(self, messages, context=None):
        CALLS.append(
            {
                "method": "extract",
                "roles": [message["role"] for message in messages],
                "has_default_model": context is not None and context.default_model is not None,
            }
        )
        return [ExtractionResult(content="extracted durable fact")]


class LoggedAddStore(TestMemoryStore):
    async def search(self, query, options=None):
        CALLS.append({"store": self.name, "method": "search", "query": query, "options": options})
        return await super().search(query, options)

    async def add(self, content, metadata=None):
        CALLS.append({"store": self.name, "method": "add", "content": content, "metadata": metadata})
        return await super().add(content, metadata)


class LoggedRawStore(LoggedAddStore):
    async def add_messages(self, messages, context=None):
        CALLS.append(
            {
                "store": self.name,
                "method": "add_messages",
                "roles": [message["role"] for message in messages],
                "texts": [
                    block["text"]
                    for message in messages
                    for block in message["content"]
                    if "text" in block
                ],
                "sequence_numbers": None if context is None else context.sequence_numbers,
            }
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        add_store = LoggedAddStore(
            name="add-only",
            path=str(Path(tmp) / "add.json"),
            extraction={"trigger": [InvocationTrigger()], "extractor": FixedExtractor()},
        )
        raw_store = LoggedRawStore(
            name="raw-messages",
            path=str(Path(tmp) / "raw.json"),
            extraction={"trigger": [InvocationTrigger()]},
        )
        asyncio.run(add_store.add("User color is blue"))
        CALLS.clear()
        model = StubModel()
        manager = MemoryManager(stores=[add_store, raw_store], add_tool_config=True)
        agent = Agent(model=model, callback_handler=None, memory_manager=manager)
        agent("What color do I like?")
        agent("Tell me again")

        async def tools():
            search = next(tool for tool in manager.tools if tool.tool_name == "search_memory")
            add = next(tool for tool in manager.tools if tool.tool_name == "add_memory")
            await search(query="blue")
            await add(entries=["Explicit tool fact"])

        asyncio.run(tools())
        report = {"calls": CALLS, "model_calls": model.calls, "model_seen": model.seen}
        assert any(
            "<memory>" in block.get("text", "")
            for block in model.seen[0][-1]["content"]
        ), "seeded memory was not injected into the first model call"
        output = Path("results/phase0_lifecycle.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, default=str, indent=2) + "\n", encoding="utf-8")
        print(f"saved {output}: {len(CALLS)} store/extractor calls, {model.calls} model calls")


if __name__ == "__main__":
    main()
