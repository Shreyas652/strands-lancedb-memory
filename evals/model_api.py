"""Resumable, bounded-cost model calls used only by the evaluation harness."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# USD per million text tokens, standard input/output, checked 2026-11-24.
# https://developers.openai.com/api/docs/models/gpt-4o-mini
# https://developers.openai.com/api/docs/models/gpt-4o
MODEL_PRICES = {
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4o-2024-08-06": (2.50, 10.00),
}
API_URL = "https://api.openai.com/v1/chat/completions"


class ModelAPIError(RuntimeError):
    """A model call failed; the API key and response body are never logged."""


class BudgetExceededError(RuntimeError):
    """The next model call could exceed the configured evaluation budget."""


def load_key(env_file: Path | None = None) -> str:
    if env_file is not None and env_file.exists():
        keys = []
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(".env contains a malformed assignment")
            name, value = line.split("=", 1)
            if name.strip() == "OPENAI_API_KEY":
                keys.append(value.strip())
        if len(keys) != 1 or not keys[0]:
            raise ValueError(".env needs exactly one nonempty OPENAI_API_KEY entry")
        return keys[0]
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError("set OPENAI_API_KEY in the process or an ignored .env file")
    return key


def request_hash(
    model: str,
    prompt: str,
    max_tokens: int,
    response_format: dict[str, Any] | None = None,
) -> str:
    details: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if response_format is not None:
        details["response_format"] = response_format
    canonical = json.dumps(details, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def billed_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_rate, output_rate = MODEL_PRICES[model]
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


@dataclass(frozen=True)
class CallResult:
    text: str
    response_model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    prompt_sha256: str
    request_sha256: str
    cached: bool


class ModelClient:
    """Calls pinned models and durably records each response for exact resume."""

    def __init__(self, *, key: str, ledger: Path, budget_usd: float = 20.0) -> None:
        if budget_usd <= 0 or budget_usd > 20:
            raise ValueError("budget_usd must be in (0, 20]")
        self._key = key
        self.ledger = ledger
        self.budget_usd = budget_usd
        self.records: dict[str, dict[str, Any]] = {}
        self.spent_usd = 0.0
        if ledger.exists():
            for line in ledger.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                digest = record["request_sha256"]
                if digest in self.records:
                    raise ValueError("duplicate request in model ledger")
                self.records[digest] = record
                self.spent_usd += float(record["cost_usd"])
        if self.spent_usd > budget_usd:
            raise BudgetExceededError("existing ledger already exceeds the budget")

    def _fetch(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "User-Agent": "strands-lancedb-memory-evaluation",
            },
        )
        for attempt in range(3):
            try:
                # Large structured replies may need longer to finish generating.
                timeout_seconds = 180 if max_tokens > 4096 else 60
                with urllib.request.urlopen(
                    request, timeout=timeout_seconds
                ) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise ModelAPIError(f"{model} returned HTTP {exc.code}") from None
            except (TimeoutError, urllib.error.URLError) as exc:
                if attempt == 2:
                    raise ModelAPIError(
                        f"{model} request failed after retries"
                    ) from exc
            time.sleep(2**attempt)
        raise AssertionError("unreachable retry state")

    def call(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> CallResult:
        if model not in MODEL_PRICES:
            raise ValueError("model price is not configured")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        digest = request_hash(model, prompt, max_tokens, response_format)
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if digest in self.records:
            record = self.records[digest]
            return CallResult(
                text=record["text"],
                response_model=record["response_model"],
                input_tokens=record["input_tokens"],
                output_tokens=record["output_tokens"],
                cost_usd=record["cost_usd"],
                prompt_sha256=prompt_sha256,
                request_sha256=digest,
                cached=True,
            )
        # UTF-8 byte length conservatively bounds text tokens. Add framing room.
        conservative_input_tokens = len(prompt.encode("utf-8")) + 32
        reserve = billed_cost(model, conservative_input_tokens, max_tokens)
        if self.spent_usd + reserve > self.budget_usd:
            raise BudgetExceededError("next call could exceed the configured budget")
        response = (
            self._fetch(model, prompt, max_tokens)
            if response_format is None
            else self._fetch(model, prompt, max_tokens, response_format)
        )
        try:
            response_model = response["model"]
            if response_model != model:
                raise ValueError("response model differs from the pinned request")
            text = response["choices"][0]["message"]["content"].strip()
            usage = response["usage"]
            input_tokens = int(usage["prompt_tokens"])
            output_tokens = int(usage["completion_tokens"])
            if not text or input_tokens < 0 or output_tokens < 0:
                raise ValueError("empty or invalid model response")
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise ModelAPIError("model response was incomplete") from exc
        cost = billed_cost(model, input_tokens, output_tokens)
        record = {
            "request_sha256": digest,
            "prompt_sha256": prompt_sha256,
            "model": model,
            "response_model": response_model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        }
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records[digest] = record
        self.spent_usd += cost
        return CallResult(
            text,
            response_model,
            input_tokens,
            output_tokens,
            cost,
            prompt_sha256,
            digest,
            False,
        )
