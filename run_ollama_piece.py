"""Run one pipeline piece through a local Ollama server.

This is intentionally isolated from ``content_pipeline``. It injects a small
Ollama adapter into the existing runner at runtime and does not change the
Gemini production path or project dependencies.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from content_pipeline.config import load_config
from content_pipeline.models import Plan, PlanSlot
from content_pipeline.orchestrator import PipelineRunner
from content_pipeline.providers.base import ProviderResponse, ProviderUsage

T = TypeVar("T", bound=BaseModel)


class OllamaAdapter:
    mode = "live"

    def __init__(self, model: str, base_url: str, timeout: float, rubric_ids: list[str] | None = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rubric_ids = rubric_ids or []

    @staticmethod
    def _schema(model: type[BaseModel] | None) -> dict[str, Any] | str:
        return model.model_json_schema() if model is not None else "json"

    @staticmethod
    def _parse(content: str, response_model: type[T] | None) -> Any:
        text = content.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Ollama returned invalid JSON")
            value = json.loads(text[start:end + 1])
        if response_model is None:
            return value
        # Small local models sometimes express rubric scores as percentages
        # despite the prompt's 0-5 contract. Normalize that representation in
        # this Ollama-only adapter; the core contracts remain strict.
        if getattr(response_model, "__name__", "") == "ReviewResult" and isinstance(value, dict):
            aliases = {
                "grounding": "accuracy", "evidence_integrity": "accuracy",
                "evidence-integrity": "accuracy", "mapping_accuracy": "accuracy",
                "hard_rules": "accuracy", "hard-rules": "accuracy",
                "style": "voice", "style_and_formatting": "clarity",
            }
            for item in value.get("scores", []) or []:
                if isinstance(item, dict):
                    criterion = str(item.get("criterion", "")).strip()
                    item["criterion"] = aliases.get(criterion, criterion)
                    if isinstance(item.get("score"), (int, float)) and item["score"] > 5:
                        item["score"] = item["score"] / 20
            if isinstance(value.get("weighted_score"), (int, float)) and value["weighted_score"] > 5:
                value["weighted_score"] = value["weighted_score"] / 20
        try:
            return response_model.model_validate(value)
        except (ValidationError, TypeError, ValueError) as exc:
            raise ValueError(f"Ollama response failed schema validation: {exc}") from exc

    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/api/chat", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}: {exc}") from exc
        if not isinstance(result, dict) or result.get("error"):
            raise RuntimeError(f"Ollama error: {result.get('error', 'invalid response')}")
        return result

    async def complete(self, prompt: str, *, system: str | None = None,
                       response_model: type[T] | None = None,
                       context: dict[str, Any] | None = None,
                       temperature: float = 0, **_: Any) -> ProviderResponse:
        user = prompt
        if context:
            user += "\n\n<CONTEXT_JSON>\n" + json.dumps(context, default=str) + "\n</CONTEXT_JSON>"
        if context and context.get("agent") == "reviewer" and self.rubric_ids:
            user += (
                "\n\nFor the ReviewResult, use exactly these criterion IDs: "
                + ", ".join(self.rubric_ids)
                + ". Score every criterion from 0 to 5, never percentages."
            )
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": user})
        body = {"model": self.model, "messages": messages, "stream": False,
                "format": self._schema(response_model),
                # Qwen can place its reasoning in a separate field and leave
                # content empty for schema requests. The pipeline needs only
                # the validated answer, so disable that extra channel.
                "think": False,
                "options": {"temperature": temperature}}
        last: BaseException | None = None
        for attempt in range(3):
            try:
                result = await asyncio.to_thread(self._request, body)
                message = result.get("message") or {}
                content = message.get("content") if isinstance(message, Mapping) else None
                if not content and isinstance(message, Mapping):
                    content = message.get("thinking")
                if not content:
                    raise ValueError("Ollama returned an empty message")
                prompt_tokens = result.get("prompt_eval_count")
                output_tokens = result.get("eval_count")
                usage = ProviderUsage(
                    input_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
                    output_tokens=int(output_tokens) if output_tokens is not None else None,
                    cost_usd=0.0,
                )
                if (getattr(response_model, "__name__", "") == "ResearchPacket"
                        and context and context.get("verified_packet")):
                    # ResearchAgent deliberately treats model synthesis as
                    # advisory. Preserve the verified KB packet instead of
                    # allowing a small local model to rewrite source IDs.
                    data = response_model.model_validate(context["verified_packet"])
                else:
                    data = self._parse(str(content), response_model)
                return ProviderResponse(
                    data=data, usage=usage,
                    model=self.model, raw=result,
                )
            except Exception as exc:
                last = exc
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Ollama request failed after 3 attempts: {last}") from last

    async def aclose(self) -> None:
        return None


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one content piece through local Ollama")
    parser.add_argument("--type", required=True, dest="piece_type")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--output", default="output/ollama-piece")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3.5:4b-q4_K_M"))
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("OLLAMA_TIMEOUT", "180")))
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    # PipelineRunner's normal provider selector remains Gemini-only. This
    # process-local override keeps that production path untouched.
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["GEMINI_IMAGE_MODELS"] = ""
    os.environ["GEMINI_IMAGE_FALLBACK"] = "pillow"
    config = load_config(Path.cwd())
    adapter = OllamaAdapter(args.model, args.base_url, args.timeout,
                            rubric_ids=[criterion.id for criterion in config.rubric.criteria])
    runner = PipelineRunner(config, mode="live", output=Path(args.output).resolve(), max_concurrency=1)
    runner.provider = adapter
    runner.researcher.provider = adapter
    runner.reviewer.provider = adapter
    runner.visual.provider = adapter
    for writer in runner.writers.values():
        writer.provider = adapter
    try:
        summary = await runner.run(Plan(slots=[PlanSlot(id="piece_1", type=args.piece_type, focus=args.topic)]))
        return 0 if summary.dropped == 0 else 2
    finally:
        await runner.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run(_args())))
