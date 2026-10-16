"""Run the existing six-slot weekly plan through local Ollama.

This is a local test harness. The production ``run_week.py`` Gemini path is
left unchanged; the Ollama adapter is injected only in this process.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from content_pipeline.config import load_config
from content_pipeline.orchestrator import PipelineRunner
from run_ollama_piece import OllamaAdapter


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the weekly plan through local Ollama")
    parser.add_argument("--plan", default="plan.json")
    parser.add_argument("--output", default="output/ollama-week")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3.5:4b-q4_K_M"))
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("OLLAMA_TIMEOUT", "180")))
    parser.add_argument("--max-concurrency", type=int, default=1)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    # Keep the normal provider selector untouched; this process replaces the
    # constructed provider instances before any model call is made.
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["GEMINI_IMAGE_MODELS"] = ""
    os.environ["GEMINI_IMAGE_FALLBACK"] = "pillow"
    root = Path.cwd()
    config = load_config(root, plan_path=args.plan)
    adapter = OllamaAdapter(
        args.model,
        args.base_url,
        args.timeout,
        rubric_ids=[criterion.id for criterion in config.rubric.criteria],
    )
    runner = PipelineRunner(
        config,
        mode="live",
        output=Path(args.output).resolve(),
        max_concurrency=max(1, args.max_concurrency),
    )
    runner.provider = adapter
    runner.researcher.provider = adapter
    runner.reviewer.provider = adapter
    runner.visual.provider = adapter
    for writer in runner.writers.values():
        writer.provider = adapter
    try:
        summary = await runner.run(config.plan)
        return 0 if summary.dropped == 0 else 2
    finally:
        await runner.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run(_args())))
