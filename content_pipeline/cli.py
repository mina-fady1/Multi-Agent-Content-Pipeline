"""Command-line entry points for week and single-piece runs."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from .config import load_config
from .orchestrator import run_pipeline


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _overrides(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"--set must use slot_id=topic, got: {raw}")
        key, value = raw.split("=", 1)
        if not key.strip() or not value.strip():
            raise ValueError(f"--set must use a non-empty slot ID and topic, got: {raw}")
        result[key.strip()] = value.strip()
    return result


def _build_parser(piece: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SanadSquad grounded content pipeline")
    if piece:
        parser.add_argument("--type", required=True, dest="piece_type")
        parser.add_argument("--topic", required=True)
    else:
        parser.add_argument("--plan", default="plan.json")
        parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", choices=["demo", "live"], default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _execute(args: argparse.Namespace, *, piece: bool) -> int:
    root = _root()
    if load_dotenv:
        load_dotenv(root / ".env")
    config = load_config(root)
    provider_name = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider_name != "gemini":
        raise SystemExit("Only Gemini is supported. Set LLM_PROVIDER=gemini.")
    configured_key = os.getenv("GEMINI_API_KEY")
    mode = args.mode or ("live" if configured_key else None)
    if mode is None:
        raise SystemExit(f"No {provider_name.upper()} API key found. Pass --mode demo for an offline deterministic run.")
    if mode == "live" and not configured_key:
        raise SystemExit(f"Live mode with LLM_PROVIDER={provider_name} requires its API key. Pass --mode demo for offline execution.")
    if piece:
        if args.piece_type not in config.post_types:
            raise SystemExit(f"Unknown post type: {args.piece_type}")
        from .models import Plan, PlanSlot
        plan = Plan(slots=[PlanSlot(id="piece_1", type=args.piece_type, focus=args.topic)])
        output = Path(args.output or (root / "output" / "piece-run"))
        overrides = None
    else:
        plan = config.plan if args.plan == "plan.json" else config.load_plan(Path(args.plan))
        output = Path(args.output or (root / "output" / "week-run"))
        overrides = _overrides(args.set)
    summary = asyncio.run(
        run_pipeline(
            config,
            plan,
            mode=mode,
            output=output if output.is_absolute() else root / output,
            overrides=overrides,
            max_concurrency=args.max_concurrency or int(os.getenv("MAX_CONCURRENT_REQUESTS", "2")),
            seed=args.seed,
        )
    )
    return 0 if summary.dropped == 0 else 2


def week_main() -> int:
    args = _build_parser().parse_args()
    try:
        return _execute(args, piece=False)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise SystemExit(str(exc)) from exc


def piece_main() -> int:
    args = _build_parser(piece=True).parse_args()
    try:
        return _execute(args, piece=True)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        raise SystemExit(str(exc)) from exc

