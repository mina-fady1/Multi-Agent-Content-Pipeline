"""Runtime configuration and input validation.

Configuration is deliberately data driven: changing the voice, rubric, prompts or
post type files does not require changing Python code.  The helpers in this module
are small and side effect free, which makes them useful from both CLIs and tests.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from .models import BrandVoice, KBItem, Plan, PostType, Rubric


class ConfigError(ValueError):
    """Raised when an input/configuration file cannot be validated."""


def _env_path(root: Path, name: str, default: str | Path) -> Path:
    raw = os.getenv(name)
    path = Path(raw) if raw else Path(default)
    return path if path.is_absolute() else root / path


def _front_matter(path: str | Path) -> tuple[dict[str, Any], str]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Unable to read configuration {path}: {exc}") from exc
    # Accept the usual --- YAML front matter delimiter.  A document with no
    # front matter remains useful as prose, with an empty machine section.
    match = re.match(r"^\s*---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)(.*)$", text, re.S)
    if not match:
        return {}, text.strip()
    try:
        raw = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML front matter in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: front matter must be a YAML mapping")
    return raw, match.group(2).strip()


def _as_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigError(f"{name} must be a list of strings")
    return value


def load_brand_voice(path: str | Path) -> BrandVoice:
    values, prose = _front_matter(path)
    # Accept both the names used in the brief and a few convenient aliases.
    try:
        voice = BrandVoice(
            tone=str(values.get("tone", "clear, practical and calm")),
            banned_phrases=_as_list(values.get("banned_phrases", values.get("banned_words")), "banned_phrases"),
            banned_characters=_as_list(values.get("banned_characters", values.get("banned_chars")), "banned_characters"),
            disallow_hashtags=bool(values.get("disallow_hashtags", True)),
            approved_company_facts=_as_list(values.get("approved_company_facts", values.get("approved_facts")), "approved_company_facts"),
            allowed_urls=_as_list(values.get("allowed_urls", values.get("exact_allowed_urls")), "allowed_urls"),
            max_tweet_length=int(values.get("max_tweet_length", values.get("maximum_tweet_length", 280))),
            guidance=prose,
        )
    except (ValidationError, TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid brand voice {path}: {exc}") from exc
    return voice


def load_rubric(path: str | Path) -> Rubric:
    values, prose = _front_matter(path)
    criteria = values.get("criteria", [])
    if isinstance(criteria, dict):
        criteria = [dict(item, id=key) if isinstance(item, dict) else {"id": key, "weight": item}
                    for key, item in criteria.items()]
    try:
        rubric = Rubric(
            threshold=float(values.get("threshold", 3.0)),
            criteria=criteria,
            guidance=prose,
        )
    except (ValidationError, TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid rubric {path}: {exc}") from exc
    total = sum(c.weight for c in rubric.criteria)
    if total <= 0:
        raise ConfigError("Rubric must contain at least one criterion with positive weight")
    return rubric


def load_kb(path: str | Path, *, minimum: int = 0) -> list[KBItem]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Unable to read knowledge base {path}: {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("items", raw.get("knowledge_base", raw))
    if not isinstance(raw, list):
        raise ConfigError("kb.json must contain a list (or an 'items' list)")
    try:
        items = [KBItem.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise ConfigError(f"Invalid knowledge base: {exc}") from exc
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ConfigError("Knowledge-base IDs must be unique")
    if len(items) < minimum:
        raise ConfigError(f"Knowledge base requires at least {minimum} entries; found {len(items)}")
    return items


def load_plan(path: str | Path, post_types: Mapping[str, PostType] | None = None) -> Plan:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "slots" not in raw:
            raw = {"slots": raw.get("plan", raw)}
        plan = Plan.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ConfigError(f"Invalid plan {path}: {exc}") from exc
    if post_types:
        unknown = sorted({slot.type for slot in plan.slots} - set(post_types))
        if unknown:
            raise ConfigError(f"Unknown post type(s): {', '.join(unknown)}")
    counts = {post_type: sum(slot.type == post_type for slot in plan.slots) for post_type in ("tech-thread", "marketing")}
    if len(plan.slots) != 6 or counts["tech-thread"] != 3 or counts["marketing"] != 3:
        raise ConfigError("Weekly plan must contain exactly six slots: three tech-thread and three marketing")
    return plan


def load_post_types(directory: str | Path) -> dict[str, PostType]:
    result: dict[str, PostType] = {}
    directory = Path(directory)
    if not directory.exists():
        raise ConfigError(f"Post type directory does not exist: {directory}")
    for path in sorted(directory.glob("*.y*ml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if "prompt_path" not in raw and "writer_prompt" in raw:
                raw["prompt_path"] = raw.pop("writer_prompt")
            obj = PostType.model_validate(raw)
        except (OSError, yaml.YAMLError, ValidationError, TypeError) as exc:
            raise ConfigError(f"Invalid post type {path}: {exc}") from exc
        if obj.id in result:
            raise ConfigError(f"Duplicate post type ID: {obj.id}")
        result[obj.id] = obj
    if not result:
        raise ConfigError(f"No post type YAML files found in {directory}")
    return result


def apply_overrides(plan: Plan, overrides: Mapping[str, str]) -> Plan:
    """Return a copied plan with ``slot_id -> focus`` overrides applied."""
    known = {slot.id for slot in plan.slots}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ConfigError(f"Unknown slot ID in --set override: {', '.join(unknown)}")
    slots = [slot.model_copy(update={"focus": overrides[slot.id]}) if slot.id in overrides else slot.model_copy()
             for slot in plan.slots]
    return Plan(slots=slots)


def load_prompt(path: str | Path) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Unable to read prompt {path}: {exc}") from exc
    if not text.strip():
        raise ConfigError(f"Prompt is empty: {path}")
    return text


@dataclass(frozen=True)
class Config:
    brand_voice: BrandVoice
    rubric: Rubric
    kb: list[KBItem]
    plan: Plan
    post_types: dict[str, PostType]
    prompts_dir: Path | None = None
    pricing: dict[str, Any] | None = None

    # Friendly short aliases used by agents and callers.
    @property
    def voice(self) -> BrandVoice:
        return self.brand_voice

    def load_plan(self, path: str | Path) -> Plan:
        return load_plan(path, self.post_types)

    def hard_rules(self, draft: Any, research: Any, post_type: PostType):
        from .skills.hard_rules import check_hard_rules
        return check_hard_rules(draft, research, self.brand_voice, post_type)


def load_config(root: str | Path = ".", *, plan_path: str | Path | None = None,
                kb_path: str | Path | None = None) -> Config:
    root = Path(root)
    post_types = load_post_types(root / "post_types")
    voice = load_brand_voice(_env_path(root, "PIPELINE_BRAND_VOICE", "brand-voice.md"))
    rubric = load_rubric(_env_path(root, "PIPELINE_RUBRIC", "rubric.md"))
    kb = load_kb(kb_path or _env_path(root, "PIPELINE_KB", "kb.json"), minimum=15)
    plan = load_plan(plan_path or _env_path(root, "PIPELINE_PLAN", "plan.json"), post_types)
    if plan_path is None or Path(plan_path).resolve() == (root / "plan.json").resolve():
        counts = {post_type: sum(slot.type == post_type for slot in plan.slots) for post_type in post_types}
        if len(plan.slots) != 6 or counts.get("tech-thread", 0) != 3 or counts.get("marketing", 0) != 3:
            raise ConfigError("The weekly plan must contain exactly three tech-thread and three marketing slots")
    pricing_path = _env_path(root, "PIPELINE_PRICING", "config/model_pricing.json")
    pricing: dict[str, Any] | None = None
    if pricing_path.exists():
        try:
            value = json.loads(pricing_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                pricing = value
        except (OSError, json.JSONDecodeError):
            pricing = None
    return Config(voice, rubric, kb, plan, post_types, root / "prompts", pricing)


__all__ = [
    "Config", "ConfigError", "apply_overrides", "load_brand_voice", "load_config",
    "load_kb", "load_plan", "load_post_types", "load_prompt", "load_rubric",
]
