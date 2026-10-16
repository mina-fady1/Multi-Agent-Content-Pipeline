"""Deterministic branded visual specification and Pillow renderer."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ..config import load_prompt
from ..models import DraftPiece, ResearchPacket, VisualSpec
from ..providers.base import LLMProvider
from ..telemetry import record_provider_usage


class VisualAgent:
    def __init__(self, config: Any | None = None, *, provider: LLMProvider | None = None,
                 prompt_path: str | Path | None = None, ledger: Any = None, mode: str = "demo", **_: Any):
        if config is not None and prompt_path is None and getattr(config, "prompts_dir", None):
            prompt_path = Path(config.prompts_dir) / "visual.md"
        if provider is None and mode == "live" and config is not None:
            from ..providers.gemini_provider import GeminiProvider
            provider = GeminiProvider(pricing=getattr(config, "pricing", None))
        self.provider = provider
        self.ledger = ledger
        self.prompt = load_prompt(prompt_path) if prompt_path else ""

    def spec(self, draft: DraftPiece, research: ResearchPacket) -> VisualSpec:
        first = research.evidence[0]
        return VisualSpec(
            headline=draft.topic[:100],
            key_point=first.statement[:300],
            evidence_ids=[first.id],
            alt_text=f"Educational smart-contract security card about {draft.topic}",
        )

    def render(self, slot_or_draft: Any, draft_or_research: Any, research_or_output: Any,
               output_dir: str | Path | None = None, *, attempt: int = 1,
               spec_override: VisualSpec | None = None) -> tuple[VisualSpec, str]:
        """Render ``(slot, draft, research, run_dir)`` or ``(draft, research, run_dir)``."""
        if output_dir is None:
            draft, research, output_dir = slot_or_draft, draft_or_research, research_or_output
        else:
            draft, research = draft_or_research, research_or_output
        visual_spec = spec_override or self.spec(draft, research)
        out = Path(output_dir)
        # Callers may pass the run directory; keep images and specs separated.
        images_dir = out / "images"
        specs_dir = out / "image_specs"
        images_dir.mkdir(parents=True, exist_ok=True)
        specs_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", draft.slot_id).strip("_") or "piece"
        image_path = images_dir / f"{safe}.png"
        spec_path = specs_dir / f"{safe}.json"
        spec_path.write_text(json.dumps(visual_spec.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

        image = Image.new("RGB", (1200, 675), (12, 27, 45))
        draw = ImageDraw.Draw(image)
        # Degrade gracefully when optional system fonts are unavailable.
        try:
            title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
            body_font = ImageFont.truetype("DejaVuSans.ttf", 28)
            small_font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except OSError:
            title_font = body_font = small_font = ImageFont.load_default()
        draw.rounded_rectangle((55, 50, 1145, 625), radius=24, fill=(20, 52, 78), outline=(64, 184, 196), width=3)
        draw.text((95, 95), "SANADSQUAD  |  SECURITY NOTE", font=small_font, fill=(100, 220, 214))
        # Pillow's multiline_text handles long evidence without inventing facts.
        # Headline is bounded by the VisualSpec contract; wrap on rough word
        # boundaries so it remains legible with fallback fonts too.
        headline_words, headline_lines, headline_line = visual_spec.headline.split(), [], ""
        for word in headline_words:
            candidate = f"{headline_line} {word}".strip()
            if draw.textlength(candidate, font=title_font) > 1000 and headline_line:
                headline_lines.append(headline_line)
                headline_line = word
            else:
                headline_line = candidate
        if headline_line:
            headline_lines.append(headline_line)
        draw.multiline_text((95, 155), "\n".join(headline_lines[:2]), font=title_font, fill=(245, 249, 250), spacing=8)
        body = visual_spec.key_point
        # Simple wrapping based on measured width.
        words, lines, line = body.split(), [], ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if draw.textlength(candidate, font=body_font) > 980 and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        draw.multiline_text((95, 360), "\n".join(lines[:5]), font=body_font, fill=(218, 230, 235), spacing=10)
        draw.text((95, 575), "Grounded educational content | review assumptions before deployment", font=small_font,
                  fill=(151, 184, 197))
        image.save(image_path, format="PNG", optimize=True)
        return visual_spec, str(image_path)

    async def render_async(self, draft: DraftPiece, research: ResearchPacket,
                           output_dir: str | Path, *, attempt: int = 1) -> tuple[VisualSpec, str | None]:
        """Create a Gemini visual spec and image with ordered model fallback."""
        if self.provider is None or getattr(self.provider, "mode", "demo") != "live":
            return self.render(draft, research, output_dir, attempt=attempt)
        try:
            response = await self.provider.complete(
                "Create a concise, factual visual specification for the accepted piece.",
                system=self.prompt,
                response_model=VisualSpec,
                context={"agent": "visual", "draft": draft.model_dump(), "research": research.model_dump()},
            )
            candidate = response.data
            if self.ledger is not None:
                record_provider_usage(self.ledger, agent="visual", slot_id=draft.slot_id,
                                      attempt=attempt, result_or_error=response, mode="live",
                                      fallback_model=getattr(self.provider, "model", "unknown"))
        except Exception as exc:
            if self.ledger is not None:
                record_provider_usage(self.ledger, agent="visual", slot_id=draft.slot_id,
                                      attempt=attempt, result_or_error=exc, mode="live",
                                      fallback_model=getattr(self.provider, "model", "unknown"))
            # Visual metadata is not allowed to block an otherwise accepted
            # piece. Build a deterministic spec from verified evidence when
            # the optional visual-spec call is unavailable.
            candidate = self.spec(draft, research)
        allowed = {fact.id for fact in research.evidence}
        if not candidate.evidence_ids or not set(candidate.evidence_ids) <= allowed:
            candidate = self.spec(draft, research)
        out = Path(output_dir)
        images_dir = out / "images"
        specs_dir = out / "image_specs"
        images_dir.mkdir(parents=True, exist_ok=True)
        specs_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", draft.slot_id).strip("_") or "piece"
        image_path = images_dir / f"{safe}.jpg"
        # Persist the spec before attempting image generation so a provider
        # outage still leaves the required visual handoff artifact.
        (specs_dir / f"{safe}.json").write_text(
            json.dumps(candidate.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        image_prompt = (
            f"Create a professional 16:9 educational image for a smart-contract security post. "
            f"Topic: {candidate.headline}. Key point: {candidate.key_point}. "
            "Use a clean dark navy and teal technical illustration style. Do not add logos, "
            "unsupported claims, or readable paragraphs of text."
        )
        configured = os.getenv("GEMINI_IMAGE_MODELS", "gemini-nano-banana-2.1,gemini-3.1-flash-lite-image,gemini-3.1-flash-image")
        models = [model.strip() for model in configured.split(",") if model.strip()]
        timeout = float(os.getenv("GEMINI_IMAGE_TIMEOUT", "90"))
        failures: list[str] = []
        for image_model in models:
            try:
                response = await self.provider.generate_image(
                    image_prompt, image_path, model=image_model, timeout=timeout
                )
                if self.ledger is not None:
                    record_provider_usage(self.ledger, agent="visual", slot_id=draft.slot_id,
                                          attempt=attempt, result_or_error=response, mode="live",
                                          fallback_model=image_model)
                return candidate, str(image_path)
            except Exception as exc:
                if self.ledger is not None:
                    record_provider_usage(self.ledger, agent="visual", slot_id=draft.slot_id,
                                          attempt=attempt, result_or_error=exc, mode="live",
                                          fallback_model=image_model)
                detail = str(exc).strip() or exc.__class__.__name__
                failures.append(f"{image_model}: {detail[:240]}")
                if "limit: 0" in detail.lower() or "upgrade your tier" in detail.lower():
                    break
        if os.getenv("GEMINI_IMAGE_FALLBACK", "").strip().lower() == "pillow":
            return self.render(draft, research, output_dir, attempt=attempt, spec_override=candidate)
        # The task contract permits a written visual spec when no image API is
        # available. The orchestrator will export the accepted piece with an
        # empty Image field while retaining this spec for a later renderer.
        return candidate, None

    def generate(self, *args: Any, **kwargs: Any) -> tuple[VisualSpec, str]:
        return self.render(*args, **kwargs)


__all__ = ["VisualAgent"]

