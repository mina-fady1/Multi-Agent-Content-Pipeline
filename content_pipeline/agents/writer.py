"""Configurable technical-thread and marketing writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import load_prompt
from ..models import BrandVoice, DraftPiece, PlanSlot, PostType, ResearchPacket, Tweet, ClaimMapping, UsageRecord
from ..providers.base import LLMProvider, coerce_response
from ..telemetry import record_provider_usage


def _slot_value(slot: PlanSlot | dict[str, Any], key: str, default: Any = None) -> Any:
    if hasattr(slot, key):
        return getattr(slot, key, default)
    return slot.get(key, default) if isinstance(slot, dict) else default


class WriterAgent:
    """Base writer; subclasses select the post-type-specific fixture style."""

    kind = "tech-thread"

    def __init__(self, post_type: PostType | Any | None = None, *, provider: LLMProvider | None = None,
                 voice: BrandVoice | None = None, prompt_path: str | Path | None = None, ledger: Any = None,
                 mode: str = "demo", config: Any | None = None, kind: str | None = None, **_: Any):
        self.kind = kind or type(self).kind
        # Accept either a PostType (small unit tests) or the complete Config
        # object (application dependency injection).
        if config is None and hasattr(post_type, "post_types"):
            config = post_type
        if config is not None:
            post_type = config.post_types.get(self.kind)
            voice = voice or config.voice
            prompt_path = prompt_path or (Path(config.prompts_dir) / f"{self.kind}.md" if getattr(config, "prompts_dir", None) else None)
        if post_type is not None and not isinstance(post_type, PostType):
            post_type = None
        if provider is None and mode == "live":
            from ..providers.gemini_provider import GeminiProvider
            provider = GeminiProvider(pricing=getattr(config, "pricing", None))
        self.post_type = post_type
        self.provider = provider
        self.voice = voice
        self.prompt = load_prompt(prompt_path) if prompt_path else ""
        self.ledger = ledger
        self.mode = mode

    def _render_prompt(self, research: ResearchPacket) -> str:
        """Render the small set of documented prompt placeholders safely."""
        values = {
            "topic": research.topic,
            "min_tweets": str(getattr(self.post_type, "min_tweets", "")),
            "max_tweets": str(getattr(self.post_type, "max_tweets", "")),
        }
        rendered = self.prompt
        for key, value in values.items():
            rendered = rendered.replace("{" + key + "}", value)
        return rendered

    def _demo(self, slot: PlanSlot | dict[str, Any], research: ResearchPacket, *, attempt: int = 1,
              feedback: list[str] | None = None) -> DraftPiece:
        slot_id = str(_slot_value(slot, "id", research.slot_id))
        post_type = str(_slot_value(slot, "type", self.post_type.id if self.post_type else self.kind))
        topic = research.topic
        facts = research.evidence
        if post_type == "marketing" or self.kind == "marketing" or (self.post_type and self.post_type.structure == "single"):
            fact = facts[0]
            text = f"A practical security habit: test {topic} assumptions before deployment. {fact.statement}"[:270]
            return DraftPiece(slot_id=slot_id, type=post_type, topic=topic,
                              tweets=[Tweet(text=text[:280], claims=[ClaimMapping(claim=fact.statement, evidence_ids=[fact.id])])])
        tweets: list[Tweet] = []
        # Exactly five educational beats for deterministic demo runs.  A real
        # configured type can request 6-10; this base keeps within 5-10.
        templates = [
            f"{topic}: start with the behavior, not the label.",
            "The evidence describes the failure mode and its root cause.",
            "Audit the trust boundary and test the negative path.",
            "Use a focused regression test to keep the issue from returning.",
            "Security reviews improve when each claim points to concrete evidence.",
        ]
        for index, template in enumerate(templates):
            fact = facts[index % len(facts)]
            text = f"{template} {fact.statement}"
            claim = fact.statement
            tweets.append(Tweet(text=text[:280], claims=[ClaimMapping(claim=claim, evidence_ids=[fact.id])]))
        return DraftPiece(slot_id=slot_id, type=post_type, topic=topic, tweets=tweets)

    async def write(self, slot: PlanSlot | dict[str, Any], research: ResearchPacket, *, attempt: int = 1,
                    feedback: list[str] | None = None,
                    previous_draft: DraftPiece | None = None) -> DraftPiece:
        if self.provider is None or getattr(self.provider, "mode", "demo") == "demo":
            draft = self._demo(slot, research, attempt=attempt, feedback=feedback)
            if self.ledger is not None:
                self.ledger.add(UsageRecord(agent=f"{self.kind}-writer", slot_id=research.slot_id,
                                            attempt=attempt, model="demo-fixture-v1", mode="demo",
                                            input_tokens=0, output_tokens=0, total_tokens=0, cost_usd=0.0))
            return draft
        context = {
            "agent": self.kind, "slot_id": research.slot_id, "type": _slot_value(slot, "type", self.kind),
            "topic": research.topic, "research": research.model_dump(), "attempt": attempt,
            "feedback": feedback or [],
        }
        if previous_draft is not None:
            context["previous_draft"] = previous_draft.model_dump()
        try:
            response = await self.provider.complete(
                "Generate the requested piece from the supplied context.",
                system=self._render_prompt(research),
                response_model=DraftPiece,
                context=context,
            )
        except Exception as exc:
            if self.ledger is not None:
                record_provider_usage(self.ledger, agent=f"{self.kind}-writer", slot_id=research.slot_id,
                                       attempt=attempt, result_or_error=exc, mode="live",
                                       fallback_model=getattr(self.provider, "model", "unknown"))
            raise
        if self.ledger is not None:
            record_provider_usage(self.ledger, agent=f"{self.kind}-writer", slot_id=research.slot_id,
                                  attempt=attempt, result_or_error=response, mode="live",
                                  fallback_model=getattr(self.provider, "model", "unknown"))
        draft = coerce_response(response, DraftPiece).data
        if draft.slot_id != research.slot_id:
            raise ValueError("Writer response changed the slot ID")
        return draft

    async def revise(self, slot: PlanSlot | dict[str, Any], research: ResearchPacket, draft: DraftPiece | list[str] | None,
                     feedback: list[str] | None = None) -> DraftPiece:
        # The orchestrator controls the one-revision limit.  This method does
        # not recursively retry and always receives explicit reviewer feedback.
        if feedback is None and isinstance(draft, list):
            feedback = draft
        previous_draft = draft if isinstance(draft, DraftPiece) else None
        return await self.write(slot, research, attempt=2, feedback=feedback or [],
                                previous_draft=previous_draft)


class TechnicalWriterAgent(WriterAgent):
    kind = "tech-thread"


class MarketingWriterAgent(WriterAgent):
    kind = "marketing"


# Common aliases make the classes easy to import in small integrations.
TechnicalWriter = TechnicalWriterAgent
MarketingWriter = MarketingWriterAgent

__all__ = ["MarketingWriter", "MarketingWriterAgent", "TechnicalWriter", "TechnicalWriterAgent", "WriterAgent"]
