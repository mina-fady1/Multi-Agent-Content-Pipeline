"""Research agent: deterministic, grounded topic resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config import load_prompt
from ..models import KBItem, PlanSlot, ResearchPacket, TopicSelection
from ..providers.base import LLMProvider
from ..telemetry import record_provider_usage
from ..skills.evidence_lookup import evidence_lookup, lookup_topic


class ResearchAgent:
    def __init__(self, kb: Sequence[KBItem] | Iterable[dict[str, Any]] | Any, *, provider: LLMProvider | None = None,
                 prompt_path: str | Path | None = None, mode: str = "demo", ledger: Any = None, **_: Any):
        # PipelineRunner passes the complete config as the first positional
        # argument; accepting raw KB records keeps the class easy to unit test.
        config = kb if hasattr(kb, "kb") else None
        if config is not None:
            kb = config.kb
            prompt_path = prompt_path or (Path(config.prompts_dir) / "research.md" if getattr(config, "prompts_dir", None) else None)
        self.config = config
        self.mode = mode
        self.ledger = ledger
        self.kb = [item if isinstance(item, KBItem) else KBItem.model_validate(item) for item in kb]
        if provider is None and mode == "live":
            from ..providers.gemini_provider import GeminiProvider
            provider = GeminiProvider()
        self.provider = provider
        self.prompt = load_prompt(prompt_path) if prompt_path else ""

    def resolve_topic(self, slot: PlanSlot | dict[str, Any], used_topics: set[str] | None = None) -> str:
        resolved_topic = getattr(slot, "topic", None) if not isinstance(slot, dict) else slot.get("topic")
        if resolved_topic and str(resolved_topic).strip() and str(resolved_topic).strip() != "unresolved":
            return str(resolved_topic).strip()
        focus = getattr(slot, "focus", None) if not isinstance(slot, dict) else slot.get("focus")
        if focus and str(focus).strip():
            # A populated focus is never silently changed.  Return it and let
            # research() fail clearly if no supporting record exists.
            return str(focus).strip()
        used = {value.casefold() for value in (used_topics or set())}
        for item in self.kb:
            candidates = {item.id.casefold(), item.name.casefold(), *(a.casefold() for a in item.aliases)}
            if not candidates & used:
                return item.name
        return self.kb[0].name if self.kb else ""

    def choose_topic(self, used_topics: set[str] | None = None) -> KBItem | None:
        used = {value.casefold() for value in (used_topics or set())}
        for item in self.kb:
            if item.id.casefold() not in used and item.name.casefold() not in used:
                return item
        return None

    def research(self, slot: PlanSlot | dict[str, Any], used_topics: set[str] | None = None) -> ResearchPacket:
        slot_id = getattr(slot, "id", None) if not isinstance(slot, dict) else slot.get("id")
        slot_id = str(slot_id)
        topic = self.resolve_topic(slot, used_topics)
        if not topic:
            raise LookupError(f"No topic available for slot {slot_id}")
        # Calling the shared skill here is intentional; ReviewerAgent uses the
        # same function to ensure both sides inspect one evidence format.
        return evidence_lookup(topic, self.kb, slot_id=slot_id)

    async def run(self, slot: PlanSlot | dict[str, Any], used_topics: set[str] | None = None) -> ResearchPacket:
        return self.research(slot, used_topics)

    async def select_topic_async(self, slot: PlanSlot | dict[str, Any], used_topics: set[str] | None = None) -> str:
        """Select an unused KB topic with Gemini, then validate it locally."""
        fallback = self.resolve_topic(slot, used_topics)
        if self.provider is None or self.mode != "live":
            return fallback
        used = {value.casefold() for value in (used_topics or set())}
        available = [item.name for item in self.kb
                     if not ({item.id.casefold(), item.name.casefold(), *(a.casefold() for a in item.aliases)} & used)]
        if not available:
            raise LookupError(f"No unused knowledge-base topic remains for slot {getattr(slot, 'id', 'slot')}")
        slot_id = getattr(slot, "id", None) if not isinstance(slot, dict) else slot.get("id")
        try:
            response = await self.provider.complete(
                "Choose exactly one topic from available_topics. Return only the structured topic selection.",
                system=self.prompt,
                response_model=TopicSelection,
                context={"agent": "research", "slot_id": str(slot_id),
                         "available_topics": available, "used_topics": sorted(used)},
            )
        except Exception as exc:
            if self.ledger is not None:
                record_provider_usage(self.ledger, agent="research", slot_id=str(slot_id), attempt=1,
                                      result_or_error=exc, mode="live",
                                      fallback_model=getattr(self.provider, "model", "unknown"))
            raise
        if self.ledger is not None:
            record_provider_usage(self.ledger, agent="research", slot_id=str(slot_id), attempt=1,
                                  result_or_error=response, mode="live",
                                  fallback_model=getattr(self.provider, "model", "unknown"))
        selected = lookup_topic(response.data.topic, self.kb)
        if selected is None or selected.id.casefold() in used:
            raise ValueError(f"Gemini selected an unavailable knowledge-base topic: {response.data.topic}")
        return selected.name

    async def research_async(self, slot: PlanSlot | dict[str, Any], used_topics: set[str] | None = None) -> ResearchPacket:
        """Ask the live provider for synthesis while retaining KB-only evidence.

        Gemini's response is advisory: source IDs and factual statements that
        enter the handoff are always taken from the verified local packet. This
        prevents generated prose from becoming evidence merely because it was
        returned by an LLM.
        """
        packet = self.research(slot, used_topics)
        if self.provider is None or self.mode != "live":
            return packet
        try:
            response = await self.provider.complete(
                "Synthesize the supplied verified evidence for this topic. Do not add facts or sources.",
                system=self.prompt,
                response_model=ResearchPacket,
                context={"agent": "research", "topic": packet.topic, "verified_packet": packet.model_dump()},
            )
        except Exception as exc:
            if self.ledger is not None:
                record_provider_usage(self.ledger, agent="research", slot_id=packet.slot_id,
                                      attempt=1, result_or_error=exc, mode="live",
                                      fallback_model=getattr(self.provider, "model", "unknown"))
            raise
        if self.ledger is not None:
            record_provider_usage(self.ledger, agent="research", slot_id=packet.slot_id,
                                  attempt=1, result_or_error=response, mode="live",
                                  fallback_model=getattr(self.provider, "model", "unknown"))
        # Validate the structured response before continuing, but never trust
        # its generated evidence as a source of truth.
        candidate = response.data
        if candidate.slot_id != packet.slot_id or candidate.topic.casefold() != packet.topic.casefold():
            raise ValueError("Gemini research response changed the slot or topic")
        return packet


__all__ = ["ResearchAgent"]
