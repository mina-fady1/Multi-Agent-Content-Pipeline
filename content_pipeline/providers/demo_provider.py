"""Deterministic offline provider used by tests and sample runs."""

from __future__ import annotations

import hashlib
from typing import Any, TypeVar

from ..models import CriterionScore, DraftPiece, EvidenceFact, ReviewResult, Tweet, VisualSpec, ClaimMapping
from .base import ProviderResponse, ProviderUsage

T = TypeVar("T")


class DemoProvider:
    mode = "demo"
    model = "demo-fixture-v1"

    def __init__(self, *, seed: int = 0):
        self.seed = seed
        self.calls = 0

    @staticmethod
    def _packet(context: dict[str, Any]) -> list[EvidenceFact]:
        packet = context.get("research")
        if packet is None:
            return []
        raw = list(packet.evidence if hasattr(packet, "evidence") else packet.get("evidence", []))
        return [item if isinstance(item, EvidenceFact) else EvidenceFact.model_validate(item) for item in raw]

    async def complete(self, prompt: str, *, system: str | None = None,
                       response_model: type[Any] | None = None,
                       context: dict[str, Any] | None = None,
                       temperature: float = 0) -> ProviderResponse:
        self.calls += 1
        context = context or {}
        data = self._make(response_model, context)
        # Demo mode explicitly reports zero real model tokens/cost.  This is
        # intentionally different from unknown usage returned by live models.
        return ProviderResponse(data=data, usage=ProviderUsage(0, 0, cost_usd=0.0), model=self.model)

    async def complete_json(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        return await self.complete(prompt, **kwargs)

    async def generate(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        return await self.complete(prompt, **kwargs)

    def _make(self, response_model: type[Any] | None, context: dict[str, Any]) -> Any:
        name = getattr(response_model, "__name__", "")
        if name == "DraftPiece":
            packet = self._packet(context)
            topic = str(context.get("topic", "security weakness"))
            slot_id = str(context.get("slot_id", "slot"))
            post_type = str(context.get("type", context.get("post_type", "tech-thread")))
            if post_type in {"marketing", "single"}:
                fact = packet[0] if packet else None
                text = f"Security is a practice: review {topic} before users depend on it."
                claims = [ClaimMapping(claim=text, evidence_ids=[fact.id])] if fact else []
                return DraftPiece(slot_id=slot_id, type=post_type, topic=topic,
                                  tweets=[Tweet(text=text[:280], claims=claims)])
            tweets: list[Tweet] = []
            for index in range(5):
                fact = packet[index % len(packet)] if packet else None
                if fact is None:
                    text = f"{topic}: verify assumptions, authorization and failure paths before deployment."
                    claims = []
                elif index == 0:
                    text = f"{topic}: {fact.statement}"
                    claims = [ClaimMapping(claim=fact.statement, evidence_ids=[fact.id])]
                elif index == 1:
                    text = f"Root cause to investigate: {fact.statement}"
                    claims = [ClaimMapping(claim=fact.statement, evidence_ids=[fact.id])]
                else:
                    text = f"Audit cue {index}: test the concrete behavior described in the evidence."
                    claims = [ClaimMapping(claim="evidence", evidence_ids=[fact.id])]
                tweets.append(Tweet(text=text, claims=claims))
            return DraftPiece(slot_id=slot_id, type=post_type, topic=topic, tweets=tweets)
        if name == "ReviewResult":
            approved = not bool(context.get("violations"))
            grounding = not bool(context.get("grounding_problems"))
            approved = approved and grounding
            reason = [] if approved else list(context.get("violations", [])) + list(context.get("grounding_problems", []))
            return ReviewResult(
                scores=[CriterionScore(criterion="clarity", score=4.0 if approved else 2.0, reason="Deterministic demo review")],
                approved=approved, grounding_passed=grounding, reasons=reason,
                feedback=["Tie each factual claim to a specific evidence statement."] if not approved else [],
                weighted_score=4.0 if approved else 2.0,
            )
        if name == "VisualSpec":
            topic = str(context.get("topic", "smart-contract security"))
            evidence = self._packet(context)
            return VisualSpec(headline=topic[:100], key_point=(evidence[0].statement if evidence else "Review security assumptions before deployment")[:300],
                              evidence_ids=[evidence[0].id] if evidence else ["demo:evidence"],
                              alt_text=f"Educational security card about {topic}")
        return {"ok": True, "context_hash": hashlib.sha256(repr(context).encode()).hexdigest()[:12]}


__all__ = ["DemoProvider"]
