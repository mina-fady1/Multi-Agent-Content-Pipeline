"""Knowledge-base evidence retrieval and lightweight grounding checks.

This skill is intentionally shared by Research and Reviewer.  Retrieval is
deterministic and KB-only by default; callers can add a verified URL source
explicitly, but unverified web text is never silently treated as evidence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from ..models import DraftPiece, EvidenceFact, KBItem, ResearchPacket, SourceRef


def _items(kb: Sequence[KBItem] | Iterable[dict[str, Any]]) -> list[KBItem]:
    return [item if isinstance(item, KBItem) else KBItem.model_validate(item) for item in kb]


def _terms(value: str) -> set[str]:
    # Punctuation is a separator, so flash-loan and flash loan ground alike.
    return {term for term in re.findall(r"[a-z0-9]+", value.casefold()) if len(term) > 2}


def lookup_topic(topic: str, kb: Sequence[KBItem] | Iterable[dict[str, Any]]) -> KBItem | None:
    """Resolve a human topic to the most relevant KB item.

    Exact IDs, names and aliases win.  A bounded token overlap fallback keeps
    the command line ergonomic while avoiding arbitrary replacement of a
    populated focus: a caller can detect ``None`` and drop that slot.
    """
    query = topic.strip().lower()
    if not query:
        return None
    records = _items(kb)
    for item in records:
        if query in {item.id.lower(), item.name.lower(), *(a.lower() for a in item.aliases)}:
            return item
    q = _terms(query)
    if not q:
        return None
    best: tuple[int, int, KBItem] | None = None
    for index, item in enumerate(records):
        haystack = _terms(" ".join([item.id, item.name, item.category, *item.aliases, item.description]))
        score = len(q & haystack)
        if score:
            candidate = (score, -index, item)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    return best[2] if best else None


def build_research_packet(slot_id: str, topic: str, kb: Sequence[KBItem] | Iterable[dict[str, Any]]) -> ResearchPacket:
    item = lookup_topic(topic, kb)
    if item is None:
        raise LookupError(f"No verified knowledge-base evidence found for topic: {topic}")
    source_id = f"kb:{item.id}"
    source = SourceRef(id=source_id, kind="kb", locator=item.id, title=item.name, verified=True,
                       references=list(item.references))
    statements = [
        (f"{item.name} ({item.severity}) is a {item.category} weakness: {item.description}", "description"),
        (f"A typical root cause is {item.root_cause}.", "root_cause"),
    ]
    statements.extend((point, f"point-{i}") for i, point in enumerate(item.evidence_points))
    evidence = [EvidenceFact(id=f"ev:{item.id}:{suffix}", statement=statement,
                             source_ids=[source_id]) for statement, suffix in statements if statement.strip()]
    return ResearchPacket(slot_id=slot_id, topic=topic, evidence=evidence, sources=[source])


def evidence_lookup(topic: str, kb: Sequence[KBItem] | Iterable[dict[str, Any]], slot_id: str = "slot") -> ResearchPacket:
    """Alias used by agents and external integrations."""
    return build_research_packet(slot_id, topic, kb)


def verify_claim_support(draft: DraftPiece, research: ResearchPacket) -> list[str]:
    """Return grounding problems for claim/evidence mappings.

    This is deliberately conservative: an ID must exist, and at least one
    meaningful token from the claim must occur in the cited evidence.  Semantic
    entailment still needs reviewer/model judgment and is documented as a
    limitation by the application.
    """
    evidence = {fact.id: fact for fact in research.evidence}
    problems: list[str] = []
    for index, tweet in enumerate(draft.tweets):
        if not tweet.text.strip():
            problems.append(f"tweet {index}: empty text")
        for mapping in tweet.claims:
            if not mapping.claim.strip() or not mapping.evidence_ids:
                problems.append(f"tweet {index}: claim mapping is empty or has no evidence IDs")
                continue
            missing = [eid for eid in mapping.evidence_ids if eid not in evidence]
            if missing:
                problems.append(f"tweet {index}: unknown evidence IDs: {', '.join(missing)}")
                continue
            claim_terms = _terms(mapping.claim)
            supported = _terms(" ".join(evidence[eid].statement for eid in mapping.evidence_ids))
            # Short claims such as "reentrancy" can have one token; requiring
            # one shared token avoids treating a forged citation as grounded.
            if claim_terms and not (claim_terms & supported):
                problems.append(f"tweet {index}: claim is not supported by cited evidence")
            if claim_terms and not (claim_terms & _terms(tweet.text)):
                problems.append(f"tweet {index}: mapped claim is absent from tweet text")
    return problems


__all__ = ["build_research_packet", "evidence_lookup", "lookup_topic", "verify_claim_support"]


# A small class facade is convenient for dependency injection in applications.
class EvidenceLookup:
    def __init__(self, kb: Sequence[KBItem] | Iterable[dict[str, Any]]):
        self.kb = _items(kb)

    def topic(self, topic: str) -> KBItem | None:
        return lookup_topic(topic, self.kb)

    def packet(self, slot_id: str, topic: str) -> ResearchPacket:
        return build_research_packet(slot_id, topic, self.kb)

