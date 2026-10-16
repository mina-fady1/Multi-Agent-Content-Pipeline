"""Validated handoff contracts; policy failures remain reviewable drafts."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KBItem(Contract):
    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(min_length=1)
    severity: str = Field(min_length=1)
    category: str = Field(min_length=1)
    description: str = Field(min_length=1)
    root_cause: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    evidence_points: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class PlanSlot(Contract):
    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    type: str = Field(min_length=1)
    focus: str | None = None


class Plan(Contract):
    slots: list[PlanSlot] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> "Plan":
        ids = [slot.id for slot in self.slots]
        if len(set(ids)) != len(ids):
            raise ValueError("Slot IDs must be unique")
        return self


class ResolvedSlot(Contract):
    id: str
    type: str
    focus: str | None
    topic: str
    source_id: str | None
    resolution_error: str | None = None


class TopicSelection(Contract):
    topic: str = Field(min_length=1)


class SourceRef(Contract):
    id: str
    kind: Literal["kb", "url", "user"]
    locator: str
    title: str
    verified: bool = True
    references: list[str] = Field(default_factory=list)


class EvidenceFact(Contract):
    id: str
    statement: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)


class ResearchPacket(Contract):
    slot_id: str
    topic: str
    evidence: list[EvidenceFact] = Field(min_length=1)
    sources: list[SourceRef] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_references(self) -> "ResearchPacket":
        sources = {s.id for s in self.sources if s.verified}
        if len(sources) != len(self.sources):
            raise ValueError("Duplicate or unverified sources")
        if len({fact.id for fact in self.evidence}) != len(self.evidence):
            raise ValueError("Duplicate evidence IDs")
        for fact in self.evidence:
            if not set(fact.source_ids) <= sources:
                raise ValueError("Evidence references an unknown source")
        return self


class ClaimMapping(Contract):
    claim: str
    evidence_ids: list[str]


class Tweet(Contract):
    text: str
    claims: list[ClaimMapping] = Field(default_factory=list)


class DraftPiece(Contract):
    slot_id: str
    type: str
    topic: str
    tweets: list[Tweet]


class RuleViolation(Contract):
    code: str
    message: str
    tweet_index: int | None = None


class CriterionScore(Contract):
    criterion: str
    score: float = Field(ge=0, le=5)
    reason: str


class ReviewResult(Contract):
    scores: list[CriterionScore]
    approved: bool
    grounding_passed: bool
    reasons: list[str]
    feedback: list[str]
    weighted_score: float = 0
    violations: list[RuleViolation] = Field(default_factory=list)


class VisualSpec(Contract):
    headline: str = Field(min_length=1, max_length=100)
    key_point: str = Field(min_length=1, max_length=300)
    evidence_ids: list[str] = Field(min_length=1)
    alt_text: str = Field(min_length=1, max_length=500)


class FinalPiece(Contract):
    draft: DraftPiece
    review: ReviewResult
    research: ResearchPacket
    visual_spec: VisualSpec
    image: str | None


class DraftAttempt(Contract):
    attempt: int
    draft: DraftPiece
    review: ReviewResult | None = None
    created_at: str = Field(default_factory=utc_now)


class StageEvent(Contract):
    stage: str
    status: str
    attempt: int | None = None
    detail: str | None = None
    timestamp: str = Field(default_factory=utc_now)


class SlotOutcome(Contract):
    slot_id: str
    type: str
    topic: str
    status: Literal["shipped", "dropped"]
    stage: str
    reasons: list[str] = Field(default_factory=list)
    research: ResearchPacket | None = None
    attempts: list[DraftAttempt] = Field(default_factory=list)
    final: FinalPiece | None = None
    events: list[StageEvent] = Field(default_factory=list)


class UsageRecord(Contract):
    agent: str
    slot_id: str
    attempt: int
    model: str
    mode: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    thinking_tokens: int | None = Field(default=None, ge=0)
    tool_tokens: int | None = Field(default=None, ge=0)
    thoughts_tokens: int | None = Field(default=None, ge=0)
    tool_use_prompt_tokens: int | None = Field(default=None, ge=0)
    tool_use_response_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    status: str = "completed"
    timestamp: str = Field(default_factory=utc_now)


class RunSummary(Contract):
    schema_version: str = "1.0"
    run_id: str
    started_at: str
    completed_at: str
    provider_mode: str
    planned: int
    shipped: int
    dropped: int
    csv_rows: int
    live_model_calls: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None = None
    thinking_tokens: int | None = None
    estimated_cost_usd: float | None
    unknowns: list[str]
    outcomes: list[SlotOutcome]


class BrandVoice(Contract):
    tone: str
    banned_phrases: list[str]
    banned_characters: list[str]
    disallow_hashtags: bool
    approved_company_facts: list[str]
    allowed_urls: list[str]
    max_tweet_length: int = Field(ge=1, le=280)
    guidance: str = ""


class RubricCriterion(Contract):
    id: str
    weight: float = Field(gt=0)
    minimum: float = Field(ge=0, le=5)
    description: str


class Rubric(Contract):
    threshold: float = Field(ge=0, le=5)
    criteria: list[RubricCriterion] = Field(min_length=1)
    guidance: str = ""

    @model_validator(mode="after")
    def unique_criteria(self) -> "Rubric":
        if len({c.id for c in self.criteria}) != len(self.criteria):
            raise ValueError("Rubric criterion IDs must be unique")
        return self


class PostType(Contract):
    id: str
    prompt_path: str
    structure: Literal["thread", "single"]
    min_tweets: int = Field(ge=1)
    max_tweets: int = Field(ge=1)
    criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_count_range(self) -> "PostType":
        if self.max_tweets < self.min_tweets:
            raise ValueError("max_tweets must be >= min_tweets")
        if self.structure == "single" and (self.min_tweets != 1 or self.max_tweets != 1):
            raise ValueError("Single post types must specify one tweet")
        return self
