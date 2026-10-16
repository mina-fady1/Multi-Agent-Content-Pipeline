"""Reviewer: criterion scoring plus deterministic grounding verification."""

from __future__ import annotations

from typing import Any

from ..config import load_prompt
from ..models import CriterionScore, DraftPiece, PostType, ResearchPacket, ReviewResult, RuleViolation, UsageRecord
from ..skills.evidence_lookup import evidence_lookup, verify_claim_support
from ..telemetry import record_provider_usage


class ReviewerAgent:
    def __init__(self, config: Any, *, mode: str = "demo", ledger: Any = None, provider: Any = None):
        self.config = config
        self.mode = mode
        self.ledger = ledger
        if provider is None and mode == "live":
            from ..providers.gemini_provider import GeminiProvider
            provider = GeminiProvider(pricing=getattr(config, "pricing", None))
        self.provider = provider
        self.prompt = load_prompt(config.prompts_dir / "reviewer.md") if getattr(config, "prompts_dir", None) else ""

    async def review(self, draft: DraftPiece, research: ResearchPacket,
                     violations: list[RuleViolation], post_type: PostType, attempt: int = 1) -> ReviewResult:
        grounding_problems = verify_claim_support(draft, research)
        # Resolve the topic through the same shared evidence skill used by
        # ResearchAgent.  The returned packet is used as a consistency check;
        # all claim-level checks still inspect the actual handoff packet.
        if getattr(self.config, "kb", None):
            try:
                evidence_lookup(research.topic, self.config.kb, slot_id=research.slot_id)
            except LookupError as exc:
                grounding_problems.append(str(exc))
        # Reviewer examines the entire draft even when Python has hard failures.
        hard_reasons = [v.message for v in violations]
        reasons = hard_reasons + grounding_problems
        scores = []
        for criterion in self.config.rubric.criteria:
            score = 2.0 if reasons else 4.5
            scores.append(CriterionScore(criterion=criterion.id, score=score,
                                         reason="Hard-rule or grounding issue found" if reasons else "Clear, grounded draft"))
        weighted = (sum(s.score * c.weight for s, c in zip(scores, self.config.rubric.criteria)) /
                    max(sum(c.weight for c in self.config.rubric.criteria), 1e-9))
        approved = not reasons and weighted >= self.config.rubric.threshold
        model_result = None
        if self.mode == "live" and self.provider is not None:
            try:
                response = await self.provider.complete(
                    "Review the draft against the rubric and return the structured review.",
                    system=self.prompt,
                    response_model=ReviewResult,
                    context={"agent": "reviewer", "draft": draft.model_dump(), "research": research.model_dump(),
                             "violations": [v.model_dump() for v in violations], "attempt": attempt},
                )
            except Exception as exc:
                if self.ledger is not None:
                    record_provider_usage(self.ledger, agent="reviewer", slot_id=research.slot_id,
                                           attempt=attempt, result_or_error=exc, mode="live",
                                           fallback_model=getattr(self.provider, "model", "unknown"))
                raise
            model_result = response.data
            if self.ledger is not None:
                record_provider_usage(self.ledger, agent="reviewer", slot_id=research.slot_id,
                                      attempt=attempt, result_or_error=response, mode="live",
                                      fallback_model=getattr(self.provider, "model", "unknown"))
            # The model supplies judgment, while Python's grounding and hard
            # rules remain non-overridable gates.
            scores = model_result.scores or scores
            # Recompute the weighted score from the configured rubric. The
            # model's reported aggregate is useful audit data but must not be
            # allowed to bypass the rubric or be replaced accidentally when
            # it is legitimately zero.
            score_by_id = {score.criterion: score.score for score in scores}
            missing = [criterion.id for criterion in self.config.rubric.criteria
                       if criterion.id not in score_by_id]
            if missing:
                reasons.append(f"Reviewer omitted rubric criteria: {', '.join(missing)}")
            below_minimum = [
                f"{criterion.id} scored {score_by_id[criterion.id]:g}, below minimum {criterion.minimum:g}"
                for criterion in self.config.rubric.criteria
                if criterion.id in score_by_id and score_by_id[criterion.id] < criterion.minimum
            ]
            reasons.extend(below_minimum)
            weighted = sum(
                score_by_id.get(criterion.id, 0.0) * criterion.weight
                for criterion in self.config.rubric.criteria
            ) / max(sum(criterion.weight for criterion in self.config.rubric.criteria), 1e-9)
            if not model_result.grounding_passed:
                reasons.append("Reviewer marked grounding as uncertain")
            # Explanatory reasons are often positive even when Gemini returns
            # an inconsistent boolean approval flag. Approval is therefore
            # derived from the validated score, grounding result, and Python
            # policy gates; the model's reasons remain available in the audit.
            approved = (
                model_result.grounding_passed
                and not reasons
                and weighted >= self.config.rubric.threshold
            )
        if approved:
            feedback: list[str] = []
        else:
            feedback = (list(model_result.feedback) if model_result is not None and model_result.feedback else
                        ["Rewrite only the flagged claims, keeping each factual claim tied to evidence."])
        if self.ledger is not None and model_result is None:
            self.ledger.add(UsageRecord(agent="reviewer", slot_id=research.slot_id,
                                        attempt=attempt, model="demo-fixture-v1", mode=self.mode,
                                        input_tokens=0, output_tokens=0, total_tokens=0, cost_usd=0.0)
                            if self.mode == "demo" else UsageRecord(agent="reviewer", slot_id=research.slot_id,
                                        attempt=attempt, model="deterministic-policy", mode=self.mode,
                                        input_tokens=None, output_tokens=None, cost_usd=None))
        return ReviewResult(scores=scores, approved=approved,
                            grounding_passed=(model_result.grounding_passed if model_result is not None
                                              else not grounding_problems),
                            reasons=reasons, feedback=feedback, weighted_score=weighted,
                            violations=violations)


__all__ = ["ReviewerAgent"]
