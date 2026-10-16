"""The run coordinator: validation, isolated concurrent slots, one rewrite, export."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import dump_model, write_json
from .exporters import export_csv
from .models import (
    DraftAttempt,
    FinalPiece,
    Plan,
    PlanSlot,
    ResolvedSlot,
    SlotOutcome,
    StageEvent,
    RunSummary,
)
from .telemetry import UsageLedger
from .agents.research import ResearchAgent
from .agents.reviewer import ReviewerAgent
from .agents.visual import VisualAgent
from .agents.writer import MarketingWriterAgent, TechnicalWriterAgent, WriterAgent
from .providers.gemini_provider import GeminiProvider
from .skills.evidence_lookup import lookup_topic


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineError(RuntimeError):
    """An error scoped to one slot, suitable for an isolated outcome."""


class PipelineRunner:
    def __init__(self, config: Any, *, mode: str, output: Path, max_concurrency: int = 4, seed: int = 7):
        self.config = config
        self.mode = mode
        self.output = output
        self.max_concurrency = max(1, max_concurrency)
        self.seed = seed
        self.ledger = UsageLedger()
        provider_name = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        if provider_name != "gemini":
            raise ValueError("Only Gemini is supported; set LLM_PROVIDER=gemini")
        if mode == "live":
            self.provider = self._make_provider(os.getenv("GEMINI_MODEL"))
            research_model = os.getenv("GEMINI_RESEARCH_MODEL")
            research_provider = (self._make_provider(research_model)
                                 if research_model else self.provider)
        else:
            self.provider = None
            research_provider = None
        self.provider_name = provider_name if mode == "live" else "demo"
        self.researcher = ResearchAgent(
            config.kb,
            provider=research_provider,
            mode=mode,
            ledger=self.ledger,
            prompt_path=(config.prompts_dir / "research.md" if config.prompts_dir else None),
        )
        self.writers = {
            "tech-thread": TechnicalWriterAgent(config.post_types.get("tech-thread"), provider=self.provider,
                                                voice=config.voice, ledger=self.ledger,
                                                prompt_path=(config.prompts_dir / "tech-thread.md" if config.prompts_dir else None)),
            "marketing": MarketingWriterAgent(config.post_types.get("marketing"), provider=self.provider,
                                              voice=config.voice, ledger=self.ledger,
                                              prompt_path=(config.prompts_dir / "marketing.md" if config.prompts_dir else None)),
        }
        # A new YAML post type gets a generic config-driven writer automatically;
        # specialized writers above remain available for the required types.
        for type_id, post_type in config.post_types.items():
            if type_id not in self.writers:
                if config.prompts_dir:
                    configured_prompt = Path(post_type.prompt_path)
                    prompt_path = (config.prompts_dir.parent / configured_prompt
                                   if configured_prompt.parts and configured_prompt.parts[0] == "prompts"
                                   else config.prompts_dir / configured_prompt)
                else:
                    prompt_path = None
                self.writers[type_id] = WriterAgent(post_type, provider=self.provider, kind=type_id,
                                                    voice=config.voice, ledger=self.ledger,
                                                    prompt_path=prompt_path)
        review_provider = self.provider
        if mode == "live":
            review_model = os.getenv("GEMINI_REVIEW_MODEL")
            if review_model:
                review_provider = self._make_provider(review_model)
        self.reviewer = ReviewerAgent(config, mode=mode, ledger=self.ledger, provider=review_provider)
        self.visual = VisualAgent(config=config, provider=self.provider, ledger=self.ledger, mode=mode)

    def _make_provider(self, model: str | None = None) -> GeminiProvider:
        return GeminiProvider(model=model, pricing=self.config.pricing)

    async def aclose(self) -> None:
        providers = {id(provider): provider for provider in (
            self.provider,
            getattr(self.researcher, "provider", None),
            getattr(self.reviewer, "provider", None),
            getattr(self.visual, "provider", None),
        ) if provider is not None}
        for provider in providers.values():
            close = getattr(provider, "aclose", None)
            if close is not None:
                result = close()
                if hasattr(result, "__await__"):
                    await result

    async def run(self, plan: Plan, overrides: dict[str, str] | None = None) -> RunSummary:
        started = now()
        self.output.mkdir(parents=True, exist_ok=True)
        overrides = overrides or {}
        known = {slot.id for slot in plan.slots}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValueError(f"Unknown slot ID in --set: {unknown[0]}")
        effective_slots = [
            PlanSlot(id=s.id, type=s.type, focus=overrides.get(s.id, s.focus)) for s in plan.slots
        ]
        effective_plan = Plan(slots=effective_slots)
        resolved = await self._resolve(effective_plan)
        write_json(self.output / "resolved_plan.json", {"schema_version": "1.0", "slots": resolved})
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [asyncio.create_task(self._run_slot(slot, semaphore)) for slot in resolved]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outcomes: list[SlotOutcome] = []
        for slot, result in zip(resolved, results):
            if isinstance(result, BaseException):
                detail = str(result) or type(result).__name__
                outcomes.append(
                    SlotOutcome(
                        slot_id=slot.id,
                        type=slot.type,
                        topic=slot.topic,
                        status="dropped",
                        stage="orchestrator",
                        reasons=[f"Unhandled slot error: {detail}"],
                        events=[StageEvent(stage="orchestrator", status="error", detail=detail)],
                    )
                )
            else:
                outcomes.append(result)
        self._write_artifacts(outcomes, effective_plan, started)
        finals = [outcome.final for outcome in outcomes if outcome.final is not None]
        image_dir = self.output / "images"
        image_dir.mkdir(exist_ok=True)
        # Export only after all pieces and images have been finalized.
        row_count = export_csv(finals, self.output / "posts.csv")  # type: ignore[arg-type]
        input_tokens, output_tokens, cost, unknowns = self.ledger.totals()
        total_tokens, thinking_tokens = self.ledger.extended_totals()
        completed = now()
        summary = RunSummary(
            run_id=self.output.name,
            started_at=started,
            completed_at=completed,
            provider_mode=self.mode,
            planned=len(outcomes),
            shipped=len(finals),
            dropped=len(outcomes) - len(finals),
            csv_rows=row_count,
            live_model_calls=self.ledger.live_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            thinking_tokens=thinking_tokens,
            estimated_cost_usd=cost,
            unknowns=unknowns,
            outcomes=outcomes,
        )
        write_json(self.output / "run_summary.json", summary)
        print(
            f"planned={summary.planned} shipped={summary.shipped} dropped={summary.dropped} "
            f"csv_rows={summary.csv_rows} mode={summary.provider_mode} "
            f"input_tokens={summary.input_tokens} output_tokens={summary.output_tokens} "
            f"cost_usd={summary.estimated_cost_usd if summary.estimated_cost_usd is not None else 'unknown'}"
        )
        for outcome in outcomes:
            if outcome.status == "dropped":
                print(f"dropped {outcome.slot_id}: {'; '.join(outcome.reasons)}")
        return summary

    async def _resolve(self, plan: Plan) -> list[ResolvedSlot]:
        # Reserve explicitly focused topics first so blank slots cannot collide
        # with a user-selected focus later in the plan.
        used: set[str] = set()
        for focused in plan.slots:
            if focused.focus:
                item = lookup_topic(focused.focus, self.config.kb)
                if item:
                    used.add(item.id)
        resolved: list[ResolvedSlot] = []
        for slot in plan.slots:
            topic = (slot.focus or "").strip()
            if topic:
                item = lookup_topic(topic, self.config.kb)
                resolved.append(
                    ResolvedSlot(id=slot.id, type=slot.type, focus=slot.focus, topic=topic,
                                 source_id=f"kb:{item.id}" if item else None,
                                 resolution_error=None if item else f"No verified knowledge-base evidence found for focus: {topic}")
                )
                continue
            try:
                topic = (await self.researcher.select_topic_async(
                    PlanSlot(id=slot.id, type=slot.type, focus=None), used
                ) if self.mode == "live" else
                         self.researcher.resolve_topic(PlanSlot(id=slot.id, type=slot.type, focus=None), used))
            except Exception as exc:
                resolved.append(
                    ResolvedSlot(id=slot.id, type=slot.type, focus=None, topic="unresolved",
                                 source_id=None, resolution_error=f"Gemini topic selection failed: {exc}")
                )
                continue
            item = next((candidate for candidate in self.config.kb if candidate.name == topic), None)
            if item is None:
                resolved.append(
                    ResolvedSlot(
                        id=slot.id,
                        type=slot.type,
                        focus=None,
                        topic="unresolved",
                        source_id=None,
                        resolution_error="No unused knowledge-base topic remains",
                    )
                )
            else:
                used.add(item.id)
                resolved.append(
                    ResolvedSlot(
                        id=slot.id,
                        type=slot.type,
                        focus=None,
                        topic=item.name,
                        source_id=f"kb:{item.id}",
                    )
                )
        return resolved

    async def _run_slot(self, slot: ResolvedSlot, semaphore: asyncio.Semaphore) -> SlotOutcome:
        async with semaphore:
            events = [StageEvent(stage="resolve", status="completed")]
            if slot.resolution_error:
                return SlotOutcome(
                    slot_id=slot.id,
                    type=slot.type,
                    topic=slot.topic,
                    status="dropped",
                    stage="resolve",
                    reasons=[slot.resolution_error],
                    events=events + [StageEvent(stage="research", status="error", detail=slot.resolution_error)],
                )
            try:
                research_slot = PlanSlot(id=slot.id, type=slot.type, focus=slot.topic)
                if self.mode == "live":
                    research = await self.researcher.research_async(research_slot)
                else:
                    research = self.researcher.research(research_slot)
                events.append(StageEvent(stage="research", status="completed"))
            except Exception as exc:
                return SlotOutcome(
                    slot_id=slot.id,
                    type=slot.type,
                    topic=slot.topic,
                    status="dropped",
                    stage="research",
                    reasons=[str(exc)],
                    events=events + [StageEvent(stage="research", status="error", detail=str(exc))],
                )
            writer = self.writers.get(slot.type)
            ptype = self.config.post_types.get(slot.type)
            if writer is None or ptype is None:
                return SlotOutcome(
                    slot_id=slot.id,
                    type=slot.type,
                    topic=slot.topic,
                    status="dropped",
                    stage="write",
                    research=research,
                    reasons=[f"Unknown post type: {slot.type}"],
                    events=events + [StageEvent(stage="write", status="error", detail="unknown post type")],
                )
            attempts: list[DraftAttempt] = []
            feedback: list[str] = []
            previous_draft = None
            for attempt_no in (1, 2):
                try:
                    draft = await (
                        writer.write(slot, research)
                        if attempt_no == 1
                        else writer.revise(slot, research, previous_draft, feedback)
                    )
                    previous_draft = draft
                    violations = self.config.hard_rules(draft, research, ptype)
                    review = await self.reviewer.review(draft, research, violations, ptype, attempt=attempt_no)
                    attempts.append(DraftAttempt(attempt=attempt_no, draft=draft, review=review))
                    events.extend(
                        [
                            StageEvent(stage="write", status="completed", attempt=attempt_no),
                            StageEvent(
                                stage="review",
                                status="approved" if review.approved else "rejected",
                                attempt=attempt_no,
                                detail="; ".join(review.reasons),
                            ),
                        ]
                    )
                    accepted = review.approved and not violations
                    if accepted:
                        if self.mode == "live":
                            visual_spec, image_path = await self.visual.render_async(draft, research, self.output)
                        else:
                            visual_spec, image_path = self.visual.render(draft, research, self.output)
                        # The visual agent returns a real path; keep the CSV portable
                        # by storing it relative to this run directory.
                        if image_path is not None:
                            try:
                                image_path = str(Path(image_path).relative_to(self.output))
                            except ValueError:
                                image_path = str(Path(image_path))
                            image_file = Path(image_path)
                            if not image_file.is_absolute():
                                image_file = self.output / image_file
                            if not image_file.is_file():
                                return SlotOutcome(
                                    slot_id=slot.id, type=slot.type, topic=slot.topic,
                                    status="dropped", stage="visual",
                                    reasons=["Visual agent returned a missing image path"],
                                    research=research, attempts=attempts,
                                    events=events + [StageEvent(stage="visual", status="error", detail="missing image")],
                                )
                        final = FinalPiece(
                            draft=draft,
                            review=review,
                            research=research,
                            visual_spec=visual_spec,
                            image=image_path,
                        )
                        events.append(StageEvent(stage="visual", status="completed"))
                        return SlotOutcome(
                            slot_id=slot.id,
                            type=slot.type,
                            topic=slot.topic,
                            status="shipped",
                            stage="export",
                            research=research,
                            attempts=attempts,
                            final=final,
                            events=events,
                        )
                    feedback = review.feedback or review.reasons or [v.message for v in violations]
                    if attempt_no == 2:
                        return SlotOutcome(
                            slot_id=slot.id,
                            type=slot.type,
                            topic=slot.topic,
                            status="dropped",
                            stage="review",
                            reasons=review.reasons + [v.message for v in violations],
                            research=research,
                            attempts=attempts,
                            events=events,
                        )
                except Exception as exc:
                    return SlotOutcome(
                        slot_id=slot.id,
                        type=slot.type,
                        topic=slot.topic,
                        status="dropped",
                        stage="write" if attempt_no == 1 else "rewrite",
                        reasons=[str(exc)],
                        research=research,
                        attempts=attempts,
                        events=events + [StageEvent(stage="write", status="error", attempt=attempt_no, detail=str(exc))],
                    )
            raise AssertionError("unreachable")

    def _write_artifacts(self, outcomes: list[SlotOutcome], plan: Plan, started: str) -> None:
        write_json(self.output / "research_log.json", {
            "schema_version": "1.0", "started_at": started,
            "packets": [dump_model(item.research) for item in outcomes if item.research is not None],
        })
        write_json(self.output / "drafts.json", {
            "schema_version": "1.0", "attempts": [
                {"slot_id": item.slot_id, "attempts": dump_model(item.attempts)} for item in outcomes
            ],
        })
        write_json(self.output / "review_report.json", {
            "schema_version": "1.0", "reviews": [
                {"slot_id": item.slot_id, "attempts": [dump_model(a.review) for a in item.attempts]}
                for item in outcomes
            ],
        })
        write_json(self.output / "final_content.json", {
            "schema_version": "1.0", "pieces": [dump_model(item.final) for item in outcomes if item.final]
        })
        write_json(self.output / "usage.json", {
            "schema_version": "1.0", "records": dump_model(self.ledger.records)
        })
        self._write_image_specs(outcomes)

    def _write_image_specs(self, outcomes: list[SlotOutcome]) -> None:
        spec_dir = self.output / "image_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        for item in outcomes:
            if item.final:
                write_json(spec_dir / f"{item.slot_id}.json", item.final.visual_spec)


async def run_pipeline(config: Any, plan: Plan, *, mode: str, output: Path, overrides: dict[str, str] | None = None, max_concurrency: int = 4, seed: int = 7) -> RunSummary:
    runner = PipelineRunner(config, mode=mode, output=output, max_concurrency=max_concurrency, seed=seed)
    try:
        return await runner.run(plan, overrides)
    finally:
        await runner.aclose()

