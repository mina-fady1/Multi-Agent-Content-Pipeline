"""Usage aggregation with explicit unknowns for missing provider data/prices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import UsageRecord
from .providers.base import ProviderUsage


@dataclass
class UsageLedger:
    records: list[UsageRecord] = field(default_factory=list)

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)

    @property
    def live_calls(self) -> int:
        return sum(1 for record in self.records if record.mode == "live")

    def totals(self) -> tuple[int | None, int | None, float | None, list[str]]:
        if not self.records:
            return 0, 0, 0.0, []
        input_values = [item.input_tokens for item in self.records if item.input_tokens is not None]
        output_values = [item.output_tokens for item in self.records if item.output_tokens is not None]
        input_known = len(input_values) == len(self.records)
        output_known = len(output_values) == len(self.records)
        cost_known = all(item.cost_usd is not None for item in self.records)
        unknowns: list[str] = []
        # Preserve the usage we did receive. A failed request (for example an
        # image request rejected before generation) often has no token
        # metadata, but that must not hide successful text-call usage.
        in_total = sum(input_values) if input_values else None
        out_total = sum(output_values) if output_values else None
        cost = sum(item.cost_usd or 0 for item in self.records) if cost_known else None
        if not input_known:
            unknowns.append("input token total is a known subtotal; one or more provider calls omitted usage")
        if not output_known:
            unknowns.append("output token total is a known subtotal; one or more provider calls omitted usage")
        if any(item.total_tokens is None for item in self.records):
            unknowns.append("total token count is a known subtotal; one or more provider calls omitted usage")
        if not cost_known:
            unknowns.append("one or more calls use an unconfigured model price")
        return in_total, out_total, cost, unknowns

    def extended_totals(self) -> tuple[int | None, int | None]:
        """Return known subtotals for total and thinking-token usage."""
        total_values = [item.total_tokens for item in self.records if item.total_tokens is not None]
        thinking_values = [item.thinking_tokens if item.thinking_tokens is not None else item.thoughts_tokens
                           for item in self.records]
        known_thinking = [value for value in thinking_values if value is not None]
        total = sum(total_values) if total_values else None
        thinking = sum(known_thinking) if known_thinking else None
        return total, thinking


def record_provider_usage(
    ledger: UsageLedger,
    *,
    agent: str,
    slot_id: str,
    attempt: int,
    result_or_error: Any,
    mode: str,
    fallback_model: str = "unknown",
) -> None:
    """Record every provider request, including bounded retry attempts.

    GeminiProvider exposes ``request_usages`` on both successful responses and
    provider exceptions. Older adapters are normalized to their final usage;
    a call with no metadata remains unknown rather than becoming zero.
    """
    usages = getattr(result_or_error, "request_usages", None)
    if not usages:
        usage = getattr(result_or_error, "usage", None)
        usages = [usage] if usage is not None else [None]
    model = getattr(result_or_error, "model", None) or fallback_model
    for usage in usages:
        usage = usage or ProviderUsage()
        ledger.add(UsageRecord(
            agent=agent,
            slot_id=slot_id,
            attempt=attempt,
            model=model,
            mode=mode,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            total_tokens=usage.total_tokens,
            thinking_tokens=getattr(usage, "thinking_tokens", None) or getattr(usage, "thoughts_tokens", None),
            tool_tokens=getattr(usage, "tool_tokens", None),
            thoughts_tokens=getattr(usage, "thoughts_tokens", None),
            tool_use_prompt_tokens=getattr(usage, "tool_use_prompt_tokens", None),
            tool_use_response_tokens=getattr(usage, "tool_use_response_tokens", None),
            cost_usd=usage.cost_usd,
            status="completed" if not isinstance(result_or_error, BaseException) else "failed",
        ))

