"""Provider abstraction used by agents.

Agents depend on this tiny interface rather than an SDK.  Demo and live
providers return the same result shape, including explicitly unknown usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass(slots=True)
class ProviderUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    cost_usd: float | None = None
    # Gemini reports a total that can include thinking/tool-use tokens.  Keep
    # that value when the provider supplies it while retaining the historical
    # input/output/cost fields used by the rest of the pipeline.
    reported_total_tokens: int | None = None
    thinking_tokens: int | None = None
    tool_tokens: int | None = None
    # Alias retained because UsageRecord historically used this spelling.
    thoughts_tokens: int | None = None
    tool_use_prompt_tokens: int | None = None
    tool_use_response_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.reported_total_tokens is not None:
            return self.reported_total_tokens
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class ProviderResponse:
    data: Any
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    model: str = "unknown"
    raw: Any = None
    # One entry per provider attempt, including attempts whose returned
    # content was malformed.  Existing positional construction remains
    # compatible because this field is additive at the end.
    request_usages: list[ProviderUsage] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.request_usages:
            self.request_usages = [self.usage]

    @property
    def input_tokens(self) -> int | None:
        return self.usage.input_tokens

    @property
    def output_tokens(self) -> int | None:
        return self.usage.output_tokens

    @property
    def cost_usd(self) -> float | None:
        return self.usage.cost_usd

    # Dict-like conveniences for integrations that do not need a typed model.
    def __getitem__(self, key: str) -> Any:
        if isinstance(self.data, dict):
            return self.data[key]
        return getattr(self.data, key)


class LLMProvider(Protocol):
    mode: str
    model: str

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_model: type[T] | None = None,
        context: dict[str, Any] | None = None,
        temperature: float = 0,
    ) -> ProviderResponse: ...

    async def complete_json(self, *args: Any, **kwargs: Any) -> ProviderResponse: ...


def coerce_response(
    value: ProviderResponse | T | dict[str, Any], response_model: type[T] | None = None
) -> ProviderResponse:
    """Normalize custom test providers and provider responses."""
    if isinstance(value, ProviderResponse):
        if response_model is not None and not isinstance(value.data, response_model):
            value.data = response_model.model_validate(value.data)
        return value
    data: Any = response_model.model_validate(value) if response_model is not None else value
    return ProviderResponse(data=data)


__all__ = ["LLMProvider", "ProviderResponse", "ProviderUsage", "coerce_response"]
