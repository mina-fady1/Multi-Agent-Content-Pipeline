"""Google Gemini provider adapter.

The rest of the pipeline talks to :class:`~content_pipeline.providers.base.LLMProvider`;
this module is the only place that knows about ``google-genai``.  The SDK is
imported lazily so demo mode does not need to import the client just to load
the package.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import random
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .base import ProviderResponse, ProviderUsage

T = TypeVar("T", bound=BaseModel)


def _gemini_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a Gemini-compatible subset of a Pydantic JSON schema.

    Pydantic emits ``additionalProperties: false`` for strict models. The
    Gemini API's response-schema wire format does not accept that keyword,
    including when it appears inside nested array/object definitions.
    """
    schema = model.model_json_schema()

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if key != "additionalProperties"
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(schema)


class GeminiProviderError(RuntimeError):
    """Base error raised for a failed Gemini request."""

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        request_usages: list[ProviderUsage] | None = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.request_usages = list(request_usages or [])


class GeminiAuthenticationError(GeminiProviderError):
    """The key is missing or Gemini rejected it."""


class GeminiBlockedError(GeminiProviderError):
    """Gemini returned a safety or policy block instead of content."""


class GeminiResponseError(GeminiProviderError):
    """Gemini returned empty, malformed, or schema-invalid content."""


class _RetryableResponseError(GeminiResponseError):
    """A response problem that may be worth retrying (within the bound)."""


def _value(obj: Any, key: str, default: Any = None) -> Any:
    """Read an SDK object or a dict uniformly (also useful with test doubles)."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    try:
        return getattr(obj, key, default)
    except Exception:
        # A few SDK convenience properties (notably ``text`` for a blocked
        # response) raise when no candidate exists.  Treat those as absent so
        # the provider can report a useful blocked/empty response error.
        return default


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _enum_text(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value)).upper()


class GeminiProvider:
    """Async adapter for the official ``google-genai`` SDK.

    ``client`` is injectable for offline tests.  In normal use it is created
    lazily from ``GEMINI_API_KEY`` and shared by every call on this provider.
    A single provider can optionally route calls to per-agent models through
    ``agent_models`` or an explicit ``model=`` argument.
    """

    mode = "live"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        max_retries: int = 2,
        timeout: float = 60.0,
        client: Any = None,
        pricing: dict[str, Any] | None = None,
        agent_models: Mapping[str, str] | None = None,
        model_overrides: Mapping[str, str] | None = None,
        max_concurrent_requests: int | None = None,
        semaphore: Any = None,
    ) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.max_retries = max(0, int(max_retries))
        self.timeout = float(timeout)
        self.pricing = pricing or {}
        self.agent_models = dict(agent_models or model_overrides or {})
        # Environment overrides are convenient for callers that construct a
        # provider directly instead of routing model selection through the
        # orchestrator.  Explicit constructor overrides win.
        for env_name, aliases in {
            "RESEARCH": ("research",),
            "REVIEW": ("review", "reviewer"),
            "WRITER": ("writer", "tech-thread", "marketing"),
            "VISUAL": ("visual",),
        }.items():
            configured = os.getenv(f"GEMINI_{env_name}_MODEL")
            if configured:
                for alias in aliases:
                    self.agent_models.setdefault(alias, configured)
        self._client = client
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if max_concurrent_requests is None:
            try:
                max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
            except (TypeError, ValueError):
                max_concurrent_requests = 2
        try:
            self.max_concurrent_requests = max(1, int(max_concurrent_requests))
        except (TypeError, ValueError):
            self.max_concurrent_requests = 2
        self._semaphore = semaphore or asyncio.Semaphore(self.max_concurrent_requests)
        self.semaphore = self._semaphore

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise GeminiAuthenticationError(
                    "GEMINI_API_KEY is required for live Gemini mode; use --mode demo for offline execution"
                )
            try:
                from google import genai
            except (
                ImportError
            ) as exc:  # pragma: no cover - dependency is installed in live environments
                raise RuntimeError(
                    "Install the 'google-genai' package to use Gemini live mode"
                ) from exc
            client_kwargs: dict[str, Any] = {"api_key": self._api_key}
            # google-genai retries 429/5xx by default (currently five
            # attempts).  The provider owns the bounded retry policy, so turn
            # SDK retries off to avoid multiplying attempts and backoff.
            try:
                from google.genai import types

                http_values: dict[str, Any] = {
                    "retry_options": types.HttpRetryOptions(attempts=1),
                }
                if self.timeout > 0:
                    http_values["timeout"] = int(self.timeout * 1000)
                client_kwargs["http_options"] = types.HttpOptions(**http_values)
            except (ImportError, AttributeError, TypeError):
                # Older SDK releases accept a dict-shaped HttpOptions.  Keep
                # the key lazy and allow those versions to use the base client.
                client_kwargs["http_options"] = {
                    "retry_options": {"attempts": 1},
                    **({"timeout": int(self.timeout * 1000)} if self.timeout > 0 else {}),
                }
            try:
                self._client = genai.Client(**client_kwargs)
            except TypeError as exc:
                # A small SDK-compatible test double or an older release may
                # not accept http_options; retaining the key-only fallback is
                # preferable to failing before the request can be attempted.
                if "http_options" not in str(exc).lower():
                    raise
                self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _select_model(self, model: str | None, context: Mapping[str, Any] | None) -> str:
        if model:
            return model
        context = context or {}
        agent = context.get("agent") or context.get("agent_name") or context.get("role")
        if agent and str(agent) in self.agent_models:
            return self.agent_models[str(agent)]
        return self.model

    @staticmethod
    def _user_prompt(prompt: str, context: Mapping[str, Any] | None) -> str:
        if not context:
            return prompt
        # Context is data, not instructions.  Delimiting it keeps fetched or
        # user-supplied text from being interpreted as a new system directive.
        return f"{prompt}\n\n<CONTEXT_JSON>\n{json.dumps(context, default=str)}\n</CONTEXT_JSON>"

    @staticmethod
    def _config(system: str | None, response_model: type[T] | None, temperature: float) -> Any:
        values: dict[str, Any] = {
            "response_mime_type": "application/json",
            "temperature": temperature,
        }
        if system:
            values["system_instruction"] = system
        if response_model is not None:
            values["response_schema"] = _gemini_schema(response_model)
        # GenerateContentConfig is preferred for real SDK calls.  A dict is a
        # documented SDK input too and keeps injected clients/import-light
        # tests independent of a locally installed google package.
        try:
            from google.genai import types
        except ImportError:
            return values
        return types.GenerateContentConfig(**values)

    async def _generate(self, *, model: str, prompt: str, config: Any) -> Any:
        client = self.client
        aio = getattr(client, "aio", None)
        models = getattr(aio, "models", None) if aio is not None else None
        if models is None:
            # This fallback makes small SDK-compatible test doubles convenient;
            # official asynchronous clients expose ``client.aio.models``.
            models = getattr(client, "models", None)
        method = getattr(models, "generate_content", None)
        if method is None:
            raise RuntimeError("Gemini client does not expose aio.models.generate_content")
        async with self._semaphore:
            result = method(model=model, contents=prompt, config=config)
            if inspect.isawaitable(result):
                if self.timeout > 0:
                    return await asyncio.wait_for(result, timeout=self.timeout)
                return await result
            return result

    @staticmethod
    def _blocked_reason(response: Any) -> str | None:
        feedback = _value(response, "prompt_feedback")
        reason = _enum_text(_value(feedback, "block_reason"))
        if reason and reason not in {"NONE", "BLOCK_REASON_UNSPECIFIED"}:
            return reason
        candidates = _value(response, "candidates", []) or []
        for candidate in candidates:
            finish = _enum_text(_value(candidate, "finish_reason"))
            if finish in {
                "SAFETY",
                "BLOCKLIST",
                "PROHIBITED_CONTENT",
                "SPII",
                "RECITATION",
            }:
                return finish
        return None

    @classmethod
    def _response_text(cls, response: Any) -> tuple[Any, str | None]:
        """Return SDK ``parsed`` data when available, otherwise response text."""
        parsed = _value(response, "parsed")
        text: Any = None
        try:
            text = _value(response, "text")
        except Exception:
            # Some SDK response.text properties raise when no candidate exists.
            text = None
        if text is None:
            candidates = _value(response, "candidates", []) or []
            fragments: list[str] = []
            for candidate in candidates:
                content = _value(candidate, "content")
                for part in _value(content, "parts", []) or []:
                    fragment = _value(part, "text")
                    if fragment:
                        fragments.append(str(fragment))
            text = "".join(fragments) or None
        return parsed, str(text) if text is not None else None

    @staticmethod
    def _parse_json(response: Any, response_model: type[T] | None) -> Any:
        parsed, text = GeminiProvider._response_text(response)
        if parsed is None and not text:
            raise _RetryableResponseError("Gemini returned an empty response")
        if parsed is None:
            assert text is not None
            cleaned = text.strip()
            # Models occasionally wrap JSON in a markdown fence even when the
            # response MIME type is set.  Removing only the fence preserves
            # strict JSON validation while handling that harmless formatting.
            if cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = cleaned[3:-3].strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].lstrip()
            try:
                parsed = json.loads(cleaned)
            except (json.JSONDecodeError, TypeError) as exc:
                raise _RetryableResponseError(f"Gemini returned invalid JSON: {exc}") from exc
        if response_model is not None:
            try:
                if isinstance(parsed, response_model):
                    return parsed
                return response_model.model_validate(parsed)
            except (ValidationError, TypeError, ValueError) as exc:
                raise _RetryableResponseError(
                    f"Gemini JSON failed schema validation: {exc}"
                ) from exc
        return parsed

    def _usage(self, response: Any, model: str) -> ProviderUsage:
        metadata = _value(response, "usage_metadata") or _value(response, "usageMetadata")
        prompt = _as_int(_value(metadata, "prompt_token_count"))
        output = _as_int(_value(metadata, "candidates_token_count"))
        total = _as_int(_value(metadata, "total_token_count"))
        cached = _as_int(_value(metadata, "cached_content_token_count"))
        thinking = _as_int(_value(metadata, "thoughts_token_count"))
        if thinking is None:
            thinking = _as_int(_value(metadata, "thinking_token_count"))
        tool_prompt = _as_int(_value(metadata, "tool_use_prompt_token_count"))
        tool_response = _as_int(_value(metadata, "tool_use_response_token_count"))
        tool_tokens = None
        if tool_prompt is not None or tool_response is not None:
            tool_tokens = (tool_prompt or 0) + (tool_response or 0)
        usage = ProviderUsage(
            input_tokens=prompt,
            output_tokens=output,
            cached_input_tokens=cached,
            reported_total_tokens=total,
            thinking_tokens=thinking,
            tool_tokens=tool_tokens,
            thoughts_tokens=thinking,
            tool_use_prompt_tokens=tool_prompt,
            tool_use_response_tokens=tool_response,
        )
        usage.cost_usd = self._cost(model, usage, thinking=thinking, tool_tokens=tool_tokens)
        return usage

    def _cost(
        self, model: str, usage: ProviderUsage, *, thinking: int | None, tool_tokens: int | None
    ) -> float | None:
        models = (
            self.pricing.get("models", self.pricing) if isinstance(self.pricing, Mapping) else {}
        )
        rates = models.get(model, {}) if isinstance(models, Mapping) else {}
        if not isinstance(rates, Mapping):
            return None

        def rate(*names: str) -> float | None:
            for name in names:
                value = rates.get(name)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
            return None

        in_rate = rate("input", "input_usd_per_million")
        out_rate = rate("output", "output_usd_per_million")
        cached_rate = rate("cached_input", "cached_input_usd_per_million")
        thinking_rate = rate(
            "thoughts", "thinking", "thoughts_usd_per_million", "thinking_usd_per_million"
        )
        tool_rate = rate("tool", "tool_use", "tool_usd_per_million")
        if (
            usage.input_tokens is None
            or usage.output_tokens is None
            or in_rate is None
            or out_rate is None
        ):
            return None
        cached = min(usage.cached_input_tokens or 0, usage.input_tokens)
        # A cached category has its own billable rate.  Falling back to the
        # regular input rate would invent a price when Gemini's cache rate is
        # unavailable, so report cost as unknown instead.
        if cached and cached_rate is None:
            return None
        if thinking and thinking_rate is None:
            return None
        if tool_tokens and tool_rate is None:
            return None
        input_cost = (usage.input_tokens - cached) * in_rate
        input_cost += cached * (cached_rate or 0.0)
        # candidates_token_count already excludes thoughts and tool-use
        # tokens.  Do not subtract thinking from it a second time.
        cost = input_cost + usage.output_tokens * out_rate
        if thinking is not None:
            cost += thinking * (thinking_rate or 0.0)
        if tool_tokens is not None:
            cost += tool_tokens * (tool_rate or 0.0)
        return cost / 1_000_000

    @staticmethod
    def _is_auth_error(exc: BaseException) -> bool:
        code = _value(exc, "status_code") or _value(exc, "code") or _value(exc, "status")
        code_text = _enum_text(code)
        message = str(exc).lower()
        return code_text in {
            "401",
            "403",
            "unauthenticated",
            "permission_denied",
            "forbidden",
        } or any(
            marker in message
            for marker in (
                "api key",
                "invalid key",
                "authentication",
                "unauthorized",
                "permission denied",
                "forbidden",
            )
        )

    @staticmethod
    def _is_transient(exc: BaseException) -> bool:
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
            return True
        code = _value(exc, "status_code") or _value(exc, "code") or _value(exc, "status")
        code_text = _enum_text(code)
        message = str(exc).lower()
        # A quota/billing denial will not be fixed by sending the same request
        # again. Retry ordinary 429 rate limiting, but fail quota exhaustion
        # immediately so a run does not consume its retry budget pointlessly.
        if any(
            marker in message
            for marker in (
                "exceeded your current quota",
                "quota exceeded",
                "free_tier_requests",
                "daily limit",
                "billing details",
            )
        ):
            return False
        if code_text in {
            "408",
            "429",
            "500",
            "502",
            "503",
            "504",
            "resource_exhausted",
            "unavailable",
            "deadline_exceeded",
        }:
            return True
        return any(
            marker in message
            for marker in (
                "429",
                "408",
                "500",
                "502",
                "rate limit",
                "too many requests",
                "503",
                "504",
                "service unavailable",
                "temporarily unavailable",
                "timeout",
                "timed out",
            )
        )

    @staticmethod
    def _is_blocked_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "safety block",
                "blocked by safety",
                "prohibited content",
                "content blocked",
                "generation blocked",
            )
        )

    def _safe_error_text(self, exc: BaseException) -> str:
        message = str(exc) or exc.__class__.__name__
        secrets = {self._api_key, os.getenv("GEMINI_API_KEY")}
        for secret in secrets:
            if secret:
                message = message.replace(secret, "[REDACTED]")
        # SDK exceptions can echo very large response bodies or prompts. Keep
        # audit/error artifacts bounded while retaining status information.
        return message[:1000]

    @staticmethod
    def _annotate_error(
        error: GeminiProviderError,
        *,
        model: str,
        request_usages: list[ProviderUsage],
    ) -> GeminiProviderError:
        error.model = model
        error.request_usages = list(request_usages)
        return error

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_model: type[T] | None = None,
        context: dict[str, Any] | None = None,
        temperature: float = 0,
        model: str | None = None,
        agent: str | None = None,
    ) -> ProviderResponse:
        selection_context = dict(context or {})
        if agent:
            selection_context["agent"] = agent
        selected_model = self._select_model(model, selection_context)
        user_prompt = self._user_prompt(prompt, context)
        config = self._config(system, response_model, temperature)
        last_error: BaseException | None = None
        request_usages: list[ProviderUsage] = []
        for attempt in range(self.max_retries + 1):
            had_response = False
            try:
                response = await self._generate(
                    model=selected_model, prompt=user_prompt, config=config
                )
                had_response = True
                usage = self._usage(response, selected_model)
                # Record usage before inspecting content: blocked, empty, and
                # malformed responses can still be billable API responses.
                request_usages.append(usage)
                blocked = self._blocked_reason(response)
                if blocked:
                    raise GeminiBlockedError(f"Gemini blocked generation ({blocked})")
                data = self._parse_json(response, response_model)
                return ProviderResponse(
                    data=data,
                    usage=usage,
                    model=selected_model,
                    raw=response,
                    request_usages=list(request_usages),
                )
            except GeminiAuthenticationError as exc:
                if self._api_key and not had_response:
                    request_usages.append(ProviderUsage())
                if str(exc) and "GEMINI_API_KEY" in str(exc):
                    # Keep the actionable missing-key guidance while still
                    # attaching the selected model and usage state.
                    message = self._safe_error_text(exc)
                else:
                    message = f"Gemini authentication failed: {self._safe_error_text(exc)}"
                error = GeminiAuthenticationError(
                    message, model=selected_model, request_usages=request_usages
                )
                raise error from exc
            except GeminiBlockedError as exc:
                error = self._annotate_error(
                    exc, model=selected_model, request_usages=request_usages
                )
                error.args = (self._safe_error_text(error),)
                raise error
            except Exception as exc:
                if not had_response and not (
                    isinstance(exc, GeminiAuthenticationError) and not self._api_key
                ):
                    request_usages.append(ProviderUsage())
                if self._is_auth_error(exc):
                    error = GeminiAuthenticationError(
                        f"Gemini authentication failed: {self._safe_error_text(exc)}",
                        model=selected_model,
                        request_usages=request_usages,
                    )
                    raise error from exc
                if self._is_blocked_error(exc):
                    error = GeminiBlockedError(
                        f"Gemini blocked generation: {self._safe_error_text(exc)}",
                        model=selected_model,
                        request_usages=request_usages,
                    )
                    raise error from exc
                last_error = exc
                transient = self._is_transient(exc) or isinstance(exc, _RetryableResponseError)
                if not transient or attempt >= self.max_retries:
                    if isinstance(exc, GeminiProviderError):
                        error = self._annotate_error(
                            exc, model=selected_model, request_usages=request_usages
                        )
                        error.args = (self._safe_error_text(error),)
                    else:
                        error = GeminiProviderError(
                            f"Gemini request failed after {attempt + 1} attempts: "
                            f"{self._safe_error_text(exc)}",
                            model=selected_model,
                            request_usages=request_usages,
                        )
                    raise error from exc
            if attempt < self.max_retries:
                # Give quota and transient service errors time to recover.
                base = min(8.0, 1.0 * (2**attempt))
                await asyncio.sleep(base + random.uniform(0.0, min(0.25, base * 0.25)))
        # The loop always returns or raises, but retaining this guard keeps type
        # checkers and unusual injected clients from seeing an implicit None.
        raise GeminiProviderError(
            f"Gemini request failed: {self._safe_error_text(last_error) if last_error else 'unknown error'}",
            model=selected_model,
            request_usages=request_usages,
        ) from last_error

    async def complete_json(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        return await self.complete(prompt, **kwargs)

    async def generate(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        return await self.complete(prompt, **kwargs)

    async def generate_image(
        self,
        prompt: str,
        output_path: str | Path,
        *,
        model: str,
        timeout: float | None = None,
    ) -> ProviderResponse:
        """Generate an image through Gemini's Interactions image endpoint."""
        request_timeout = self.timeout if timeout is None else float(timeout)
        body: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "response_format": {
                "type": "image",
                "mime_type": "image/jpeg",
                "aspect_ratio": "16:9",
                "image_size": "1K",
            },
        }
        async with self._semaphore:
            result = asyncio.to_thread(self._create_image_rest, body, request_timeout)
            result = await asyncio.wait_for(result, timeout=request_timeout) if request_timeout > 0 else await result
        usage = self._usage(result, model)
        output_image = _value(result, "output_image")
        encoded = _value(output_image, "data")
        if not encoded:
            raise GeminiResponseError("Gemini image response did not contain image data", model=model,
                                      request_usages=[usage])
        try:
            image_bytes = base64.b64decode(encoded)
        except (ValueError, TypeError) as exc:
            raise GeminiResponseError("Gemini returned invalid image data", model=model,
                                      request_usages=[usage]) from exc
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image_bytes)
        return ProviderResponse(data=str(path), usage=usage, model=model, raw=result,
                                request_usages=[usage])

    def _create_image_rest(self, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Call the Gemini Interactions image endpoint with a bounded HTTP request."""
        if not self._api_key:
            raise GeminiAuthenticationError(
                "GEMINI_API_KEY is required for live Gemini image generation"
            )
        request = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise GeminiProviderError(
                f"Gemini image request failed with HTTP {exc.code}: {self._safe_error_text(RuntimeError(detail))}",
                model=str(body["model"]),
            ) from exc
        except urllib.error.URLError as exc:
            raise GeminiProviderError(
                f"Gemini image transport failed: {self._safe_error_text(exc)}",
                model=str(body["model"]),
            ) from exc

    async def aclose(self) -> None:
        client = self._client
        if client is None:
            return
        aio = getattr(client, "aio", None)
        close = getattr(aio, "aclose", None) if aio is not None else None
        if close is None:
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    async def close(self) -> None:
        """Alias matching clients that use ``close`` rather than ``aclose``."""
        await self.aclose()


__all__ = [
    "GeminiProvider",
    "GeminiProviderError",
    "GeminiAuthenticationError",
    "GeminiBlockedError",
    "GeminiResponseError",
]
