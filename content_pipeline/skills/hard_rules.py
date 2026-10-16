"""Deterministic, model-independent publication policy checks."""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

from ..models import BrandVoice, DraftPiece, PostType, ResearchPacket, RuleViolation
from .evidence_lookup import verify_claim_support

URL_RE = re.compile(r"https?://[^\s<>\]\[(){}\"']+", re.I)
HASHTAG_RE = re.compile(r"(?<!\w)#\w+", re.UNICODE)


def _normal_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return ""
    if parsed.username is not None or parsed.password is not None:
        return ""
    # Keep path/query/fragment exact while normalising scheme and host.  This
    # prevents evil.example.com from passing an allowlist for example.com.
    host = parsed.hostname.lower()
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme.lower(), host, parsed.path, parsed.query, parsed.fragment))


def _violation(code: str, message: str, index: int | None = None) -> RuleViolation:
    return RuleViolation(code=code, message=message, tweet_index=index)


def check_hard_rules(
    draft: DraftPiece,
    research: ResearchPacket | None,
    voice: BrandVoice,
    post_type: PostType | None = None,
    *,
    image: str | None = None,
    require_image: bool = False,
) -> list[RuleViolation]:
    """Check a draft against all machine-enforceable publication constraints."""
    errors: list[RuleViolation] = []
    if post_type is not None:
        count = len(draft.tweets)
        if not post_type.min_tweets <= count <= post_type.max_tweets:
            errors.append(_violation("tweet_count", f"Expected {post_type.min_tweets}-{post_type.max_tweets} tweets; got {count}"))
    elif not draft.tweets:
        errors.append(_violation("tweet_count", "Draft contains no tweets"))

    evidence_ids = {fact.id for fact in research.evidence} if research else set()
    for i, tweet in enumerate(draft.tweets):
        text = tweet.text or ""
        if not text.strip():
            errors.append(_violation("empty_post", "Tweet text is empty", i))
        if len(text) > min(280, voice.max_tweet_length):
            errors.append(_violation("tweet_length", f"Tweet exceeds {min(280, voice.max_tweet_length)} characters", i))
        lowered = text.casefold()
        for phrase in voice.banned_phrases:
            if phrase and phrase.casefold() in lowered:
                errors.append(_violation("banned_phrase", f"Banned phrase present: {phrase}", i))
        for char in voice.banned_characters:
            if char and char in text:
                errors.append(_violation("banned_character", f"Prohibited character present: {char}", i))
        if voice.disallow_hashtags and HASHTAG_RE.search(text):
            errors.append(_violation("hashtag", "Hashtags are not allowed", i))
        urls = URL_RE.findall(text)
        allow = {_normal_url(url) for url in voice.allowed_urls if _normal_url(url)}
        for url in urls:
            if not allow or _normal_url(url.rstrip(".,;:")) not in allow:
                errors.append(_violation("unsupported_url", f"URL is not exactly allowlisted: {url}", i))
        if not tweet.claims:
            errors.append(_violation("missing_claim_mapping", "Every tweet must map factual claims to evidence", i))
        for claim in tweet.claims:
            if not claim.claim.strip() or not claim.evidence_ids:
                errors.append(_violation("malformed_claim", "Claim mapping must include claim text and evidence IDs", i))
            elif research is None:
                errors.append(_violation("missing_research", "Research packet is required for claims", i))
            elif any(eid not in evidence_ids for eid in claim.evidence_ids):
                unknown = [eid for eid in claim.evidence_ids if eid not in evidence_ids]
                errors.append(_violation("unknown_evidence", f"Unknown evidence IDs: {', '.join(unknown)}", i))

    if research is not None:
        for problem in verify_claim_support(draft, research):
            # Existing deterministic errors are intentionally deduplicated by
            # message/code while retaining a useful claim-grounding code.
            if "mapped claim is absent from tweet text" in problem:
                errors.append(_violation("claim_not_in_tweet", problem))
            elif "not supported" in problem:
                errors.append(_violation("unsupported_claim", problem))
            elif "unknown evidence" in problem and not any(
                v.code == "unknown_evidence" and v.message.casefold().endswith(problem.rsplit(": ", 1)[-1].casefold())
                for v in errors
            ):
                errors.append(_violation("unknown_evidence", problem))

    if require_image and (not image or not __import__("pathlib").Path(image).is_file()):
        errors.append(_violation("missing_image", "Accepted content must have an existing image"))
    return errors


def validate_draft(*args, **kwargs) -> list[RuleViolation]:
    return check_hard_rules(*args, **kwargs)


__all__ = ["check_hard_rules", "validate_draft"]
