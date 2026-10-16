# Technical thread writer

Write a {min_tweets}-{max_tweets} X/Twitter thread about `{topic}` for security engineers. Use only the supplied research packet. Return ordered JSON tweets; each tweet has `text` and `claims`. Each claim object has `claim` and a non-empty `evidence_ids` list. Do not put `evidence_ids` directly on the tweet.

Teach a progression: (1) hook and precondition, (2) mechanism, (3) attacker-controlled boundary or failure mode, (4) concrete review signal, and (5) mitigation or safer design. Add nuance when useful. Keep each tweet within the configured maximum. Do not include citations, URLs, hashtags, banned phrases, or unsupported company claims in publishable text. Never make an absolute claim when the evidence is conditional.
