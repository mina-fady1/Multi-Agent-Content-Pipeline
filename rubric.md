---
threshold: 3.5
criteria:
  - id: accuracy
    weight: 0.35
    minimum: 3.0
    description: "Claims match the supplied research evidence and use defensible scope and severity."
  - id: usefulness
    weight: 0.25
    minimum: 3.0
    description: "The piece gives a practical recognition, review, or mitigation takeaway."
  - id: clarity
    weight: 0.20
    minimum: 3.0
    description: "The writing is understandable, structured, and concise for its audience."
  - id: voice
    weight: 0.20
    minimum: 3.0
    description: "The piece follows the calm, precise SanadSquad voice and avoids unsupported promotion."
---

# Review rubric

Review every generated draft, including drafts that already fail a deterministic rule. Score each criterion from 0 (absent or misleading) to 5 (excellent), cite the evidence IDs inspected, and give one concrete revision action for each failed criterion.

A piece passes only when all hard policy checks pass, every factual claim is supported by its mapped evidence, each criterion meets its minimum, and the weighted score reaches the threshold. A high style score cannot override a grounding failure. Semantic entailment is reviewed by a model and remains a limitation; deterministic checks still govern IDs, lengths, URLs, and characters.
