# SanadSquad content pipeline

This repository generates a week of grounded X/Twitter content for a smart-contract security team. It runs the same orchestration, research, writer, reviewer, hard-rule, visual, audit, and CSV paths in deterministic **demo** mode and in **live** mode through Gemini. Live mode uses Gemini text models and Gemini image models; demo mode uses local Pillow rendering. An isolated **Ollama** test harness lets you run the full pipeline against a local model without changing the Gemini production path or project dependencies.

## Install

Use Python 3.11 or newer:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Unix: source .venv/bin/activate
pip install -r requirements.txt
```

The Python package lives directly in `content_pipeline/` so the repository is
importable and easy to navigate without an extra `src/` layer. Agents,
providers, and policy skills are grouped inside that package; top-level JSON,
Markdown, YAML, and prompt files remain the pipeline's editable inputs.

Copy `.env.example` to `.env` for live settings. Keep secrets out of source files, logs, and artifacts. The sample `approved_company_facts` and `allowed_urls` lists are intentionally empty, so marketing output cannot imply client work, metrics, certifications, guarantees, or outbound links.

## Project structure

```
content_pipeline/
  __init__.py          – Package root (version 0.1.0)
  cli.py               – argparse entry points for week and piece runs
  config.py            – Data-driven configuration and input validation
  models.py            – Pydantic contracts for every handoff
  orchestrator.py      – Async run coordinator with slot isolation
  artifacts.py         – JSON artifact persistence helpers
  exporters.py         – Scheduler-friendly CSV export
  telemetry.py         – UsageLedger, token/cost aggregation
  agents/
    research.py        – KB-grounded research and Gemini topic selection
    writer.py          – Technical-thread, marketing, and generic writers
    reviewer.py        – Rubric scoring + deterministic grounding gate
    visual.py          – Visual spec + Pillow/Gemini image rendering
  providers/
    base.py            – LLMProvider protocol, ProviderResponse, ProviderUsage
    gemini_provider.py – Async Gemini adapter with retry, backoff, image gen
    demo_provider.py   – Deterministic offline provider for tests/samples
  skills/
    evidence_lookup.py – Shared KB retrieval and claim-support verification
    hard_rules.py      – Model-independent publication policy checks
config/
  model_pricing.json   – Gemini pricing entries (intentionally unset)
post_types/
  tech-thread.yaml     – Thread post type (5-10 tweets)
  marketing.yaml       – Single-post marketing type
prompts/
  research.md          – Research agent system instructions
  tech-thread.md       – Technical thread writer prompt
  marketing.md         – Marketing writer prompt
  reviewer.md          – Reviewer agent prompt
  visual.md            – Visual agent prompt
```

## Gemini setup

1. Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey). Check that the selected Google Cloud project has access to the model and enough quota.
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`:

   ```dotenv
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your-key
   GEMINI_MODEL=gemini-3.8-flash
   MAX_CONCURRENT_REQUESTS=2
   ```

`GEMINI_MODEL` is configurable; use a model available to the project rather than silently substituting another model. Optional `GEMINI_RESEARCH_MODEL`, `GEMINI_REVIEW_MODEL`, `GEMINI_WRITER_MODEL`, and `GEMINI_VISUAL_MODEL` variables can override text agents. The Gemini provider reads these from the environment and routes per-agent requests automatically via its `agent_models` map. `GEMINI_IMAGE_MODELS` is an ordered comma-separated fallback list for image generation, and `GEMINI_IMAGE_TIMEOUT` controls the timeout per image model. Set `GEMINI_IMAGE_FALLBACK=pillow` only when you want a live Gemini text run to continue with a local image if image quota is unavailable. If visual-model or image calls fail, the grounded visual specification is retained. The same key is shared by all agents. `MAX_CONCURRENT_REQUESTS` bounds provider pressure; command-line `--max-concurrency` can further lower the bound for one run.

The editable input paths can be overridden with `PIPELINE_BRAND_VOICE`, `PIPELINE_KB`, `PIPELINE_RUBRIC`, `PIPELINE_PLAN`, and `PIPELINE_PRICING`. Relative values are resolved from the project root; explicit CLI paths still take precedence.
3. Run a live command below. The CLI loads `.env` automatically. To use the deterministic offline path, pass `--mode demo` explicitly.

Gemini is the only supported live provider. The pipeline does not use OpenAI or Anthropic credentials or SDKs.

## Run the sample

```bash
python run_week.py --plan plan.json --output output/sample-week --mode demo
python run_piece.py --type tech-thread --topic "reentrancy" --mode demo
```

For a live Gemini run, the command-line interface is unchanged:

```bash
python run_piece.py --type tech-thread --topic "reentrancy"
python run_week.py --plan plan.json --output output/week-42
python run_week.py --plan plan.json --set thread_2="oracle manipulation"
```

For an isolated local Ollama test without changing the Gemini pipeline, use
`run_ollama_piece.py` or `run_ollama_week.py`. These harnesses inject an Ollama
adapter into the existing runner at runtime and use Pillow for images:

```powershell
python run_ollama_piece.py --type tech-thread --topic "reentrancy" --output output/ollama-piece
python run_ollama_week.py --plan plan.json --output output/ollama-week --max-concurrency 1
```

The Ollama harnesses accept `--model` (default `qwen3.5:4b-q4_K_M`), `--base-url` (default `http://127.0.0.1:11434`), and `--timeout` (default 180 seconds). Environment variables `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, and `OLLAMA_TIMEOUT` set the defaults. The adapter normalizes small-model quirks such as percentage-range scores and rubric criterion aliases, retries up to three times with linear backoff, and bypasses Gemini image generation by forcing `GEMINI_IMAGE_FALLBACK=pillow`.

A week has exactly six slots in `plan.json` (three `tech-thread` and three `marketing`). Demo mode selects empty focuses deterministically from unused KB entries; live Gemini asks the Research Agent to choose from the unused KB entries and validates that choice locally. A supplied focus is retained. Prompt placeholders such as `{topic}` and `{min_tweets}` are rendered from the selected slot and post-type configuration before a live request. Override a slot without editing the plan:

```bash
python run_week.py --plan plan.json --set thread_2="oracle manipulation" --mode demo
```

An unknown slot ID or post type fails before execution. In live Gemini mode, the configured model and `GEMINI_API_KEY` are required. Gemini transport failures such as rate limits (`429`), temporary service errors (`503`), and timeouts use bounded exponential backoff with jitter; permanent quota or billing denials fail immediately instead of repeating the request. Empty, blocked, or malformed structured responses are reported as provider failures rather than passed to the next agent. These transport retries are separate from the one allowed content rewrite after reviewer rejection. A failed slot is isolated from the other slots, and live failures never silently turn into demo output.

The CLI returns exit code `0` when all slots ship and exit code `2` when any slot is dropped.

## Inputs and extension points

- `kb.json` is an offline list of 18 vulnerability records. Each record has a stable ID, severity, category, root cause, aliases, five concise evidence points, and primary references. Research packets carry source IDs; writers map every factual claim to those IDs.
- `brand-voice.md` has YAML front matter for tone, banned phrases/characters, hashtags, approved facts, allowlisted URLs, and maximum tweet length. Edit this file to change policy without changing Python.
- `rubric.md` has YAML criteria with per-criterion weights, per-criterion minimums, and the weighted approval threshold. A piece is rejected if any individual criterion score falls below its minimum, even when the weighted average exceeds the threshold.
- `post_types/*.yaml` is the data-driven registry. Add a YAML file and prompt path to add a type; the orchestrator discovers it without a type-specific branch and assigns a generic `WriterAgent`.
- `prompts/*.md` contains all model instructions and is read at runtime.
- `config/model_pricing.json` contains Gemini model entries with `input`, `output`, and applicable cached, thoughts, and tool-token categories left as `null`. Add reviewed USD rates per million tokens for the exact model before relying on live cost estimates; unknown rates remain `null` and are reported as `unknown`.

## Outputs

Each run directory contains resolved plan, research log, draft/review attempts, final content, run summary, usage records, image specs, generated `images/*` when available, and `posts.csv`. Live mode generates accepted-piece JPEG images with Gemini's image models via an ordered fallback list; demo mode renders deterministic 1200×675 PNG cards with Pillow. If live image generation is unavailable and the Pillow fallback is disabled, the accepted piece is exported with a blank image field and its grounded visual spec remains available for later rendering. Image model fallback attempts are bounded by `GEMINI_IMAGE_TIMEOUT` and the ordered `GEMINI_IMAGE_MODELS` list.

Completed Ollama runs are saved under `output/ollama-piece/` and `output/ollama-week/` by default.

`posts.csv` has exactly `Post Text,Image` headers. A technical thread contributes one row per tweet in order; its image path appears only on the first row. A marketing piece contributes one row with its image. Thread IDs, ordering, evidence maps, and scheduling metadata remain in `final_content.json` because this intentionally minimal CSV cannot represent a thread. Image paths are run-relative files; a scheduler that requires hosted URLs must upload them separately.

The console summary reports planned, shipped, and dropped pieces with reasons, provider mode, input/output token totals, and cost (`$0.00` only for demo; live unknowns are reported as `unknown`).

## Migration note

The provider boundary uses Google Gemini through the official `google-genai` SDK. Agent responsibilities, prompts, Pydantic response contracts, evidence IDs, deterministic hard rules, one-rewrite policy, Pillow rendering, CSV columns, audit artifacts, and per-slot fault isolation remain unchanged. Pricing is intentionally unset until Gemini rates are reviewed.

## Quality controls

Python hard rules enforce banned text/characters, hashtags, empty posts, exact allowlisted URLs, tweet lengths, thread counts, evidence IDs, and existing image paths. The reviewer scores accuracy, usefulness, clarity, and voice and checks grounding. Each rubric criterion has a per-criterion minimum; a score below any minimum triggers rejection regardless of the weighted average. A rejected draft gets at most one writer revision using reviewer feedback; a second rejection drops that slot without running the visual agent. Failures are isolated per slot.

The sample is educational content, not a security audit or a guarantee that a protocol is safe.
