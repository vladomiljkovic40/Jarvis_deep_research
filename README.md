# JARVIS — Deep Research Pipeline

A self-hosted, multi-source research pipeline built on [Open WebUI](https://github.com/open-webui/open-webui) Pipelines, running against a tunneled Ollama-compatible model (`qwen3.6:35b-a3b`, MoE). Instead of synthesizing a single answer from a handful of sources (the standard "deep research" pattern), this pipeline searches broadly, reads many sources in parallel, and presents a **ranked, per-source digest** with full provenance — because small/local models fabricate far more readily during cross-source synthesis than during single-source extraction.

An optional, clearly-labeled synthesis section can be layered on top, with structural safeguards to limit (not eliminate) hallucination risk.

## Why this architecture

Frontier hosted models win on synthesis quality and source curation. This project's bet: a local model can't out-reason a frontier model per-call, but it can out-iterate one — reading far more sources per query than a single-pass frontier response would, and being fully auditable about what it found and didn't find. Depth via iteration, not depth via intelligence.

## Architecture
Open WebUI (chat UI)
|
v
Pipelines server -- deep_research.py
|
v
Ollama-compatible endpoint (your own tunnel/host) -- qwen3.6:35b-a3b
|
v
SearXNG (local) -- web search
|
v
Jina Reader (r.jina.ai) -- fetch/extraction, bypasses most bot-gating

**Pipeline stages per run:**
1. **Plan** - decompose the topic into sub-questions/aspects, generate search queries with a per-query result budget (model-allocated, code-enforced against a total ceiling).
2. **Search + read** (parallel) - fetch via Jina Reader with direct-HTTP fallback, extract facts grounded to a single source only, score relevance, extract publication year.
3. **Gap-fill** - further iterations targeting under-covered aspects.
4. **Digest** - rank by relevance score, dedupe by DOI/title, build a per-source summary list (no cross-source merging).
5. **Question coverage** - descriptive tally of which parts of the original question got covered, thin, or missed (reads only already-extracted fields, not raw content).
6. **Synthesis (optional, experimental)** - cross-source synthesis from already-extracted summaries only, mandatory citations, code-level check flags any sentence with no citation marker.

## Key design decisions

| Decision | Why |
|---|---|
| No default cross-source synthesis | Small/local models hallucinate far more when merging claims across sources than when summarizing one source in isolation. |
| Date filter parsed from query, enforced on extracted publication year in code | The model is unreliable at self-constraining on date ranges; a deterministic post-extraction filter is not. |
| Search-total budget with model-chosen allocation, code-enforced ceiling and floor | The model tends to under-search by default; a prompt-only instruction to use the full budget proved unreliable, so a code-level top-up and a separate overshoot clip enforce it. |
| Domain-level exclusion for confirmed-unfetchable sources | Evidenced from fetch logs: a consistent short paywall-stub response across dozens of URLs on one domain, 100% failure rate, no free workaround. Excluding at the search stage saves the fetch and the wasted LLM call. |
| Reasoning-mode enabled with logging, but not fed back into later prompts | Improved scoring/extraction quality but consumed much more context; num_ctx was raised to accommodate. |
| Synthesis is additive, not a replacement | Preserves the trustworthy per-source digest as the primary output; synthesis is a labeled, lower-confidence bonus section. |

## Known limitations (evidenced, not theoretical)

- **"What research gaps exist" is structurally hard to answer.** Consistently landed at "thinly covered" across every test run regardless of source volume, because source pages rarely state their own limitations explicitly, and single-source extraction can't infer a gap that requires noticing an absence across many papers.
- **Relevance scoring is not the same as source-quality gating.** A well-written low-quality page and a peer-reviewed paper can score similarly if both are topically on-point.
- **Citation-presence is not citation-accuracy.** The synthesis check confirms a claim has a citation marker, not that the citation is correct.
- **Free-tier tunnel infrastructure is a real reliability constraint.** Longer runs increase exposure to tunnel drops and session limits.
- **Model self-constraint is unreliable in general.** Observed repeatedly: search-query syntax leaking through despite explicit prohibition, budget overshoot/undershoot despite explicit targets, task-prompt bleed-through from the host UI's own housekeeping calls. Working pattern: treat every prompt instruction as a strong hint, not a guarantee - add a code-level check for anything that must actually hold.

## Requirements

- Open WebUI + Pipelines server
- SearXNG instance (local, JSON format enabled)
- An Ollama-compatible endpoint serving a thinking-capable model (bring your own - not included here)
- Python: `requests` (everything else is stdlib)

## Configuration (env vars, with fallback defaults in-file)

| Var | Purpose |
|---|---|
| `DR_OLLAMA_URL` | Ollama OpenAI-compatible endpoint |
| `DR_MODEL` | Model name as registered with your endpoint |
| `DR_USE_JINA` | `1`/`0` - route fetches through Jina Reader (default on) |

## Valves (tunable per-run in Open WebUI)

- `total_search_budget` - total URL-fetch slots across initial planning
- `follow_up_iterations` - gap-filling rounds
- `min_content_length` - minimum fetched-content length before an LLM call is attempted
- `max_workers` - parallel fetch/extract concurrency
- `min_relevance_score` - display threshold
- `min_sources_shown` - floor, so output is never empty

## Known open items

- Fixed-domain tunnel migration to remove free-tier interstitial/rotation issues - evaluated, not completed.
- Second keyword pre-filter to reduce reliance on per-source LLM relevance calls - considered, not implemented.
