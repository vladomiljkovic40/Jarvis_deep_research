"""
title: JARVIS Deep Research
author: vlado
version: 2.2
description: Multi-iteration research. Searches, reads, and presents best-matching
             sources as ranked per-source summaries. No cross-source synthesis.
             Parallel fetch+extract. Date-range enforced on extracted publication
             year. Jina Reader fetch with direct fallback. Loud LLM-error surfacing.
"""

import os
import re
import json
import requests
from datetime import datetime
from typing import Iterator, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

OUTPUT_DIR = "/home/vlado/documents"
SEARXNG_URL = "http://localhost:8888/search"

# Config is env-driven so the ngrok URL (which rotates every Kaggle session) is set
# in ONE place (start-jarvis.sh) instead of hardcoded here. Fallbacks = current values.
OLLAMA_URL = os.getenv("DR_OLLAMA_URL",
                       "https://your-tunnel-url.ngrok-free.dev/v1")
MODEL = os.getenv("DR_MODEL", "qwen3.6:35b-a3b")

# Jina Reader proxy: bypasses most bot-gating/JS and returns clean text.
# Set DR_USE_JINA=0 to disable and fetch directly.
USE_JINA = os.getenv("DR_USE_JINA", "1") == "1"
JINA_PREFIX = "https://r.jina.ai/"

CTX_SMALL = 24576  # raised from 8192: thinking-mode reasoning was exhausting context before the formatted answer, causing silent truncation -> empty content -> false skips

CURRENT_YEAR = datetime.now().year


class LLMUnreachable(Exception):
    """Raised when an LLM call fails, so failure is loud instead of silent 0-sources."""


class Pipeline:
    class Valves(BaseModel):
        max_search_queries: int = Field(default=8)  # kept as fallback for gap-fill queries
        results_per_query: int = Field(default=5)       # kept as fallback for gap-fill queries
        total_search_budget: int = Field(default=40)     # total URL slots for initial planning; model allocates freely
        follow_up_iterations: int = Field(default=2)
        min_content_length: int = Field(default=300)
        max_workers: int = Field(default=4)
        min_relevance_score: int = Field(default=40)
        min_sources_shown: int = Field(default=5)
        output_format: str = Field(default="md")

    def __init__(self):
        self.valves = self.Valves()
        self._date_range: Optional[Tuple[int, int]] = None
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- LLM ----------

    def _ollama(self, messages, temperature=0.3, num_ctx=CTX_SMALL) -> str:
        """Raises LLMUnreachable on transport failure. think=True enables reasoning;
        reasoning is logged to a sidecar file, not fed back into prompts."""
        try:
            r = requests.post(
                f"{OLLAMA_URL}/chat/completions",
                json={
                    "model": MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": False,
                    "think": True,
                    "options": {"num_ctx": num_ctx},
                },
                timeout=300,
            )
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            reasoning = msg.get("reasoning", "")
            if reasoning:
                self._log_reasoning(reasoning)
            return msg.get("content", "")
        except Exception as e:
            raise LLMUnreachable(f"{OLLAMA_URL} model={MODEL}: {e}")

    def _log_reasoning(self, reasoning: str):
        """Sidecar log of model reasoning - inspect manually, not read by pipeline."""
        try:
            with open(self._reasoning_log_path, "a", encoding="utf-8") as f:
                f.write(reasoning.strip() + "\n\n---\n\n")
        except Exception:
            pass

    # ---------- Date range parsing ----------

    def _parse_date_range(self, topic: str) -> Optional[Tuple[int, int]]:
        """Extract an explicit year range from the query. Returns None if the user
        gave no years -> no date filtering (does not invent a range)."""
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", topic)]
        low_since = re.search(r"\b(?:since|after|from)\s+(19\d{2}|20\d{2})\b", topic, re.I)
        if low_since:
            return (int(low_since.group(1)), CURRENT_YEAR)
        if not years:
            return None
        if len(years) == 1:
            return (years[0], years[0])
        return (min(years), max(years))

    # ---------- Search / fetch ----------

    def _sanitize_query(self, q: str) -> str:
        """Strip boolean operators / quotes / date-range colons that made engines
        return nothing (observed: iteration-2 gap queries fetched 0 sources)."""
        q = q.replace('"', " ")
        q = re.sub(r"\b(AND|OR|NOT)\b", " ", q)
        q = q.replace("(", " ").replace(")", " ")
        q = re.sub(r"\d{4}\.\.\d{4}", " ", q)  # "2023..2025" range syntax
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def _search(self, query, n=5):
        try:
            r = requests.get(
                SEARXNG_URL,
                params={"q": self._sanitize_query(query), "format": "json",
                        "language": "en", "safesearch": 0},
                timeout=15,
            )
            # ScienceDirect confirmed 100% paywall-stub failure across every URL
            # tried in real runs (evidenced: dozens of fetches all returning ~559
            # char consent/cookie stubs, zero real content). Excluded at the
            # search stage to skip both the wasted fetch AND the wasted LLM call.
            EXCLUDED_DOMAINS = ("sciencedirect.com",)
            results = r.json().get("results", [])[:n]
            return [
                {"title": x.get("title", ""), "url": x.get("url", ""),
                 "snippet": x.get("content", "")}
                for x in results
                if x.get("url") and not any(d in x["url"] for d in EXCLUDED_DOMAINS)
            ]
        except Exception:
            return []

    def _fetch_url(self, url) -> str:
        log = os.path.join(OUTPUT_DIR, "fetch_debug.log")
        def dbg(msg):
            try:
                with open(log, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
        if USE_JINA:
            try:
                r = requests.get(JINA_PREFIX + url, timeout=30,
                                 headers={"User-Agent": "Mozilla/5.0 (research bot)"})
                dbg(f"JINA {r.status_code} len={len(r.text)} {url}")
                if r.status_code == 200 and len(r.text) > 200:
                    return re.sub(r"\s+", " ", r.text).strip()[:8000]
            except Exception as e:
                dbg(f"JINA EXC {type(e).__name__}: {e} {url}")
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (research bot)"})
            text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", r.text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            clean = re.sub(r"\s+", " ", text).strip()
            dbg(f"DIRECT {r.status_code} len={len(clean)} {url}")
            return clean[:8000]
        except Exception as e:
            dbg(f"DIRECT EXC {type(e).__name__}: {e} {url}")
            return ""

    # ---------- Planning ----------

    def _plan_queries(self, topic):
        budget = self.valves.total_search_budget
        prompt = (
            f'Identify the distinct sub-questions or aspects within this research topic '
            f'(a topic may have one aspect, or several - e.g. a technical question, a '
            f'career question, and a venue question are three separate aspects).\n\n'
            f'For each aspect, propose search queries AND how many results to pull for '
            f'each query (more results for aspects where good sources are sparse or hard '
            f'to predict, fewer for aspects where a search term hits precisely). Give '
            f'every distinct aspect at least 2 queries - do not let one aspect dominate '
            f'at the expense of others.\n\n'
            f'Use the FULL budget of {budget} total results across all queries - do not '
            f'economize or use fewer than the budget allows. It is better to fetch too '
            f'many candidate sources and let a later relevance filter discard the weak '
            f'ones, than to under-search. Distribute the full {budget} across your queries.\n\n'
            f'Topic: {topic}\n\n'
            "Use plain-language keyword queries only. Do NOT use boolean operators "
            "(AND/OR), quotes, or date-range syntax.\n"
            'Return ONLY a JSON array of objects, each with "query" and "n_results" keys. '
            'No explanation.\n'
            'Example: [{"query": "query one", "n_results": 5}, {"query": "query two", "n_results": 3}]'
        )
        result = self._ollama([{"role": "user", "content": prompt}], temperature=0.2)
        parsed = []
        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                arr = json.loads(match.group())
                if isinstance(arr, list) and arr:
                    for item in arr:
                        if isinstance(item, dict) and "query" in item:
                            q = str(item["query"])
                            n = item.get("n_results", 5)
                            try:
                                n = max(1, int(n))
                            except Exception:
                                n = 5
                            parsed.append({"query": q, "n_results": n})
        except Exception:
            pass

        if not parsed:
            return [{"query": topic, "n_results": min(budget, self.valves.results_per_query)}]

        # Code-level floor: prompt asks the model to use the full budget, but this
        # model is not reliable at self-constraining toward a target (same class of
        # issue as boolean-query leakage and budget-overshoot elsewhere in this file).
        # If it undershoots by more than 20%, top up the last query's n_results
        # (cheapest fix - avoids re-planning) so real search volume approaches budget.
        total_planned = sum(p["n_results"] for p in parsed)
        if total_planned < budget * 0.8:
            parsed[-1]["n_results"] += (budget - total_planned)

        # Code-level enforcement: model is not reliable at self-constraining totals
        # (same class of issue as boolean-query syntax earlier). Scale down
        # proportionally if it overshoots the budget rather than trusting the prompt.
        total = sum(p["n_results"] for p in parsed)
        if total > budget:
            scale = budget / total
            for p in parsed:
                p["n_results"] = max(1, round(p["n_results"] * scale))

        return parsed

    def _identify_gaps(self, topic, memory):
        facts_summary = "\n\n".join(f"[{m['url']}] {m['summary'][:300]}" for m in memory[:10])
        prompt = (
            f'Review this research on: "{topic}"\n\nFound so far:\n{facts_summary}\n\n'
            "Identify 3 specific aspects not yet covered. "
            "Use plain-language keyword queries only. Do NOT use boolean operators "
            "(AND/OR), quotes, or date-range syntax.\n"
            'Return ONLY a JSON array of 3 search queries.\n'
            'Example: ["gap query 1", "gap query 2", "gap query 3"]'
        )
        result = self._ollama([{"role": "user", "content": prompt}])
        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                arr = json.loads(match.group())
                if isinstance(arr, list):
                    return [str(x) for x in arr]
        except Exception:
            pass
        return []

    # ---------- Per-source extraction ----------

    def _extract(self, topic, result) -> Optional[dict]:
        content = self._fetch_url(result["url"])
        if len(content) < self.valves.min_content_length:
            return None

        prompt = (
            f'Question being researched: "{topic}"\n\n'
            f"Source URL: {result['url']}\n"
            f"Source content (use ONLY this, do not add outside knowledge):\n{content[:4000]}\n\n"
            "Assess how well THIS source answers the question, then summarise it.\n"
            "Rules:\n"
            "- Score based on how well this source addresses the SPECIFIC aspect it covers, "
            "not the entire multi-part question. A source that thoroughly covers one legitimate "
            "aspect of the question should score 60+ even if it does not address other aspects.\n"
            "- Use only facts present in the content above. Do not invent numbers, dates, or claims.\n"
            "- If the content is irrelevant to the question, reply with exactly: IRRELEVANT\n\n"
            "Otherwise reply in EXACTLY this format:\n"
            "SCORE: <integer 0-100, how directly this source answers the question>\n"
            "PUBLISHED: <4-digit publication year if stated in the content, else UNKNOWN>\n"
            "ADDRESSES: <one short line: which part of the question this source covers>\n"
            "SUMMARY:\n"
            "- <fact or point, with any specific number/date from the source>\n"
            "- <3 to 6 bullets total>"
        )
        raw = self._ollama([{"role": "user", "content": prompt}], temperature=0.2)

        if raw.strip().upper().startswith("IRRELEVANT"):
            return None
        score = self._parse_score(raw)
        if score < 15:
            return None
        summary = self._parse_summary(raw)
        if not summary:
            return None

        return {
            "url": result["url"],
            "title": result.get("title") or self._domain(result["url"]),
            "score": score,
            "year": self._parse_year(raw),
            "addresses": self._parse_line(raw, "ADDRESSES"),
            "summary": summary,
        }

    # ---------- Parsing helpers ----------

    @staticmethod
    def _parse_score(raw) -> int:
        m = re.search(r"SCORE:\s*(\d{1,3})", raw, re.IGNORECASE)
        return max(0, min(100, int(m.group(1)))) if m else 30

    @staticmethod
    def _parse_year(raw) -> Optional[int]:
        m = re.search(r"PUBLISHED:\s*(19\d{2}|20\d{2})", raw, re.IGNORECASE)
        return int(m.group(1)) if m else None

    @staticmethod
    def _parse_line(raw, label) -> str:
        m = re.search(rf"{label}:\s*(.+)", raw, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _parse_summary(raw) -> str:
        m = re.search(r"SUMMARY:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
        body = m.group(1).strip() if m else raw.strip()
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        bullets = [ln if ln.startswith(("-", "*", "•")) else f"- {ln}" for ln in lines]
        return "\n".join(bullets[:6])

    @staticmethod
    def _domain(url) -> str:
        m = re.search(r"https?://([^/]+)", url)
        return m.group(1) if m else url

    # ---------- Date-range gate (code-level, not model judgment) ----------

    def _date_verdict(self, src) -> str:
        """Returns 'keep', 'drop', or 'flag' (in-range / out-of-range / unknown-year)."""
        if not self._date_range:
            return "keep"
        yr = src.get("year")
        if yr is None:
            return "flag"
        lo, hi = self._date_range
        return "keep" if lo <= yr <= hi else "drop"

    # ---------- Parallel batch ----------

    def _process_batch(self, topic, results, seen):
        todo = []
        for res in results:
            if res["url"] in seen:
                continue
            seen.add(res["url"])
            todo.append(res)
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=self.valves.max_workers) as pool:
            futures = {pool.submit(self._extract, topic, res): res for res in todo}
            for fut in as_completed(futures):
                res = futures[fut]
                try:
                    src = fut.result()
                except LLMUnreachable:
                    raise  # propagate: this is a systemic failure, abort loudly
                except Exception:
                    src = None
                if not src:
                    yield f"    skipped: {self._domain(res['url'])}\n", None
                    continue
                verdict = self._date_verdict(src)
                if verdict == "drop":
                    yield f"    dropped (year {src['year']} out of range): {self._domain(src['url'])}\n", None
                elif verdict == "flag":
                    src["date_flag"] = True
                    yield f"    kept (year unverified): {self._domain(src['url'])} ({src['score']})\n", src
                else:
                    yield f"    kept: {self._domain(src['url'])} ({src['score']})\n", src

    # ---------- Dedup ----------

    @staticmethod
    def _doi(url) -> Optional[str]:
        m = re.search(r"(10\.\d{4,9}/[^\s/?#]+)", url)
        return m.group(1).lower() if m else None

    def _dedup(self, memory):
        """Collapse the same paper reached via different URLs (DOI match, or same
        title prefix). Keep the highest-scoring copy. Observed: Springer #2 and
        PubMed #4 were the same review counted twice."""
        best = {}
        for m in memory:
            key = self._doi(m["url"]) or re.sub(r"\W+", "", m["title"].lower())[:40]
            if key not in best or m["score"] > best[key]["score"]:
                best[key] = m
        return list(best.values())

    # ---------- Optional synthesis (experimental, higher hallucination risk) ----------

    def _synthesize_findings(self, topic, kept_sources) -> str:
        """Optional cross-source synthesis. Only sees already-extracted per-source
        summaries (never raw content), must cite [N] per claim, and gets a code-level
        citation-presence check afterward - claims with no citation marker are flagged,
        not silently trusted. Does not verify citation ACCURACY, only presence."""
        if not kept_sources:
            return ""

        numbered = "\n\n".join(
            f"[{i}] {m['title']}{' (' + str(m['year']) + ')' if m.get('year') else ''} "
            f"- relevance {m['score']}/100\n{m['summary']}"
            for i, m in enumerate(kept_sources, 1)
        )

        prompt = (
            f'You are synthesizing findings from the following grounded, per-source '
            f'summaries only, to help answer this research question:\n'
            f'"{topic}"\n\n'
            f'{numbered}\n\n'
            "Rules:\n"
            "- Every claim in your synthesis must be traceable to at least one specific "
            "source above. Cite sources inline using [N] matching the numbers above.\n"
            "- If sources disagree, state the disagreement explicitly rather than picking a side.\n"
            "- Do NOT introduce any fact, number, date, or claim that is not present in "
            "the summaries above. You are working from summaries, not original source text - "
            "do not add specificity beyond what the summaries state.\n"
            "- If the provided sources are insufficient to answer part of the question, "
            "say so explicitly rather than filling the gap.\n\n"
            "Write a synthesis of 200-400 words."
        )
        result = self._ollama([{"role": "user", "content": prompt}], temperature=0.3)

        sentences = re.split(r"(?<=[.!?])\s+", result.strip())
        checked = []
        for s in sentences:
            if s.strip() and not re.search(r"\[\d+\]", s):
                checked.append(s.strip() + " [unsupported claim - no source cited]")
            else:
                checked.append(s.strip())
        return " ".join(checked)

    # ---------- Digest ----------

    def _analyze_coverage_gaps(self, topic, memory) -> str:
        """Descriptive tally, not synthesis: reviews only the ADDRESSES fields already
        extracted per-source (never raw content) and reports which named aspects of
        the original question got thin or no coverage. Cannot fabricate findings -
        it can only notice absence of coverage, which the extracted ADDRESSES fields
        make directly observable."""
        addresses_list = "\n".join(f"- {m.get('addresses','')}" for m in memory if m.get("addresses"))
        if not addresses_list:
            return ""
        prompt = (
            f'Original research question: "{topic}"\n\n'
            f'Here is what each gathered source was found to address (one line per source):\n'
            f'{addresses_list}\n\n'
            "List the distinct aspects or sub-questions in the ORIGINAL question above. "
            "For each aspect, state whether it is: WELL COVERED, THINLY COVERED, or NOT COVERED, "
            "based ONLY on whether the source-addresses list above mentions it. "
            "Do not add outside knowledge or guess at findings - only report presence or "
            "absence of coverage.\n\n"
            "Reply in this format, one line per aspect:\n"
            "- <aspect name>: <WELL COVERED / THINLY COVERED / NOT COVERED> - <one line why>"
        )
        result = self._ollama([{"role": "user", "content": prompt}], temperature=0.2)
        return result.strip()

    def _build_digest(self, topic, memory) -> str:
        memory = self._dedup(memory)
        ranked = sorted(memory, key=lambda m: m["score"], reverse=True)
        kept = [m for m in ranked if m["score"] >= self.valves.min_relevance_score]
        if len(kept) < self.valves.min_sources_shown:
            kept = ranked[: self.valves.min_sources_shown]

        lines = [f"# Research: {topic}\n",
                 f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}._\n"]
        if self._date_range:
            lines.append(f"_Date filter: {self._date_range[0]}–{self._date_range[1]} "
                         f"(enforced on source publication year)._\n")

        scores = [m["score"] for m in kept]
        lines.append("## Coverage\n")
        lines.append(f"- Sources presented: {len(kept)} (of {len(memory)} after dedup)")
        if scores:
            lines.append(f"- Relevance range: {min(scores)}-{max(scores)} / 100")
        facets = [m["addresses"] for m in kept if m.get("addresses")]
        if facets:
            lines.append("- Aspects covered:")
            for f in facets:
                lines.append(f"  - {f}")
        lines.append("")

        gaps = self._analyze_coverage_gaps(topic, memory)
        if gaps:
            lines.append("## Question Coverage\n")
            lines.append("_What aspects of your original question were and were not addressed "
                         "by the gathered sources (based only on their stated scope, not their content)._\n")
            lines.append(gaps)
            lines.append("")

        lines.append("## Best-matching sources\n")
        for i, m in enumerate(kept, 1):
            yr = m.get("year")
            flag = " [year unverified]" if m.get("date_flag") else ""
            yrtxt = f" ({yr})" if yr else ""
            lines.append(f"### {i}. {m['title']}{yrtxt} — relevance {m['score']}/100{flag}")
            if m.get("addresses"):
                lines.append(f"*Addresses:* {m['addresses']}")
            lines.append(m["summary"])
            lines.append(f"Source: {m['url']}\n")
        return "\n".join(lines)

    def _save_output(self, topic, content) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c for c in topic[:30] if c.isalnum() or c in " -_").strip()
        path = os.path.join(OUTPUT_DIR, f"research_{safe}_{ts}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    # ---------- Main ----------

    def pipe(self, user_message: str, model_id: str, messages: list, body: dict) -> Iterator[str]:
        topic = user_message.strip()
        # Guard: Open WebUI fires title/tags/follow-up generation against whatever
        # model is "current" in a conversation. If deep_research is current, these
        # housekeeping prompts land here instead of a real research query.
        _task_markers = ("### Task:", '"follow_ups"', '"title":', '"tags":')
        if any(m in topic for m in _task_markers):
            yield '{"error": "not a research query - task-generation prompt detected"}'
            return
        seen, memory = set(), []
        t0 = datetime.now()
        self._reasoning_log_path = os.path.join(
            OUTPUT_DIR, f"reasoning_{t0.strftime('%Y%m%d_%H%M%S')}.log")
        self._date_range = self._parse_date_range(topic)

        yield f"Deep Research: {topic}\n\n"
        if self._date_range:
            yield f"Date filter: {self._date_range[0]}-{self._date_range[1]} (from query)\n\n"
        else:
            yield "Date filter: none (no year given in query)\n\n"

        try:
            yield "[1/4] Planning search queries\n"
            queries = self._plan_queries(topic)
            yield f"  {len(queries)} queries generated\n\n"

            yield "[2/4] Searching and reading sources\n"
            for i, q in enumerate(queries, 1):
                query, n_results = q["query"], q["n_results"]
                yield f"  query {i}/{len(queries)} (n={n_results}): {query}\n"
                for line, src in self._process_batch(topic, self._search(query, n_results), seen):
                    yield line
                    if src:
                        memory.append(src)
            yield f"\n  {len(memory)} sources kept\n\n"

            yield "[3/4] Filling gaps\n"
            for it in range(self.valves.follow_up_iterations):
                yield f"  iteration {it + 1}/{self.valves.follow_up_iterations}\n"
                for gap_query in self._identify_gaps(topic, memory):
                    yield f"  query: {gap_query}\n"
                    for line, src in self._process_batch(topic, self._search(gap_query, 3), seen):
                        yield line
                        if src:
                            memory.append(src)
            yield f"\n  {len(memory)} sources total\n\n"

        except LLMUnreachable as e:
            yield f"\nABORTED: LLM unreachable — {e}\n"
            yield "Check the tunnel URL / model name / Kaggle session.\n"
            return

        yield "[4/4] Ranking and building digest\n"
        if not memory:
            yield "\nNo usable sources found. Try a broader or differently worded topic.\n"
            return

        digest = self._build_digest(topic, memory)

        yield "Building experimental synthesis (higher hallucination risk - verify against sources above)\n"
        ranked_for_synthesis = sorted(memory, key=lambda m: m["score"], reverse=True)
        kept_for_synthesis = [m for m in ranked_for_synthesis if m["score"] >= self.valves.min_relevance_score]
        if len(kept_for_synthesis) < self.valves.min_sources_shown:
            kept_for_synthesis = ranked_for_synthesis[: self.valves.min_sources_shown]
        synthesis = self._synthesize_findings(topic, kept_for_synthesis)
        if synthesis:
            digest += (
                "\n\n## Synthesis (experimental - higher hallucination risk than the "
                "per-source summaries above; unmarked claims below have no cited source)\n\n"
                f"{synthesis}\n"
            )

        elapsed = (datetime.now() - t0).total_seconds()
        digest += f"\n\n_Run time: {elapsed/60:.1f} min._\n"
        path = self._save_output(topic, digest)

        yield f"\nSaved to: {path}\n"
        yield f"Run time: {elapsed/60:.1f} min\n\n---\n\n"
        yield digest
