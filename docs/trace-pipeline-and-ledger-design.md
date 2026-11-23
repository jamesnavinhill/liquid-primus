# Trace Pipeline & Ledger — Design

Status: design, agreed in principle 2026-11-23. Infrastructure shape only;
ledger semantics deliberately deferred. Nothing here is built yet except the
capture and sync layers, which already exist and are described as found.

## Purpose

Two products come out of the gateway's trace stream:

1. **Datasets in prime shape** — traces processed once, correctly, so they can
   be dropped into SFT/RL work at any later time without re-excavation.
2. **A global ledger** — project progress, preferences, and durable state,
   folded out of processing. The ledger is the queryable index. The traces
   themselves are never analytically queried; operational lookups ("latest
   status", "fix this issue") go through PostHog's and Sentry's own MCP/API
   tools over a lookback of days, which needs nothing built.

## Audit findings (verified 2026-11-22/23, corrected where noted)

- **Capture is healthy.** The gateway uses LiteLLM's native PostHog callback,
  subclassed (`AgencyPostHogLogger` in agency `config/litellm/agency_tool_schema_hooks.py`),
  with `turn_off_message_logging: false` — full prompt/completion bodies are
  captured by design.
- **The "null content" finding was a wrong-table artifact.** PostHog stores
  heavy LLM content in `posthog.ai_events` (columns `input`, `output_choices`),
  not on the `events` table, and `$ai_output` is not a `$ai_generation`
  property (the property is `$ai_output_choices`). Live check 2026-11-23:
  6,445/6,445 generation rows in `ai_events` over the trailing 2 days carry
  both fields. Earlier audit conclusions drawn from the `events` table are
  superseded by this paragraph.
- **Payload bounding is deliberate.** Events over a 700 KB budget are windowed
  (first 2 + last 6 messages, 24K-char per-message cap) and stamped
  `$ai_payload_bounded=true` — agency Decision 013 (2026-08-21), adopted after
  oversized events were dropped at ingestion and full-payload batches OOM-killed
  the gateway host. ~12K historical events are bounded; for those, the middle
  messages of the largest threads exist in neither PostHog nor the lake.
- **The ~73% payload presence in R2 exports is composition, not loss.** The
  export job drops nothing. Events without payload keys are failed calls (no
  response object) and non-chat modalities (image generations, ~10K calls in
  the audited window). Exact per-cause split unmeasured; measurable by a
  grouped pass over the export files if ever needed.
- **Retention is the real risk, and it is already mitigated.** `ai_events`
  content carries a 30-day default retention in PostHog. Permanence comes from
  the lake sync (below), so the sync's cadence and reliability are the load-
  bearing property of the whole system.
- **Langfuse is closed.** Retired by agency Decision 018 (2026-08-30); its
  August corpus expired under free-tier retention and is unrecoverable. Its
  metadata shadow in PostHog (31,325 events, Aug 13–31) remains usable as a
  one-month extension of router/outcome data.
- **Assets as inventoried:** R2 `agency-raw` daily PostHog exports
  2026-09-20 → present (44,627 generations in the audited window) and
  `agency-curated` traces (1,491 rows / 1,077 traces, full text, from
  2026-10-07). Sentry holds the gateway/sandbox failure taxonomy. Cloudflare
  holds no LLM request content. The Neon lakehouse schemas exist but hold no
  tables; Neon is not part of this design (see Ledger).

## Architecture

Industry-standard lakehouse shape, batch instead of streaming (deviation
justified by volume: ~45K calls per two months). No hand-rolled replacements
for vendor features: PostHog and Sentry keep doing everything they are good
at; the hand-owned pieces are only the record and the processing.

1. **Capture (exists).** Gateway emits one record per call via the LiteLLM
   PostHog callback subclass. Unchanged, except the open item below.
2. **Fan-out (exists).**
   - *PostHog* — working surface: trace views, LLM evaluations, cost/latency
     insights. Agents query it directly via its MCP/API for operational work.
   - *Sentry* — errors and alarms, including alarms for this pipeline itself.
   - *R2* — the record of record, via `scripts/sync_trace_lakehouse.py`
     (agency repo: `posthog.ai_events` → `agency-raw`, curated set →
     `agency-curated`) plus the PostHog-side batch export. Append-only,
     full fidelity as captured.
3. **Processing (to build).** A versioned, re-runnable job that reads the
   lake and produces:
   - dataset builds (router/outcome table, SFT candidates, eval replay set),
     each with a data card: counts, filters applied, provenance;
   - ledger updates (below).
   Datasets are builds, not stores: regenerated from the lake on demand.
4. **Watchdog (to formalize).** The sync and processing jobs report failure
   to Sentry. A missed sync day or a completeness drop is an alarm, not a
   discovery made months later. (The August Langfuse loss is the reference
   failure: ingestion retired deliberately, archive expired silently.)

## The ledger

Purpose: the one queryable index — of *state*, not of traces. Project
progress, preferences, and durable facts, updated as a byproduct of trace
processing and of other obvious signal sources (git history is the standing
example: progress and preference signal that never appears in traces gets
parsed in from there).

Shape: the standard event-sourcing pattern.

- **Append-only event log.** Every entry is an immutable event in a standard
  envelope: `id`, `ts`, `source` (trace processing / git / relay / manual),
  `subject` (project or entity), `kind`, `payload`, and `provenance`
  (pointers to the evidence: trace IDs, commit SHAs, session IDs). No entry
  is ever edited; corrections are new events.
- **Folded materialized view.** Current state is a fold over the log —
  derived, rebuildable at any time by replaying it, never the system of
  record itself. The view is what agents read.
- **One writer.** Only the processing pipeline appends. Anything else that
  wants the ledger changed emits source material; it does not write.
- **Physical home follows shape, decided at build time.** The ledger is
  small (state, not content). Candidates: the log as partitioned files in R2
  with the view materialized alongside; a small Postgres (Neon) if a served
  query endpoint is ever wanted — that would be a deliberate, named exception
  to the manual-use rule, not a default. D1 is not standard for this and is
  not a candidate.

Semantics — what kinds of events exist, how preferences are extracted, how
conflicts and ageing fold — are explicitly deferred. This document fixes the
infrastructure the semantics will run on, nothing more.

## Open items

1. **Oversized-thread full text.** If the middle messages of >700 KB threads
   are ever wanted in the lake, the fix is a separate direct-to-R2 write of
   oversized payloads at the same hook point in
   `agency_tool_schema_hooks.py` — an addition, not a change to Decision 013.
   Design decision, not a repair. Default: not built.
2. **Sync reliability.** Confirm the lake sync's schedule and failure
   reporting; add the Sentry alarm. This is the highest-value small job in
   the system, because of the 30-day `ai_events` retention.
3. **Quality signal.** No source records whether an output was good; every
   outcome is operational (success/latency/cost). Processing must therefore
   filter on proxies now, and any judged quality (e.g. PostHog evaluations,
   Kaggle Benchmarks credit) attaches to traces as new provenance-bearing
   records, never as edits to the raw lake.

## Decisions recorded

- Industry standard is the north star; deviations are named with reasons
  (batch landing instead of streaming: volume).
- PostHog is the working surface and short-lookback index; Sentry is errors
  and alarms; R2 is the record; the ledger is the state index. No component
  holds a second job by accident after this.
- No capture rewrite: the existing callback and sync are the foundation.
  The remaining engineering is sync reliability, processing, and the ledger.

## Appendix (2026-11-23): as-built infra map + anomaly dispositions

Built the same night this design was approved. What exists now:

- **Capture** — unchanged: LiteLLM native PostHog callback (subclassed,
  Decision 013 bounding) + Sentry integration. Content lands in
  `posthog.ai_events`; retention ~30 days.
- **Sync** — `agency/scripts/sync_trace_lakehouse.py` on a GitHub Actions
  daily cron, now `--days 3 --offset 1` (last 3 completed days, re-synced
  daily; deterministic per-date partitions make re-runs safe). Fail-loud:
  fetch errors raise, exit is non-zero, no silent local-only degrade.
  Per-date manifests at `agency-raw/manifests/dt=<date>.json`. HogQL fetch
  paginates (the endpoint silently caps unpaginated queries at 100 rows).
  Pinned deps in `agency/requirements-lakehouse.txt`.
- **Watchdog + receiver** — `agency/src/lakehouse-watchdog/` on the yrka
  Cloudflare account, independent of the sandbox: twice-daily manifest /
  count / partition verification with Sentry + PostHog alarms, and the
  GitHub webhook receiver appending push events to the ledger log at
  `agency-curated/ledger/events/dt=<date>/<delivery-id>.json`.
- **Ledger** — log live (push tributary); batch tributary (trace
  processing fold) not yet built; semantics still deferred, per design.
- **Sentry** — project `agency` consolidated to the 3 live environments
  (galaxy / cf / local); 4 dead-lane environments hidden; defaults aligned
  to `agency-local` at every source.

Anomalies from the audit, with dispositions:

| # | Anomaly | Disposition |
| --- | --- | --- |
| A1 | Sync cron synced the partial current day, never yesterday in full | **Fixed** — `--days 3 --offset 1` |
| A2 | Silent-green failures (fetch errors returned [], exit 0) | **Fixed** — FetchError, per-date isolation, non-zero exit, failure manifests |
| A3 | Two PostHog→R2 pipelines; the hourly CDP Batch Export (95,877-event firehose) is configured only in PostHog's UI, ownerless in-repo | **Open** — document/own it or retire it; needs an explicit call, not a silent default |
| A4 | Sentry issues fetch ignored the date (hardcoded 24h window) | **Fixed** — lastSeen date window + pagination |
| A5 | Sentry spans capped at 100/day | **Fixed** — cursor pagination |
| A6 | Sync workflow installed unpinned dependencies | **Fixed** — `requirements-lakehouse.txt` |
| A7 | Decision 013 text stale (claims Langfuse full fidelity; wrong callback registration) | **Open** — amend the decision text |
| A8 | Neon lakehouse identity mess (two project names; schemas prepared for a pipeline never built; empty) | **Open** — Neon stays manual-use only; the empty skeleton should be dropped or explicitly adopted, not left ambiguous |
| A9 | Dead `src/tracing` pointer + empty reserved packages | **Open** — cosmetic; delete on next tidy |
| A10 | Hand-rolled PostHog/Sentry `trace.ts` duplicated across Workers (now 3 copies incl. lakehouse-watchdog), against the "official integrations only" doc line | **Open** — extract a shared package or amend the doc line to match reality |
| A11 | Standby (sideway) lane observability parity unverifiable from the repo | **Open** — verify on the lane itself |
| A12 | Stale Langfuse mentions in docs | **Open** — cosmetic sweep |
| A13 | Observability credentials in ≥5 projections, no manifest / regeneration path for the GH Actions lakehouse secret set | **Open** — write the credential manifest |
| A14 | Possibly-dead `POSTHOG_PROJECT_TOKEN` | **Open** — verify and remove if dead |
| A15 | PostHog self-driving scouts half-wired (PR repo source not connected) | **Open** — finish or switch off |
| A16 | Oct-11 changelog describes a Sentry events extraction the code no longer has | **Open** — correct the record |
| A17 | HogQL 100-row truncation of every historical ai_events sync (found by the 2026-11-23 backfill drill: 100 fetched vs 1,160 actual for 2026-11-20) | **Fixed** — LIMIT/OFFSET pagination; days inside the 30-day retention are repairable by re-run, older truncated partitions are not recoverable from PostHog |

Open items 2 (sync reliability) from the list above is therefore closed;
items 1 and 3 stand.
