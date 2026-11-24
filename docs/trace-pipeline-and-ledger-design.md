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

Anomalies from the audit, with dispositions (A7–A14, A16 re-worked
2026-11-24 by the cleanup job; per-anomaly evidence below the table):

| # | Anomaly | Disposition |
| --- | --- | --- |
| A1 | Sync cron synced the partial current day, never yesterday in full | **Fixed** — `--days 3 --offset 1` |
| A2 | Silent-green failures (fetch errors returned [], exit 0) | **Fixed** — FetchError, per-date isolation, non-zero exit, failure manifests |
| A3 | Two PostHog→R2 pipelines; the hourly CDP Batch Export (95,877-event firehose) is configured only in PostHog's UI, ownerless in-repo | **Open** — document/own it or retire it; needs an explicit call, not a silent default |
| A4 | Sentry issues fetch ignored the date (hardcoded 24h window) | **Fixed** — lastSeen date window + pagination |
| A5 | Sentry spans capped at 100/day | **Fixed** — cursor pagination |
| A6 | Sync workflow installed unpinned dependencies | **Fixed** — `requirements-lakehouse.txt` |
| A7 | Decision 013 text stale (claims Langfuse full fidelity; wrong callback registration) | **Fixed 2026-11-24** — both stale claims amended in place with dated correction notes (Sentry is the remaining full-fidelity surface; registration is `callbacks:`, not success/failure_callback) |
| A8 | Neon lakehouse identity mess (two project names; schemas prepared for a pipeline never built; empty) | **Documented + labels fixed 2026-11-24** — true state verified live (see below); retired-project labels in `live-topology.md` / `d1-system-map.mmd` / `.env.example` corrected; no credential changes |
| A9 | Dead `src/tracing` pointer + empty reserved packages | **Fixed 2026-11-24** — `src/tracing/`, `src/admin/`, `src/evals/` stubs deleted (zero references); the two `trace.ts` header comments pointing at `src/tracing` repointed at `docs/proxy/observability.md` |
| A10 | Hand-rolled PostHog/Sentry `trace.ts` duplicated across Workers (now 3 copies incl. lakehouse-watchdog), against the "official integrations only" doc line | **Plan written 2026-11-24** (dedup plan with file/line evidence below); doc line annotated with the standing exception; no refactor executed unsupervised |
| A11 | Standby (sideway) lane observability parity unverifiable from the repo | **Verified 2026-11-24** — parity confirmed on the lane itself after the OTel bridge re-land: PostHog `logs` records under `deployment.environment=agency-cf` (108 in trailing 2 days, latest 14:23 UTC same day), Sentry issues tagged `environment:agency-cf`; stale "bridge disabled" doc statuses corrected |
| A12 | Stale Langfuse mentions in docs | **Fixed 2026-11-24** — live docs swept (`INSIGHTS_SPEC.md` rail, `benchmarks.md` TTFT source, `target-architecture.md` Neon fact); retirement-aware mentions, tombstones, legacy/archived records left as history |
| A13 | Observability credentials in ≥5 projections, no manifest / regeneration path for the GH Actions lakehouse secret set | **Manifest written 2026-11-24** — 8-projection manifest + GH Actions regeneration commands in agency `docs/operations/credentials.md`; no secret values moved |
| A14 | Possibly-dead `POSTHOG_PROJECT_TOKEN` | **Proven dead 2026-11-24** — duplicate of `POSTHOG_API_KEY` (identical phc_ value, digests match), zero code readers; flagged with evidence, secret NOT deleted (vault entry `agency` untouched); `.env.example` template line removed |
| A15 | PostHog self-driving scouts half-wired (PR repo source not connected) | **Open** — finish or switch off (not in the 2026-11-24 cleanup scope) |
| A16 | Oct-11 changelog describes a Sentry events extraction the code no longer has | **Annotated 2026-11-24** — correction banner per repo convention; the `dataset=events` extraction never existed in any revision of the script (verified against the session's own commit `92b7aa5` and HEAD); the 100-count "backfill numbers" flagged as truncation artifacts (see A17) |
| A17 | HogQL 100-row truncation of every historical ai_events sync (found by the 2026-11-23 backfill drill: 100 fetched vs 1,160 actual for 2026-11-20) | **Fixed** — keyset pagination on (timestamp, uuid) (OFFSET is rejected for personal API keys); days inside the 30-day retention are repairable by re-run, older truncated partitions are not recoverable from PostHog |

## Cleanup evidence (2026-11-24)

Per-anomaly verification and evidence for the rows above. All "live" checks
were run 2026-11-24 through the agency MCP toolbox (PostHog HogQL, Neon SQL,
Sentry, `gh`).

- **A7** — `_ops/decisions/decisions.md` Decision 013 said "Langfuse and
  Sentry keep full fidelity at 100%" (stale since Decision 018 retired
  Langfuse 2026-08-30) and "registered as the success/failure callback"
  (wrong at decision time: `config/litellm/config.yaml:991-1006` registers
  `agency_posthog_logger` under `callbacks:` precisely because dotted
  instances in `success_callback` land in a sync-only list the async proxy
  path never iterates — found and fixed 2026-08-22, recorded in the
  2026-08-22 follow-up of `session-2026-08-21-neon-timeouts-identity-aws-sweep.md`).
  Both claims amended in place with dated correction notes.
- **A8** — True state from records + live probes: the live project is
  `agency-lakehouse` (`wild-tree-40675272`, branch `production`,
  db `neondb`, role `neondb_owner`, region `aws-us-east-2`, org
  `org-lively-recipe-39051864`). The name `rough-voice-79290966`
  (the original 2026-08 lakehouse project, `aws-us-east-1` per
  `_ops/roadmaps/_archive/ROADMAP-2026-08-cloud-foundation.md:47`) is
  retired and absent from the org; `.env`/Worker bindings were repointed
  2026-10-10 (`session-2026-10-10-waf-skip-neon-repoint-…`). Live checks
  2026-11-24: `pg_settings` on the deployed MCP SQL binding shows endpoint
  `ep-damp-mud-axvwm6mg`, branch `br-muddy-base-ax07i3ya`; schemas `raw`,
  `curated`, `kb`, `public` all exist and hold **zero tables**
  (`information_schema.tables` count 0) — "schemas prepared for a pipeline
  never built" confirmed; the `vector` extension 0.8.6 is installed.
  Text fixes: `live-topology.md` (2 nodes named the retired project + a
  `Neon — aws-us-east-1` zone label → corrected to agency-lakehouse /
  aws-us-east-2 with an explicit "index pipeline NOT built" note),
  `diagrams/d1-system-map.mmd` same node, `.env.example` placeholder
  endpoints moved from the retired `ep-shiny-wind…us-east-1` form to the
  live `ep-damp-mud-axvwm6mg…us-east-2` form. No credentials touched.
- **A9** — `src/tracing/__init__.py` (docstring pointing at a nonexistent
  `src/proxy/tracing.py`), `src/admin/__init__.py`, `src/evals/__init__.py`:
  zero references anywhere in the repo (workflows, tests, scripts, Docker,
  packaging) — deleted. The two Workers' `trace.ts` headers that pointed at
  "src/tracing notes" (`agent-sandbox-watchdog/src/trace.ts:3`,
  `scheduler-worker/src/trace.ts:3`) now point at
  `docs/proxy/observability.md`.
- **A10** — dedup plan (to be executed supervised; no refactor done):
  - Evidence: three near-identical 101-line copies —
    `src/agent-sandbox-watchdog/src/trace.ts`,
    `src/scheduler-worker/src/trace.ts`,
    `src/lakehouse-watchdog/src/trace.ts`. Identical structure in all
    three: `TraceEnv` interface at line 14, `SURFACE` const at line 20,
    `timeoutFetch` at 22-29, `capture()` (PostHog `/capture/` POST) at
    line 33, `reportError()` (PostHog capture + raw Sentry store-envelope
    POST) at line 56. `diff` shows the only deltas are the header comment,
    `SURFACE` value, `$lib` string, the PostHog event name inside
    `reportError` (`watchdog.alarm` vs `scheduler.error`), and the Sentry
    `sdk.name` string. Each Worker imports it relatively (e.g.
    `lakehouse-watchdog/src/index.ts:43`); each is an independent wrangler
    project (own `package.json`/`wrangler.jsonc`; no root workspace).
  - Recommended path: single shared source `src/shared/trace.ts`,
    identity (surface / `$lib` / sdk name / error-event name) passed by
    each caller as a small config object, since dashboards reference the
    per-worker event names. Update the three imports, delete the three
    copies. Wrangler's esbuild bundler follows relative imports outside
    the Worker dir, so no registry, no workspace, no new dependency.
  - Alternative (policy-aligned, externally blocked): official
    `@sentry/cloudflare` + Cloudflare Workers Logs OTLP destination →
    PostHog Logs. The destinations API returned 404 on the yrka account
    as of 2026-11-23 (`docs/proxy/observability.md` note) — revisit once
    the dashboard step is done.
  - Verification when executed: per-Worker `npm run typecheck`, deploy via
    each README's path, then confirm per-surface events still land
    (PostHog `events` by `$lib`; the agent-sandbox-watchdog README's `/check`
    alarm drill exercises the Sentry envelope path end-to-end).
  - Doc line: `docs/proxy/observability.md` "official native integrations
    only" now carries the standing-exception annotation pointing here.
- **A11** — Standby-lane parity, verified live 2026-11-24: the OTel bridge
  re-land (commit `7dc6829`, deployed standby-first per
  `session-2026-11-24-reupgrade.md`) is observable from PostHog: `logs`
  table rows with `service_name='agency-gateway'` and resource
  `deployment.environment='agency-cf'` — 108 records in the trailing 2
  days, latest 2026-11-24T14:23:32Z (post-deploy); the Galaxy lane
  (`agency-galaxy`) shows 2,756. Sentry holds standby-lane issues under
  `environment:agency-cf` (5 issues, last seen 2026-11-20/21). Trailing
  2-day `$ai_generation` shows the standby surface family
  (`distinct_id='agency-cf-vscode'`, 14 events) plus deploy-verification
  smokes (`smoke-ops`, 13:31Z). Caveat kept honest: `agency-cf-*` keys are
  also accepted on Galaxy since the 2026-10-26 reconciliation, so the
  load-bearing lane marker is the OTel resource environment — which only
  the standby sets to `agency-cf`. Stale "bridge disabled" statuses in
  `docs/proxy/observability.md` (wiring #6, Logs/Distributed Tracing rows)
  and the "no OpenTelemetry in the stack" line in
  `docs/architecture/live-topology.md` were corrected to the re-landed,
  both-lanes-live state.
- **A12** — Fixed mentions: `docs/design/INSIGHTS_SPEC.md:23` (rail
  mockup listed Langfuse as an observability source; now PostHog, matching
  the §2.1 retirement note), `docs/insights/benchmarks.md:36` (TTFT
  "via Langfuse tracing" → PostHog `$ai_generation` + Sentry spans),
  `docs/architecture/target-architecture.md:88` ("Langfuse-class app
  needs" → neutral). Left as history: decision bodies already amended by
  Decision 018's canon-amendments block, tombstone rows, `_ops/_legacy`,
  `_ops/roadmaps/_archive`, and changelog session records.
- **A13** — Manifest written into agency `docs/operations/credentials.md`
  ("Lakehouse & observability secret set — manifest"): 8 enumerated
  projections with owners (vault `agency` canonical; local `.env`; GH
  Actions repo secrets — 9 lakehouse-sync secrets verified live via
  `gh secret list`, set 2026-10-11, matching the 9 `os.environ` reads in
  `scripts/sync_trace_lakehouse.py:63-224,496-593`; the
  `lakehouse-watchdog`, `agent-sandbox-scheduler`, `agent-sandbox-watchdog`
  Worker secrets; the `agency-mcp-tools` toolbox bindings per
  `deploy_mcp_worker.py` `SECRET_MAP`; the two gateway runtimes), plus the
  regeneration path (`vault get agency` → `gh secret set` × 9 → green sync
  run → watchdog reconciliation). No secret values moved or printed.
- **A14** — `POSTHOG_PROJECT_TOKEN`: no code reader anywhere
  (`scripts/`, `src/`, `config/`, `tests/`, `.github/` — zero hits). Its
  historical consumers were the pre-LiteLLM proxy modules (git `-S`
  history: `6f3dfb0`, `f5ec20c`, both long gone). Vault entry `agency`
  still declares it, holding a value byte-identical to `POSTHOG_API_KEY`
  (same 48-char `phc_` capture token; digests compared without printing).
  GH Actions secrets and the three Workers' secrets do not include it.
  Action taken: flagged in the manifest + appendix; the vault declaration
  deliberately NOT deleted per the standing rule; the duplicate template
  line removed from `.env.example` with an explanatory note.
- **A16** — Correction banner added to
  `_ops/.changelog/session-2026-10-11-full-trace-pipeline-lakehouse-closure.md`
  (the repo's convention, cf. the 2026-11-24 outage-record corrections):
  the record's §2.B claims a Sentry error-events (`dataset=events`)
  extraction that never existed in any revision — the script at the
  session's own commit `92b7aa5` fetches only spans (`dataset=spans`) and
  issues (the `/issues/` endpoint), as does HEAD; the §3 "100 ai_events /
  100 spans" backfill counts are flagged as the 100-row truncation
  artifacts later found and fixed (A17). File kept as the session record.

Open items 2 (sync reliability) from the list above is therefore closed;
items 1 and 3 stand.
