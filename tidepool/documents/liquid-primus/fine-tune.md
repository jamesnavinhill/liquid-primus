# Fine-tune — our workflows, the stacks we specialize for, and the data map

This doc defines **what the models should be good at** (the specialization target) and maps
the data that teaches it. The *how* (LoRA vs full, which recipe, what mix ratio) is Primus's
to design — this is the spec of the outcome, not a training pipeline.

## Who's consuming these models

A single agent gateway (litellm/langfuse routing) sits in front of our work. Everything —
research, planning, implementing, auditing, tool calls, code execution, browser-use — flows
through that harness. On top of it runs a **voice/video agent** (ElevenLabs + Deepgram
voice; Anam + Vivix video) that delegates work to the gateway via APIs/MCP. The goal for
this project: fill the API-cost line with a fleet of small, local, stack-trained models that
the gateway can route to per-task, including one hybrid "brain" that covers most tasks alone.

Task classes the fleet must serve (this is what "our stack" means, at the underlying-tech
level — deliberately not pinned to one vendor's SDK):

1. **Code production & review** — TypeScript/JavaScript/React/Node/Next.js, Python,
   Electron/Tauri IPC + Vite configs, shell (PowerShell + bash). Produces idiomatic code
   for *our* repo shapes and reviews/audits diffs.
2. **Tool & function calling** — strict structured output (JSON/tool-call templates) for
   MCP servers, REST APIs, and our own harness tools; multi-step chaining; handling bad or
   empty tool results without hallucinating the missing data. (This is the top-priority
   axis per `readme.md` — and a documented community pain point with small models:
   confidently asserting wrong tool output. Our models should be trained to flag, not
   assert, on malformed tool returns.)
3. **Agentic multi-step work** — research → plan → implement → audit loops, long-horizon
   task state, file-system reasoning (folder structure, paths, git diffs).
4. **Data & telemetry** — SQL for Postgres (Supabase/Neon) *and* ClickHouse, Mongo/Agg
   pipelines; reading raw error payloads (Sentry-style) and event streams (PostHog-style)
   into debugging plans. Telemetry/tracing concepts (OpenTelemetry semantics).
5. **Vision/screenshot understanding** — reading screens, editor state, UIs, documents,
   charts (the browser-use subagent's sense organ). This is the VL tier's job.
6. **Infra & platform ops** — Docker (Windows + Linux), cloud deployment (Cloudflare-first,
   with AWS/Oracle), edge runtimes, CI — producing correct config and diagnosing deploys.

Aim at the *underlying technology* (SQL, JSON schemas, MCP, IPC, containers, tracing
semantics) rather than one vendor's idiosyncrasies, so the fleet stays useful as the stack
shuffles — this is a deliberate owner decision, not a suggestion.

## What "specialized to our stack" should produce

- Code output that matches our conventions and repo shapes without being asked
  (framework idioms, file layout, naming) — measured on a small in-house eval built from
  our own open-source repos (not committed here; Primus can be pointed at the repos to
  extract SFT pairs: e.g. files showing Electron `invoke()` IPC, Tauri command definitions,
  Vite/Next configs, ClickHouse schema + query pairs).
- Tool calls that are structurally valid on first attempt and that **refuse to fabricate**
  when a tool returns garbage.
- Debugging plans that read a real error payload (Sentry issue JSON, stack trace, event
   stream) to the likely root-cause locus, not a generic checklist.
- SQL that is dialect-correct (Postgres vs ClickHouse) and references only real
  schema objects (no invented columns) — the classic text-to-SQL failure.
- (VL) Screen grounding: point at a UI element named in natural language; read a chart or
  table from a screenshot; describe editor state.

## The fine-tune options (menu, not a plan)

All first-party-supported for this family (`Liquid4All/cookbook` notebooks; VL-3B support
added ~2026-08):

- **SFT (LoRA or full)** — `sft_with_trl.ipynb`, `sft_with_unsloth.ipynb`,
  `sft_for_vision_language_model(_with_trl).ipynb` (VL-3B).
- **DPO** — `dpo_with_trl.ipynb` (preference; pairs generated on Liquid's own
  `antidoom-mix` prompt set).
- **GRPO / RL** — `grpo_for_verifiable_tasks.ipynb`, `grpo_with_unsloth.ipynb` — verifiable
  rewards (schema validity, SQL execution, test pass) are ideal for our tool-call and
  data axes; our harness *is* the environment for agentic rollouts (Liquid's 2.6B agentic-RL
  used black-box harnesses with a trajectory-capturing proxy — our gateway can serve the
  same role).
- **CPT** — `cpt_text_completion_with_unsloth.ipynb` if domain-continual pretraining turns
  out to be needed (likely not, before trying SFT+RL).
- **Distillation** — specialist-teacher → single-student MOPD (Liquid's 2.6B recipe) and
  QAD for the quant step (`docs/quants.md`).
- **In-house extraction** — the cheapest high-value data: mining our own repos + our
  gateway's past transcripts (research/plan/implement/audit trajectories, tool calls with
  results) into SFT/RL pairs. This is the data no public dataset has, and it is the core
  of "specialized to our workflows."

## Dataset map (full verified catalog in `datasets/`)

- **Tool/function calling (SFT+eval)**: `Salesforce/xlam-function-calling-60k` (60k,
  execution-verified, CC-BY — the real one; *not* `Trelis/Function_Calling_Extended`,
  which is 59 paid rows — correct the outline), Plus: BFCL eval set, ToolSandbox,
  τ²-Bench as eval.
- **Code (SFT)**: `m-a-p/CodeFeedback-Filtered-Instruction` (OpenCodeInterpreter,
  Apache-2.0) for iterative code + execution feedback; in-house repo mining for
  our-framework idioms; LiveCodeBench/HumanEval/MBPP as eval.
- **Data/telemetry (SFT)**: `b-mc2/sql-create-context` (78k NL+CREATE→SQL, CC-BY,
  built to prevent column hallucination — exactly our Postgres/ClickHouse need),
  `Clinton/Text-to-SQL-v1` (100k+, Apache); BIRD as eval. In-house: our ClickHouse
  schema + real query patterns.
- **Instruction structure & anti-doom (SFT/preference)**: `LiquidAI/ifstruct-v1.0`
  (Apache, structured-output compliance), `LiquidAI/antidoom-mix-v1.0` (Apache, prompts
  for antidoom preference training), Liquid's IFBench/Multi-IF as eval.
- **Vision (SFT/eval for VL tier)**: ScreenSpot-v2 (GUI grounding), DocVQA/ChartQA/
  OCRBench (document reading), MMStar/MMMU/BLINK (general vision) — see `datasets/` for
  the verified IDs.
- **Agentic (RL env + eval)**: our own gateway trajectories (in-house); τ²-Bench /
  τ³-Bench / Claw-Eval / BrowseComp+ as published agentic evals; `sapbot/ask-my-agent-bench-2`
  (already in the 1.2B card's eval results).

## Open questions for Primus

- What fraction of the fine-tune should be in-house (our repos + transcripts) vs public,
  per task axis? (Our read: in-house dominates for code/tool-calling; public fine for
  general IF + SQL + code-iteration.)
- Does the 65K-vocab 1.2B need tokenizer expansion before heavy tool-calling SFT (long
  identifiers/JSON tokenize badly at 65K) — or is 128K-vocab 2.6B the right "dev
  assistant" and 1.2B the pure router?
- How much agentic RL (in-harness rollouts) vs SFT to get the tool-result-reliability
  behavior (flag, don't assert) — the community reports are that SFT alone doesn't give
  small models this habit.
- Whether to train per-task specialists then MOPD-fuse (Liquid's recipe) or do one
  mixed SFT + multi-reward RL (their VL recipe). Both are published; we want the one
  that survives 4-bit best.
