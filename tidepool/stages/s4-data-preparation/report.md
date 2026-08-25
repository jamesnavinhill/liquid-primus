# Stage 4: Data preparation

_Status: in progress (s4.1 and s4.2 complete on measured numbers; the s4.3 splits job is running)._

## Headline

The full-corpus scan came back clean over 1,047,820 rows in 11 corpora, and reading it
changed four things that had been written from samples. The two text-to-SQL sets turn out to
be nearly the same data: 75,985 of the smaller set's 78,577 questions also appear in the
larger one, so 97% of it is a subset, and a per-corpus split would have put the same question
on both sides of the divide through two different files. Benchmark contamination is low at
799 rows in a million and concentrated in one place: 560 rows of the code corpus overlap
HumanEval prompts, and they come out by id before anything trains. Two defects I had sized
from samples measured at roughly half the estimate, and both stay in the pipeline at the
smaller size. The mislabelled refusals got sharper: 1,942 of 1,968 rows in one refusal family
answer a question the same corpus answers with a real call elsewhere, making that family
98.7% self-contradictory, so it is dropped. Token counts fix one training setting the plan had
left open, a sequence length of at least 4,096, because a tenth of the multi-turn rows
truncate at 2,048 and they truncate exactly where the tool call sits. The defensible trainable
mix stands at 68,405 rows against the 60,000 the gated set would have supplied.

## Operator input (no-checkpoint, 2026-08-25)

The previous run ended its turn to flag two access questions. The reply was an
automatic decline: _"Full Self Driving: no operator is answering this. Proceed on your
own recommendation, and record in the stage report which option you took and why."_

Both questions are now answered from the environment rather than from the operator, so
nothing was left waiting:

- **Their own repository and gateway data**: reachable. The GitHub credentials on file
  authenticate as `jamesnavinhill` with access to 71 repositories, 19 of them public.
  See "In-house corpus" below for the option taken and its boundary.
- **The gated public tool-calling set**: not reachable, and worked around without
  waited on. See "The two gated sets" below.

**Option taken, and why.** Proceed on an openly licensed substitute mix, treat both
gated sets as recorded limitations with a stated one-click fix, and build the in-house
corpus from the operator's public non-forked repositories only. Waiting on either gate
would idle the project against a decision that costs nothing to reverse later: if the
terms are accepted, the gated rows are additive to a mix that already exceeds the
planned volume, and GPQA slots back into the guardrail suite without disturbing
anything measured before it.

## Work log

- 2026-08-25 · s4.1 · Access verified for 19 datasets by authorized byte-range download,
  not by metadata read; two refusals identified and substituted; schemas and row counts
  recorded below.
- 2026-08-25 · s4.2 · Diagnostics job `e92de716` completed over 1,047,820 rows in 11 corpora,
  zero corpora failed, 12m31s, CPU only, 0 GPU-h. 16 artifacts copied into
  `eda/diagnostics-e92de716/`. Four sampled claims corrected, one reader bug found in our own
  code, three group keys found collapsed, one sequence-length floor established. Ledger entry
  appended to `budget.json`.
- 2026-08-25 · s4.2 · Follow-up: the s4.2 flagged row-id lists were located. All twelve
  uploaded successfully, to the job's eval-results prefix rather than its artifacts prefix,
  which is why the artifact listing appeared to be missing them. They are recorded but not
  reachable through the download path, so `s4.3` re-derives them and asserts each count
  against the recorded one. The `s4.3` job writes its id lists as plain artifacts instead.
- 2026-08-25 · s4.3 · Splits job `7d70b957` queued on CPU (8 vCPU / 32 GB, gcp, no
  accelerator, 240 minutes requested). It materializes the group-disjoint splits and settles
  the four things `s4.2` left open. Design choices made and recorded below: the SQL pair and
  the Hermes `func_calling` pair each share a group namespace; `CodeFeedback` is re-keyed on
  the single most distinctive API symbol per answer; `antidoom` is keyed on the upstream
  sample inside its source; split targets are fractions of all rows, so a collapsed group
  costs train rather than the held-out sets. Verified on fixtures before queueing: the
  antidoom adapter reads a `conversations` row where the old one returned an empty string,
  the code key is stable when an unrelated import is added, the split rule is deterministic
  and group-disjoint, and a deliberately oversized group is routed wholesale to train.

## s4.1 Access, licenses, schemas

Verified on 2026-08-25 with the credentials on file. Metadata is public even for a
gated repository, so **every row below was checked by an authorized range request
against the actual data file**, and `HTTP 206` means the bytes came back.

### Reachable: training side

| Dataset | Rows | License | Schema | Role |
| ------- | ---- | ------- | ------ | ---- |
| `Team-ACE/ToolACE` | 11,300 | Apache-2.0 | `system`, `conversations` | tool calling; the ToolACE paper (`2409.00920v2`) the plan already takes its hyperparameters from |
| `argilla/Synth-APIGen-v0.1` | 49,402 | Apache-2.0 | `func_name`, `func_desc`, `tools`, `query`, `answers`, `model_name`, `hash_id` | tool calling, APIGen three-stage verified recipe |
| `NousResearch/hermes-function-calling-v1` | 1,893 + 1,893 + 5,209 + 1,342 + 1,241 | Apache-2.0 | `conversations`, `tools`, `category`, `subcategory`, `task`, `schema` | multi-turn tool calling and the only JSON-schema-conditioned **training** signal in the mix |
| `nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1` | 285 MB `train.jsonl` | CC-BY-4.0 | agentic function-calling episodes | prompt source for the reinforcement-learning arm |
| `m-a-p/CodeFeedback-Filtered-Instruction` | 156,526 | Apache-2.0 | `query`, `answer`, `resource`, `lang` | iterative code with execution feedback |
| `b-mc2/sql-create-context` | 78,577 | CC-BY-4.0 | `question`, `context`, `answer` | schema-grounded text-to-SQL, built against invented column names |
| `Clinton/Text-to-sql-v1` | 635 MB `texttosqlv2.jsonl` | Apache-2.0 | `instruction`, `input`, `response` | text-to-SQL scale filler. **Note the canonical id is `Clinton/Text-to-sql-v1`**; the capitalization in `overview.md` 307-redirects |
| `LiquidAI/antidoom-mix-v1.0` | 598 MB `train.jsonl` | Apache-2.0 | `id`, `conversations` | prompts only, for preference pairs |

### Reachable: evaluation side

| Benchmark | Where | License | Note |
| --------- | ----- | ------- | ---- |
| BFCLv3 / BFCLv4 | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | Apache-2.0 | per-category files present, including `BFCL_v3_multi_turn_base.json` |
| IFStruct | `LiquidAI/ifstruct-v1.0` | Apache-2.0 | **test split only, 2,000 prompts, no gold responses** (see the correction below) |
| IFEval | `google/IFEval` | Apache-2.0 | |
| IFBench | `allenai/IFBench_test` (+ `allenai/IFBench_multi-turn`) | n/a | the id in `overview.md` (`allenai/IFBench`) does not resolve |
| Multi-IF | `facebook/Multi-IF` | **CC-BY-NC-2.0** | non-commercial. Fine as an internal guardrail; it must not be redistributed in the reproducibility package, and the model card cannot imply a commercial licence flows from it |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | MIT | |
| HumanEval | `openai/openai_humaneval` | MIT | |
| MBPP | `google-research-datasets/mbpp` | CC-BY-4.0 | `full` and `sanitized` configs |
| ToolSandbox | not on Hugging Face | n/a | Apple's code repository, so harness work, tracked into `s5.1` |

### The two gated sets

Both are `gated: auto`, which means access is granted the moment a signed-in user accepts
the terms on the dataset page. Neither has been accepted on the account whose token we
hold, so both return `HTTP 403 … you are not in the authorized list`.

| Refused | Wanted for | What replaces it now |
| ------- | ---------- | -------------------- |
| `Salesforce/xlam-function-calling-60k` (60k) | the primary tool-calling training set | ToolACE + Synth-APIGen + Hermes = **70,387 rows**, all Apache-2.0 |
| `Idavidrein/gpqa` (Diamond, 198 items) | one of five guardrail metrics | nothing. MMLU-Pro carries the knowledge guardrail alone until the terms are accepted |

**Why no attempt was made to get around either.** Accepting a licence on someone's
behalf is a legal act, not a configuration step, so it is not ours to click. Ungated
third-party re-uploads of the xlam data exist and CC-BY-4.0 does permit redistribution,
but using one to route around the publisher's own access gate is a governance call for
the operator, and the substitute mix makes it unnecessary: it is larger than the gated
set, uniformly Apache-2.0, and execution-verified by the same APIGen and ToolACE recipes
the plan already cites.

**What changes if the terms are accepted later.** The xlam rows are additive to the
supervised mix and would add an arm without replacing one, so nothing measured
before that point is invalidated. GPQA Diamond is 198 items and re-scores in minutes;
adding it late costs one evaluation pass per finalist, not a re-run of the study.

### Correction: IFStruct is eval-only

`overview.md` lists `LiquidAI/ifstruct-v1.0` as `train + eval`. It is neither trainable
nor optional to get right: the repository ships a single `test` split of 2,000 prompts
and **no gold responses at all**, because scoring is done by a validator in Liquid's own
eval repository rather than against reference text. So it cannot be trained on even in
principle, and training on it would additionally contaminate a headline metric.

The structured-output *training* signal therefore comes from the Hermes JSON-mode
configs (`json_mode_agentic` 1,342 + `json_mode_singleturn` 1,241 = 2,583 rows, each
carrying an explicit `schema` field). Its `entity_type` field (values such as
`test__recipe`, `test__escaping__short_story_chapter`) is kept as a subgroup key for the
per-taxonomy reporting `overview.md` asks for.

### In-house corpus: reachable, and bounded to public non-forked repositories

The open question in `overview.md` (_does the in-house stack corpus exist, and where_)
is answered. The credentials authenticate as `jamesnavinhill` with access to 71
repositories. `fine-tune.md` authorizes the use directly: _"Primus can be pointed at the
repos to extract SFT pairs."_

**Option taken: mine the 11 public, non-forked repositories only.** Roughly 270 MB of
the operator's own code, in the languages the specialization targets.

| Repository | Language | Size | Stack axis it carries |
| ---------- | -------- | ---- | --------------------- |
| `vidz` | Rust | 133 MB | Tauri commands, IPC |
| `komorebi` | TypeScript | 92 MB | desktop app shape |
| `liquid-primus` | n/a | 19 MB | this project's own docs |
| `sherlock` | TypeScript | 12 MB | agent/tooling |
| `thinking-machine` | TypeScript | 6 MB | agent/tooling |
| `luna` | TypeScript | 2 MB | voice/video layer |
| `Realtor` | TypeScript | 2 MB | Next.js app |
| `docs` | HTML | 1 MB | documentation idiom |
| `changelog`, `elements`, `spellbook`, `puzzler` | TypeScript | < 1 MB each | component and config idiom |

**Why public-only, and why it is the reversible choice.** Two reasons, both about the
deliverable, and not about the data. The 8 forked repositories carry other projects'
idioms, not the operator's, so they would teach the wrong thing. And the 52 private
repositories are excluded because the artifacts of this project are mirrored to a
**public** Hugging Face repository at the operator's request: a model trained on private
source can emit it, and that is not a property to discover after publishing weights.
`fine-tune.md` points the in-house evaluation at _"our own open-source repos"_
specifically, which is the same boundary. Including the private repositories later is a
one-line change to the mining scope; un-publishing weights trained on them is not.

**What is still missing, and it is the more valuable half.** Gateway trajectories
(research → plan → implement → audit loops with real tool calls and returns) live in a
running litellm/langfuse service, not in a repository, and no credential on file reaches
them. Nothing in this stage can substitute for them: they are the only source of the
operator's *own* tool-call distribution, as against public tool-call data plus their code
idiom. The public-data-only ablation (`C2` in the plan) is what keeps the claim honest
either way, and the stack-idiom probe measures what the repository mining does buy.

## Outputs

- This report. Sample-level exploratory work lands in `eda/` at `s4.2`.

## s4.2 Exploratory analysis

**Status: closed on measured numbers.** Job `e92de716-f8d5-402c-bef5-615ea1a8d2c3`
(`s4-data-diagnostics`, experiment `tidepool`) completed at 09:59:24Z after 12m31s,
`completion_status: success`, over **1,047,820 rows across 11 corpora with zero corpora
failed**. Every figure below is that job's, measured over all rows except where a line says
otherwise; the earlier sample-based estimates in `eda/scouting-notes.md` are superseded and
four of them were wrong. Its 16 artifacts are copied verbatim into
`eda/diagnostics-e92de716/`: eleven per-corpus JSON files, the benchmark n-gram index
summary, the cross-corpus duplicate table, and three figures.

**Provenance and settings.** Tokenizer `LiquidAI/LFM2.5-1.2B-Instruct`, decontamination at
13-gram, seed 20260825, token statistics over all rows for any corpus under 80,000 rows and
over a seeded 40,000-row subsample above that (the four large corpora say which in their own
`token_stats_coverage` field). CPU only, 8 vCPU / 32 GB, no accelerator, so the run cost
**zero GPU-hours** against the 200 GPU-h allowance; its wall clock is in `budget.json`.

### Corrections to the sampled analysis

Four claims written from samples did not survive measurement. All four are recorded here
rather than quietly overwritten, because two of them were the basis for a preprocessing step.

| Claim from samples | Measured | Effect |
| ------------------ | -------- | ------ |
| "A quarter of sampled `Synth-APIGen` rows answer with prose where a call belongs" | **6,000 of 49,402 = 12.1%** refusal rows | The defect is real and half the estimated size. The filter is unchanged; the volume it removes is smaller. |
| "A JSON-only reader mislabels roughly one row in eight of ToolACE as having no tools" | **748 of 11,300 = 6.6%** (369 rows in four non-JSON serializations, 379 unclassified) | Same defect, half the rate. Five parsers stay required: 6.6% of the strongest tool corpus still reads as empty-toolset abstention without them. |
| "`LiquidAI/antidoom-mix-v1.0` is unreadable; its row endpoint returns an internal error" | **The file loads: 478,229 rows.** Our own reader was at fault | Corrected below. It is a bug on our side, not a broken dataset. |
| "Neither xlam-lineage defect appears at a rate worth reporting" | One is **exactly zero**, the other is **small but real** | Stated precisely below rather than as a single negative. |

### The reader bug, and why it is recorded as a finding

`antidoom-mix-v1.0` loaded all 478,229 rows and produced **one distinct prompt and zero
tokens**. Its schema is `id` + `conversations`; the generic prompt-only adapter this stage
pointed at it looks for `prompt`, `text`, `instruction` or `query` and finds none of them, so
every row yielded the empty string. The consequences are specific and both are false
negatives, not wrong numbers: its reported contamination of **0 rows is not a result**, and
its group-size distribution (one group of 478,229) is an artifact of the same empty field.

The fix is a two-line adapter that walks `conversations` for the first user turn, the same
shape the Hermes adapter already uses. Because antidoom feeds only the `s5` preference rung
and nothing in `s4` trains on it, the re-measurement is folded into the `s4.3` split job
rather than paying for a second diagnostics run. **Its prompts do not enter the preference
rung until that pass reports a real contamination number**, which is the only thing the bad
read actually blocked.

### Leakage, worst first

**1. The two text-to-SQL corpora are very nearly the same data.** 75,985 prompts appear in
both `Clinton/Text-to-sql-v1` (262,208 rows) and `b-mc2/sql-create-context` (78,577 rows) —
**96.7% of the smaller set is contained in the larger one**. Two things follow. Counting them
as 340,785 rows of independent SQL supervision was wrong; the real distinct volume is roughly
265,000. And a split drawn per-corpus would place the same question on both sides through two
different files, which is exactly the inflation `overview.md`'s split strategy exists to
prevent. The repair is available and cheap: both corpora are keyed by the same normalized
table-surface signature, so `s4.3` splits them **jointly under one group index** rather than
independently, and the containment becomes harmless.

**2. Benchmark contamination is low in aggregate and concentrated in one place.**
799 rows of 1,047,820 (**0.076%**) carry a 13-gram overlap with a benchmark prompt set.

| Corpus | Contaminated rows | Rate | By benchmark |
| ------ | ----------------: | ---: | ------------ |
| `CodeFeedback-Filtered-Instruction` | 759 | 0.485% | HumanEval 560, BFCLv3 143, MMLU-Pro 36, MBPP full 20, MBPP sanitized 10 |
| `Synth-APIGen-v0.1` | 36 | 0.073% | BFCLv3 34, HumanEval 1, MMLU-Pro 1 |
| `Team-ACE/ToolACE` | 4 | 0.035% | BFCLv3 4 |
| all others | 0 | 0 | — |

The aggregate is reassuring and the distribution is the actionable part: **560 code rows
overlap HumanEval**, a benchmark this project reports as a guardrail. At 0.485% the effect on
a trained score is likely small, but it is removable by row id at no cost, so it is removed
rather than argued about. The nine benchmark indices the scan ran against are in
`eda/diagnostics-e92de716/eval_index.json`, with per-benchmark unique-13-gram counts;
`allenai/IFBench_multi-turn` was **not** indexed and is carried into `s4.3` as the one gap in
the decontamination coverage.

**3. Three group keys collapse, and a collapsed key is a silent leak.** A group split only
works where the key partitions the corpus. Three do not:

| Corpus | Groups | Largest group | Why |
| ------ | -----: | ------------: | --- |
| `CodeFeedback` | **4** | 40.2% of rows | Keyed on `resource`, which has four values. Unusable as written. |
| `Clinton/Text-to-sql-v1` | 154,976 | 19,996 rows (7.6%) | Rows whose context declares no `CREATE TABLE` all hash to the same empty-schema key. |
| `Synth-APIGen-v0.1` | 45,617 | 2,068 rows (4.2%) | The empty-toolset rows share one key, for the same reason. |

`CodeFeedback` is the one that has to change: four groups cannot be split three ways without
surrendering a whole programming language to the test side. `s4.3` re-keys it on the query's
normalized code surface (imported modules plus called API symbols), which is the same
"specialize to the underlying technology" principle the operator's standing note states, and
reports the resulting group count before splitting on it. The two empty-key groups are
handled by routing each collapsed group wholesale to one side, which is what group-disjointness
requires anyway.

By contrast the keys that work, work well: ToolACE 10,394 groups at 3.4% largest,
`sql-create-context` 72,942 groups at 0.017% largest, Hermes glaive 3,406 at 0.5%.
Group-size distributions are in `fig-group-sizes.png`.

**4. `hermes-function-calling-v1` ships one corpus twice.** `func_calling` and
`func_calling_singleturn` are 1,893 rows each and share **all 1,161 of their distinct
prompts**, with identical token and call statistics; only the turn segmentation differs
(7,706 turns against 5,679). One config is dropped. That is the single deduction the raw
volume figure was already carrying, and it is now measured rather than inferred from equal
row counts.

### Label quality

**The mislabelled over-refusals in `Synth-APIGen-v0.1` are real, and sharper than the
sample suggested.** 6,000 rows refuse. They split into three families by exact string:
2,068 `no_tools` (honest: the toolset is empty), 1,968 `missing_params`, and 1,964 `other`.
Of the `missing_params` family, **1,942 of 1,968 — 98.7% — fail the contradiction test**: the
same query appears elsewhere in the corpus answered with a call. A family that is
98.7% self-contradictory is not a labelling edge case, and training on it teaches exactly the
over-refusal the project has a pre-registered false-flag ceiling of 0.15 on. Those 1,942 rows
are dropped. The `no_tools` family is kept and is valuable: it is honest abstention, which is
the behaviour `H3` is trying to buy.

The flagged row ids themselves did get written. The job saved twelve id lists (one per
corpus, plus a combined index) holding every contaminated row, every row whose call names an
absent tool, and every suspect refusal, and all twelve uploaded without error. They landed
under the job's eval-results prefix rather than its artifacts prefix, and the artifact
download path serves only the latter, so they are recorded but not retrievable by the route
the rest of this stage's files came down. `s4.3` therefore recomputes the flags from code
instead of plumbing a second retrieval path to reach them. Recomputing is safe here because
every test is fully decidable and seed-independent: an exact canned string, a set membership
against the declared toolset, an exact n-gram hash against a fixed benchmark index. Each
recomputed count is asserted against the recorded count in
`eda/diagnostics-e92de716/per_corpus_*.json`, and any mismatch fails the job, which turns the
unreachable lists into a check on the new ones rather than a loss.

**ToolACE's 1,928 refusal rows are a different phenomenon and are kept.** None of them
declares the target tool, none contradicts a call elsewhere, and all 1,928 fall in the
`other` family. Together with the 3,727 rows whose assistant turn is natural language rather
than a call, they read as genuine conversational turns rather than mislabelled refusals.
Nothing is dropped from ToolACE on refusal grounds.

### Output-format defects in ToolACE

**Non-identifier call names are the largest single preprocessing requirement in the mix:
8,626 of 18,371 calls (47.0%)** name functions like `Market Trends API` or
`Get Cars Information`. BFCL's pythonic categories score by parsing an abstract syntax tree,
so 47% of this corpus's calls, as written, teach output that the benchmark's own parser
rejects. `s4.4` normalizes names to identifiers in the training target and the data card
records the mapping.

The five schema serializations are confirmed and quantified: JSON 10,552, YAML-ish block 156,
markdown 76, LaTeX `tabular` 69, XML-ish 68, unclassified 379. Assistant output splits
pythonic-bracket 10,013, natural language 3,727, other bracket forms 68, custom delimiters 11.
The variation is kept deliberately — reading a tool contract in whatever shape it arrives is
the point — and `fig-tool-shape.png` plots the per-corpus shape.

### The two suspected xlam-lineage defects, measured

Both were carried in from the `s3` plan as mandatory repair passes. Measured over all
72,280 tool and structured rows:

- **Whitespace after the opening bracket: exactly 0 occurrences** (`lead_space: 0`,
  `trail_space: 0`). The repair pass is removed from the pipeline. Keeping a pass for a defect
  measured at zero would be cargo cult.
- **Calls naming a function absent from their own toolset: small but real.** ToolACE 288 calls
  across 113 rows (1.6% of its calls), `Synth-APIGen` 7 calls across 5 rows, Hermes glaive
  3 calls across 3 rows. 298 calls in total. The pass stays, scoped to dropping those 121 rows
  rather than attempting a repair, because at that volume a repair is not worth the risk of
  inventing a target.

### One genuine reader failure to fix

`hermes-function-calling-v1` config `glaive_func_calling` has **865 adapter parse failures of
5,209 rows (16.6%)** — the only corpus with a nonzero failure count. Its calls use a
`<tool_call>` wrapper the adapter reads, so the 865 are rows where that wrapper is absent or
malformed. `s4.3` reports the failure mode before those rows are either recovered or dropped;
they are not silently counted as trainable.

### Sequence length, which changes an s5 setting

Token statistics under the LFM2.5 tokenizer set a floor the plan had not fixed. Hermes
`func_calling` prompts run **p50 382, p90 3,128, max 3,218 tokens** before the schema block
(mean 259) or the call (mean 58) are added. ToolACE schema blocks reach p90 789 and max 3,010.
Clinton SQL contexts reach p99 1,417 and max 2,595.

**So max sequence length must be at least 4,096 for the supervised arms.** At 2,048 — a
plausible default for a 1.2B model — roughly a tenth of the multi-turn Hermes rows truncate,
and they truncate at the end, which is where the tool call is. That is recorded here as an
`s5.1` smoke-test assertion rather than left to a default.

Median cost is otherwise low and the mix is cheap to train: APIGen prompts p50 33 tokens with
a 227-token schema block, `sql-create-context` p50 15 with a 23-token context, ToolACE p50 40
with a 394-token schema block.

### Trainable volume, after the deductions this stage can defend

| Step | Rows |
| ---- | ---: |
| Tool and structured-output rows, raw | 72,280 |
| Drop the duplicated Hermes config | −1,893 |
| Drop the 1,942 self-contradictory over-refusals | −1,942 |
| Drop the 40 contaminated tool rows | −40 |
| **Defensible trainable total** | **68,405** |

That still exceeds the 60,000 rows the gated `xlam` set would have supplied, so the `s4.1`
conclusion holds on measured numbers rather than declared row counts.

**What this table deliberately does not do is deduplicate prompts**, and the reason is a
measurement this job did not make. It counted duplicate *prompts* (APIGen 12,474 rows in
6,175 groups, Hermes glaive 3,501 in 349, ToolACE 269 in 134, Clinton 157,531 in 78,364), not
duplicate *examples*. In APIGen the two are known to differ: the contradiction test works
precisely because the same query recurs against a different toolset with a different answer,
and those are legitimately different training examples. Collapsing on the prompt alone would
delete the corpus's most useful variation. `s4.3` measures duplication on the
`(prompt, toolset, answer)` triple, and `s4.4` decides the dedup rule against that number.
Prompt-level counts put the eventual figure somewhere between 58,000 and 68,405, and the
stage will not narrow that from a sample.

### Where to look

- Numbers: `eda/diagnostics-e92de716/per_corpus_*.json`, one file per corpus, every statistic
  cited above with its `n`.
- Figures: `fig-tool-shape.png`, `fig-contamination.png`, `fig-group-sizes.png` in the same
  directory.
- Live: `lab job info e92de716 -e tidepool` for the score dict,
  `lab job task-logs e92de716 -e tidepool` for the per-corpus commentary,
  `lab job chart -e tidepool` to export the experiment's run chart.
- The s4.2 flagged id lists are recorded at the job's eval-results prefix. They are visible in
  `lab --format json job info e92de716 -e tidepool` under `job_data.eval_results` (twelve
  files) and are not served by `lab job download`, which reads the artifacts prefix only.

## s4.3 Splits

_Job `7d70b957` running. Everything below is design and rationale; no measured split numbers
are recorded until the job completes and its assertions pass._

The split is group-disjoint, deterministic and size-aware. Groups are ordered by a seeded
hash of the group key, then walked to fill test to its row target, then val, then train, so
the realized row shares land close to 5/5/90 without any group being divided. Two decisions
in that rule are worth stating because the obvious version of each is wrong for this mix.

**Group namespaces, not corpora.** `s4.2` measured that 75,985 of `sql-create-context`'s
78,577 questions also appear in `Clinton/Text-to-sql-v1`, and that the two Hermes
`func_calling` configs share all 1,161 of their distinct prompts. Splitting each corpus on its
own would put the same question on both sides of the divide through two different files. Both
pairs already share a group key, so they are given a shared namespace and split jointly: `sql`
covers 340,785 rows across the two SQL sets, `hermes_fc` covers 3,786 across the two duplicated
configs, `hermes_json` covers the two JSON-mode configs. Eight namespaces in total.

**Split targets are fractions of all rows, and a collapsed group costs train.** A group larger
than 2% of its namespace cannot be split without dominating whichever side it lands on, so it
is routed wholesale to train: Clinton's 19,996-row no-`CREATE TABLE` group and APIGen's
2,068-row empty-toolset group are the two `s4.2` identified. Sizing test and val against total
rows rather than against the splittable remainder means those collapsed rows come out of train,
leaving the held-out sets at their intended absolute size. A cap at 40% of the splittable pool
keeps a mostly-collapsed namespace from handing its entire remainder to eval.

**`CodeFeedback` is re-keyed on the API surface each answer exercises.** The `resource` field
gave four groups with the largest at 40.2% of the corpus, which is not a split key. The new key
is the single least frequent API symbol the answer touches, measured over the whole corpus, so
a group reads as "every answer whose most distinctive symbol is `np.linalg.svd`". One symbol
rather than three: a three-symbol key moves when an answer imports something incidental, and
two answers calling the same function through different incidental imports belong together.
Symbols seen once corpus-wide are dropped as typos before the minimum is taken, bare imports
are weighted below actual calls, and rows with no surviving symbol fall back to the names they
define and then to the language alone, with the tier mix reported so a collapsed tail stays
visible. Keying on what a row does with which technology, rather than on which generator
produced it, is also the axis the operator asked the fleet to specialize along.

**`antidoom` is keyed on the upstream sample inside its source.** Keying on the source alone
would make each group a large share of 478,229 rows, so the collapse rule would route every
one of them to train and leave no held-out prompts at all. The leak this corpus can actually
carry is the same upstream item arriving twice, which the sample id prevents. The
source-and-config histogram is reported separately, so a source-disjoint split stays available
to `s4.4` if the preference rung wants one.

**What the job settles beyond the splits.** The re-read `antidoom` prompts give it a real
contamination number for the first time, along with its upstream source mix, its per-source
licence, and the `heldout_policy` field each row carries, which names the benchmark that
source must not be used against. Duplication is measured on the
`(prompt, toolset, answer)` triple, so `s4.4` sets the dedup rule against a number rather than
against the prompt-level bound of 58,000 to 68,405 rows. The 865 Hermes glaive adapter failures
are categorized by row shape (role sequence, whether the `tools` field is absent or empty,
whether a schema sits in the system turn) before anything decides to recover or drop them.
Both `allenai/IFBench_multi-turn` configs join the decontamination index, closing the gap
`s4.2` recorded. A leakage audit counts prompts and triples that appear in more than one split,
which a group-disjoint split can still do when one prompt carries two group keys, so the number
is measured rather than assumed to be zero.

**Every `s4.2` count is re-derived and checked.** Row counts, mismatch rows, refusal rows,
suspect over-refusals, same-query contradictions and non-identifier call names are asserted
against the recorded values; a mismatch on any of them raises and leaves the job failed, so
nothing downstream can cite splits built on counts that did not reproduce. Group counts and
prompt counts for the two re-keyed corpora are marked superseded rather than checked, and the
glaive parse-failure count is reported rather than asserted, since the recovery pass is
expected to move it.
## Next steps

- `s4.3` group-aware splits, materialized by one job that also clears the four items `s4.2`
  left open: the fixed antidoom reader and its real contamination number, the re-keyed
  `CodeFeedback` group index, `(prompt, toolset, answer)` duplication so `s4.4` can set the
  dedup rule on a measured basis, and the 865 Hermes glaive parse failures.
- `s4.4` preprocessing: canonicalize the five ToolACE serializations and the `dict`/`object`
  type divergence, normalize the 8,626 non-identifier call names, drop the 1,942
  self-contradictory refusals and the 799 contaminated rows by id, and drop the 121 rows whose
  calls name an absent tool. The whitespace repair pass is removed, having measured at zero.
- `s4.5` data card, then autonomous sign-off.
