# Stage 4: Data preparation

_Status: complete. s4.1 through s4.5 are done on measured numbers, the data card is
written, and the s4.5 sign-off was taken here under autonomous mode. Stage 5 is next._

## Headline

The training corpus is built and measured: **518,330 supervised examples, 201.5 million
trainable tokens, and a separate pool of 447,053 prompts**, with zero examples landing on both
sides of the train and held-out line either before or after rendering. Eleven differently
shaped source corpora now render into one conversation format, and the canonicalizer met **no
declared parameter type it did not recognize**, at any nesting depth, in any of them.

Two measurements change what the experimentation stage does. The general-quality corpus carries
**no assistant responses at all** — all 447,053 of its renderable rows end on a user turn, so it
is a prompt set and not training data, and the planned replay control has no corpus to replay
from. Replay will instead draw on the base model's own frozen completions over that pool, which
is what a prompt set with no shipped responses is for. And the mix as assembled is weighted
against the project's own priority order: SQL and code take **79% of the training tokens**,
tool calling takes 20%, and structured output takes under 1%, on a project whose second headline
metric is structured-output validity. A role-balanced sampling arm becomes the reference for the
supervised sweep, with the raw mix kept as the explicit comparison.

Getting the splits here caught a leak a group-disjoint split cannot prevent on its own. Groups
were kept whole, which is the standard guarantee, and the assignment still put **39,161 prompts
on both sides of the train/held-out line**, a 4.99% leak rate, because one prompt can carry two
different group keys. Those 42,702 rows are set aside rather than counted and shipped.

Reading the corpora correctly also closed the four questions the earlier scan left open.
Benchmark contamination is 25× larger than first reported, 20,166 rows rather than 799, and 96%
of it sits in the general-quality corpus whose reader had previously found no prompt field at
all. Duplication, measured on the question-toolset-answer triple rather than on the question
alone, removed 149,822 rows. The code corpus re-keys from 4 unusable groups into 52,286, with
the largest falling from 40.2% of the corpus to 3.7%. And every count the earlier scan recorded
was re-derived and checked, 110 of 110 passing.

One thing the split makes plain and the plan should absorb: held-out is small on the axis that
matters most, at 451 test rows of tool calling out of 11,550. In-mix held-out numbers have to be
reported per capability rather than pooled, and the tool-calling verdict has to rest on the
external benchmarks and on the two purpose-built probe sets built here.

Those probes are now built and gated: **434 hand-written items**, and **none of them shares a
13-gram** with any of the 494,341 rows a model will be trained on. The gates paid for
themselves on the first attempt, which found that two of the five tool-corruption modes were
ignoring the nesting depth they were supposed to sit at, making 40 of 290 items duplicates of
each other. Scored as first built, the one hypothesis those items exist to test would have
returned a null result for a reason that had nothing to do with any model.

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
- 2026-08-25 · s4.4 · Preprocessing job `4674a2ec` completed over 1,047,820 rows, 34m29s,
  CPU only, 0 GPU-h, 0 assertion failures and 0 purity violations. 518,330 examples kept and
  201.5M trainable tokens estimated; 447,053 rows routed to the prompt pool because they carry
  no assistant turn. Two findings changed the plan and are recorded under "s4.4 Preprocessing
  — measured": the general-quality corpus is a prompt set, and the token mix runs against the
  project's own priority order. Both decisions were taken here, unreviewed, under autonomous
  mode. Artifacts in `preprocess-4674a2ec/`; splits and pool uploaded to `tidepool/s4.4/`.
- 2026-08-25 · s4.4 · Probe build attempt 1, job `32fc86ad`, failed two of its own three
  gates and is superseded. One failure was a real defect in the build (two corruption modes
  ignored envelope depth, so 40 of 290 probe-A items were duplicates); the other was a wrong
  assertion, which banned the deliberate question-sharing between the corrupted and
  contradicted arms. Both fixed in the build script, the assertion replaced by two narrower
  checks. Ledger entry appended.
- 2026-08-25 · s4.4 · Probe build attempt 2, job `05bbcd49`, is the build of record.
  2m45s, CPU only, 0 GPU-h, 0 assertion failures. 434 items, 0 sharing a 13-gram with the
  training split over an index of 494,341 rows and 13,345,986 unique 13-grams, 0 duplicate
  items and 0 questions under two scenarios. Items at `tidepool/s4.4/probes/`; summary and
  score in `probes-05bbcd49/`.
- 2026-08-25 · s4.5 · Data card written from the measured numbers, covering sources and
  licence position, the deliberate absences, the split guarantee and the leak it does not
  cover, the per-consumer slices the experiment matrix draws on, and the known defects. The
  sign-off is a checkpoint in the normal flow; autonomous mode is on for this project, so it
  was taken here rather than sent, and the decision is recorded under "s4.5 sign-off" below.
- 2026-08-25 · mirror · Project state through `s4.2` pushed to
  `github.com/jamesnavinhill/liquid-primus` under `tidepool/`, per the standing operator note
  to mirror code and docs there as they land. Commit `61d65e0`: the overview, the checklist,
  the initial prompt, the compute ledger, the stage reports for `s0` through `s4`, the research
  plan, the reader brief, the 16 diagnostics artifacts verbatim, and the code of both queued
  jobs exactly as submitted. Two things are deliberately left out and said so in the mirror's
  own readme: the roughly 230 converted paper full texts and the per-paper reading notes, since
  republishing third-party full texts is not a call this stage should make on its own. Weights
  and quantized builds have the Hugging Face half of that note and do not exist yet.

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
The `missing_params` family is exactly the set of refusals that name a tool their own toolset
declares — `refusal_with_target_declared` is 1,968, the same number — so every one of them
declines to call a function it was offered. Of those, **1,942 name a function requiring two
or fewer arguments**, which is a refusal citing missing parameters against a call that needs
at most two of them.

Two things that reading kept blurred together, and they are separate measurements. The
`required <= 2` gate is a plausibility test on the individual row. The contradiction test is
a corpus-level one, and it says something different: **5,999 of APIGen's 43,103 distinct
questions — 13.9% — appear somewhere with a refusal and somewhere with a real call.** An
earlier draft of this section reported the 1,942-of-1,968 ratio as if it were the
contradiction result. It is not; the two tests were never intersected.

`s4.3` flags both conditions independently on every row, so the intersection is available for
the first time, and it is the number that should carry the drop: a row that both refuses
against an easily-callable tool *and* has its own question answered with a call elsewhere is
mislabelled on two independent readings. `s4.4` drops on the flag rather than on the ratio,
and the intersection is reported next to each condition alone. Training on these teaches
exactly the over-refusal the project has a pre-registered false-flag ceiling of 0.15 on.

The `no_tools` family is kept and is valuable: it is honest abstention, which is the
behaviour `H3` is trying to buy.

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
| Drop the 1,942 suspect over-refusals | −1,942 |
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
- In a browser: the compute backend's own console at <https://app.lab.cloud>, workspace
  `69347550-a587-440b-869f-61575113f1d0`, experiment `tidepool`. Every job this project
  queues appears there under its short id with live status, its logs, and its charts, so the
  numbers quoted in this report can be read off the source rather than taken on trust. The
  two job ids to look for so far are `e92de716` (s4.2 diagnostics) and `7d70b957` (s4.3
  splits).

## s4.3 Splits

_Measured. Job `cc0fad09` (COMPLETE, CPU only, ~17 min, 0 GPU-hours) passed **110 of 110**
`s4.2` reproduction assertions and both held-out purity checks. Attempt 1, job `7d70b957`,
failed five of those assertions on definitional drift and is superseded; nothing from it is
cited. Artifacts are in `splits-cc0fad09/`, and the manifest is in shared storage at
`tidepool/s4.3/splits.jsonl.gz` (20.3 MB, one line per row)._

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
### Measured — the split, and what it cost to keep held-out clean

1,046,955 rows across 8 namespaces:

| | train | val | test | set aside |
| --- | ---: | ---: | ---: | ---: |
| rows | 942,029 | 31,708 | 30,516 | 42,702 |
| share | 89.98% | 3.03% | 2.91% | 4.08% |

The group-disjoint assignment alone landed on 90/5/5, at 52,426 val and 52,500 test. Then the
purity check ran, and it is the number this substage exists to have produced: **39,161 prompts
appeared in more than one split**, a 4.99% prompt leak rate, alongside 24,891 straddling
triples. A group-disjoint split can do that legitimately, because the same prompt can carry two
different group keys and land on both sides. Inside train it is harmless. Across the held-out
line it is the leak the stage was built to prevent, and counting it without acting on it would
have been recording a known contamination and shipping it anyway.

So it is enforced: a straddling prompt keeps its train rows and its held-out rows are set aside,
and 566 test rows whose prompts straddled only val and test were demoted to val so the two stay
mutually disjoint. That costs 42,702 rows, 4.08% of the mix, and it comes almost entirely out of
the held-out sets rather than out of train. The re-check on the written manifest reports **0
prompts shared between train and held-out and 0 shared between val and test**, which is what the
enforcement guarantees by construction and is verified rather than assumed.

| corpus | rows | train | val | test | set aside |
| --- | ---: | ---: | ---: | ---: | ---: |
| toolace | 11,300 | 10,170 | 561 | 557 | 12 |
| apigen | 49,402 | 44,460 | 1,979 | 2,028 | 935 |
| hermes_fc | 1,893 | 1,703 | 95 | 95 | 0 |
| hermes_fc_st | 1,893 | 1,703 | 95 | 95 | 0 |
| hermes_glaive | 4,344 | 3,908 | 47 | 45 | 344 |
| hermes_json_ag | 1,342 | 1,199 | 60 | 83 | 0 |
| hermes_json_st | 1,241 | 1,123 | 70 | 48 | 0 |
| sql_ctx | 78,577 | 69,069 | 299 | 175 | 9,034 |
| sql_clinton | 262,208 | 237,563 | 3,236 | 2,649 | 18,760 |
| codefeedback | 156,526 | 140,726 | 6,607 | 6,656 | 2,537 |
| antidoom | 478,229 | 430,405 | 18,659 | 18,085 | 11,080 |

`sql_ctx` keeps 78,577 training rows and contributes 474 held-out rows, because 75,985 of its
questions also appear in Clinton and the purity rule sends every straddling pair's held-out side
away. The joint `sql` namespace still yields 3,535 val and 2,824 test rows, so the capability is
evaluated; it is evaluated on Clinton's phrasing.

### Measured — held-out is dominated by the general-quality corpus

Splitting the held-out sets by what they test, which the split rule never did because it works
on groups rather than roles:

| role | val | test |
| --- | ---: | ---: |
| tool calling | 2,777 | 2,820 |
| structured output | 130 | 131 |
| SQL | 3,535 | 2,824 |
| code | 6,607 | 6,656 |
| general prompts | 18,659 | 18,085 |

The project's first priority has the second-smallest held-out slice, and structured output has
131 test rows. Two consequences, both for `s6` rather than for this stage. In-mix held-out
numbers must be reported per role and never pooled, since a pooled figure would be 59% a
corpus whose job is to detect regression rather than to measure tool calling. And the real
tool-calling verdict has to come from the external benchmarks, which is what BFCL, IFStruct and
the IFBench family are indexed for. `s4.4`'s probe sets are built for the same reason: the
in-mix held-out tool slice is too small and too easy to carry the claim on its own.

### Measured — the four `s4.2` open questions, closed

**`antidoom` contamination is real and it is the largest in the mix.** Read correctly, 20,166 of
478,229 prompts share a 13-gram with a benchmark, 4.22% of the corpus and 96% of all
contamination found anywhere. `s4.2` reported 0, entirely because its reader found no prompt
field. The corpus draws on 20 upstream sources, MIT for 299,211 rows and Apache-2.0 for 179,018,
and every row carries a `heldout_policy` string naming what it must not be evaluated against:
50,000 MMLU rows say to use `auxiliary_train` only, 20,000 IFStruct rows say to reserve the
`test__*` taxonomies, and the GSM8K and MATH slices each reserve their test splits. Those
policies agree with the decontamination index rather than contradicting it, which is the useful
finding: the contamination is upstream drift, not a policy being ignored.

**`CodeFeedback` re-keys cleanly.** 52,286 groups where `resource` gave 4, and the largest group
is `bare:python` at 3.72% of the corpus where the old largest was 40.2%. 142,097 rows key on an
actual API symbol, 12,036 fall back to the bare language, and 2,393 to a symbol the answer
defines. The corpus touches 3,601 distinct modules and 194,473 distinct called symbols, headed
by numpy, re, math, random and java. A key on what a row does with which technology, which is
the axis the operator asked the fleet to specialize along.

**Duplication, on the triple rather than the prompt.** 1,046,955 rows carry 897,133 distinct
`(prompt, toolset, answer)` triples: 277,906 rows sit in 128,084 duplicate groups, and exact
triple dedup would remove **149,822 rows**, 14.3% of the mix. Prompt-level duplication is much
larger at 262,847 rows over 784,108 distinct prompts, and the gap between the two is exactly the
variation the mix is supposed to carry, the same question asked against a different toolset or
answered in a different language. `s4.2` could only bound this between 58,000 and 68,405 rows
and said so. Cross-corpus prompt sharing is concentrated in two pairs, `sql_clinton | sql_ctx`
at 75,985 and `antidoom | codefeedback` at 28,210, with the Hermes config pair at 1,161 and
exactly two other pairs in the whole mix.

**The 865 glaive parse failures reproduce exactly** and are categorized by row shape in
`extras.json`; whether they are recovered or dropped is `s4.4`'s call and is scoped there.
`sql_clinton` contributes 2 further parse failures, which is the whole of the rest of the mix.

### Measured — the reproduction check

110 checks, 0 hard failures. Every `s4.2` count that could be re-derived was: row counts,
contamination, mismatch rows, refusal rows, suspect over-refusals, same-query contradictions and
non-identifier call names, per corpus. Getting there took one failed attempt and five definition
alignments, written up below. Three quantities are now reported under two readings, the `s4.2`
one that the assertion checks and a broader one recorded as a new measurement, so neither
definition is silently lost.

## Next steps

- `s4.3` group-aware splits, materialized by one job that also clears the four items `s4.2`
  left open: the fixed antidoom reader and its real contamination number, the re-keyed
  `CodeFeedback` group index, `(prompt, toolset, answer)` duplication so `s4.4` can set the
  dedup rule on a measured basis, and the 865 Hermes glaive parse failures.
- `s4.4` preprocessing: canonicalize the five ToolACE serializations and the `dict`/`object`
  type divergence, normalize the 8,626 non-identifier call names, drop the 1,942
  suspect over-refusals and the 799 contaminated rows by id, and drop the 121 rows whose
  calls name an absent tool. The whitespace repair pass is removed, having measured at zero.
- `s4.5` data card, then autonomous sign-off.

### Work log — 2026-08-25, the recompute caught a definition drift, not a data problem

Job `7d70b957` re-derives all six of the `s4.2` counts it asserts against, and while it was
still running the per-corpus log showed four of them landing on different numbers. Reading
both jobs side by side, every one of the four is a definition that moved between the two
runs. None is a disagreement about the data.

| Count | `s4.2` | first `s4.3` pass | why they differ |
| ----- | -----: | ----------------: | --------------- |
| ToolACE calls naming an absent tool | 113 rows | 87 rows | `s4.2` took the declared set as the union of the adapter's name list and the parsed schema objects. `s4.3` read the schema objects alone, so the 369 ToolACE rows whose schema is YAML, XML, markdown or LaTeX have no parsed objects, read as having declared nothing, and were skipped. |
| APIGen suspect over-refusals | 1,942 | 1,968 | `s4.2` only called a refusal suspect when the tool it named required two or fewer arguments, on the reasoning that a refusal is only suspicious when the call was plausibly makeable. `s4.3` dropped the gate. |
| ToolACE same-query contradictions | 0 | 159 | `s4.2` ran the test on APIGen alone. Its zero everywhere else is an absence of measurement, not a measured zero. |
| APIGen same-query contradictions | 5,999 | 12,046 | `s4.2` counted contradicting *queries*; `s4.3` counted the *rows* under them, about two per query. |

Row counts, refusal counts, contamination counts and the non-identifier call-name count all
reproduced exactly, which is what makes the four stand out as definitional.

Each is now realigned to the `s4.2` definition, so the assertion tests reproduction rather
than testing whether two slightly different questions give the same answer. Where the newer
reading is the more useful measurement, both are recorded: `tool_name_mismatch_rows` beside
`tool_name_mismatch_rows_named_only`, `suspect_over_refusals` beside
`target_present_on_refusal`, `label_contradictions_same_query` beside
`label_contradiction_rows`. The contradiction assertion is now scoped to APIGen and marked
`NEW` for every other corpus, since there is no earlier number there to reproduce.

Two smaller fixes went in with them. `FLAG_CONTRA` was defined but never applied to any row:
the contradiction verdict is only knowable once a whole corpus has been read, and the
per-row records were built before that, so the bit is now stamped on in the post-pass for
both the ordinary and the re-keyed corpora. And the drop set for `s4.4` is fixed as
`FLAG_SUSPECT`, the gated 1,942, matching the trainable-volume table above rather than the
broader 5,999-query contradiction set.

The first pass is left unpublished. Its split shapes were healthy — a clean 90/5/5 in every
namespace, with the two known overlapping corpus pairs forced whole to train — but a job
whose assertions fail cannot be the source of a number this project cites, so the run is
discarded and the corrected job requeued. Both passes are CPU-only and charge nothing
against the GPU allowance.

### Work log — 2026-08-25, held-out purity is now enforced, not just counted

The discarded pass also settled a question the design section had left open, and the answer
was worth acting on before requeueing. A group-disjoint split is not a prompt-disjoint
split. The same question recurs in the mix against a different toolset, which is a different
group key and a legitimately different training example, so nothing in the group rule stops
it landing on both sides of the line. The first pass counted how often that happened and
reported it. Counting is the right thing to do inside `train`, where the recurrence is the
useful variation the corpus was built to carry. It is the wrong thing to do across the
held-out line, where a model that memorized the question during training would score on it
at eval without ever having read the schema.

So the split now runs in two passes. The group-disjoint assignment is made first and kept
verbatim on every row as `s0`. Then any prompt found on both sides of the line keeps its
`train` rows and has its `val` and `test` rows set aside under a `drop` label; a prompt
straddling only `val` and `test` is demoted wholly to `val`. The straddling rows are not
moved into `train`, which would import held-out-shaped examples into the training mix; they
are simply not used. The cost falls entirely on the held-out sets, which is the right place
for it, since eval integrity is worth more here than eval size.

The guarantee is then re-derived from the written assignment rather than asserted from the
code that produced it, and two counts — prompts shared between `train` and either held-out
set, and prompts shared between `val` and `test` — must both come back zero or the job
fails alongside the reproduction assertions. Keeping `s0` next to the enforced label means
`s4.4` can still see what the group rule alone decided, and the before-and-after row counts
are both reported, so the price of the enforcement is visible rather than absorbed.

One measurement to expect from the corrected run that `s4.2` could not make: contamination
in `antidoom`. `s4.2` recorded zero there, but only because its adapter returned no prompt
text for that corpus, so no n-grams were harvested and nothing could match. The new adapter
reads the conversation properly, which means `antidoom` is checked against the benchmark
index for the first time. Whatever it returns is a new number, not a change to an old one.

Attempt 2 is job `cc0fad09-0721-4a9e-b6e5-a05de7557c8d`, same CPU-only shape, nothing
charged against the GPU allowance.

## s4.4 Preprocessing — design

_Written while `s4.3` runs. The decisions below are settled; the counts they operate on come
from `s4.3` and are filled in once that job passes._

### What gets canonicalized, and what deliberately does not

`s4.2` found five different serializations of the same tool contract inside ToolACE alone:
JSON for 10,552 rows, then a YAML-ish block, markdown, a LaTeX `tabular`, and an XML-ish
form for another 369, with 379 unclassified. The obvious move is to rewrite them all into
one shape. The pipeline does not do that, because reading a tool contract in whatever shape
it arrives is a capability this project wants rather than noise it wants gone. **The prompt
keeps its native serialization.**

What is canonicalized is everything downstream of reading it. All five forms are parsed into
one internal schema, that schema is what the absent-tool and name-agreement checks run
against, and the training *target* is rendered in a single form regardless of how the
contract arrived. The model therefore sees five input shapes and learns one output shape,
which is the asymmetry the gateway actually needs.

The `dict`/`object` divergence is handled in the same internal schema: `dict` becomes
`object`, `list` and `tuple` become `array`, `str` becomes `string`, `int` becomes
`integer`, `float` and `double` become `number`, `bool` becomes `boolean`. The mapping is
recorded and the type vocabulary is re-counted afterwards, so a type the mapping missed
shows up as a survivor rather than passing silently.

### The canonical call target is JSON, and why

The target is a JSON object naming the function and its arguments, inside a `tool_call`
delimiter. Two reasons, in order.

The operator's standing instruction is to specialize to the underlying technologies — JSON
schemas, MCP, IPC — rather than to any one vendor's SDK. A JSON call object is what MCP
carries on the wire and what almost every inference stack parses, so training the target in
that shape specializes to the protocol. A pythonic bracket form, which is how 10,013 of
ToolACE's assistant turns are written, is a benchmark's scoring convention rather than a
protocol.

Second, it is the shape the eval can adapt *to* rather than *from*. BFCL scores several
categories by parsing a pythonic abstract syntax tree; converting a well-formed JSON call
into that form is mechanical and lossless, and the `s6` harness does it. Converting a
pythonic string back into typed JSON arguments is the lossy direction, and doing it at
training time would bake every parse ambiguity into the labels.

The cost is that ToolACE's pythonic argument strings must be parsed into typed arguments
once, at preprocessing, rather than never. `s4.2`'s adapter only counted arguments; the
pipeline needs their names and values. Any row whose argument string does not parse is
dropped rather than guessed at, and the drop count is reported per corpus.

### Names

**8,626 of ToolACE's 18,371 calls name functions no parser will accept** — `Market Trends
API`, `Get Cars Information`. Normalization lowercases, replaces every run of characters
outside `[a-z0-9_]` with a single underscore, strips the edges, and prefixes `fn_` to
anything starting with a digit; collisions inside one toolset get a numeric suffix. The
declaration and the call are rewritten with the *same* map in the same pass, so a call that
matched its declaration before still matches after. The full map ships in the data card.

### The drop set, by flag rather than by re-derivation

`s4.3` writes one flag byte per row, so the pipeline drops by reading bits rather than by
recomputing tests a third time.

| Flag | Meaning | Action |
| ---- | ------- | ------ |
| contaminated | shares a 13-gram with a benchmark prompt | drop, every split |
| tool-name mismatch | calls a tool its own toolset never declared | drop |
| suspect over-refusal | refuses while naming a tool it could plausibly have called | drop **only with** contradiction |
| same-query contradiction | the same question answered two ways across the corpus | keep alone, drop **with** suspect |
| adapter parse failure | the reader got nothing usable out of the row | recover, else drop |
| held-out purity | prompt straddled the train/held-out line | already set aside by `s4.3` |

Two of those rows read together rather than separately, which is a change from the first
draft of this section. Either flag alone is weak evidence. A refusal that names a callable
tool can still be the right answer for reasons the declared schema does not show, and a
question carrying both labels can be two legitimately different requests that happen to
share a prompt string. A row carrying both is a corpus contradicting itself about a call it
could have made, which is the exact failure this project exists to fix and is the one thing
that must not be trained on. So the drop rule is the conjunction, and the count for each
condition alone is reported beside it.

**Measured, from the `s4.3` manifest.** The flag bits are set on 20,965 contaminated rows,
12,205 contradiction rows, 1,942 suspect over-refusals, 121 tool-name mismatches and 2 adapter
parse failures. The intersection the rule turns on is **1,942**, which is every suspect row:
each of them also carries the contradiction flag, so the conjunction and the suspect flag alone
select exactly the same rows on this data. The conjunction is still the rule that ships, because
it is the one that states what makes those rows wrong, and a later corpus where the two
conditions come apart will be dropped correctly rather than by a coincidence that held once.
Contradiction alone covers 10,263 further rows and those are kept.

Contradiction alone is kept deliberately. Once the unjustified refusals are gone, a question
answered differently against a different toolset is the variation the corpus exists to
carry, and collapsing it would delete APIGen's most useful signal.

### Deduplication

`s4.2` could only count duplicate *prompts* and said so. `s4.3` counts duplicate
`(prompt, toolset, answer)` triples, which is the quantity a dedup rule should act on: the
same question against a different toolset with a different answer is two examples, not one.
The rule is to keep one row per triple, chosen by lowest row id so the choice is
reproducible, and the removed count is reported against the `s4.2` prompt-level bracket of
58,000 to 68,405 so it is visible whether the truth landed inside it.

### The `glaive` recovery

865 of 5,209 rows in the Hermes `glaive_func_calling` config return nothing from the reader.
`s4.3` reports the shape of those rows — which turn roles are present, whether a schema
appears in the system turn, whether a `tool_call` wrapper appears anywhere. The pipeline
writes one recovery path against whatever that report shows and drops what it cannot
recover, with both counts reported. Nothing is counted as trainable on the strength of a
reader that returned nothing.

## s4.4 Preprocessing — measured

Job `4674a2ec`, COMPLETE, 34m29s, CPU only, **0 assertion failures and 0 purity
violations**. Every number below is from that job's `preprocess_summary.json`; nothing
here is estimated.

| | Rows |
| --- | ---: |
| Read from the 11 source corpora | 1,047,820 |
| No manifest entry (set aside at `s4.3`, or unreadable) | 42,917 |
| Dropped by flag rule | 21,001 |
| Unrenderable | 4,371 |
| Duplicate rendered text | 13,270 |
| Routed to the on-policy prompt pool | 447,053 |
| **Kept as supervised rows** | **518,330** |
| — train | 494,341 |
| — val | 12,439 |
| — test | 11,550 |

13 rows exceeded the 4,096-token window and were dropped rather than truncated. The
longest kept example is 3,812 tokens.

### The canonicalizer met no schema type it did not know

`unknown_schema_types` is **empty**. Across all eleven corpora, at every nesting depth,
every declared parameter type mapped onto the seven-name JSON Schema vocabulary the
pipeline trains. That is the strongest available evidence that the tool contracts are
being read rather than approximated, and it is the assertion that would have caught a
silent normalization failure.

### The general-quality corpus contributes no supervised rows at all

The open question from `s4.2` was whether `antidoom-mix` is training data or a prompt
set. It is a prompt set: **all 447,053 renderable rows end on a user turn**, and not one
carries an assistant response. The corpus is 478,229 rows; 20,096 were dropped as
contaminated and the rest went to the prompt pool.

Two consequences, and both change what `s5` does.

The trainable corpus is **518,330 rows, not the 1,004,253** the `s4.3` split sizes
suggested. Roughly half the manifest was prompts. That is not a loss: the pool is exactly
what the preference rung needs, and it is now sized and separated rather than discovered
mid-training.

The replay control in the plan (`C5`, replay fraction ∈ {0%, 1%, 5%}) has no corpus to
replay from. There is no general-instruction data with responses anywhere in the mix.
**Decision, taken here and recorded unreviewed:** replay draws on the base model's own
greedy completions over a sample of the prompt pool, generated once as a job at `s5.1`
and frozen for the whole sweep. That is self-distillation replay, it is what a prompt set
with no shipped responses is designed for, and it keeps the replay distribution on-policy
for the checkpoint being protected. The alternative, pulling a fourth-party instruction
corpus in at `s5`, would add a licence surface and an unmeasured contamination surface
this stage has no budget left to audit.

### Per-corpus, and the mix that comes out of it

| Corpus | Read | Kept | train / val / test | Flag drops | Unrenderable | Dedup |
| ------ | ---: | ---: | ------------------ | ---: | ---: | ---: |
| `toolace` | 11,300 | 9,098 | 8,181 / 466 / 451 | 117 | 2,067 | 3 |
| `apigen` | 49,402 | 46,350 | 42,346 / 1,979 / 2,025 | 41 | 1,856 | 0 |
| `hermes_fc` | 1,893 | 1,668 | 1,504 / 86 / 78 | 0 | 224 | 0 |
| `hermes_fc_st` | 1,893 | 1,100 | 910 / 95 / 95 | 0 | 61 | 732 |
| `hermes_glaive` | 5,209 | 3,833 | 3,748 / 43 / 42 | 3 | 162 | 2 |
| `hermes_json_ag` | 1,342 | 1,301 | 1,163 / 59 / 79 | 0 | 0 | 41 |
| `hermes_json_st` | 1,241 | 1,241 | 1,123 / 70 / 48 | 0 | 0 | 0 |
| `sql_ctx` | 78,577 | 69,543 | 69,069 / 299 / 175 | 0 | 0 | 0 |
| `sql_clinton` | 262,208 | 243,446 | 237,561 / 3,236 / 2,649 | 2 | 0 | 0 |
| `codefeedback` | 156,526 | 140,750 | 128,736 / 6,106 / 5,908 | 742 | 1 | 12,492 |
| `antidoom` | 478,229 | 0 | 0 / 0 / 0 | 20,096 | 0 | 0 |

### The mix is weighted against the project's own priority order

| Role | Rows | Share | Est. tokens | Share |
| ---- | ---: | ---: | ---: | ---: |
| SQL | 312,989 | 60.4% | 87.2M | 43.3% |
| Code | 140,750 | 27.2% | 72.0M | 35.7% |
| Tool calling | 62,049 | 12.0% | 40.8M | 20.2% |
| Structured output | 2,542 | 0.5% | 1.5M | 0.7% |
| **Total** | **518,330** | | **201.5M** | |

The brief's priority order is tool-calling reliability first, then on-device efficiency,
then no regression in general quality. The corpus as assembled puts 79% of its tokens on
SQL and code, and under 1% on the axis one of the two headline metrics measures
(`IFStruct` first-attempt schema validity, target +5.0 points). Uniform sampling over
this mix would spend most of the training budget on the two roles the project cares about
least.

That is an artifact of what is publicly available at scale, not of a choice made here:
the SQL corpora are large because SQL corpora are large. It is recorded now, before any
training, because it is cheap to fix by sampling weight and expensive to diagnose after a
sweep comes back flat on structured output.

**Decision, taken here and recorded unreviewed.** `s5.3` gains one control: the reference
arm `C1` trains on a role-balanced sample rather than on the raw mix, with per-role
sampling weights set so tool calling and structured output together take at least half
the token budget, and the raw-mix arm is kept as an explicit comparison. The exact
weights are an `s5.1` calibration against a single epoch's token count, not a number to
invent here. The falsification condition on `H1` is unchanged.

### Length, and the sequence budget

Sampled over 178,241 kept rows: mean 428 tokens, median 333, p90 920, p99 1,744, max
3,812. The 4,096 window set at `s4.2` costs 13 rows in the whole corpus, which settles
that setting. `hermes_fc` is the one corpus that runs long (mean 2,392, p90 3,425) and it
is 1,668 rows, so it does not move the packing arithmetic.

Estimated trainable tokens: **201.5M**. One epoch over the full mix is therefore a
200M-token pass, which is the figure the `s3.3` compute estimate should be read against.

### What the renderer could not read, and why each case is a decision

4,371 rows were dropped as unrenderable, and every reason is named rather than pooled.

- **`toolace` `no_call_turn`, 1,290.** Conversations that declare tools and never call
  one. They are legitimate text, but they teach a tool-calling model to answer without
  calling, which is the failure the probe sets are built to catch.
- **`toolace` `schema_not_json`, 658 across four spellings.** The tool block in the
  system turn is YAML (142), Markdown (71), a LaTeX `tabular` (66), or unrecognized
  (379). A tool contract that is not machine-readable cannot be canonicalized, and
  guessing at one would put an invented schema in front of a real call.
- **`apigen` `no_tools_declared`, 1,856.** Rows with an answer and no tool list.
- **`hermes` `conversation_ends_off_assistant`, 325.** The last turn is not the model's,
  so there is no target.
- **`hermes` `neither_tools_nor_schema`, 122.** No contract of either kind.
- **`codefeedback` `empty_answer`, 1.**

### The `glaive` recovery worked

865 of 5,209 rows returned nothing from the `s4.2` reader and were absent from the
manifest entirely. The `s4.4` renderer reads them: `hermes_glaive` keeps **3,833 rows**,
with only 162 unrenderable, against 4,344 rows that reached the manifest. The 865 are
accounted for as `adapter_error` and are not counted as trainable.

### Deduplication, measured

13,270 rows collapsed onto rendered text already emitted, on top of the 149,822 removed
at triple level in `s4.3`. Nearly all of it is two corpora: `codefeedback` 12,492 and
`hermes_fc_st` 732. The `hermes_fc_st` figure is the interesting one — 732 of 1,893 rows,
39% — and it is the single-turn variant of `hermes_fc` collapsing onto itself once the
conversation is rendered rather than compared field by field.

### Purity survived rendering

The one thing rendering could break that splitting could not: two different source rows
becoming the same prompt after canonicalization, one in train and one in held-out.
`purity_violations.json` is **empty**. No rendered prompt lands on both sides of the
train/held-out line, and none on both sides of val/test.

## s4.4 Probe sets — design

Two evaluation sets are built here rather than at `s6`, because a probe written after
seeing a model's errors measures the model that produced them and nothing else. Both are
hand-authored, and both are checked for overlap against the rendered training split on
the same 13-gram rule the corpora were decontaminated with. Hand-authored is not the same
as held out.

### Probe A — unreliable tool returns, 290 items

The failure this project is most exposed to is not calling the wrong tool. It is calling
the right tool, getting back something broken or quietly wrong, and answering anyway with
an invented number. No public benchmark in the plan measures it: BFCL scores the call,
not the model's handling of what the call returns.

Ten scenarios, each a realistic tool with a JSON Schema, a user question, and a correct
payload. Each scenario is then damaged two ways.

**Corrupted, 150 items.** The return is unusable and the only correct behaviour is to say
so. Five modes: JSON truncated mid-object, a field's type swapped, an error envelope in
place of a body, an empty body, and a null where a number is required.

**Contradicted, 120 items.** The return is well-formed and wrong in a way the model can
detect from the request it made. Four modes: the payload describes a different entity
than the one asked about, a list is silently truncated below its own stated total, the
`as_of` timestamp is older than the window requested, and a unit is swapped against the
schema's declared unit.

**Text-only controls, 20 items.** The same questions with no tool at all, to separate a
model that has learned to hedge from one that has learned to check.

Each item is graded on values the model would have to fabricate to answer, drawn from the
scenario's *correct* payload. The build asserts that every forbidden value actually
appears in that correct payload, so a grader cannot pass by accident on a string that was
never available to fabricate.

The corrupted and contradicted arms are each rendered at **three envelope depths** —
`{"data": …}`, `{"result": {"body": …}}`, `{"envelope": {"response": {"content": …}}}` —
because the hypothesis worth testing is that a break three levels down reads as plausible
where the same break at the top level is visible. Those envelopes are shapes real gateways
add, so depth is not a synthetic parsing tax.

### Probe B — stack idiom, 144 items

Six families over the operator's own stack: SQL dialect differences (10 base items), JSON
Schema (7), MCP (5), containers (5), tracing (5) and inter-process communication (4). Each
of the 36 base items is written around one specific wrong answer a general-purpose small
model tends to give, and carries both a required pattern and a forbidden one, so partial
credit for hedging is not available. The trap each item catches is recorded next to it,
which is what makes a regression readable later.

Each base item is rendered in **four surface forms**: the question directly, the same
question framed as reviewing a colleague's answer, a terse form, and one that says the
answer is going into production. The graders do not change across forms, so a model that
answers the direct question and fails the review framing has a robustness gap rather than
a knowledge gap, and the set separates the two instead of averaging over them.

### The three gates the build must pass

**Graders are well formed.** Every regex compiles, every item has at least one required
pattern, and no item requires and forbids the same string. Each corrupted item's forbidden
values are readable off the scenario's correct payload.

**No item is the same measurement twice.** Two things are checked, and the distinction
between them is the whole point. Two items may not be byte-identical on question *and*
tool return, and one question may not appear under two different scenarios, which would
let a single failure be counted twice. A shared question *within* one scenario is
intended: the corrupted and contradicted arms deliberately ask the same thing and differ
only in what the tool hands back, which is the comparison the probe exists to make.

**No item overlaps the training split.** Checked on the same 13-gram rule the corpora were
decontaminated with, against the rendered training text. If the training split is not
available to the build, the job says so in its log rather than passing quietly, and no
probe score may be cited from a build that skipped the check.

## s4.4 Probe sets — measured

Job `05bbcd49`, CPU only, 2m45s, **0 assertion failures**. Attempt 1 (`32fc86ad`) is
recorded in the ledger and superseded; what it caught is below, because the defect it
found is the reason to keep gates in the build rather than in a reviewer's head.

**434 items**, matching the design: 290 in probe A and 144 in probe B.

| Set | Arm | Items |
| --- | --- | ---: |
| A tool_return | corrupted | 150 |
| A tool_return | contradicted | 120 |
| A tool_return | text_only (control) | 20 |
| B stack_idiom | sql_dialect | 40 |
| B stack_idiom | json_schema | 28 |
| B stack_idiom | mcp | 20 |
| B stack_idiom | containers | 20 |
| B stack_idiom | tracing | 20 |
| B stack_idiom | ipc | 16 |

Each of the five corruption modes contributes 30 items and each of the four contradiction
modes 30, so no mode can dominate a score. Probe A's depth strata are exactly balanced at
90 items per envelope depth, with the remaining 164 items sitting outside the envelope
question (the 20 text-only controls and all 144 of probe B). Probe B's four surface forms
are 36 items each.

### The held-out gate, measured rather than asserted

The build downloaded the rendered training split and indexed it: **494,341 rows,
13,345,986 unique 13-grams**, in 44.4s. Against that index, **0 of 434 probe items share a
single 13-gram** with anything a model will be trained on. The worst-overlap list the job
writes is empty. Every probe score in this project can therefore be cited as held out on
a measurement, and the check is recorded in the job's own summary artifact rather than
claimed here.

### The duplicate gate, and the defect it found

Clean on the build of record: **0 byte-identical items, 0 questions appearing under more
than one scenario**, and 10 shared-question groups, all of them the intended within-scenario
pairs.

Attempt 1 failed this gate, and correctly. Two of the five corruption modes, the error
envelope and the empty body, replaced the payload outright and ignored the envelope depth
they were supposed to be sitting at. Both were therefore identical at depths 1, 2 and 3,
and **40 of 290 probe-A items were byte-for-byte copies of another item**. Scored as
built, the depth hypothesis would have been tested on a set where a third of one arm
carried no depth signal at all, and the finding would have looked like a null result.
Both modes now sit inside the same envelope the success payload would have arrived in,
which is also what a real gateway does with an upstream failure.

Attempt 1 also failed a third assertion that was itself wrong: it flagged 10 questions as
shared between arms. That sharing is the design. The corrupted and contradicted arms ask
the same question so that the only difference between them is the tool return, and an
assertion against it would have forced the two arms apart and destroyed the comparison.
The assertion was replaced by the two narrower checks described above, which keep the real
guarantee (no item counted twice, no question under two scenarios) without banning the
intended pairing.

### Stored

Items and summary at `tidepool/s4.4/probes/probes.jsonl` and
`tidepool/s4.4/probes/probes_summary.json` in shared storage; the summary and score are
also under `stages/s4-data-preparation/probes-05bbcd49/`. The set is generated
deterministically from two banks and the build script, with no sampling and no model in
the loop, so it reproduces exactly and carries no noise between checkpoints.

## s4.5 Data card

### What this dataset is

One supervised corpus of **518,330 examples and 201.5M trainable tokens**, rendered into a
single conversation format, plus a separate pool of **447,053 prompts** with no responses.
Assembled to specialize a 1.2B on-device model for the operator's agent stack. Four roles,
in the priority order the brief sets: tool calling first, then structured output, then
text-to-SQL and code, with a general-quality guardrail on top.

### Sources, and what each one contributes

| Corpus | Licence | Role | Kept rows | Est. tokens |
| ------ | ------- | ---- | ---: | ---: |
| `argilla/Synth-APIGen-v0.1` | Apache-2.0 | tool | 46,350 | 25.8M |
| `Team-ACE/ToolACE` | Apache-2.0 | tool | 9,098 | 7.7M |
| `NousResearch/hermes-function-calling-v1` (`func_calling`) | Apache-2.0 | tool | 1,668 | 4.0M |
| `…` (`glaive_func_calling`) | Apache-2.0 | tool | 3,833 | 2.2M |
| `…` (`func_calling_singleturn`) | Apache-2.0 | tool | 1,100 | 1.0M |
| `…` (`json_mode_agentic`) | Apache-2.0 | struct | 1,301 | 1.0M |
| `…` (`json_mode_singleturn`) | Apache-2.0 | struct | 1,241 | 0.5M |
| `Clinton/Text-to-sql-v1` | Apache-2.0 | sql | 243,446 | 80.5M |
| `b-mc2/sql-create-context` | CC-BY-4.0 | sql | 69,543 | 6.6M |
| `m-a-p/CodeFeedback-Filtered-Instruction` | Apache-2.0 | code | 140,750 | 72.0M |
| `LiquidAI/antidoom-mix-v1.0` | Apache-2.0 | prompts | 0 (447,053 to the pool) | — |

`Salesforce/xlam-function-calling-60k` is **not** in the mix. It is gated and the terms have
not been accepted on the account whose token we hold; accepting a licence on someone else's
behalf is not ours to do. The substitute is larger than the gated set and uniformly
Apache-2.0.

### Licence position

Every training corpus is Apache-2.0 or CC-BY-4.0, both of which permit commercial use and
redistribution with attribution. No non-commercial data trains anything.
`facebook/Multi-IF` is CC-BY-NC-2.0 and is an internal evaluation guardrail only: never
trained on, and excluded from the reproducibility package.

### What is deliberately absent

Gateway trajectories, meaning the operator's own research-plan-implement-audit loops with
real tool calls and returns, live in a running service that no credential on file reaches.
They are the only source of the operator's actual tool-call distribution, and nothing here
substitutes for them. The public-data-only ablation (`C2`) is what keeps the resulting claim
honest.

Private repositories are excluded from the in-house mining scope. The artifacts of this
project are mirrored to a public repository, and a model trained on private source can emit
it.

### Splits, and the property they guarantee

Group-disjoint by construction across 8 group namespaces, so no near-duplicate family
straddles the train and held-out line. Group disjointness alone was not enough: it left
39,161 identical prompts on both sides, because the same question reaches the corpus through
different sources under different group keys. A second enforcement pass removes any held-out
row whose prompt also appears in train, and any test row whose prompt also appears in val.
After it, and again after rendering, **no prompt is shared across the train and held-out
line, and none across val and test**.

Held-out is small on the axis that matters most: 451 test rows of tool calling out of 11,550.
In-mix held-out numbers are reported per capability rather than pooled, and the tool-calling
verdict rests on the external benchmarks and the two probe sets built here.

### What each experiment row consumes

| Consumer | Reads | Why that slice |
| -------- | ----- | -------------- |
| `C1` reference arm | `train`, role-balanced sample | the priority order, not the corpus's own shape |
| `C7` raw-mix comparison | `train`, uniform | measures what the balancing buys |
| `C2` public-data-only | `train` minus in-house stack rows | isolates what mining the operator's repositories buys |
| `C5` replay fractions | base-model completions over the prompt pool | the pool ships no responses of its own |
| `D` preference rung | the prompt pool | prompts without gold responses is what a preference pair needs |
| `s5.3` model selection | `val` | never `test`, at any point before `s6` |
| `s6` final numbers | `test` plus the two probe sets | scored once, after selection is frozen |

### Known defects, kept and labelled

Nothing here is claimed clean. Each defect the exploratory pass found is recorded with its
measured size, and the rows carrying it are either dropped by an explicit rule or kept with a
flag later stages can condition on. Contaminated rows (20,965), tool-name mismatches (121)
and unparseable calls (2) are dropped. Rows that refuse a request the tools could have
satisfied are dropped only where the same query is also answered elsewhere in the corpus,
which is what makes the refusal wrong rather than merely cautious; on this data that
conjunction selects all 1,942 suspect rows.

4,371 rows are unrenderable and every reason is named rather than pooled. The largest single
category is 1,290 ToolACE conversations that declare tools and never call one, and 658 whose
tool block in the system turn is YAML, Markdown or a LaTeX table rather than JSON.

### Intended and unintended use

Intended: supervised fine-tuning and preference optimization of a small on-device model for
tool calling, structured output, SQL and code, with a general-quality guardrail.

Not intended: any claim about safety behaviour. The refusal handling in this corpus is tuned
for over-refusal on benign tool-satisfiable requests, and says nothing about how the
resulting model handles requests that should be refused. Nor is it a general instruction
corpus: it is weighted toward one agent stack on purpose, and the role-balancing decision at
`s4.4` weights it further.

### Where the data lives

Shared storage, as a plain file tree, read by each job that needs it:
`tidepool/s4.3/splits.jsonl.gz` (the split manifest with per-row flags),
`tidepool/s4.4/train.jsonl.gz`, `val.jsonl.gz`, `test.jsonl.gz`,
`prompt_pool.jsonl.gz`, and `tidepool/s4.4/probes/probes.jsonl` (434 items, both probe
sets, one JSON object per line). Nothing is registered or versioned; the reproducibility
package at `s7.3` ships the recipes that rebuild them, not the bytes. The probe items are
the exception worth shipping verbatim, since they are hand-written here and exist nowhere
else.

### The two probe sets, as data

434 items, written for this project, covering the failure no public benchmark in the plan
scores: what the model does after a tool returns something broken or quietly wrong. 290
items on unreliable tool returns across 10 scenarios, 5 corruption modes, 4 contradiction
modes, 3 envelope depths and a no-tool control arm; 144 items on the operator's own stack
idioms across 6 families and 4 surface framings each. Licence: ours, and released with the
project. Held out on a measurement, not an assertion: 0 items share a 13-gram with the
training split. They are evaluation-only and must never appear in a training mix, which is
why they are stored under their own prefix rather than beside the splits.

## s4.5 sign-off

Autonomous mode is on for this project, so the checkpoint here was decided rather than
sent. **Signed off, and the stage is closed.** The bar for that decision was whether
anything measured at `s4` would make an `s5` result uninterpretable, and three candidates
were weighed.

**The token mix is wrong for the brief, and the fix is a sampler, not a rebuild.** SQL and
code carry 79% of the tokens on a project whose headline metrics are tool calling and
structured-output validity. Re-collecting to fix the ratio would cost days and buy nothing
a sampling weight cannot: the rows exist, there are simply more of one kind. The reference
arm draws role-balanced, the raw mix is kept as row `C7`, and the comparison between them
becomes a result rather than an assumption.

**Structured output is thin in absolute terms, at 2,542 rows and 0.7% of tokens.** It is
the smallest slice supporting a headline metric, and no open corpus in the licence position
this project holds would move it much. Flagged as the clearest gap in the data rather than
papered over; if `s5.2` baselines show structured-output validity failing to move, the
first thing to try is synthesizing more of it from the schemas already in the tool corpora,
which is a job, not a re-scope.

**Held-out tool calling is 451 rows.** Small enough that an in-mix test number on that axis
carries a wide interval, which is why the tool-calling verdict rests on the external
benchmarks and the two probes, and why held-out numbers are reported per capability.

None of the three blocks the experiment matrix, and each is recorded where the number that
depends on it will be read. Stage 5 opens on `s5.1` smoke runs, which also generate the
base model's frozen completions over the prompt pool that `C5` replay needs.
