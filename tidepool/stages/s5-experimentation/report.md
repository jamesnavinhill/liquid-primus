# Stage 5: Experimentation

_Status: `s5.1` complete, `s5.2` in progress. The throughput the whole budget rests on is
measured, the sweep is resized to fit it, and the evaluation harness every later number comes
from is built, verified against ground truth and smoked end to end. **The reference baseline is
complete at full item counts on all four components**, and it reads below the competitor on tool
calling and at the floor on the guardrail axis. The competitor row that carries the operative
tool-calling threshold is running, the 4-bit serving path is compiling after being moved to a
different compute source, and the harness has gained a score-only replay mode so a finished run
can be re-scored without paying for generation twice._

## Headline

The reference baseline is complete on all four components at full item counts, and two of its four
numbers reset what this project is aiming at. On tool calling it scores **0.6700 over all 3,490
held-out items** in its own calling format and 0.5355 in the convention our training data uses,
behind the 1B-class competitor's **0.7795** on a 378-item sample of the same suite under identical
decoding, so on the capability ranked first here the model we were asked to improve sits behind the
one it is measured against. On the safety axis it is **at the floor with a clean floor**: it flags
3 of 434 malformed tool returns against a target of 0.70, and raises **zero** false alarms on the
30 clean control items, which is a large gap to close and no bad habit to unlearn. Structured
output reads 0.1355 over 2,000 items, a within-harness reference with no published figure that may
be quoted beside it. Instruction following reads 0.8170 over all 541 prompts, 4.5 points under the
card's 86.23, which the four-way IFEval mean plausibly reconciles and the harness never printed the
numbers to confirm. The competitor's full pass is now running, and the 4-bit serving build is
compiling on a different compute source after two launches on the previous one provisioned a
machine and ran nothing, so there is no infrastructure fault to raise. The harness also gained a
score-only replay mode: a finished run's saved text can be re-scored on CPU work alone, which is
what recovers the per-item detail this run's artifacts lost. Spend is 3.017 of 145 GPU-hours, of
which 0.28 bought nothing at all.

## Work log

- 2026-08-25 · s5.1 · Two smoke jobs queued on L4 cards, deliberately on different
  providers: the supervised run `18573ffc` on one and the replay generation `defa6156` on
  another. The GPU cap is two at a time and both want the same card, so splitting them means
  a provider being out of L4 capacity costs one smoke rather than both.
- 2026-08-25 · s5.1 · Both smoke jobs reached the GPU and are reporting. The in-flight record
  and the per-attempt log live in `runs/s5.1-smoke.md`; live loss curves are on Weights &
  Biases under the `tidepool` project
  ([supervised run](https://wandb.ai/james-navinhill/tidepool/runs/mz0evlcl)).
- 2026-08-25 · s5.1 · Repaired the in-flight job marker, which had been written in a shape
  the run scheduler cannot parse. Left as it was, the two running jobs would have been
  invisible to it and the project would have sat still while they burned GPU time. The job
  intents it carried are preserved in `runs/s5.1-smoke.md`.
- 2026-08-25 · s5.1 · Both smoke jobs finished their work, saved every artifact, and then
  raised on the last statement: the harness's `finish` call was written with a status argument
  it does not take. Fixed in both task scripts against the SDK's published signature, and both
  smokes re-queued as `c8699eaf` (supervised) and `66aca6c7` (replay). Full measurements from
  attempt 1, the failure signature, and the relaunch accounting are in `runs/s5.1-smoke.md`.
- 2026-08-25 · s5.1 · Charged both failed attempts to the ledger at 0.232 GPU-hours together.
  They spent real compute and produced real measurements, so recording them as free would
  understate the stage.
- 2026-08-25 · mirror · Project state through the `s5.1` measurements pushed to
  `github.com/jamesnavinhill/liquid-primus` under `tidepool/`, per the standing operator note
  to mirror code and docs there as they land. Commit `255e15c`: the corrected code of both
  tasks exactly as re-submitted, this report, the `s5.1` run record with its full attempt log,
  the compute ledger, and the metrics both smoke jobs saved copied in verbatim so the loss and
  throughput figures can be checked against their source. Weights have the Hugging Face half
  of that note; the smoke adapter is a 30-step artifact off 512 rows, so it is not a build
  worth publishing and none has been pushed there yet.

- 2026-08-25 · s5.1 · The supervised smoke re-ran clean: COMPLETE, a full score dict on the
  job record, all five artifacts present, and figures within 2% of attempt 1. The harness fix
  is proven, so every later run in this stage will land its numbers where the ranked dashboard
  reads them.
- 2026-08-25 · s5.1 · The replay smoke re-ran clean as well, and its counts came back
  byte-identical to the first attempt: 64 of 64 prompts usable, 64 completions, none empty,
  15,677 tokens. Hash-ranked sampling over a fixed pool with frozen greedy decoding is supposed
  to be reproducible, and now it is shown to be. Throughput moved 4%, which is instance
  variation. Both smoke paths are closed on the science.
- 2026-08-25 · s5.1 · Queued a calibration run of the same supervised code on an L40S,
  at 60 steps over 1,024 rows. The plan's whole 145 GPU-hour budget rests on an
  assumed 6,000 tokens per second on that card, and the L4 measurement suggests the assumption
  is optimistic. One short job replaces the assumption with a measurement, which is cheaper
  than discovering it eight arms into the sweep.
- 2026-08-25 · s5.1 · The calibration run was refused at launch, at no compute cost, because the
  provider jobs go to by default does not sell the card the plan's budget is written against.
  Re-queued against the first source the hardware guidance lists for that card, and it ran clean
  as `68635a5d`. Every run in the sweep inherits the same default and will need the same pinning.
- 2026-08-25 · s5.1 · **The calibration landed and the budget moved.** 3,994.2 tok/s measured on
  the 48 GB card against 6,000 assumed, so every supervised row is half again as expensive as
  planned. Re-priced the sweep three ways and resized it to a fixed 64.0M-token arm budget, which
  holds all 8 arms inside the approved 145 GPU-hours. Full arithmetic and what a larger sweep
  would cost are under "Compute" below. The standing instruction to size the sweep to available
  compute and to price the larger version before overspending is what chose this shape.
- 2026-08-25 · s5.1 · Noted a second, quieter movement: the epoch token budget came back 9%
  lower than the L4 smoke derived it (83.1M against 90.6M). The mix rules are deterministic, so
  the spread is in the mean-tokens-per-row estimator, which reads a subsample. Every projection
  is quoted against the conservative figure and full-scale runs will report the exact count.
- 2026-08-25 · s5.1 · Substage closed. Every criterion the smokes were written to test came back
  clean, the harness fix is proven on the record, and the two numbers that size the rest of the
  stage are measurements.

- 2026-08-25 · s5.2 · Built the evaluation harness the whole project scores through, as one job
  that generates and scores in the same run (task `s5-eval`, 16 files). The alternative, a GPU
  pass that hands raw text to a cheap CPU scorer, needs the orchestrator to move artifacts
  between two jobs on every evaluation, and the scoring dependencies turned out to be light
  enough that the split buys nothing. Raw completions are saved before scoring touches them, so
  a grading bug found after eight sweep arms costs a rerun over saved text and no GPU time.
- 2026-08-25 · s5.2 · Decided the baseline gets prompted in its own tool-call format as well as
  ours, and scores the better of the two. Our training data presents tool contracts in a
  different surface form than the one LFM2.5 ships with, so grading the stock checkpoint only
  our way would have inflated every delta measured against it by the cost of a format mismatch.
- 2026-08-25 · s5.2 · Defined our tool-calling number as a named subset and wrote the
  exclusions into every summary file. A fifth of the public benchmark scores a model by
  executing its calls against live third-party APIs, and its multi-turn split grades against a
  stateful environment that is not in the data; neither is reproducible here. The published
  1B-class threshold in `overview.md` is an overall on the full benchmark, so `B4` is a rerun
  of that model through this harness and the rerun becomes the threshold.
- 2026-08-25 · s5.2 · Closed a hole in the project's own success criteria. They ask for a
  false-alarm rate on clean tool returns, and all 434 probe items are defective, so the
  quantity had no denominator. Added a 30-item clean control arm generated from the same bank
  and wrapper with the payload intact, saved as an artifact with a content hash on every run,
  plus a frozen 26-pattern detector so “flagged the defect” is measured separately from
  “did not invent a value”.
- 2026-08-25 · s5.2 · Verified every grader locally on fixtures before queueing anything.
  The tool-call checker scores a ground-truth oracle at 1.0000 across 858 items and 0.000 for a
  wrong function name, an extra parameter and silence; the structured-output validator loads all
  2,000 rows and rejects empty and garbage answers with the benchmark's own error strings; the
  instruction-following registry resolves all 541 prompts. One real bug surfaced: about a dozen
  instruction classes draw a missing argument at random, so the strict and loose passes were
  grading different instructions and loose scored below strict, which cannot happen. Seeded per
  item and instruction from a stable checksum, loose now sits at or above strict and repeat runs
  are identical.
- 2026-08-25 · mirror · Harness code and this stage's record pushed to
  `github.com/jamesnavinhill/liquid-primus` under `tidepool/`, commit `5d1cc34`: all 15 files of
  the evaluation task, the `s5.2` design and verification record, the re-priced compute ledger
  and plan. A `VENDORED.md` sits beside the code naming which graders are the benchmarks' own
  files, where each came from and under what licence, so a reader can tell our code from
  theirs at a glance.
- 2026-08-25 · s5.2 · Harness smoke queued as `5fe7a828` on an L4, at 40 items per component
  over the stock instruct checkpoint. It proves the path rather than measuring the model: the
  chat template accepting tool turns, both surface forms rendering and parsing, every grader
  returning, and the artifacts landing. The card, the serving backend and the decoding settings
  are frozen here and recorded in every summary, because changing them later invalidates every
  full-precision number in the project.
- 2026-08-25 · s5.2 · The smoke got most of the way and then raised: it proved the template and
  both tool-call surface forms, scored the first pass, and died writing the second pass's
  per-item file because one completion in 378 answered with a Python set where a number belongs.
  Both raw completion files were already on the job record, which is the reason they are written
  before scoring touches them, so the fix was checked against all 378 real completions at no
  compute cost. Charged 0.17 GPU-hours; the parser now normalizes exotic literals and the
  artifact writer can no longer be the thing that loses a paid-for generation pass.
- 2026-08-25 · s5.2 · Made every capped pass sample by stride rather than take the first N
  items, and measured what the change is worth instead of assuming it. On the tool-calling
  files it matters: a head slice of 40 `live_simple` rows averages 2.35 arguments per call
  against the category's own 2.76, so it quietly screens the easy end. On the other two
  benchmarks the rows are already interleaved and it changes nothing, which is the opposite of
  what I wrote here an hour ago; the correction is in `runs/s5.2-baselines.md` with the counts.
  Striding is deterministic either way, so a capped comparison between two models stays exactly
  paired.
- 2026-08-25 · s5.2 · Split the evaluation into two named profiles and made the caps explicit
  per-component parameters. Full counts for the baseline row and the final checkpoints; a
  screening profile for ranking the eight sweep arms against each other. Ranking arms does not
  need all 2,000 structured-output rows, and it does need every probe item, because the
  guardrail criteria are pass or fail over the whole set.
- 2026-08-25 · s5.2 · Re-priced evaluation against the smoke's own measurement rather than the
  estimate in this report an hour ago, which was too low. Full item counts and the arithmetic
  are under "Compute" below: about 2.5 GPU-hours per full-precision baseline and 0.55 per
  screened arm, which puts the stage through the sweep at roughly 49 of the approved 145
  GPU-hours. Nothing needs asking for.

## s5.1 What the smoke runs are for

A smoke test that only proves the code runs leaves the expensive half untested. These two
are ordered by where the failures actually happen, and each one is an assertion the job
fails on rather than a line in a log someone might read.

**The supervised path** (`s5-sft`, one code path for the whole `C` row of the matrix, so an
ablation and the arm it is compared against cannot diverge in their code):

1. The base checkpoint loads in bf16 on an L4. T4 is excluded on purpose: it is the only
   allowed card without bf16, and a loss computed in fp16 on this model would either fail or
   quietly degrade.
2. The model's **own chat template** accepts a four-role conversation including `tool` turns.
   If it does not, an explicit fallback is used and the run records that it did, because a
   number produced under a different template is not comparable to one produced under this
   one. The template is also checked for additivity across turns, since the loss mask is
   derived from per-turn token deltas and a non-additive template would silently misplace it.
3. The loss mask covers assistant tokens and nothing else. Both degenerate cases fail the
   run: a mask over no tokens optimizes nothing, and a mask over everything trains the model
   to write the user's questions, which reads later as a hyperparameter problem.
4. The sampler's calibration runs over the real 494,341-row training split.
5. LoRA attaches, and the number of trainable parameters is recorded rather than assumed.
6. Thirty optimizer steps move validation loss down. Not moving is a failure here, not a
   result.
7. **The adapter comes back as an artifact.** Saving results is the step that fails silently
   and only costs you at full scale, so a missing archive fails the run even when the
   training itself succeeded.

**The replay path** (`s5-selfdistill`): the prompt sample is reproducible, the pool templates
cleanly, greedy generation runs, few or no completions come back empty, and the replay file
is saved. It also measures completion throughput, which is what sizes the full buffer.

## s5.1 The mix calibration

The plan was amended at `s4.4` to sample the training mix role-balanced rather than uniformly,
because the rendered corpus puts 79% of its 201.5M tokens on SQL and code while the project's
headline metrics are tool calling and structured-output validity. The weights are not picked;
they fall out of three rules, and the rules are in the code rather than in a config file
someone can quietly tune:

1. Tool calling is the first priority and is never downsampled. One full pass.
2. Structured output is upsampled, because that is the only way to give 2,542 rows any
   weight, and capped at three passes, because repetition past that buys memorization.
3. Tool calling and structured output together take at least half the epoch's tokens. SQL and
   code fill the rest, downsampled in proportion to their own sizes so neither is favoured.

The epoch budget is whatever those rules imply, which is the point: it is not a free
parameter that could be tuned after the fact to make a mix look better.

| Role | Available tokens | Passes taken | Tokens in the epoch | Share |
| ---- | ---: | ---: | ---: | ---: |
| tool | 40.8M | 1.00 | 40.8M | 45.0% |
| struct | 1.5M | 3.00 | 4.5M | 5.0% |
| sql | 87.2M | 0.28 | 24.8M | 27.4% |
| code | 72.0M | 0.28 | 20.5M | 22.6% |
| **epoch** | **201.5M** | | **90.6M** | |

Structured output at 5% is the clearest consequence of the data gap recorded at `s4.5`, and
it is a gap rather than a choice: no amount of reweighting makes 1.5M tokens into 15% of a
90M-token epoch without repeating them ten times over.

Row `C7` runs the same code with the sampler off, so what the balancing buys is measured.
Both arms draw rows in the same hash order, which keeps the comparison paired.

## s5.1 Hardware, and the note that chose it

The team's provider steering note (updated 2026-08-23) sets the card and its sources. L4 is
the smallest allowed card with bf16 support, and its source row lists three providers rather
than one, so a single provider being out of capacity does not park the run. T4 is excluded by
the note for exactly the reason that matters here: it is the only allowed card without bf16.
Nothing in this stage needs more than 24 GB yet; a 1.2B model in bf16 with LoRA and gradient
checkpointing at a 2,048-token window fits with room to spare.

### The card the plan is priced against is not sold where jobs land by default

The calibration run was refused at launch, four seconds in and at no compute cost, because the
provider a job goes to when nothing pins it does not sell the 48GB card the whole budget is
written against. It sells the 24GB card the smokes have been running on, and a set of larger
ones the hardware guidance excludes.

The fix is one flag: the guidance lists each card's own sources in try order, and the 48GB card
has three, none of them the default. Re-queued against the first of them and now provisioning.
Worth recording rather than quietly correcting, for two reasons. Every run in the `C` sweep
inherits the same default, so a card that has to be pinned once has to be pinned every time,
and a sweep that silently lands on the smaller card would produce numbers that look like the
plan's and are not. And it is the second time in this stage that a launch has failed on
something the plan could not have known: a real card row, like a real SDK signature, is
learned by trying it.

## s5.1 What the smoke established

Attempt 1 of both jobs completed all of its work. Every criterion the substage was written
to test came back clean, and the numbers below are read from the saved artifacts rather than
from a log line:

| Check | Result |
| --- | --- |
| bf16 base checkpoint on an L4 | `LiquidAI/LFM2.5-1.2B-Instruct` loaded |
| Chat template accepts four-role conversations with `tool` turns | `native`, no fallback; 2 of 64 probe rows carried a tool turn and it took them |
| Loss mask covers assistant tokens only | 83 supervised of 896 probe tokens (9.3%), so neither degenerate case |
| Sampler calibrated over the real split | 178,346 rows selected from the full 494,341-row split |
| Structured-output repetition cap honoured | 6,858 rows taken from 2,286 available, exactly 3.00 passes |
| LoRA attaches | 11,108,352 trainable of 1,181,448,960 parameters (0.940%) |
| Validation loss falls over 30 steps | 1.3223 to 0.4940, delta -0.8284 |
| `priority_share == 0.5` | 0.5, exactly the plan's floor |
| `assertion_failures == 0` | 0, on both jobs |
| Adapter returns as an artifact | `adapter.zip` listed, with four metric files beside it |
| Replay path | 64 of 64 prompts usable, 64 completions, 0 empty, 204.1 tok/s, `replay.jsonl` saved |

The template result is the one worth flagging. The run was written to fall back to an explicit
template if the model's own rejected a tool turn, and to record that it had done so, because a
number produced under a different template is not comparable to one produced under this one.
The fallback was never needed, which keeps every later run in this stage on the model's native
format.

The two numbers that size the rest of the stage are both measurements now. Supervised
throughput settled at 1,435.3 tok/s across the run, which puts the full 90.6M-token epoch at
**16.08 GPU-hours**. Replay generation ran at 204.1 completion tok/s, or **33.34 hours per
100k completions**. The 145 GPU-hour plan was built on estimates and has two of its rows
anchored; what that does to the rest of the plan is taken up under "Compute" below.

## s5.1 The one thing that broke, and the proof it is fixed

Both jobs crashed after finishing, on the same statement. The harness's completion call takes
a message and a score dict and has no status argument, and both scripts passed one, so the
call raised `TypeError` once all the work was done.

Nothing was lost. Every artifact is written before that call, so the adapter, the metrics and
the replay file were all saved and were read back to recover the table above. What the crash
did take is the job record: both jobs read `FAILED` with an empty score field, so neither
shows up in the ranked dashboard that `s5.3` and `s5.4` are decided from.

Waiving it and moving on was the tempting option, since the science had passed. The reason not
to is the shape of what comes next: a 16-hour reference run, then baselines and an 8-row sweep
ranked by that same score field. A patch that has never executed is not yet a fix, and the
cheapest possible place to prove one is a pair of smokes at 0.1 GPU-hour each. So attempt 2 is
the identical experiment with the completion call corrected in both scripts, queued as
`c8699eaf` and `66aca6c7`.

The supervised half of attempt 2 has since finished and the fix holds. The job reads COMPLETE,
its score dict is on the record where the dashboard reads it, and all five artifacts are
present. Its figures also stand as a reproducibility check on attempt 1, since the two runs
share a seed and differ only in that one line: val loss 1.3223 to 0.4771 against 0.4940, and
1,398.3 tok/s against 1,435.3, so a 16.51-hour projected epoch against 16.08. The 2% spread
across a 30-step run on shuffled data is the noise floor of a smoke, not a discrepancy to
chase. The replay half is still on GPU. The full attempt log, the failure signature and the
relaunch accounting are in `runs/s5.1-smoke.md`.
- 2026-08-25 · s5.2 · Queued the reference baseline row: `4350ce4e` runs the full-precision
  instruct checkpoint at full item counts, 3,491 tool-calling items, 2,000 structured-output
  rows, 541 instruction-following prompts and the whole 464-item probe set, in both surface
  forms. Every later delta in this project is measured against it.
- 2026-08-25 · s5.2 · Hardened the tool-call parser before letting it near a competitor. Tested
  against the call formats other 1B-class checkpoints emit, it read **none** of them, and one
  case failed silently: arguments under a `parameters` key came back as a correctly named call
  with an empty argument dict, which the grader would have marked wrong with nothing looking
  broken. Scoring the competitor with that parser would have floored it and turned a published
  threshold into a figure we invented. 14 fixtures now pass, and re-running the change over the
  756 real completions already saved confirms zero of them are read through a newly added path,
  so the extra leniency cannot manufacture a call and inflate the restraint categories.
- 2026-08-25 · s5.2 · The pre-registered threshold has two published values. Granite 4.0-1B
  scores **52.43** on Liquid's card, measured with Liquid's own weighted handler, and **54.82**
  on IBM's card for the same dense 1B checkpoint. Neither is what this harness computes.
  `overview.md` keeps 52.43 as the pre-registered figure, since moving a bar after seeing a
  second source is how a bar becomes whatever the result needs; the matched rerun is what
  settles it. Recorded in `runs/s5.2-baselines.md`.
- 2026-08-25 · s5.2 · Queued `c644aac2`, a 40-item Granite smoke, before paying for the full
  competitor pass. It answers whether the checkpoint loads on this stack, whether its own chat
  template accepts our tool rendering, and whether the hardened parser reads its real output
  rather than the fixtures written for it. 0.15 GPU-hours against the 2.5 a full pass costs.
- 2026-08-25 · s5.2 · Built the missing half of the harness. Three baseline rows are published
  4-bit GGUF files and the project's headline promise is about 4-bit builds, and neither could be
  measured: GGUF runs in llama.cpp, which publishes no prebuilt Linux CUDA binary. The evaluation
  task now carries a second backend behind the same interface, so a 4-bit row and the
  full-precision row it is compared against differ in weights and runtime and in nothing else.
  A new job compiles the backend once at a pinned tag, serves the vendor's own Q4 file on the
  eval card to prove the architecture runs, checks the server reproduces its own output at
  temperature 0, and prices a full 4-bit pass from measured throughput. The three published files
  are named explicitly in `runs/s5.2-baselines.md`, and the task refuses to pick a quantization by
  pattern.
- 2026-08-25 · s5.2 · The Granite smoke settled all three of its unknowns for 0.30 GPU-hours and
  paid for itself twice over. The competitor loads and renders our tool contract; its two surface
  forms differ by a factor of four, with 299 of 378 prose-prompted completions being the bare
  opening marker and nothing after it; and it generates at a quarter of the reference
  checkpoint's rate on the same card, which re-prices its full pass at about five hours. Its
  batch size stays at 16 rather than being raised for speed, because a baseline served
  differently from the row it is compared against is not a baseline. Ceiling raised to seven
  hours.
- 2026-08-25 · s5.2 · Found the real cause of the artifact bug, which the previous run had
  misdiagnosed. Five of eleven files logged as saved were absent from the artifact listing and
  from the download path, and the five absent were exactly the five marked as eval results.
  Marking a file that way files it under metadata the tooling cannot read back. Every save in the
  eval task is now untyped, the wrong explanation is corrected in the run log and the ledger, and
  the rule stands: a job's outputs are unproduced until `job artifacts` has listed them.
- 2026-08-25 · s5.2 · The reference row's tool-calling pass finished: **0.6700 composite over all
  3,490 held-out items** in the model's own tool template, against 0.5355 in our training
  convention. The competitor's 378-item smoke reads 0.7795 under the same protocol. On the
  capability ranked first in this project the incumbent is currently ahead of the checkpoint we
  were asked to improve, and the gap is larger than sampling noise at that sample size. Recorded
  as the finding that shapes `s5.3` rather than as a number in a table.
- 2026-08-25 · s5.2 · Queued `ad6cded5` into the freed GPU slot: the pinned-tag CUDA build of the
  4-bit serving stack, which every quantized baseline row waits on.

## s5.2 The evaluation harness, and what its numbers mean

One job definition scores every model this project reports on: the baseline row, the eight sweep
arms, the tuning runs and the final checkpoints. It is the most reused artifact in the project,
so the decisions inside it are recorded in full here and in `runs/s5.2-baselines.md`.

| Component | Items | Graded by | Metric |
| --- | --- | --- | --- |
| Tool calling (BFCLv3-AST) | ~4,400 over 11 categories, in 2 surface forms | our checker, verified against the answer key | unweighted category mean, item-weighted beside it |
| Structured output (IFStruct v1.0) | 2,000 | Liquid's own validator, vendored verbatim | first-attempt validity, mean partial score |
| Instruction following (IFEval) | 541 | Google's own instruction registry, vendored | prompt and instruction level, strict and loose |
| In-house probes | 434 graded + 30 clean control | the `s4.4` checks, plus a flag detector | flag rate, false-flag rate, stack-idiom accuracy, by envelope depth |

Four decisions shape what those numbers mean.

**The baseline is prompted in its own format as well as ours.** The `s4` training data puts a
tool contract in the system message as a JSON array and expects a JSON call back. LFM2.5 ships
with a template that takes tools as an argument and answers in Python call syntax. Scoring the
stock checkpoint only in our convention would understate it, and every delta measured against it
would carry the format mismatch as a free gain. Both forms are generated, the parser accepts
either regardless of which was prompted, and a baseline scores the better of the two.

**Our tool-calling number is a named subset.** The `exec_*` and `rest` categories score by
executing calls against live third-party APIs with real credentials, and `multi_turn_*` grades
against a stateful environment that lives in the benchmark's harness. Neither is reproducible
here, so the metric is the composite over the 8 abstract-syntax categories plus the 3 that score
restraint, equal weight per category, with the exclusions written into every summary file.
Categories differ in size by two orders of magnitude, so a size-weighted mean would mostly
report on one of them; both are printed. The absolute threshold in `overview.md` is a published
overall on the full benchmark and is therefore not comparable, which is why `B4` reruns that
model through this harness and the rerun becomes the threshold.

**Benchmark prompts go in verbatim, with no system prompt.** Adding our own structured-output
instruction would flatter every model on the structured-output benchmark and break comparability
with the published figure, even though our models are trained on it. The in-house probes carry
their own system prompt in the data, so nothing is added to them either.

**The guardrail criteria needed a denominator.** `overview.md` asks for a flag rate of at least
0.70 on malformed tool returns with a false-flag rate at most 0.15. Every graded probe item is
defective, so there was nothing to false-alarm on: a 30-item clean control arm is now built from
the same bank and envelope wrapper with the payload intact, generated deterministically and
saved with its content hash (`561a23f3c6532c4f`) on every run. Separately, the original checks
pass an item when the completion avoids the value the model would have to invent, which is a
different event from saying the return was wrong. A frozen family of 26 flag phrasings now
measures the second alongside the first, and the original check is unchanged.

### What was proven before any GPU time was spent

Fixture checks in the sandbox, none of them a result.

- The tool-call checker scores a **ground-truth oracle at exactly 1.0000 on all 858 items** of
  three categories, and 0.000 with the function name perturbed, 0.000 with one extra parameter,
  0.000 emitting no call. Restraint categories score silence and calls in the right directions.
  A checker that cannot score a perfect model as perfect quietly floors every model it grades.
- One data quirk found on the way: 7 of 3,351 ground-truth parameters carry an empty list of
  acceptable values, the benchmark's way of saying the parameter must be left empty. Read
  naively that marks a correct call wrong.
- The structured-output validator loads all 2,000 rows and returns the benchmark's own error
  strings for an empty and a garbage answer.
- **A determinism bug in the instruction-following grader, found and fixed.** About a dozen of
  its 25 instruction classes draw a missing keyword argument at random, so an unseeded run
  grades the strict and loose passes against two different instructions. The first local run
  scored loose *below* strict, which is impossible by construction. Seeded per item and
  instruction from a stable checksum, loose sits at or above strict and two runs over the same
  completions agree exactly.
- The probe graders behave correctly on real items of all three check kinds and the clean arm,
  in both the pass and the fail direction.

### The serving path is frozen here

Hugging Face `transformers`, bf16, greedy, left-padded length-sorted batches, on one L4. It is
the same code path the training jobs load a model through, which is why it was chosen: nothing
extra to install and no batching-dependent numerics to argue about later. Measured cost is
about 2.5 GPU-hours for a full pass and 0.55 for a screened one, priced under "Compute". Moving to a faster
inference server later invalidates every full-precision number taken here and means rerunning
all of them, so the backend, the card and the decoding settings are part of every recorded
result. The 4-bit builds (`B2`, `B3`, `B5`) need a separate `llama.cpp` serving path and are not
scored through this task.

## s5.2 The 4-bit half of the harness

Half of what this project promises is about 4-bit builds: three of the six baseline rows are
published GGUF files, and the success criteria ask a 4-bit form to hold the full-precision bar.
Until now only the full-precision side had a harness, so that half of the claim had no way to
be measured.

It is now a second backend on the same task rather than a second task. `backend=gguf` swaps the
generator and nothing else: item selection, prompt rendering, parsing and scoring are literally
the same code on both sides of every quantization claim. Prompts are rendered by the tokenizer
of the full-precision repo the GGUF was quantized from and tokenized by the server with the
special-token prefix suppressed, because the chat template already writes that token as text and
two of them in front of every prompt would make the comparison a comparison of prompts.

llama.cpp publishes no prebuilt Linux CUDA binary, so the backend is compiled by its own job at
a pinned tag, targeted at the one GPU architecture both project cards share, and kept as a
stored artifact that later 4-bit runs download rather than rebuild. That job does not stop at a
successful compile: it serves the vendor's own Q4 file, offloads it to the card, generates from
it, checks that two identical requests at temperature 0 return identical text, and prices a full
4-bit pass from its own measured throughput. It also keeps the quantizer and the GGUF converter
from the same tag, because our own checkpoint has to become a GGUF eventually and the converter
that makes it must match the runtime that serves it.

### The confound, stated rather than hidden

GGUF only runs in llama.cpp, so a 4-bit row and a full-precision row cannot share a backend.
What the resulting delta measures is two *deployed artifacts*, weights and runtime together,
which is what the operator's criterion is about: an 8 GB card runs a 4-bit build, not a
dequantized tensor. Quantization and runtime are still separable, and cheaply, because the vendor
publishes an F16 GGUF beside the Q4 ones: F16-in-llama.cpp against Q4-in-llama.cpp isolates
quantization, and F16-in-llama.cpp against full precision in `transformers` prices the runtime.
All three are one parameter change on the same task, and each number records which comparison it
belongs to.

## Operator input (s5.1)

The run was asked to hand back for a reply and the answer came back automatically: no operator
is reading, proceed on my own recommendation and record what I chose. There was no open
question at that point. The mix calibration described above was already derived from the rules
in the plan rather than chosen, both smoke jobs were already on GPU, and the structured-output
shortfall was a fact to report rather than a decision to take.

So the recommendation I proceeded on is the one written here: let both smoke runs finish, hold
the 5% structured-output share for the baselines so the gap is measured rather than patched,
and generate more structured-output data from the schemas the tool corpus already carries only
if validity fails to move once the baselines are in. Two decisions were taken in this run that
nobody reviewed. Re-running both smokes to prove the one-line harness fix, at about 0.2
GPU-hours in total, on the reasoning above. And putting a short L40S calibration run ahead of
the baselines, because the plan's assumed throughput is the single number the whole budget
rests on and one short job replaces it with a measurement.

## Compute

Everything before this stage ran on CPU. The two attempt-1 smokes are the first GPU charge on
the project at **0.232 GPU-hours** together, measured from job start to terminal status so
that provisioning is included; the tasks themselves ran 204.9 s and 77.2 s. Attempt 2 adds
roughly the same again. Against a 145 GPU-hour plan and a 200 GPU-hour enforced allowance
(re-read at this substage: 2 GPUs at a time, L4 among the allowed types, 0 held), the stage
has spent 0.2%.

The failed attempts are charged in full. They spent the compute and they produced the
measurements the rest of the stage is sized from, so recording them as free would make the
ledger flatter than the project.
### What the calibration measured, and what it costs

The plan's whole 145 GPU-hour budget rested on one assumed number: 6,000 training tokens per
second for a 1.2B model on one L40S, the default card for every supervised row. That number is
now measured. **It is 3,994.2 tok/s, so the plan was optimistic by 50%.** The L4 smoke had
bracketed the card at 3,600-5,000 and the truth sits at the bottom of the band, which is the
answer that costs money and the reason the run was worth 0.12 GPU-hours.

A second figure moved at the same time, and it is the one nobody was watching. The epoch token
budget came back at 83.09M here against 90.6M on the L4 smoke, a 9% spread on the number every
training row is priced from. The mix rules are deterministic, so the spread lives in the
per-role mean-tokens-per-row estimate, which is taken from a subsample whose size follows the
smoke row count. Both figures are estimates and the estimator is noisier than the budget table
implied, so everything below is priced against the conservative **90.6M**.

At the measured rate, one epoch of the role-balanced mix costs **6.30 GPU-hours**. The plan's
`C` row bought 8 training runs and 8 screenings for 38 GPU-hours, which after 2.4 hours of
screening leaves 4.45 GPU-hours per arm, or **64.0M tokens, 0.71 of one epoch**. The plan's own
estimation basis assumed each arm was 2 epochs of a 120M-token corpus. Against the mix that
`s4.4` actually rendered, 2 epochs is 181M tokens:

| Sweep shape (8 arms) | Per arm | `C` row total | Whole plan, +25% contingency |
| --- | ---: | ---: | ---: |
| 2 epochs, as the plan's basis assumed | 12.60 GPU-h | 103.2 | **227 GPU-h** |
| 1 epoch | 6.30 GPU-h | 52.8 | **164 GPU-h** |
| 64.0M tokens (0.71 epoch), the 38-hour line | 4.45 GPU-h | 38.0 | **145 GPU-h** |

The enforced allowance is 200 GPU-hours. The 2-epoch sweep is therefore not merely over its
line, it is **off the table**: it would need 227 GPU-hours and the account cannot hold it. The
1-epoch sweep fits the allowance at 164 GPU-hours but spends 19 hours past what was approved at
`s3.4`.

### The sweep is sized to fit, and here is what a larger one would cost

The standing instruction on this project is to size the first sweep to the compute actually
available and to say what a larger version would cost before spending past what has been
approved. That decides it, so the `C` sweep runs at a **fixed 64.0M-token budget per arm**,
which is the third row above: it holds the 38-hour line, it stays inside the 145 GPU-hours
approved at `s3.4`, and no operator has to be asked for anything.

Fixing a **token** budget rather than an epoch fraction is what keeps the comparison honest.
`C7` samples the raw mix uniformly and draws rows with different length statistics from `C1`'s
role-balanced sample, so equal epochs would mean unequal tokens and the mix-weighting result
would be confounded by exposure. Equal tokens, drawn in the same hash order, leaves the
sampler as the only difference between them.

Two things this trade gives up, recorded rather than buried. Each arm sees 71% of an epoch
instead of a full pass, so an arm that would only separate late in training is scored early;
the sweep ranks recipes and `s5.5` takes the winner to full length, which is the shape that
absorbs this. And the ranking rests on one seed per arm, as the plan already had it.

What a larger version costs, stated before anything is spent: the same 8 arms at one full epoch
each is **52.8 GPU-hours**, 15 hours past the `C` line, landing the plan at 164 of the 200
allowed. At two epochs each it is 103.2 GPU-hours and the plan does not fit the account at all.
A three-seed version of the sized sweep, which is what would turn the ranking into a
significance claim, is 114 GPU-hours for the `C` row alone and needs a quota raise.

One row of the sweep is still unpriced. `C3` is full-parameter SFT, not LoRA: at bf16 with Adam
states it wants roughly 14 GB of optimizer memory, which the 48 GB card holds comfortably, but
its throughput per token is lower than an adapter's and by how much is not measured. It is
budgeted at the same 64.0M tokens as its siblings and flagged here as the one arm whose cost
could overrun its share. If it does, the overrun gate catches it before the next launch.

### What evaluation costs, priced against the harness smoke

The estimate in this report an hour before the smoke ran was half a GPU-hour per model. The
smoke measured it, and the estimate was low. Both figures below are derived from job
`5fe7a828`'s own saved completions: 378 tool-calling items generated 21,000 tokens at 255
tokens per second, so 0.22 seconds per item is the unit everything else scales from.

Full item counts, read from the benchmark files rather than assumed:

| Component | Items | Passes | Est. task time |
| --- | --- | --- | --- |
| Tool calling | 3,491 across 11 categories | 2 surface forms | ~40 min |
| Structured output | 2,000 | 1 | ~30 min |
| Instruction following | 541 | 1 | ~15 min |
| Probes + clean control | 464 | 1 | ~6 min |
| | | | **~1.5 h task, ~2.5 GPU-h with provisioning** |

The screening profile caps the two large components and keeps the probe set whole: 120 items
per tool-calling category in one surface form, 400 structured-output rows, 200
instruction-following prompts, all 464 probe items. About 23 minutes of task time, **0.55
GPU-hours per arm**. Arms are screened in the training convention only, because after
supervised fine-tuning that is the model's own format; the native-form regression check is
reserved for the finalists, where it is a question worth 40 minutes and on eight arms it would
not be.

Where the stage lands, with the sweep already resized at `s5.1`:

| Row | GPU-hours |
| --- | --- |
| spent to here (`s5.1` smokes, calibration, harness smoke) | 0.88 |
| `s5.2` full baselines: `B1`, `B4`, `B6` at full counts | 7.5 |
| `B1` again at the screening profile, so arm comparisons are paired | 0.55 |
| `s5.3` sweep training, 8 arms at 64.0M tokens each | 35.6 |
| `s5.3` arm screening, 8 arms | 4.4 |
| **through `s5.3`** | **~49 of 145 approved** |

The remaining 96 GPU-hours cover what the plan allotted 66 to (preference rung, merging, the
RL arm behind its kill-line, export and final evaluation), so the stage stays inside the
approved figure and nothing needs asking for. The 4-bit baselines `B2`, `B3` and `B5` are not
in the table: they need the `llama.cpp` serving path, which is not built, and they are priced
when it is.

- 2026-08-25 · s5.2 · The first `llama.cpp` build, `ad6cded5`, failed in its own environment
  probe: no CUDA compiler in the image, and the wheel that installed one landed in a
  site-packages the job never looked in. Charged 0.18 GPU-hours as an observed bracket, since
  the platform recorded no end time.
- 2026-08-25 · s5.2 · Repaired and relaunched the serving build as `602b78a9`. Four changes:
  find the compiler by searching the filesystem and install with the same interpreter that
  searches; assemble one CUDA prefix from all four toolchain wheels, because the compiler wheel
  has no `cuda_runtime.h`; assert the running server reports a CUDA device, so a CPU fallback
  cannot pass as a build; and drop `type=` from every artifact save, which this project has
  already proven makes a file unreachable. Signatures and attempt counts in
  `runs/s5.2-baselines.md`.
- 2026-08-25 · s5.2 · `602b78a9` came back complete having run nothing: a zero-byte log, no
  artifacts, and fifteen empty polls for a job on its own machine. Ruled out the disk request and
  a broken script, charged the 0.09 GPU-hours the instance really held, and relaunched
  byte-identical as `25e75d99` so the retry tests one hypothesis and not two. A second identical
  signature is a platform fault to raise rather than a fourth build to attempt.

- 2026-08-25 · s5.2 · Attempt 4 of the serving build, `b7de2af2`, changed one value: the compute
  source, `gcp` to `aws`, with the card held at `L4:1` and `sm_89` so nothing about the
  comparison moves. It is running and compiling, which settles the question the two lost runs
  posed: they were specific to one source, and there is no fault for anyone to fix. The
  provider steering note is what pointed here, since it works a card's full source row before
  the card counts as unavailable.
- 2026-08-25 · mirror · This turn's state pushed to `github.com/jamesnavinhill/liquid-primus`
  under `tidepool/`, commit `387cf92`: the re-sourced build task and its script, the `s5.2` run
  record with the four build attempts and their diagnosis, the reference row's two full-count
  scores, the anchor limit now written into `overview.md`, and the ledger at 1.597 of 145
  GPU-hours.

## The 4-bit serving path is the stage's open blocker

Three of the six baseline rows are 4-bit GGUF, and the promise in the success criteria is that a
4-bit build holds the full-precision bar. Only one side of that comparison can currently be
measured. The first build of the `llama.cpp` serving path, job `ad6cded5`, failed inside its own
environment probe: the image ships no CUDA compiler, the wheel fallback installed one and
reported success, and the job could not find it, because it searched only `PATH` and the
site-packages of the interpreter running the script while installing with whichever `pip` was on
`PATH`. Cost 0.18 GPU-hours and produced nothing.

Repairing it surfaced three faults that had not fired yet and would each have cost a launch of
their own: the CUDA prefix handed to CMake was the compiler wheel alone, which has no
`cuda_runtime.h`; nothing asserted the served model actually reached the GPU, so a CPU fallback
would have finished green and mispriced every 4-bit measurement by roughly thirty times; and all
three artifact saves were typed, which this project has already proven puts a file where no `job`
subcommand can list it — including the tarball that is the job's whole deliverable. Attempt 2 carried
all of that: the compiler located by searching the filesystem, a synthetic CUDA prefix assembled
from all four toolchain wheels, a hard refusal to build without `cuda_runtime.h`, a hard
assertion that the running server reports a CUDA device, untyped saves, and its own
shared-storage upload to the object name the Q4 evaluation task already reads.

None of it ran. Attempt 2 was reported complete five minutes after launch having written a
zero-byte log, saved no artifacts, and recorded fifteen consecutive empty polls for a running job
on its machine: the instance was provisioned and the task never started on it. Two candidate
causes were checked and dropped — the disk request, which four other jobs in this project have
provisioned fine at the same size, and a broken script, which a local compile rules out and which
would have written a traceback rather than nothing. Across eighteen jobs in this project this is
the only one with a nonzero empty-poll count, so the launch is the suspect and the task is not.

Attempt 3, job `25e75d99`, therefore went out byte-identical. With no evidence that any task code
executed, changing the task would only confound the one question worth asking. If it returns the
same signature the repeat is itself the finding and the next move is to raise it as an
infrastructure fault, not to rebuild a fourth time.

The full narrative, including the exact failure signature of each attempt, is in
`runs/s5.2-baselines.md`. Attempt count in this substage: 3, of which one failure had a cause in
the task and one had none.

## The lost runs were not one-off, and the fix was a source rather than a marker (s5.2)

Attempt 3 of the 4-bit build repeated attempt 2's signature exactly and faster: reported complete
fourteen seconds after launch, zero-byte log, no task logs, no machine logs, no artifacts. The
entry above committed the next move to raising this as an infrastructure fault. **On re-reading
the provider steering note that commitment was wrong, and it was not followed.**

The note of 2026-08-23 lists the source row for the 24 GB card this build uses as AWS, then GCP,
then RunPod, to be worked left to right without stopping before the last entry, and it is explicit
that exhausting a row is still not a reason to park a project. Two of those three sources had
never been tried. The only reason both lost attempts landed on the one that fails is a pin this
project itself added at attempt 2, after attempt 1 provisioned there cleanly. A pin of our own is
a fix available on our side, which makes this the relaunch case and not the escalate case: parking
the project while two thirds of the card's sources were untouched would have cost the operator a
day and reached people who could not have helped.

Attempt 4 therefore changes exactly one value, checked by diffing the uploaded task against the
running one before it was applied: the compute source moves to the first entry in the row. The
card is held identical, which is what keeps the binaries and the measured throughput comparable
with every full-precision number in this project, so the note's re-measure exception does not
apply. If this attempt prints a single line, pass or fail, the build is debuggable again. If it
comes back empty the fault follows the task across two sources, and the next step is the third
and last source in the row before anything is escalated.

**This decision was taken without operator review**, under the standing instruction to proceed on
my own recommendation and record it. The steering note is what decided it, and it outranked the
plan recorded here an hour earlier.

## The structured-output result has no public anchor, and that is a limit on the claim (s5.2)

The reference row's structured-output pass finished at full counts: **first-attempt schema
validity 0.1355 over 2,000 items**, against 0.125 on the 40-item smoke. The smoke's recorded
explanation was that the small sample was to blame and the full pass would settle it upward.
Fifty times the items moved the number by a single point, so that explanation is now withdrawn.

The same section promised to re-investigate the harness if the full number came in far under the
vendor's published figure, and following that promise turned up something more useful. **The
vendor publishes no structured-output score for this checkpoint at all.** Its card carries seven
columns and this benchmark is not among them; the one published figure in the source material,
85.49, belongs to a different checkpoint of twice the size, in a table whose five models spread
from 36.25 to 85.49. There is no published number for this model to fall short of, so the trigger
never fires and the harness is not implicated.

What that leaves is a constraint rather than reassurance: on this device tier the structured-output
score has no comparable public reference, so it is a within-harness figure only and no published
number may be quoted beside it. The tool-calling metric is already in exactly that position for
the same reason. The bar every later checkpoint is measured against is this harness's own
reference row at full counts, which is what the plan specified.

Two loose ends are recorded and neither holds the substage. The harness computes a partial-credit
score alongside strict validity and only the strict one was printed, so whether the model
half-satisfies these schemas or misses them wholesale is still unknown, and it is the difference
between two quite different claims about this tier. And the per-item scored file for this run is
not retrievable, because the run launched before the typed-save fix. Both are recoverable from the
raw completions, which *are* on the job record, by re-scoring rather than regenerating. That
re-scoring step has to exist before the error analysis in stage 6 reads per-item detail, and it is
queued work rather than local work, because the partial score is a number this project would cite.

## Operator input (no checkpoint)

A prompt raised at the end of the previous wake was answered automatically rather than by a
person: _"Full Self Driving: no operator is answering this. Proceed on your own recommendation, and
record in the stage report which option you took and why."_ Two decisions followed from that and
both are recorded above in full: the lost 4-bit build was moved to a different compute source
instead of being escalated as an infrastructure fault, on the strength of the team's own provider
guidance; and the structured-output score was accepted as a within-harness reference with no public
anchor, rather than triggering another harness investigation, because the published figure it was
going to be checked against turned out to belong to a different checkpoint. Nobody reviewed either.

## The reference row is complete, and the guardrail axis is the finding (s5.2)

`4350ce4e` finished clean: four components at full item counts, zero assertion failures, 1,318,883
generated tokens at 279.1 tok/s, 85 minutes on one L4. The full table is in
`runs/s5.2-baselines.md`; three of its readings matter at stage level.

**The safety axis is at the floor, and the floor is clean.** `overview.md` asks for a flag rate of
at least 0.70 on malformed tool returns with a false-flag rate at most 0.15. The stock checkpoint
flags **3 of 434** and false-alarms on **none** of the 30 clean control items. Each number alone is
uninformative: a model that flagged nothing would also score 0.0 on the control arm. Together they
say the checkpoint is silent rather than cautious, which is the most improvable gap in the whole
matrix and the one where the control arm built at `s5.2` immediately earned its cost, since without
it the flag rate would have read as a straightforward win waiting to happen.

**Instruction following lands 4.5 points under the published figure and the gap is probably the
metric.** We measure 0.8170 prompt-level strict against the card's 86.23. IFEval is commonly
published as the mean of its four scores, and the instruction-level pair runs above the prompt-level
pair by construction. A four-way mean near 86 is consistent with what we measured, so this is a
reconciliation and not yet a demonstration: the grader computes the instruction-level pair and the
harness never printed it.

**The two surface forms are 13.5 points apart on our own checkpoint**, and the weaker one is ours.
0.6700 native against 0.5355 in the convention `s4` trains toward. The rule that every baseline is
prompted both ways and scored on the better of the two was written to be fair to competitors; it
turns out to be load-bearing for the reference row, and it means an `s4` finetune must clear 0.6700
rather than the 0.5355 it starts from on the format it is trained on.

## Re-scoring is now a mode of the harness rather than a second copy of it (s5.2)

B1's per-item files and its summary were saved before the typed-save fix and are unretrievable,
exactly as predicted when that bug was diagnosed: the six untyped saves came back and the six typed
ones did not. What is lost is all derived detail, including the structured-output mean partial score
and by-format split, the instruction-level IFEval pair, and the per-category tool-calling table.
Every one of them is a function of text that *is* retrievable.

The fix is a parameter on the eval task, not a new task. `rescore_object` points at a storage
directory of saved completions; the run then loads no weights, uses no GPU, generates nothing, and
each component reads its text from there by item id. A second task with the graders copied into it
was the obvious build and the wrong one: two copies of a scorer is how one set of completions ends
up with two different scores, and the whole point of this harness is that it must not move
underneath the numbers.

It joins on id rather than position, re-renders every prompt and checks it against the recorded
hash so a moved chat template surfaces as a counted mismatch, rebuilds the clean probe arm and
diffs it item-by-item against the source run's saved copy, and reports how many completions it
scored rather than inventing a tokens-per-second figure. Verified against a slice of B1's own
completions in the sandbox, all five branches including the failure ones. B1's six files are in
shared storage at `tidepool/s5.2/B1-fp-instruct/` at full counts.

It is not queued yet, and the reason is a real constraint rather than an oversight. The eval task
requests an L4 at the `resources` level, resources are not overridable per queue, and both
concurrent GPU slots are full. The two ways around that were worse than waiting: a CPU-only twin
task reintroduces the duplicated graders, and editing the task's resources down and back up is a
mutable-configuration dance whose failure mode is the eval task left CPU-only and the next full row
landing on a machine with no GPU. So the replay runs on the L4 when a slot frees, for about 0.2
GPU-h of a deliberately idle accelerator. Recorded as a decision taken unreviewed under autonomous
mode, with the exact queue command in `runs/s5.2-baselines.md`.

## What is running, and what B6 turns out to be (s5.2)

`c319ebb1`, B4, is the Granite 4.0-1b full pass, queued into the slot B1 freed. It stays on the
same compute source B1 ran on, unpinned. The build task was moved to another source because two of
its launches provisioned a machine and ran nothing, and the temptation was to move everything; the
evidence does not support that. All three eval launches on that source ran, the only two lost
launches in about twenty project jobs were the build task at a 200 GB disk request, and moving B4
would put the competitor row on a different image family from the reference row it is compared
against. Expect roughly five hours: Granite generates at a fifth of B1's rate at the same batch
size, and the batch size is deliberately not raised for it.

`b7de2af2`, the 4-bit serving build, is compiling. The source it was moved to ships a complete CUDA
toolkit, cmake configured for sm_89 cleanly, and the elaborate four-wheel compiler repair written
for the previous source never had to fire. Rows B2, B3, B5 and the F16 runtime reference wait on
its tarball.

`B6` was listed in the plan as conditional on the base checkpoint being published. It is:
`LiquidAI/LFM2.5-1.2B-Base`. One limit goes on the record before it runs. The base and instruct
checkpoints ship different chat templates, 1,296 bytes against 5,487, both with the same role
framing and the same tool rendering, so the harness handles either unchanged. A B1-versus-B6 delta
is therefore a delta across two checkpoints and two templates at once, and running B6 alone cannot
separate them. Forcing the instruct template onto the base weights would isolate the checkpoint by
prompting a model in a format it never saw, which is the mistake the two-surface-form rule exists to
prevent. B6 is scored under its own template and reported as what post-training bought, template
included. Nothing depends on it.

## The ledger was missing its largest entry (s5.2)

B1 had no entry in `budget.json` while three failed build attempts totalling 0.28 GPU-h were all
recorded. Added on its own reported bracket, 1.42 GPU-h against a 2.5 GPU-h estimate, so the row
came in 43% under its line. B4 is recorded as RUNNING with spend `null`, on the same rule as the
launching build: a spend is charged when known and left null when not, never estimated into the
total. Project spend is **3.017 of 145 approved GPU-hours**.

## Immediate next actions

1. Read `b7de2af2`'s log once it reports built: the offload lines after `server healthy` and the
   CUDA-device assertion, the 8-slot generated tok/s, the priced hours for a full Q4 pass, and
   whether `job artifacts` **lists** the tarball. If it is listed but the in-job storage upload
   failed, place it as `tidepool/llama-b10622-sm89.tar.gz` before queueing any 4-bit row.
2. When a GPU slot frees, queue the B1 replay with the command recorded in
   `runs/s5.2-baselines.md`. It settles four things B1 could not report: the structured-output mean
   partial score and by-format split, the instruction-level IFEval pair that would confirm or refute
   the reconciliation with the card's 86.23, and the per-category tool-calling table that `s6`'s
   error analysis reads. It also exercises the replay path on real data before `s6` depends on it.
3. Then the three 4-bit rows `B2`, `B3` and `B5` against the built serving path, plus the F16
   runtime reference that makes a Q4 quality claim a comparison rather than an assertion.
4. Then `B6` at the screening profile, `LiquidAI/LFM2.5-1.2B-Base` under its own template, with the
   template caveat carried into the summary.
5. Do not tick `s5.2` until the 4-bit rows exist. Three of the six baseline rows and the project's
   headline promise, that a 4-bit build holds the full-precision bar, all sit behind that one
   tarball.
6. Carry the 64.0M-token arm budget into `s5.3` as an explicit parameter rather than a habit, and
   pin the compute source at `resources` level for tasks asking for a 48 GB card, on the evidence
   recorded for the build task and not beyond it.
7. Before `s5.3`, re-read what B1's guardrail numbers imply for the sweep. The flag rate is the
   axis with the most headroom in the whole matrix, and the arm design was written when that
   headroom was a guess.
