# Stage 5: Experimentation

_Status: `s5.1` complete, `s5.2` in progress with three of its seven rows complete, a fourth part way and a fifth running. The
throughput the whole budget rests on is measured, the sweep is resized to fit it, and the
evaluation harness every later number comes from is built, verified against ground truth and
smoked end to end. **The reference baseline is complete at full item counts on all four
components**, and it reads below the competitor on tool calling and at the floor on the guardrail
axis. **The competitor's tool-calling component is now measured at full item counts**, and it sits 9.4
points above the reference. **Two 4-bit rows are complete**, through a `llama.cpp` path this stage built and verified; both
hold every axis except the one our training data is written in, and the gap between them prices
what quantization-aware training is worth. The runtime control that makes either interpretable is
running.
**The guardrail training set is built and clean**: three attempts, nine gates, 7,988 rows and
4.35M tokens in shared storage with a 138-item clean control arm. One sweep arm was retired as
unrunnable and replaced._

## Headline

The reference baseline is complete on all four components at full item counts, and two of its four
numbers reset what this project is aiming at. On tool calling it scores **0.6700 over all 3,490
held-out items** in its own calling format and 0.5355 in the convention our training data uses.
The 1B-class competitor has now finished the same 3,490 items under identical decoding and scores
**0.7641**, so on the capability ranked first here the model we were asked to improve sits **9.4
points behind** the one it is measured against. The full pass is kinder than the 0.7795 its
378-item screening sample suggested, and the gap it leaves is still larger than any single
intervention in the sweep is priced to close. On the safety axis the reference is **at the floor
with a clean floor**: it flags 3 of 434 malformed tool returns against a target of 0.70, and
raises **zero** false alarms on the 30 clean control items, which is a large gap to close and no
bad habit to unlearn. Structured output reads 0.1355 over 2,000 items, a within-harness reference
with no published figure that may be quoted beside it. Instruction following reads 0.8170 over all
541 prompts, 4.5 points under the card's 86.23, which the four-way IFEval mean plausibly
reconciles and the harness never printed the numbers to confirm.

**Both community and vendor 4-bit builds are in, and the gap between them is the finding.** The
vendor's quantization-aware Q4_0 build holds 95.4% of the reference on its native calling format
(0.6392 against 0.6700). The best cheap community post-training quant holds **87.1%** (0.5837).
Same binary, same card, same items, same decoding: quantization-aware training is worth **5.6
points** of tool calling on this family, which is direct evidence for the recovery rung the export
plan at `s5.6` already carries. On the `s4` text convention the two builds agree with each other
and disagree loudly with the reference, retaining **78.6%** and **81.0%** against a 93% per-axis
floor. Read naively that says 4-bit destroys the exact surface form our training data teaches,
which is what nine papers in `s2` predict. The rows differ in two things though, precision and
serving runtime, so the reading cannot be accepted yet. The control that separates them, the same
weights unquantized through the same binary, is **now running**, and every 4-bit claim downstream
waits on it. On the safety axis the community build reads exactly where the reference does: 0.0074
flagged, zero false alarms, three of four rows on the same floor.

The 4-bit path itself works, on the fifth attempt: `llama.cpp` b10622 compiled for the target
architecture in 1,039 seconds, the binaries are in shared storage, and the first full pass through
them ran at **776.7 generated tokens per second** over 1.33M tokens. The 1.71-hour price the
ledger carries for each 4-bit row came from an 8-slot concurrency measurement and is roughly three
times too high for a serial pass; the row finished in 34 minutes.

The guardrail training set is now built and clean on all nine gates, on the third attempt and on a
CPU box that held no accelerator slot: **7,988 rows and 4.35M tokens** of defective tool returns
built by the vendored probe transforms, each paired one-for-one with its own intact counterpart,
with two defect kinds held out of training and the taught phrasings split against the frozen
detector at **0.6838 coverage** so the flag rate cannot become a recall test on our own wording.
Nothing quotes a value its damaged response no longer carries, nothing shares a question with a
probe, and the **138-item clean control arm** is four times the frozen one, which is what lets the
plan's 0.15 false-alarm ceiling be measured rather than asserted.

**One sweep arm was retired.** `C2` was specified as "public data only", subtracting the in-house
stack corpus to isolate what it buys. `s4.2` authorized that mining and `s4.4` recorded that it
produced zero supervised rows, so `C2` would have spent 4.45 GPU-hours reproducing its own
reference arm byte for byte. The cell now runs **`C2'`, the reference arm with the guardrail block
removed**, which isolates a claim the project is making: the amended gate requires the probe flag
rate to reach 0.35 from a baseline of 0.0074, and without `C2'` the sweep would pass or miss that
gate with no evidence about which part of the recipe moved it.

Spend is **4.442 of 145 GPU-hours**, of which 0.68 bought nothing at all.

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

- 2026-08-25 · mirror · The reference row's result and the harness's new replay mode pushed to
  `github.com/jamesnavinhill/liquid-primus` under `tidepool/`, commit `8f1009b`: all of `s5-eval`
  including `replay.py`, the `s5.2` run record with B1's full table and the replay design, the
  corrected ledger, and this report.

## The 4-bit build compiled, and the check that judged it was wrong (s5.2)

Attempt 4 configured cleanly, compiled llama.cpp for our card in 1,038 seconds, collected all four
binaries at 70-82 MB, downloaded the vendor's quantization-aware 4-bit model and reached a healthy
server. It then failed at 45% on a guard I wrote: the assertion that the server had initialized a
CUDA device rather than falling back to the CPU, where every 4-bit number would be correct and
every hour quoted from it wrong.

The guard fired on evidence it should not have trusted. A CPU-only build of this project takes a
few minutes and produces much smaller binaries, so the compile almost certainly did include CUDA.
More tellingly, the startup text the guard searched was missing the model loader's own output as
well as its CUDA lines, on a server that had just loaded a model and answered a health check. A log
with no loader lines is a log read before the writer flushed it, so the guard proved nothing either
way.

The costlier defect is the ordering. Packing the tarball, saving it as an artifact and pushing it to
shared storage were all the last step of the job, after verification, so a failed check discarded
seventeen minutes of successful compiling. Every earlier attempt had died before the compile, which
is why that ordering had never been exercised.

Attempt 5 (`905755d3`, same source, same pinned tag and card) makes three changes. The binaries are
packed and stored the moment they exist, with a `verified` flag that stays false until every check
passes, so a stored tarball is not yet a licence to serve from it. The device is proved by asking
the runtime to enumerate its backends, which returns on its own and cannot be read early, with the
driver, the linked CUDA libraries and the CMake cache recorded beside the answer; the run fails hard
only when the enumeration really answered and no CUDA device was in it, or when none of the three
signals is positive. And a throughput floor of 150 generated tokens per second across eight slots is
asserted after the batched measurement. The floor is the guard that actually protects the pricing: a
1.2B 4-bit model on this card sits in the high hundreds and on a couple of CPU threads in the tens,
and unlike the log check it also catches a build that reaches the device but is too slow for the
hours it quotes to mean anything.

The startup log is still captured, still saved as an artifact, and now saved *before* anything is
asserted on it. It reports rather than decides.

- 2026-08-25 · mirror · The build's diagnosis and attempt 5 pushed to
  `github.com/jamesnavinhill/liquid-primus` under `tidepool/`, commit `82dceca`: the rewritten
  build script and its config, this stage's record and run log, the corrected ledger and the task
  list.

## One of the three 4-bit rows is not the comparison it appears to be (s5.2)

With the build compiling, the CPU-only work worth doing was pinning exactly which file each 4-bit
row loads and which repository renders its prompts. All three publish a dozen quantizations, and
the harness refuses to pick one by pattern at run time.

| Row | What it loads | Prompts rendered by |
| --- | ------------- | ------------------- |
| B2 | the vendor's quantization-aware `QAD-Q4_0` | `LiquidAI/LFM2.5-1.2B-Instruct` |
| B3 | unsloth's `UD-Q4_K_XL`, the best cheap community quant | `LiquidAI/LFM2.5-1.2B-Instruct` |
| B5 | Nova's function-calling fine-tune at `Q4_K_M` | `NovachronoAI/LFM2.5-1.2B-Nova-Function-Calling` |
| runtime ref | the same weights at `BF16`, unquantized | `LiquidAI/LFM2.5-1.2B-Instruct` |

Two of those were decisions. The Nova repository publishes no `Q4_0`, so B5 is a K-quant, which
leaves B2-against-B3 as the matched pair the plan wanted and B5 as a third point. The runtime
reference is taken at bf16 rather than f16 because B1 was served bf16, and holding the dtype fixed
makes the serving runtime the only thing that moved between them.

The finding is about B5. The Nova checkpoint ships its own chat template, a third of the size of
Liquid's, and it renders tool contracts as a plain `List of tools: [...]` line inside the system
prompt with none of the framing LFM2.5 uses and no declared convention for the call itself. So a
B5-against-B1 gap contains a different fine-tune, a third tool surface form and 4-bit
quantization, all at once, and it cannot be read as a verdict on a competitor's 4-bit calling
quality. The harness's existing policy, scoring every baseline under both its own rendering and
ours and taking the better, handles the prompting half. Separating the fine-tune from the
quantization needs a full-precision Nova row, about 1.4 GPU-hours, which the plan does not call
for and nothing downstream needs; recorded as an option for `s6`'s error analysis and not queued.
B5 stays what the plan made it, with the caveat carried into its summary.

The four queue commands are written out in `runs/s5.2-baselines.md`, ready to fire, and none of
them may run until a build reports `verified: true`.

## The guardrail gap is in the data, not the checkpoint (s5.2)

B1's flag rate of 0.0074 against a pre-registered 0.70 is the largest headroom in the matrix, so
the question is which sweep arm closes it. Measured on the rendered corpus the sweep will train
on, the answer is none of them.

| | rendered val | rendered train |
| --- | ---: | ---: |
| rows | 12,439 | 494,341 |
| rows containing a tool-result turn | 164 (1.32%) | 5,221 (1.06%) |
| ...whose assistant text carries doubt or failure language | 2 | 75 (0.015%) |

The 75 are a keyword net rather than a finding: read individually they are an email-validation
tool reporting a suspicious domain, a transaction hash handed back, a travel itinerary. The number
of training rows that show the model a broken tool return and an assistant saying so is, as far as
this measurement reaches, zero. The corpus teaches calling tools at length and almost never teaches
what to do with what comes back, which explains B1's 0.0074 without reference to the checkpoint at
all.

Three parts of the `s3.2` matrix rest on the opposite assumption. `C4` and `C5` are both listed as
moving guardrail deltas, and neither can teach a behaviour absent from the mix; they protect one
that is already there, which is what `C5` was always most useful for. And the stop/go gate into
`D`, "no guardrail worse than −3.0", cannot fail at a flag rate of 0.0074, while the question the
project actually pre-registered, whether the rate reaches 0.70, is not in the gate at all.

**Decision.** `s5.3` gains a data component before its arms are frozen: a purpose-built training
set for unreliable tool returns, generated as a job, on scenarios disjoint from the 434 probe items
and decontaminated against them on the same 13-gram rule the corpora were cleaned with. The
criteria stay as pre-registered and the arms stay as designed. Two conditions travel with it, both
for the paper rather than a footnote. The probes stop being a measure of untaught behaviour, which
is what specializing a model means and is not contamination while the items stay disjoint, but the
write-up must not imply a capability appeared on its own. And the risk inverts to false alarms: the
cheap way to reach 0.70 is to complain about everything, the 0.15 false-flag ceiling is what
forbids it, and 30 clean control items cannot bound a rate at 0.15 usefully. The control arm needs
enlarging by hand, on the same scenario set, before any arm trained on the new supervision is
scored. Taken autonomously and recorded here; the full measurement is in
`runs/s5.2-baselines.md`.

- 2026-08-25 · mirror · The 4-bit row pinning and the guardrail measurement pushed to
  `github.com/jamesnavinhill/liquid-primus` under `tidepool/`, commits `69a8b8b` and `56a9900`:
  the exact file and prompt-rendering repo for each 4-bit row with the B5 caveat, the corpus
  measurement behind the guardrail decision, the design note for where that supervision comes
  from without a teacher model, and the amended task list.

## Immediate next actions

1. Read `905755d3` in this order: whether `job artifacts` **lists** the tarball, which now
   happens right after the compile and is the thing four attempts have failed to deliver; the
   device enumeration block with its driver, linker and CMake evidence; the 8-slot generated tok/s
   against the 150 floor; and the priced hours for a full Q4 pass. If the tarball is listed but the
   in-job storage upload failed, place it as `tidepool/llama-b10622-sm89.tar.gz` before queueing
   any 4-bit row, and do not serve from it unless the summary reports `verified: true`.
2. When a GPU slot frees, queue the B1 replay with the command recorded in
   `runs/s5.2-baselines.md`. It settles four things B1 could not report: the structured-output mean
   partial score and by-format split, the instruction-level IFEval pair that would confirm or refute
   the reconciliation with the card's 86.23, and the per-category tool-calling table that `s6`'s
   error analysis reads. It also exercises the replay path on real data before `s6` depends on it.
3. Then the three 4-bit rows `B2`, `B3` and `B5` against the built serving path, plus the bf16
   runtime reference that makes a Q4 quality claim a comparison rather than an assertion. The
   commands are written out and the files are pinned; they go two at a time behind whichever of
   the build and B4 finishes first.
4. Then `B6` at the screening profile, `LiquidAI/LFM2.5-1.2B-Base` under its own template, with the
   template caveat carried into the summary.
5. Do not tick `s5.2` until the 4-bit rows exist. Three of the six baseline rows and the project's
   headline promise, that a 4-bit build holds the full-precision bar, all sit behind that one
   tarball.
6. Carry the 64.0M-token arm budget into `s5.3` as an explicit parameter rather than a habit, and
   pin the compute source at `resources` level for tasks asking for a 48 GB card, on the evidence
   recorded for the build task and not beyond it.
7. Before `s5.3`'s arms are frozen, build the two things the guardrail measurement above
   requires: a decontaminated training set for unreliable tool returns, and an enlarged clean
   control arm to hold the false-flag ceiling honest once the model has been taught to flag.
   Both are generation jobs and neither needs a GPU slot of the size the sweep does.
8. Re-word the stop/go gate into `D` so it tests the criterion the project pre-registered. As
   written it guards against a guardrail regression that a flag rate of 0.0074 makes impossible,
   and says nothing about reaching 0.70.

## Operator input: a decline, and what I did with it (s5.2)

The `.awaiting` this run answered was a synthetic `no-checkpoint` request carrying last run's
guardrail-data finding, and the reply was a `decline`: *"Full Self Driving: no operator is
answering this. Proceed on your own recommendation, and record in the stage report which option
you took and why."* Both markers are retired. The recommendation on the table was the one already
recorded above, to give `s5.3` a purpose-built training set for unreliable tool returns plus an
enlarged clean control arm before the arms are frozen. I took it unchanged and built both. What
follows is the design, because a decline means nobody reviewed it and the record has to carry the
reasoning instead of an approval.

## The build works, and the 4-bit rows are unblocked (s5.2)

`905755d3` came back success on the fifth attempt, and all three of the changes it carried held.

| | |
| --- | --- |
| build | `llama.cpp` b10622, `CMAKE_CUDA_ARCHITECTURES=89`, static, CUDA 13.2, 1,039 s |
| tarball | **listed by `job artifacts` and in shared storage** at `tidepool/llama-b10622-sm89.tar.gz`, 175.7 MB |
| device evidence | all four true: backend enumeration (`CUDA0: NVIDIA L4, 22,565 MiB`), driver, linker, CMake cache |
| single stream | prefill 17,443 tok/s, generate 261.2 tok/s on a 1,532-token prompt |
| 8 slots, 24 requests | **196 generated tok/s**, 12,717 including prefill, against a 150 floor |
| determinism | two identical requests returned identical output |
| priced | a full Q4 pass at **1.71 h** of task time on this card |
| spend | 0.369 h against a 0.75 h line, 51% under |

The one soft spot is worth naming: the server's startup log carried no CUDA line in the 14 lines
captured, so the runtime's use of the GPU rests on the device probe and the throughput floor rather
than on the server announcing it. At 196 generated tok/s over 8 slots on a 1.2B model that is not
a CPU, and the redesigned check exists precisely because reading a server's log at the wrong moment
is what discarded attempt 4's compile. Recorded as evidence-by-throughput, not as a log line.

**`B2` is queued** as `efa9719d` into the slot the build freed: the vendor's quantization-aware
`QAD-Q4_0`, prompts rendered by `LiquidAI/LFM2.5-1.2B-Instruct`, full item counts, gcp, priced 1.71
h. It goes first of the three rather than `B3` or `B5` because it is the exact file the build served
and measured, so a launch failure would be about the harness rather than about an unproven weights
file, and because it is the row the project's headline promise rests on. The card stays `L4`. The
provider steering note (updated 2026-08-23T22:48Z) allows substituting upward when a card is
unavailable, with a narrow exception for a number that will be compared against one already
measured on specific hardware; `B2` is exactly that comparison against `B1`, `L4` was available,
and holding the card fixed is what keeps the 4-bit-versus-full-precision gap attributable to the
quantization.

Order for the remaining slots, decided rather than asked: `B3`, then `B5`, then the bf16 runtime
reference, then `B6`, and the `B1` re-score last. The re-score was listed ahead of the 4-bit rows
in the previous plan, written when the build was still unproven and a GPU slot had nothing better
to do. It now would: the re-score buys analysis detail `s6` wants, while `B2`, `B3` and `B5` are
the three rows that gate `s5.2` and the promise the project was funded on. Convenience yields to
the gate.

## The guardrail training set, and the three things that stop it being circular (s5.3)

`a42a6c21` is running, CPU only, holding no accelerator slot and charging nothing to the ledger.
It produces two artifacts under `tidepool/s5.3/tooldata/`: the paired training set, and a
`clean_corpus` control arm.

**Where the supervision comes from.** Corrupting a valid tool return is a deterministic transform,
so the correct reply is a function of which transform ran. No teacher model, no sampling, no
grading of generated text. The job reads the rendered training split, keeps rows where a tool
return has a JSON body, a quotable headline value and a real assistant answer after it, damages
the return, and writes the templated reply that damage demands. Rows whose real reply is *another
tool call* are dropped: teaching "say it is broken" where the corpus itself emits a call would
train prose over a call site.

**Three anti-circularity measures.** A guardrail trained on its own measurement is not a guardrail,
and each of these is a fatal gate in the job rather than a note.

1. *The transforms are vendored, not reimplemented.* `defects.py` imports the probe generator
   `build.py` and calls `build.corrupt` directly, so the five structural corruption modes in
   training are literally the code that produced the graded probe items. A reimplementation would
   leave a train/test difference nobody could measure.
2. *Two defect kinds are held out.* `null_leaf` and `stale_as_of` never appear in training, which
   leaves 60 of the 290 tool-return probe items testing defect kinds the model was never taught.
   The flag rate can then be reported on taught and untaught kinds separately, and the untaught
   half is the one that says whether anything generalised.
3. *Target phrasings are split against the frozen detector.* Half the templates use wording one of
   the 26 frozen patterns fires on; the rest say the same thing in wording the detector is blind
   to. Without the split, a trained flag rate would report how well our own phrasing was memorised.
   The job fails if coverage falls outside `[0.60, 0.90]`, or if fewer than three defect families
   have a target the detector misses.

**The false-alarm guard is structural, not a hyperparameter.** Every defective row is paired 1:1
with its own source row, intact, carrying the corpus's real answer as the target. So the contrast
the model sees is one payload, damaged and not, with two different correct behaviours. That is a
deliberate deviation from the probes' 30:434 clean-to-defective proportion, recorded here because
it is a design choice and not an oversight: the probes' proportion is right for *measuring* a
false-flag rate and wrong for *teaching* the boundary, and the cheap way to reach a 0.70 flag rate
is to complain about everything.

**The other gates.** No target may quote a value the damaged response no longer carries, which is
the exact fabrication the probes forbid, checked per row against the intact payload's leaves rather
than annotated. No held-out mode may appear. No trained mode may end with zero surviving rows. The
clean-to-defective ratio must stay within 10% of 1:1. Every source row must come from the train
split. The control arm must reach 200 items. Decontamination runs on s4's own 13-gram rule
(`blake2b`, digest 8, punctuation stripped) against the probe items' request surface, with a
separate whole-row overlap *fraction* limit of 0.30 rather than an absolute rule, because the depth
envelope and the `silently_truncated` wrapper are boilerplate shared with the probes by
construction and one shared gram there means nothing.

**What the fixture caught.** Two bugs, both of the kind that would have produced a plausible
training set. Two templates tagged as detector-visible matched no frozen pattern, so the measured
coverage would have been reported against a mislabelled pool. And `wrong_entity` perturbed an
identifier by overwriting its last character with a literal `9`, which for every id already ending
in 9 returned the original string: 46 of 300 fixture rows asserted a mismatch between a value and
itself, with targets reading *"reports acct_00229 where the call asked for acct_00229"*. The
perturbation now shifts the character and refuses to emit a defect that collapsed onto the
original. On a 600-row fixture the job ends with 0 gate failures, coverage 0.65, and all seven
trained modes present.

**Two consequences carried forward.** First, the enlarged control arm has never been generated for
`B1`, and score-only replay cannot cover items that did not exist when the completions were
written, so `B1` needs a small supplementary generation pass over the new arm (a few GPU-minutes)
before any trained arm's false-flag rate has a baseline to be compared against. Second, the eval
task needs the arm wired in: a `clean_control_object` parameter and additive keys in `summarize()`.
`false_flag_rate_clean` reads only the frozen `("tool_return", "clean")` bucket, so `B1`'s recorded
0.0000 and the frozen 30-item arm are untouched by construction, and the new arm reports beside
them rather than over them.

## The stop/go gate into `D` now tests what the project pre-registered (s5.2)

Amended in `plan.md`, autonomously, and fixed before any arm is trained on the new supervision so
it stays pre-registered rather than fitted. The gate read: gain ≥ 1.5 points on BFCLv3 over `B1`
with no guardrail worse than −3.0. Two holes. A −3.0 clause is a regression guard, and `B1`'s
measured flag rate of 0.0074 leaves nothing to regress from, so an arm that learns nothing about
broken tool returns clears it by sitting at the floor. And the criterion `overview.md` actually
pre-registered, a 0.70 flag rate with false flags under 0.15, was tested nowhere on the path into
`D`. Two clauses added: a probe flag rate of **≥ 0.35** on malformed returns, and a false-flag rate
of **≤ 0.15** on clean ones.

The asymmetry is the decision. `0.35` rather than `0.70` because 0.70 is the `s6` success criterion
on the *final* checkpoint, and demanding it of a screening-profile supervised arm would kill the
preference rung for not already being the finished result; half the distance from a floor of 0.0074
is a real signal the axis moved and is reachable by supervision alone. The false-flag ceiling is
carried at the pre-registered 0.15 with no relaxation, scored on the enlarged `clean_corpus` arm
rather than the 30-item one, because the cheap route to a high flag rate is to complain about
everything and the ceiling is the only clause forbidding it.

Also carried forward, from the previous run's action 6: the 64.0M-token per-arm budget becomes an
explicit `s5.3` task parameter rather than a number reproduced by habit, and `resources`-level
compute pinning stays limited to tasks asking for a 48 GB card, which is the only class the
evidence covers. Neither is actionable until the `s5.3` sweep task is authored, which waits on
`s5.2` closing.

**Deliberately not done this run: wiring the new control arm into `s5-eval`.** The change is
additive (a `clean_control_object` parameter, extra keys in `summarize()`, `false_flag_rate_clean`
untouched because it reads only the frozen `("tool_return", "clean")` bucket), so it would not move
a recorded number. It is still the wrong moment. Two rows are mid-flight on that task and four more
follow, and editing an evaluation harness in the middle of a baseline matrix is how you end up with
`B1` scored by one set of graders and `B5` by another. The patch waits for `s5.2` to close, at
which point all six baseline rows share identical graders and the arm is added once, before
`s5.3`'s arms are scored.

- 2026-08-25 · mirror · Pushed to `github.com/jamesnavinhill/liquid-primus` under `tidepool/`,
  commit `994e9c0`: the whole `s5-tooldata` job (the generator, the corpus-adapted defect
  taxonomy, the split target pools and its config), the amended stop/go gate in `plan.md`, this
  stage's record with the build's verification table, the corrected ledger and the task list. The
  three `s5-sft-l40s` source files, which earlier mirror commits had missed, went with it.

## The guardrail generator failed six of its own gates, and both causes were real (s5.3)

`a42a6c21` ran the tool-data generator end to end in six minutes, wrote no training rows, and
failed six of its nine gates. The gates are the point: a version of this that had quietly
emitted two thousand plausible rows would have gone into the sweep and surfaced as an
unexplained result at `s6`. Two bugs, both in the safety machinery rather than the generation.

**Every row was thrown away as contaminated.** `decontamination: kept 0 of 2166,
{"request_gram_collision": 2166}`. A rule that fires on 100% of candidates is measuring
something other than contamination. It was: the request surface is the system contract plus the
user turns, and the probe set's system preamble was written at `s4` *to match the corpus's own
rendering convention*, so that the probes would sit in distribution. The colliding 13-gram,
recovered by re-running the rule locally against the held-out split, is
`a function calling assistant you are given a set of tools inside tools` — the preamble itself,
carried by 270 of the 434 probe items and by every corpus row that renders a tool call. The
rule read a deliberate match as leakage.

What identifies a probe item is its question, so the request rule now runs over the user turns
alone, and drops any gram carried by more than half the probe items. Measured over the 156
held-out rows that carry a tool return:

| | rows flagged | of 156 |
| --- | ---: | ---: |
| old rule (system + user, any shared gram) | 156 | 100.0% |
| new rule (questions only, boilerplate dropped) | 0 | 0.0% |
| whole-row overlap fraction over 0.30 | 0 | 0.0% |
| carries a value a probe forbids an answer to quote | 13 | 8.3% |

Provenance, because the table matters: those four counts were measured in the orchestration
sandbox against a downloaded copy of the held-out split, as a diagnosis of a failing rule. They
carry no job id and none of them is a citable result. The citable version is `f11afe02`'s own
decontamination counters over the training split, and if they disagree with this table the job
is right. What the local measurement was for was deciding whether to requeue at all.

Zero on the question rule is the correct answer, not a weakened one: no held-out corpus row
asks a probe's hand-written question. The payload and the schema are still covered twice over.
The overlap fraction runs to 0.15 at its worst against a 0.30 limit, so the fraction net has
room to catch a near-duplicate without firing on shared boilerplate. And a new exact check on
the 22 values the probes forbid an answer to quote drops 13 rows that genuinely echo one. The
df filter removed nothing on this data (no gram reaches df 217); it stays as a cheap guard, and
the fix that mattered was dropping the system turn.

**63% of targets were recorded as quoting a value the broken response no longer carries.**
`target_leaked_value: 1371` of 2,166, and the gate failed on any count at all. The check
derives its forbidden list from the intact payload's leaves and tests each as a case-folded
substring of the target text. Two things were wrong with that. Small values are not evidence:
a leaf `0` or `1` or `ok` says nothing about whether a model read the payload, and testing `0`
as a substring of English matches the array index in *"the value at result.0.value"*. And the
test was a substring rather than a token match, so `ok` matched `broken`. The fix filters the
forbidden list to identifying values (four characters of mixed content, or three digits) and
matches on whole token runs under the same normalization the gram hashing uses. The
`silently_truncated` wording no longer quotes the probe fixture's *1 of 47* either, which was
factually wrong about a corpus payload as well as colliding.

The gate itself was also wrong, in a smaller way. Leaking rows are *dropped*, so the written
set is clean whatever the count, and failing the run on any count at all confuses a coincidence
with a defect. It now fails above a 10% rate, which is what a template systematically
interpolating payload values looks like, and a separate assertion checks that no surviving
target quotes a forbidden value. Over all 210 scenario × mode × depth cells of the probe bank,
that assertion holds at zero.

**One more finding, not a bug.** `skipped_wrong_entity: 494` against 83 kept: a response is
only detectably about the wrong entity when it echoes something the call asked for, and 86% of
corpus tool returns echo nothing. Because the mode was assigned by hash *before* the transform
was attempted, those 494 sources were discarded rather than given a different defect. Mode
assignment now walks the rotation from the hashed starting point, so a source that cannot carry
`wrong_entity` carries the next mode that applies. On a 210-source fixture whose payloads echo
no argument at all, every source now gets a mode and six families are represented.

A 23-check fixture over the real probe bank covers all of it: the length rule, the token-run
matcher, the forbidden-list filter, the old and new request rules against a corpus-shaped row,
the rotation, and the zero-leak assertion. Requeued as `f11afe02`.

### What the corpus can actually supply

Worth recording before the sweep is sized. The harvest found 5,734 tool-result turns in
494,341 training rows, of which 4,031 carry a JSON object body with a quotable headline value
and a real assistant reply. At one defective row and one clean counterpart per source, the
ceiling is about 4,000 pairs, not the 24,000 the parameter allows. Two corpora supply them,
not the three that carry tool turns. If 4,000 pairs proves too thin, the next source is the
validation split, which is a decision for the sweep and not for the generator.

## B2 provisioned a machine and ran nothing, and the queue commands under-ran the counts (s5.2)

Reported as running in the previous note; it was not. `efa9719d` came back `COMPLETE` with
`progress: 0`, no artifacts and no logs, after five minutes and six seconds, with
`provider_empty_jobs_polls: 15` in its job record. The cluster started and the remote command
never executed. The run log already carries the same signature twice, on 4-bit build attempts
2 and 3, both on gcp. Relaunched as `c4091ac9`, pinned to aws — the first entry in the provider
note's L4 row (note updated 2026-08-23), on the same `L4:1` card B1 ran on, so the note's
comparison exception does not apply.

Reading that job record turned up a second problem, which would have been the more expensive
one. The queue commands written into `runs/s5.2-baselines.md` for B2, B3, B5 and the bf16
runtime reference passed `-p profile=full` and no limit overrides. In `s5-eval/main.py:327-338`
`profile` is a label written into the summary; the counts come from `limit_per_component` and
the four per-component overrides, and the task default is 40 with every override at `-1`. All
four rows would have run 40 items per component against B1's 6,466 and reported themselves as
full passes. The relaunch passes all five counts explicitly, the recorded commands are
corrected, and the run log now says why in the place someone would copy them from. B1's own
full pass used `-p limit_per_component=0`, so nothing already recorded is affected.

- 2026-08-25 · mirror · Pushed to `github.com/jamesnavinhill/liquid-primus` under `tidepool/`,
  commit `bba6212`: the three fixed generator files and their config, the two check scripts
  (`fixture_test.py`, the 23-check unit fixture, and `decon_check.py`, the local diagnosis that
  is explicit in its own docstring about not being a result), this stage's record of both empty
  runs, the corrected queue commands in `runs/s5.2-baselines.md`, the ledger and the task list.

## Attempt 2 got the hard part right and failed two gates on the easy parts (s5.3)

`f11afe02` wrote 5,202 rows and 2.87M tokens, and cleared seven of the nine gates including the
one the whole guardrail axis rests on: **detector coverage 0.704**, inside the `[0.60, 0.90]`
band. The taught phrasings split as designed against the frozen 26-pattern detector, so a flag
rate measured on this training will be a measurement of the behaviour rather than a recall test
of our own wording. Decontamination behaved: 0 rows dropped for sharing a question gram, 25
dropped for carrying a value a probe forbids an answer to quote. Held-out modes absent, clean
pairing exactly 1:1, no source drawn from a non-train split, three or more undetected families.

Two gates failed and both had a real cause.

**The no-fabrication check discarded 1,405 of 4,031 sources, 34.9%.** Not a template quoting
payload values, which is what the gate was written to catch, and not a coincidence of spelling
either, which is what the previous fix addressed. The mechanism, found by running the rotation
and the target selection over the held-out split and printing what actually collided: **corpus
tool responses routinely echo the called function's own name back inside the payload.** A
`get_stock_price` return carries `"name": "get_stock_price"`. `empty_body` deletes the payload,
so that name lands on the list of values the response no longer carries; and the target names
the tool, because naming the tool is how a reply says which call failed. So the check was
recording the model for quoting a string sitting in its own prompt, in the tool call, three
turns up. `empty_body` survived once in 4,031 attempts. Measured over the 73 held-out sources
that carry a usable tool return, 19 leaked, 12 of them every single `empty_body` case.

The fix follows from the check's own premise: a value the model can read anywhere in its prompt
is not evidence of fabrication, so the forbidden list is scoped to the whole prompt rather than
to the damaged tool body alone. Over the same 73 sources that takes the leak count from 19 to
**0**. And a leak now falls through the mode rotation instead of discarding the source, exactly
as a non-applicable transform already did, so a source whose first defect happens to collide
gets a different defect rather than being thrown away.

**The clean control arm reached 68 items against a floor of 200.** The floor was unreachable,
not missed: the test split holds 11,550 rows, 179 with a tool return, 73 usable, 68 after
decontamination. A gate the data cannot satisfy is a bug in the gate. The arm now draws from
both held-out splits, which are held out the same way and roughly the same size, and the floor
is 100 with the reason stated: the arm exists to hold the plan's 0.15 false-flag ceiling
tighter than the frozen 30-item arm can, and 68 barely does that while roughly 140 does.

Two local diagnostics are checked in beside the generator, both explicit in their own
docstrings that they produce no citable numbers: `decon_check.py`, which measured the
contamination rule, and `leak_check.py`, which found the function-name collision. The fixture
is up to 26 checks and includes the function-name case as a regression. Attempt 3 is `7248507b`.

- 2026-08-25 · mirror · Pushed to `github.com/jamesnavinhill/liquid-primus` under `tidepool/`,
  commit `109c309`: the prompt-scoped forbidden list and the two-split control arm, the third
  local check script (`leak_check.py`), the 26-check fixture, this stage's record of attempt 2,
  the ledger and the task list.

## The guardrail training set is built and clean on all nine gates (s5.3)

`7248507b` finished in 4 m 53 s on CPU and passed every gate. `SCORE` reads
`{"rows": 7988, "train_tokens": 4351790, "detector_coverage": 0.6838, "control_n": 138,
"gate_failures": 0}`. Both files are in shared storage:
`tidepool/s5.3/tooldata/tooldata_train.jsonl.gz` and
`tidepool/s5.3/tooldata/clean_control.jsonl`.

The three defects that killed attempts 1 and 2 are all gone from the counters, and their
absence is the evidence:

| what failed before | attempt 1 | attempt 2 | attempt 3 |
|---|---|---|---|
| rows dropped on a probe-question 13-gram | 2,166 of 2,166 | 0 | **0** |
| targets quoting a value the damaged response lost | 1,371 of 2,166 | 1,405 of 4,031 | **0** |
| clean control arm | not reached | 68, floor 200 | **138**, floor 100 |
| gates failed | 6 of 9 | 2 of 9 | **0 of 9** |

`target_leaked_value` does not appear in the generation counters at all, so the leak rate is
0.0000 against a ceiling of 0.10, and `no_mode_applied` is likewise absent: every one of the
4,031 harvested sources got a defect. The only decontamination drop left is 37 rows carrying a
value some probe forbids, which is the tighter of the two nets doing exactly its job.

**Detector coverage landed at 0.6838**, inside the `[0.60, 0.90]` band and slightly below
attempt 2's 0.704. The number is the fraction of defective targets the frozen 26-pattern
detector fires on; the band exists so the training set teaches both the phrasings the detector
recognises and a third of a corpus it does not, which is what keeps the s6 measurement from
collapsing into a test of the detector's own vocabulary. 2,756 targets came from the
`DETECTED` pool and 1,275 from `PARAPHRASE`.

**One imbalance worth naming for the sweep.** `silently_truncated` carries 1,042 of the 4,031
rows against roughly 580 for each of the other five common modes, because it is the mode that
never skips: the 494 sources whose payload echoes no call argument cannot take `wrong_entity`,
and the rotation hands them to the next mode in order. `wrong_entity` ends up at 91. The
defect kinds are still all present and the two held-out kinds (`null_leaf`, `stale_as_of`) are
absent by construction, so the s6 split between taught and untaught kinds is intact. If the
sweep shows the model over-fitting the truncation phrasing specifically, rotating from a
per-source random offset rather than a fixed mode order is the one-line fix.

The control arm is 138 items, 70 from the validation split and 68 from test, and it carries
the probe item schema (`arm`, `check`, `defect`, `depth`, `id`, `messages`, `mode`, `probe`,
`scenario`) so it drops into the existing probe scorer without a second code path. Wiring
`clean_control_object` into `s5-eval` and giving B1 a supplementary pass over it stays queued
behind s5.2, since changing the harness while four baseline rows are mid-flight would put the
rows on two different versions of it.

**The corpus ceiling held.** 494,341 training rows yielded 5,734 tool turns, 4,031 of them
with a parseable JSON object and a quotable headline value. 3,994 pairs survive
decontamination, giving 7,988 rows and 4.35M tokens. The `s5.3` sweep parameter allows 24,000;
the data supplies about 4,000, and the validation split is the next source if the arms prove
thin.

- 2026-08-25 · mirror · Pushed to `github.com/jamesnavinhill/liquid-primus` under `tidepool/`,
  commit `56280c1`: attempt 3's passing record, the ledger and the task list. The generator
  code itself went up unchanged in `109c309`, since attempt 3 ran exactly the code that commit
  carries.
- 2026-08-25 · s5.2 · A checkpoint marker arrived paired with a decline: no operator is
  answering, and the instruction was to proceed on my own recommendation and record which option
  I took. Both markers are retired. The recommendation I proceeded on is the one below, and every
  judgement in this turn was made unreviewed under autonomous mode.
- 2026-08-25 · s5.2 · **B2 is complete and verified.** The vendor's quantization-aware Q4_0
  build, through the `llama.cpp` binary this stage compiled, finished in 34 minutes 14 seconds
  with 0 assertion failures and all fourteen artifacts listed, including the four scored files
  and both summary forms. It came in 67% under its estimate because the estimate was taken from
  an 8-slot concurrency measurement and a serial pass runs at 776.7 tokens per second. The full
  row, its retention against the reference, and the four readings it supports are in
  `runs/s5.2-baselines.md`. The one number that matters most is the `s4` text convention at
  **78.6% retention**, well under the 93% per-axis floor.
- 2026-08-25 · s5.2 · **Promoted the runtime control ahead of the competitor quant.** B1 ran on
  `transformers` and B2 on `llama.cpp`, so the 78.6% figure could be quantization or it could be
  the serving stack, and nothing downstream about 4-bit export is interpretable until the two are
  separated. The bf16-through-`llama.cpp` row now runs next, ahead of the Nova Q4 row, which
  answers a different question and loses nothing by waiting one slot.
- 2026-08-25 · s5.2 · **The competitor threshold is 0.7641, not 0.7795.** Granite 4.0-1B has
  finished all 3,490 tool-calling items at full count and is part way through structured output.
  The full pass reads 3.9 points kinder than its screening sample, and it still leaves our
  reference 9.4 points behind. The rest of its row lands later today.
- 2026-08-25 · s5.2 · **Retired sweep arm `C2` and replaced it with `C2'`.** `C2` subtracts an
  in-house corpus that `s4.4` measured at zero rows, so it would have reproduced its own reference
  arm exactly. The freed cell tests the guardrail block instead, which is the supervision the
  amended stop/go gate turns on. Recorded in `../s3-research-plan/plan.md`.
- 2026-08-25 · s5.2 · B3 (unsloth UD-Q4_K_XL) queued into the slot B2 freed, pinned to the cloud
  the provider note names first for this card and carrying all five item-count overrides. Both
  GPU slots are occupied, so the remaining rows wait on capacity rather than on a decision.
- 2026-08-25 · s5.3 · **The sweep task is authored, registered and tested.** `s5-sft-sweep`
  (`37186b22`) runs all eight arms from one code path selected by a single `arm` parameter, so
  the specification of the sweep is one readable table rather than eight command lines that can
  drift apart. What it adds over the calibration task: the guardrail rows wired in as a second
  source in the same id space, the 64.0M-token per-arm budget as an explicit parameter the run
  asserts against itself, an entropy-weighted loss, a full-parameter mode, and a replay hook.
  The design record, the per-role token table and the queue order are in `runs/s5.3-sweep.md`.
- 2026-08-25 · s5.3 · Three choices in it are worth naming because each avoids a silent way the
  sweep could have measured the wrong thing. The guardrail block comes **out of** the budget
  rather than on top of it, so `C2'` is cost-matched and the comparison is guardrail supervision
  against an equal weight of ordinary supervision. `C4`'s entropy weights are centred on the
  batch mean, holding the mean weight at 1, so `entropy_beta` cannot act as a hidden
  learning-rate multiplier. And every arm is scored on plain cross-entropy over the held-out
  split, never on its own training objective, because `C4` reweights what it optimizes.
- 2026-08-25 · s5.3 · A CPU fixture test runs in about a second and covers the three ways this
  sweep could be quietly invalid: arms costing different amounts, the guardrail-off arm getting
  guardrail rows anyway, and selection order drifting between arms. **25 checks, all passing.**
  The calibration numbers in it are the real measured corpus sizes, so the token table in the
  design record is the one the arms will actually run.
- 2026-08-25 · s5.3 · Recorded a prerequisite that is not yet met: **the self-distillation replay
  set does not exist.** Its task passed a 64-prompt smoke at `s5.1` and has never been run at
  size. `C5b` needs 3.2M replay tokens, so the generation run is sized at roughly 8,000 prompts.
  Six of the eight arms can run without it, and an arm queued for replay with no replay set now
  fails loudly rather than training the reference recipe under a `C5` label.
- 2026-08-25 · mirror · Pushed to `github.com/jamesnavinhill/liquid-primus` under `tidepool/`,
  commit `b8daccb`: B2's verified row and its retention table, the promoted runtime control, the
  Granite threshold at full item counts, the retired `C2` and its replacement, and the whole
  `s5-sft-sweep` source tree including the fixture test. The paper corpus and the compressed
  splits stay out of the repository, as they have on every previous push.
- 2026-08-25 · s5.2/s5.3 · Wired the enlarged clean control arm into the eval harness. The plan
  sets a ceiling of 0.15 on false alarms over clean tool returns, and the frozen clean arm is 30
  items, which is too few to place a number against that ceiling with any confidence: one extra
  flag moves the rate by 3.3 points. `s5.3` built a 138-item clean corpus alongside the guardrail
  training data, and `s5-eval` now reads it through a new `clean_control_object` setting. The
  harness reports the two arms separately (`false_flag_rate_clean` stays the frozen 30, the new
  `false_flag_rate_clean_corpus` covers the 138) and pools them in `false_flag_rate_all_clean`.
- 2026-08-25 · s5.2/s5.3 · The wiring is **authored and tested but deliberately not applied**,
  because two baseline rows are in flight against the current harness and editing a task under a
  running job is how a table stops being self-consistent. It goes to the server once `s5.2`'s
  rows are down, and `B1` then gets a supplementary pass over the new arm so the reference has a
  number on the same footing as everything measured after it.
- 2026-08-25 · s5.2/s5.3 · The additivity claim is a test, not an assertion. `test_probes_additive.py`
  builds a scored run with and without the new arm and checks that **none of the eight
  pre-existing summary keys move**, that the new keys read empty when the arm is unconfigured, and
  that the envelope-depth breakdown stays scoped to the original arms. That last one was a real
  break rather than a hypothetical: the depth table aggregated over every `tool_return` row, so
  adding 138 clean rows would have silently changed what `depth` meant for the four baseline rows
  already published. The corpus arm now gets its own `depth_clean_corpus` bucket. Nine checks,
  all passing, and the default stays empty so a re-run of an already-measured row is
  byte-comparable to what it produced before.
- 2026-08-25 · mirror · Pushed commit `ed31483` to `github.com/jamesnavinhill/liquid-primus`: the
  eval-harness change, its additivity test, and the updated task list. The clone still held a copy
  of the paper and note corpora from an earlier sync, and those were removed rather than pushed,
  keeping the repository to code and written record as it has been on every push.
- 2026-08-25 · s5.3 · Sized the replay generation run and fixed the thing that would have made
  it useless. The sweep's sampler reads every source through one code path: it keys rows by
  `role`, sizes each role from an `n_tok` field, and opens a file by suffix. The replay writer
  emitted neither `n_tok` nor gzip, so the buffer would have indexed as **zero tokens** and both
  replay arms would have trained the reference recipe under a `C5` label while reporting a
  replay fraction. The writer now counts tokens with the training tokenizer and the training
  template, writes `replay.jsonl.gz`, and places it in shared storage at
  `tidepool/s5.3/replay`, since the arms load replay from storage rather than from a job
  artifact. A run that generates the buffer and cannot place it now fails loudly.
- 2026-08-25 · s5.3 · The run at size is 8,000 prompts, and the hash-rank sample keeps the same
  salt as the `s5.1` smoke, so the set is a strict superset of the 64 prompts already generated
  twice with byte-identical output. On the smoke's 245 completion tokens per prompt the buffer
  lands near 2.0M completion tokens and roughly twice that in training tokens, against `C5b`'s
  3.2M-token dose, which is a little under one pass. The run asserts on that ratio: a buffer
  small enough for `C5b` to replay it more than twice over would measure memorization of a small
  set as much as it measures replay. The card is an L40S rather than the smoke's L4, chosen on
  throughput, and it is pinned to aws per the provider note of 2026-08-23.
- 2026-08-25 · s5.2 · Noted a cost worth naming: the competitor row `B4` is running on an **L4**,
  where the rest of `s5.2` runs on L40S, and Granite 4.0's hybrid layers get no optimized path in
  transformers. It generates at 75 tok/s against B3's 1,743, so a row that takes B3 about an hour
  will take B4 something like seven, and it has held a GPU slot since 14:58Z. Decided to let it
  finish rather than restart it on a faster card: the tool-calling composite it already produced
  is the operative external threshold for the whole project, its remaining three components are
  the axes the project's other goals are measured on, and a restart trades a certain 2.2 spent
  GPU-hours for an uncertain speedup. Recorded as an unreviewed autonomous decision. Any later
  competitor row on a non-Liquid architecture gets pinned to a 48GB card up front.
- 2026-08-25 · mirror · Pushed commit `83e6726`: the replay generator's fix and its sizing, the
  sweep design record's new section on it, and the confirmation that every remaining baseline
  row's weights, quant file and tokenizer resolve before the row is queued.

## Operator input (no checkpoint, second)

A second prompt raised at the end of the previous wake was answered automatically rather than by a
person: _"Full Self Driving: no operator is answering this. Proceed on your own recommendation, and
record in the stage report which option you took and why."_ The prompt had flagged three things and
asked for nothing that only a person could settle, so the recommendation each carried was taken as
written. The 138-item clean control arm stays reported beside the frozen 30-item arm rather than
folded into it, so every published number keeps its meaning. The replay generator's fix stands and
the run is sized at 8,000 prompts. The competitor row on the slow card was left to finish. One
thing was decided this wake that the prompt had not asked about, and it is set out below: the
harness edit was applied earlier than the previous plan said, which changes what the remaining
queue costs. Nobody reviewed any of it.

## B3 is in, and quantization-aware training is worth 5.6 points (s5.2)

`95eb4820` finished clean at 17:20:02Z: four components at full item counts, zero assertion
failures, 1,382,474 generated tokens at 709.2 tok/s, 38 minutes on one L4 against a 1.71-hour
ledger price. The full table is in `runs/s5.2-baselines.md`; the readings are these.

**Quantization-aware training buys 5.6 points of tool calling on this family.** B2, the vendor's
QAD build, scores 0.6392 on the format the family ships with. B3, the best cheap community
post-training quant, scores **0.5837**. Same binary, same card, same tokenizer, same 3,490 items,
same decoding: what differs is that one was quantization-aware while it trained. That is the
cleanest single-variable comparison `s5.2` has produced, and it is direct evidence for the
recovery rung in the `s5.6` export plan. Post-training quantization alone gives up real
tool-calling accuracy here, and something has to buy it back.

**Two independent 4-bit builds agree on the text convention, and the agreement is the reason the
control matters.** B2 retains 78.6% of the reference on the `s4` text form; B3 retains 81.0%.
Different producers, different quantization methods, landing within 2.4 points of each other. The
agreement is equally consistent with a real property of 4 bits on this family and with a property
of the `llama.cpp` runtime they share, which is exactly the ambiguity `B1r` was promoted to
resolve. It is now running as `c3886996`, first row in the queue, same weights unquantized through
the same binary on the same host family.

**The safety axis still has not moved.** B3 flags 0.0074 of 434 malformed tool returns and raises
zero false alarms, both identical to the reference. Three of the four rows measured so far sit at
exactly the same floor. The distance from that floor to the pre-registered stop/go criterion of
0.35 is the whole of what the guardrail supervision built at `s5.3` has to close, and nothing in
the baseline set suggests any of it comes for free.

## The clean-control edit went up early, and it takes three rows off the re-score list (s5.2)

The previous plan was to hold the `clean_control_object` edit until every `s5.2` row was down,
because a row measured against a changed harness is not comparable to one measured before it. B3
finishing changed the arithmetic: only `c319ebb1` was still scored against the old harness, and it
runs on a machine that received its copy of the task at launch. The edit went up at 17:19Z.

The check that made it safe to do at all is the additivity test, re-run against the applied tree:
none of the eight pre-existing summary keys move, the new keys read empty when the arm is not
configured, and the envelope-depth breakdown stays scoped to the original arms. An unconfigured
run is byte-comparable to the row it replaces, so `c319ebb1` is unaffected whatever it reads. The
arm itself was verified in shared storage before the edit was applied — 138 items, every one
tagged `clean_corpus`, which is what the harness's new gate checks — because a mis-tagged arm
would have failed a row after it had already spent its GPU hour.

What it buys: `B1r`, `B5` and `B6` each carry the 138-item arm on their first pass, so the
supplementary re-score list is **B1 through B4** rather than all seven rows. Three GPU passes
saved on a queue where the two slots are the scarce resource, at no cost to anything already
measured. Recorded as an unreviewed autonomous decision.

## B4 is slow, advancing, and being left alone (s5.2)

`c319ebb1` has held a slot since 14:58Z and is 976 of 2,000 through structured output at 77 tok/s,
up from 816 at 17:10Z, so it is moving at roughly 160 items per ten minutes and is not wedged. Two
components remain after this one. The estimate is terminal around 21:00–22:00Z. The decision to
let it finish rather than restart it on a faster card was made at 17:10Z and stands: it has already
produced the tool-calling composite that is the operative external threshold for the project, and a
restart trades certain spent hours for an uncertain speedup.
