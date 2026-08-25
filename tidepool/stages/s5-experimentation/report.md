# Stage 5: Experimentation

_Status: `s5.1` complete, `s5.2` in progress. The throughput the whole budget rests on is
measured, the sweep is resized to fit it, and the evaluation harness every later number comes
from is built and verified against ground truth. Its end-to-end smoke is on the GPU now._

## Headline

The one assumption the whole compute budget rested on turned out to be wrong by half. A short
calibration run measured **3,994 training tokens per second on the 48 GB card against the 6,000
the plan assumed**, which makes one pass over the training mix cost 6.3 GPU-hours instead of
4.2. At that rate the planned 8-arm sweep would need 103 GPU-hours where it was given 38, and a
full two-pass version would need 227 against an account that holds 200, so it could not have
run at all. The sweep is therefore resized to a fixed 64 million tokens per arm: all 8 arms
survive, every comparison stays paired at equal exposure, and it fits the approved 145
GPU-hours with nothing to ask for. Both smoke paths are closed on the science first: validation
loss fell 1.32 → 0.28 over 60 steps, the model's own chat template accepted tool-call turns
with no fallback, and the replay path reproduced its 64 completions and 15,677 tokens exactly
across two runs. Attention then moved to `s5.2`, where the one evaluation harness that scores
every model in this project was built and checked before it was allowed to spend GPU time: its
tool-call grader scores a ground-truth oracle at exactly 1.0000 over 858 items and 0.000 on
three deliberate degradations, and a determinism bug in the vendored instruction-following
grader was caught locally by a score that is impossible by construction. Spend so far is 0.71
of 145 GPU-hours.

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
- 2026-08-25 · s5.2 · Harness smoke queued as `5fe7a828` on an L4, at 40 items per component
  over the stock instruct checkpoint. It proves the path rather than measuring the model: the
  chat template accepting tool turns, both surface forms rendering and parsing, every grader
  returning, and the artifacts landing. The card, the serving backend and the decoding settings
  are frozen here and recorded in every summary, because changing them later invalidates every
  full-precision number in the project.

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
extra to install and no batching-dependent numerics to argue about later. A screening pass is
roughly 5 million generated tokens, about half a GPU-hour per model. Moving to a faster
inference server later invalidates every full-precision number taken here and means rerunning
all of them, so the backend, the card and the decoding settings are part of every recorded
result. The 4-bit builds (`B2`, `B3`, `B5`) need a separate `llama.cpp` serving path and are not
scored through this task.

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

## Immediate next actions

1. Read the harness smoke `5fe7a828` against its own assertions: the model's template must
   accept tool turns with no fallback, both surface forms must render and parse, every grader
   must return, and the artifacts must be on the job record. The rates it reports are not
   results at 40 items per component and the summary marks the run as limited.
2. Then the baseline row at full item counts: `B1` the full-precision instruct checkpoint,
   `B4` the 1B-class competitor that carries the absolute threshold, and `B6` the base
   non-instruct checkpoint, all through the same task with the model swapped by parameter.
   `B2`, `B3` and `B5` are 4-bit builds and wait on the `llama.cpp` serving path.
3. Pin the provider in every task that asks for the 48 GB card. The stored calibration task
   carries `resources.compute_provider: aws`, which is where the field validates; at the top
   level it is rejected. Left unset a sweep arm lands on a provider that does not sell the card
   and fails at launch, or worse, on a smaller card whose numbers are not comparable.
4. Carry the 64.0M-token arm budget into the `s5.3` sweep tasks as a parameter, not a
   convention, so no arm can quietly run longer than the one it is compared against.
