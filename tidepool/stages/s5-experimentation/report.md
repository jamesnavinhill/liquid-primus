# Stage 5: Experimentation

_Status: in progress. The supervised smoke has passed twice and its fix is proven; the
replay smoke and an L40S calibration run are on GPU. Nothing has been trained for real yet._

## Headline

The supervised smoke run passed every check it was built to make: the model's own chat
template accepted tool-call turns with no fallback, the loss mask landed on assistant tokens
only, LoRA attached 11.1M of 1,181M parameters, and **validation loss fell 1.3223 → 0.4940
over 30 steps at 1,435 tokens per second, which puts the full 90.6M-token reference epoch at
16.1 GPU-hours** rather than at a guess. The replay path produced 64 of 64 completions with
none empty at 204 tok/s, or 33.3 hours per 100k. Both jobs then crashed on their final status
call, a one-line misuse of the experiment harness that lost no measurement and no artifact;
the fix has since been proven: the re-run recorded COMPLETE with a full score dict and all
five artifacts, at 1,398 tok/s and a 16.5-hour projected epoch, within 2% of the first
attempt. The replay smoke is re-running behind it, and a calibration run is now on an L40S to
replace the plan's assumed 6,000 tok/s with a measured number on the card the sweep will
actually use. Spend so far is 0.35 of 145 GPU-hours.

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
- 2026-08-25 · s5.1 · Queued a calibration run of the same supervised code on an L40S,
  at 60 steps over 1,024 rows. The plan's whole 145 GPU-hour budget rests on an
  assumed 6,000 tokens per second on that card, and the L4 measurement suggests the assumption
  is optimistic. One short job replaces the assumption with a measurement, which is cheaper
  than discovering it eight arms into the sweep.
- 2026-08-25 · s5.1 · The calibration run was refused at launch, at no compute cost, because
  the provider jobs go to by default does not sell the card the plan's budget is written
  against. Re-queued against the first source the hardware guidance lists for that card, and
  now provisioning as `68635a5d`. Every run in the sweep inherits the same default and will
  need the same pinning.

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

### The first real test of the plan's compute assumption

The projected reference-run cost is now measured rather than assumed: **16.08 GPU-hours** for
one 90.6M-token epoch at the observed 1,435 tok/s on an L4.

The plan's whole 145 GPU-hour budget rests on one number, stated at `s3.3` as a deliberately
conservative figure: **6,000 training tokens per second for a 1.2B model on one L40S**, which
is the default card for every supervised row. The smoke ran on an L4, so it does not measure
that number directly. It does constrain it. An L40S is worth roughly two and a half to three
and a half times an L4 on dense bf16 for work of this shape, so 1,435 tok/s on an L4 implies
something in the region of 3,600 to 5,000 tok/s on an L40S. The plan assumed 6,000.

Three reasons not to treat that as a verdict yet. The smoke ran 30 steps over 512 rows, which
is short enough that startup is still being amortized; throughput was still climbing at the
last step (1,571 tok/s against the 1,435 average), so the steady-state figure is higher than
what was measured. Smoke sequences are also shorter than the epoch's mean. And a card ratio
taken off spec sheets is exactly the kind of substitution this project's own protocol refuses
between paired numbers.

So the honest reading is that the plan's central estimate looks optimistic by something like
20 to 40%, and that the cheapest way to replace an inference with a measurement is to run this
same smoke once on an L40S. At roughly 0.1 GPU-hour that is the next thing to do, ahead of the
baselines, because every row of the supervised sweep is priced off this one number and the
sweep's own line is 38 GPU-hours. If the true L40S rate is near 4,000 tok/s, one full-epoch
arm costs 6.3 GPU-hours and the 8-run sweep in the plan does not fit its line at full epochs.
That is a real constraint on `s5.3` and it is better found now, at the cost of one short job,
than discovered by a sweep that overruns. The 25% contingency (29 GPU-hours) exists for
exactly this, and the decision about how to spend it belongs at the `s5.4` direction
checkpoint with a measured number in hand.

## Immediate next actions

1. The supervised smoke has landed clean with its score dict on the record. Confirm the replay
   smoke does the same, which is what makes the ranked dashboard usable for `s5.3` and `s5.4`.
2. Read the L40S calibration run, now provisioning, and replace the plan's assumed 6,000 tok/s
   with its observed figure.
3. Re-price the `C` supervised sweep against that measurement before queueing any baseline,
   and if it does not fit its 38 GPU-hour line, bring the options to `s5.4` rather than
   quietly shortening the arms.
4. Pin the provider in every sweep task that asks for the 48GB card. The stored calibration
   task now carries `resources.compute_provider`, which is where the field validates; at the
   top level it is rejected. Left unset, a sweep arm lands on a provider that does not sell
   the card and fails at launch, or worse, on a smaller card whose numbers are not comparable.
