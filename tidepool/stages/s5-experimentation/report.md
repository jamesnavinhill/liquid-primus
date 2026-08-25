# Stage 5: Experimentation

_Status: in progress. `s5.1` has its measurements; both smoke jobs are being re-run to
prove one fix. Nothing has been trained for real yet._

## Headline

The supervised smoke run passed every check it was built to make: the model's own chat
template accepted tool-call turns with no fallback, the loss mask landed on assistant tokens
only, LoRA attached 11.1M of 1,181M parameters, and **validation loss fell 1.3223 → 0.4940
over 30 steps at 1,435 tokens per second, which puts the full 90.6M-token reference epoch at
16.1 GPU-hours** rather than at a guess. The replay path produced 64 of 64 completions with
none empty at 204 tok/s, or 33.3 hours per 100k. Both jobs then crashed on their final status
call, a one-line misuse of the experiment harness that lost no measurement and no artifact;
both are being re-run with it fixed, at about 0.1 GPU-hour each, because the ranked dashboard
the sweep depends on reads exactly the field that crash left empty. Spend so far is 0.23 of
145 GPU-hours.

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

The projected reference-run cost is now measured rather than assumed: **16.08 GPU-hours** for
one 90.6M-token epoch at the observed 1,435 tok/s on an L4. The plan's supervised sweep budget
of 38 GPU-hours therefore covers roughly two full-epoch arms on this card, which is a real
constraint for `s5.3` and is taken up there rather than here: either the sweep arms run
sub-epoch, or they run on a faster card, and the choice is the operator's at the direction
decision.
