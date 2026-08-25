# Stage 5: Experimentation

_Status: in progress. `s5.1` smoke runs are in flight on two providers; nothing has been
trained for real yet._

## Headline

The supervised pipeline is written and two smoke runs are in flight, one on the training
path and one on the replay-generation path, at a few minutes of GPU each. The mix
calibration the plan called for is settled and is a consequence of three stated rules
rather than a hand-picked set of weights: **tool calling takes 45.0% of the epoch's tokens,
structured output 5.0%, SQL 27.4% and code 22.6%**, on a 90.6M-token epoch. Tool calling and
structured output together take exactly the 50% floor the plan set, and structured output
reaches only 5% because the corpus holds 1.5M tokens of it and repetition is capped at three
passes.

## Work log

- 2026-08-25 · s5.1 · Two smoke jobs queued on L4 cards, deliberately on different
  providers: the supervised run `18573ffc` on one and the replay generation `defa6156` on
  another. The GPU cap is two at a time and both want the same card, so splitting them means
  a provider being out of L4 capacity costs one smoke rather than both.

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

Everything before this stage ran on CPU, so 0.0 of the 200 GPU-hour allowance is spent. The
two smoke jobs are the first GPU charge on the project, at well under an hour between them.
The projected cost of a full single-epoch reference run is measured by the smoke rather than
guessed: the supervised job records its own throughput and writes the implied hours for the
90.6M-token epoch into its metrics, which is what the next budget entry will be sized from.
