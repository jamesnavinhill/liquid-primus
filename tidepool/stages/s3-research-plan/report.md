# Stage 3: Research plan

_Status: complete._

## Headline

The plan is five falsifiable hypotheses and a 32-row experiment matrix that fits inside the
200 GPU-hour allowance with room to spare: 116 hours of planned work plus 25% contingency,
145 in total, leaving 55 unspent as margin. Most of the budget goes where the literature says
the win is, an eight-run supervised sweep at 38 hours, and the riskiest line, a 30-hour
reinforcement-learning arm, carries three hard kill lines that end it early and return its
budget rather than spending it on a collapse. Doing the same study with three seeds per arm,
a 2.6-billion-parameter transfer, and a full-size reinforcement-learning treatment would take
roughly 370 hours and needs a larger allowance, so none of that starts without one.

## Work log

- 2026-08-25 · s3.1 · Five hypotheses (H1–H5) with why-it-might-work, why-it-might-not, the
  metric each moves and the condition that falsifies it → `plan.md`.
- 2026-08-25 · s3.2 · Experiment matrix: 6 baseline rows, 8 supervised sweep rows, 4
  preference rows, 3 merging rows, 3 reinforcement-learning rows, 4 export/recovery rows, and
  the final evaluation set, each with the metric it moves and explicit stop/go gates
  → `plan.md`.
- 2026-08-25 · s3.3 · Allowance read (200 GPU-h, 2 GPUs at a time), hardware chosen against
  the team's provider steering note, matrix costed at 116 GPU-h + 25% contingency = 145
  → `plan.md` and `../../budget.json`.
- 2026-08-25 · s3.4 · Plan and compute cap approved under autonomous mode; recorded below.

## The plan in one page

The scope says: take `LFM2.5-1.2B-Instruct`, make it reliably call tools and emit valid
structured output for the operator's own stack, and ship it as a 4-bit GGUF under 1.5 GB that
holds the quality of its full-precision parent. Stage 2 read 212 papers against that and left
a clear ordering of bets.

**The supervised rung carries the plan.** The strongest precedent in the corpus takes a
1-billion-parameter model from 9.6% to 88.9% schema accuracy with a LoRA fine-tune and no
constrained decoding, so eight of the matrix's training runs are supervised: a reference arm,
a public-data-only arm that establishes the honest floor for the "specialized to our stack"
claim, a full-parameter arm, an entropy-weighted-loss arm, a three-way replay ablation, and a
rank check. A gate sits at the end of it: at least 1.5 points of BFCLv3 gain with no guardrail
worse than −3.0, or hypothesis one is falsified and the project reports that instead of
spending on the next rung.

**Flagging a broken tool return is treated as a preference-rung behaviour.** Four preference
objectives are tried, all of them variants rather than textbook DPO, because textbook DPO is
proven to push down the probability of the response it is supposed to prefer and measures as
the weakest offline objective at this scale. The pairs are two-sided: honest abstention as the
chosen response, against both a fabricated call and a needless refusal as rejected, which is
the construction that cut fabricated-call rate from 90.2% to 55.8% in the closest study.

**Reinforcement learning is an arm, not a plank.** The two studies nearest this model size
report outright collapse, so this arm runs only from a supervised warm start, with a
decomposed graded reward, and under three health signals checked every 25 steps: policy
entropy, the fraction of sampled actions that are valid, and how often the model repeats
itself. Two consecutive breaches ends the arm and its remaining hours go to quantization
recovery.

**Compression is last, and recovery is budgeted.** All quality training happens in full
precision; the 4-bit export follows; then post-training quantization as a warm start,
distillation from this project's own full-precision checkpoint, and an error-targeted
preference pass of a few hundred examples, which recovered a collapsed model in minutes in
one published run. Two GGUF formats and two quantizer group sizes are compared, since format
choice moves throughput more than this project's own efficiency tolerance does.

**Where to look while it runs.** Every row is a Transformer Lab job with an id, so job status,
task logs and machine logs are openable per run; the sweep uses the task's own sweep block
rather than parallel one-off jobs, so a sweep appears as one queryable group. Weights and
Biases credentials are on file and training runs log to it, which is the dashboard to watch
for the loss curves and the reinforcement-learning health signals. Artifacts are mirrored to
the operator's own GitHub and Hugging Face repositories as they land.

## Hardware and the steering note

The team's provider steering note (`updated_at` 2026-08-23T22:48:37) shaped the hardware
choice and is followed rather than second-guessed. Its selection rule is memory first then
throughput, its warning is that one allowed card lacks bf16 and must not be used for training,
and it flags that the 40 GB high-bandwidth card is a throughput pick rather than a capacity
upgrade. Combining that with the enforced type list gives: 48 GB cards as the default for
training (three independent sources, comfortable for full-parameter fine-tuning of a 1.2B
model), the 40 GB high-bandwidth card for the two bandwidth-bound row groups, 24 GB cards for
evaluation-only rows, and no training on the card without bf16. One project-specific rule
overrides card convenience: both sides of any paired comparison are measured on the same card,
and a mid-study substitution triggers re-measurement of the earlier side rather than a
cross-hardware comparison. The note prescribes that same discipline for paired numbers.

## Sizing, and what a larger version costs

The operator's standing preference is to size the first sweep to the compute that is actually
available and to price the bigger version before spending past what is approved. The enforced
allowance is 200 GPU-hours with two GPUs at a time. The matrix costs 116 GPU-hours; with 25%
contingency the plan is 145, leaving 55 unspent, held specifically for the one thing most
likely to need it, which is re-measuring a baseline after a card substitution.

Priced up front, three larger versions: three seeds on every sweep arm adds ~100 GPU-hours; a
transfer of the winning recipe to the 2.6-billion-parameter tier adds ~45; a full-size
reinforcement-learning treatment rather than a kill-line-guarded arm adds ~80. All three
together is ~370 GPU-hours against a 200-hour allowance, so each needs a quota raise and none
begins without one.

## Plan and compute approval (s3.4)

Autonomous mode is on, so this input checkpoint was decided here rather than escalated.

**Decision: approve the plan at the s3.3 estimate of 145 GPU-hours, unchanged.**

**Reasoning.** The estimate sits inside the enforced allowance with 55 GPU-hours of margin, so
approving it commits nothing the platform would refuse and asks for nothing past what is
already granted. It is traceable to the matrix rather than invented: every phase line is a
row count times a per-row cost, with the per-row cost derived from a deliberately conservative
throughput assumption for this architecture. The largest single line item is also the most
speculative one, and it is the only line with hard kill conditions attached, which means the
realistic spend is lower than the plan rather than higher. And the three ways to make the
study bigger are priced and named, so a decision to spend past this plan can be made on
numbers later instead of discovered mid-sweep.

**What would have justified escalating instead.** Two things: an estimate that exceeded the
enforced allowance, which would be spending past approval, or a matrix that could not be made
to fit, which would be a rescope. Neither applies. If a mid-study overrun does push the total
past 145 GPU-hours, that becomes an escalation at the moment it is foreseen, not after.

No operator reviewed this decision.

## Outputs

- `plan.md` — hypotheses, the full experiment matrix with stop/go gates, protocol freeze,
  hardware choice, budget derivation, and the priced larger versions.
- `../../budget.json` — planned ledger, 145 GPU-hours, empty entries.

## Next steps

Stage 4 prepares data. Two items carry in as open and both are access questions rather than
research questions: whether the in-house stack corpus can be reached, and whether the gated
public tool-calling set is reachable with the credentials on file. Whichever public set is
obtained gets a mandatory repair-and-validate pass first, since two papers document schema
mismatches and a parser-breaking whitespace defect in the most likely candidate.
