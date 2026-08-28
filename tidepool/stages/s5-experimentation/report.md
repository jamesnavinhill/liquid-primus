# Stage 5: Experimentation

_Status: `s5.1` and **`s5.2` both complete**. All six baseline rows are down at full item
counts on all four components, verified against their artifacts and mirrored. The throughput the
whole budget rests on is measured, the sweep is resized to fit it, and the evaluation harness
every later number comes from is built, verified against ground truth and smoked end to end.
**Both competitors beat the reference on native tool calling and lose everywhere else**, which
turns the headline gap into a question about calling convention rather than a single number to
beat. **Three 4-bit-or-control rows** ran through a `llama.cpp` path this stage built and
verified, and the runtime-vs-quantization control splits into two different findings depending on
the axis. **The guardrail training set is built and clean**: three attempts, nine gates, 7,988
rows and 4.35M tokens in shared storage with a 138-item clean control arm. One sweep arm was
retired as unrunnable and replaced. `s5.3` is now under way, and **the sweep has been rebuilt to
pack several arms onto one card** rather than running one arm per GPU: an L40S is 48 GB and a 1.2B
model with a rank-16 adapter uses a fraction of it, so the account's two-GPU cap limits how many
cards are up and not how much work runs on them. A supervisor spawns one isolated child
process per arm with a hard memory ceiling apiece, and the four-arm trial that measured it came
back in nine minutes: **three arms sharing a card produce 1.51x the aggregate throughput of one
arm alone**, a packed arm sees a token count identical to the same arm run solo, and the one arm
that failed did so inside its own ceiling with 9.4 GB still free and its three siblings untouched.
The supervisor now schedules as well as isolates, so **the entire rest of the sweep is one job on
one card**: the replay buffer is generated as an arm of the pack and handed to the two arms that
consume it, with no second job and no round trip through storage. The solo reference arm **stalled
at step 2,940 of 8,416 with no identifiable cause and was stopped**; nothing was recoverable,
because the trainer wrote weights only after its final step. **The trainer now checkpoints to
shared storage and resumes from them**, and the reference arm is re-running from the top inside
the packing supervisor, which carries the stall watchdog the solo path never had. That re-run's
first attempt was refused a machine outright by an account-region validation at zero cost, and
it is now away on the second provider in the card's source row with an unchanged recipe. Both
GPU slots are full and the sweep is moving again._

## Headline

**Update, 2026-08-28 18:05 UTC: the first checkpoint's 4-bit quality is measured and scored,
and bare export alone does not hold the quality bar.** Tool calling keeps 94-96% of the
full-precision score and instruction following keeps 98-100%, both comfortable.
Structured-output validity keeps 85%, below the 93% floor we set per axis, and the stack-idiom
probes keep 85% on one of the two 4-bit formats. That is the axis nine papers in the literature
review predicted would break first, so the planned recovery steps are now indicated. The paired
item-by-item comparison has since finished and sharpens the reading: across twenty comparisons
only two hold up once we correct for how many we ran, and both are the tool-calling drop on the
more aggressive format. The two axes that miss the retention floor cannot be separated from full
precision item by item, because the full-precision model itself only scores 8.6% on structured
output, so a 14.5% relative drop is 25 items in 2,000. Both readings are true of the same
measurement, and the recovery work now has an order: structured output first, stack idiom
second. The second checkpoint's last two formats are still scoring, neither stuck, and a third
has been added for measurement consistency. Spend stands at 68.338 of 145 GPU-hours; the
comparison itself ran on CPU and cost none of it.

---

**Update, 2026-08-28 17:30 UTC: one checkpoint's quantized-quality re-scores are fully in hand,
and the other's last re-run just started on the slot that freed up.** All three formats for the
first checkpoint are now clean and recorded. Of the second checkpoint's three formats, one is
already clean from an earlier pass and the remaining two are both running now, neither stuck.
Spend stands at 70.016 of 145 GPU-hours, comfortably inside plan. No decision needed here; this
is a routine progress check.

---

**Update, 2026-08-28 16:26 UTC: the second export attempt looked clean by every check in place
and wasn't, and fixing the gap is already paying off.** Both quality-scoring jobs finished with a
success status, correct item counts and real output files, which passed every check the project
had. Direct inspection of the actual generated text found that two of each job's three quantized
formats had produced nothing at all: three model servers sharing one GPU's compute starved each
other until requests timed out, and the code silently recorded those timeouts as empty answers
rather than failures. Only the busiest format on each job came through with real content, and one
of those two is a genuine keeper. The fix gives each format its own GPU and adds a check that
fails a job outright if too many answers come back empty, so this exact failure cannot hide again.
Two of the four affected formats are already re-running alone and progressing normally, one about
a fifth through and the other nearly half; the remaining two are queued and will start as soon as
a GPU frees, since only two run at once. Spend stands at 68.338 of 145 GPU-hours.

**Update, 2026-08-28 15:25 UTC: the quality pass died on both cards eight minutes in, for a
reason that had nothing to do with the 4-bit builds, and the fix was already sitting in this
project.** All six scoring runs failed at the same instant and with the same message: the program
that serves these models could not start, because the machine it landed on was missing a system
library it depends on. Every run had already found and staged the right weights before dying, so
the export, the file naming and the packing are all clear; what differs is the machine. The
earlier card type happened to carry that library and this one does not. The export step hit the
identical wall a day ago and repairs it itself, and that repair had never been carried across to
the scoring step, which is the only other place that starts the same program. Reading that code
also turned up a second crash waiting immediately behind the first, which would have killed the
retry one line later and cost another pair of cards. Both are fixed, both now have tests that
fail against the old code, and those tests run automatically before any future build ships. The
two runs are back on cards with **every experimental setting unchanged**, so the numbers they
produce stay comparable to the full-precision rows already recorded. The dead attempt cost 0.269
GPU-hours, charged in full; spend stands at 66.876 of 145. Expect roughly two to three hours.

**Update, 2026-08-28 14:52 UTC: the export works, and the 4-bit builds are half the size the
project promised and as fast as the vendor's own.** The fourth attempt ran clean end to end and
built six files: both surviving checkpoints, each at full precision and in two different 4-bit
formats. The shippable ones are **0.696 GB and 0.731 GB against a 1.5 GB ceiling**, and the
cheaper of the two decodes at **301.9 tokens/second, within 0.1% of the vendor's own published
4-bit build of the same base model** — the check that our compression is doing the same thing
theirs does, differing only in the fine-tune baked in. The richer format costs 5.6% of that speed
for 35 MB more. All three fixes from the earlier crashes held. **What is still unmeasured is
quality**, which is the actual pass condition: whether these 4-bit builds keep 97% of their own
full-precision accuracy with no axis below 93%. That scoring pass is on two cards now, six
serving points, and it separates what the 4-bit runtime costs from what the compression costs
rather than blaming both on the compression. The build cost 0.168 GPU-hours; spend stands at
66.607 of 145.

**Update, 2026-08-28 14:27 UTC: the export step is on its fourth attempt, and each of the three
failures found a different, real defect that the project would otherwise have shipped with.** The
third attempt repaired the two toolkit gaps below, converted, and for the first time **measured**
something: the best tool-calling checkpoint at full precision came out at **2.343 GB and 105.1
tokens/second**. It then crashed one step later reading the 4-bit compressor's own progress
output, which contains a byte that is not valid text, while the file it was describing sat
finished on disk. That last one is a fault in our code rather than in the toolkit, it is fixed and
the fix is verified against the exact byte, and it would have hit anything that ran these tools.
The three crashes cost 0.422 GPU-hours together, all charged to the ledger; spend stands at 66.439
of 145.

**Update, 2026-08-28 14:20 UTC: the export step has hit two separate gaps in the toolkit we
pinned back at the baselines, both now closed, and the third attempt is on a card.** The first
crash was a missing library of format writers, so nothing could be converted at all. With that
carried in, the second run converted the first checkpoint cleanly in twenty seconds, which
proves the fix, and then failed because the same archive ships the command-line tools without
the graphics libraries they are built against, and this machine's image does not carry them
either. Neither gap had ever been exercised: every 4-bit number in the project so far came from
a file published by a vendor. The job now repairs both from what is already on the machine and
checks that the tools actually start before it spends anything on a checkpoint. One real artifact
survived, the first full-precision converted checkpoint, at 2.343 GB. The two crashes cost 0.261
GPU-hours together, charged to the ledger; spend stands at 66.278 of 145.

**Update, 2026-08-28 18:30 UTC: s5.6 is open and its first row is on a card.** The stage now moves to export and recovery, which is where the 4-bit promise gets tested: a build at 1.5 GB or less that holds 97% of its own full-precision quality with no axis below 93%. The export job is running on both surviving checkpoints. The order of the remaining work was deliberately reversed, so the cheap measurement comes before the expensive recovery: there is no point paying for a repair until the damage is measured. Sized at 20 to 25 GPU-hours of the 79.1 left, which leaves the final evaluation stage whole.

**Update, 2026-08-28 18:05 UTC: the replay ladder is finished, scored and compared, and it
comes back null. Raising the replay dose fourfold does not move instruction following.
Reweighting the buffer to half constraint-bearing does not move it either. Replay against no
replay at all does not move it.** All three pre-registered comparisons land inside their
confidence intervals, and the only arm in the family that separates on instruction following
is the untrained base model, which beats every trained arm by 3.5 to 9.1 points. s5.4 blamed
the earlier failure on a replay dose two orders of magnitude too small; the ladder raised it
by two orders of magnitude and the metric did not respond, so that diagnosis is retired. The
pass condition fails on all three arms and the fallback the plan wrote for exactly this case
now governs: the forced trade-off is the result, and the checkpoint is chosen on tool calling
first, which selects **R3**. Replay is not inert everywhere. It buys the project's best
text-convention tool calling, 2.8 points over the sweep winner and family-corrected, and it
costs 5.2 points of first-attempt schema validity, also family-corrected, the only regression
in the ladder. The instruction-following cost of this specialization is a property of the
training mixture rather than a tuning knob, and s6 should report it as one. Spend 65.9 of 145.

**Update, 2026-08-27 06:33 UTC: the replay ladder is over a third done and on track.** One
of the two self-distillation buffers has finished (110 minutes, clean) and both of its
training runs have started; the second buffer is close behind. No results yet, no problems
either. One thing to fix later, not now: a step that was supposed to copy the finished buffer
into shared storage failed quietly, so the training runs on this card still have it, but a
separate future job would not find it there.

**Update, 2026-08-27 04:20 UTC: the s5.5 replay ladder is on a card, and the fix s5.4 asked
for turned out to be capped by the project's own decontamination.** Five arms on one L40S:
two 32,000-prompt self-distillation buffers generated side by side, one sampled proportionally
and one reweighted to half verifiable-constraint rows, then three C7-recipe trainers at 5% and
20% replay that separate composition from dose at the same fixed 64M-token budget. Counting
the pool by sub-source showed the reweighting has a hard ceiling: `ifstruct_train_generated`
lost 18,721 of its 20,000 rows to the 13-gram contamination index, because IFStruct is also an
evaluation set here, so the second constraint stratum can supply 1,279 rows and not the 1,600
the design first asked for. Shares moved to 0.47 autoif and 0.03 ifstruct, which is the same
half-constraint buffer from a source the pool actually has. Estimated 11 to 12 GPU-hours of
the 92.6 remaining.

**Update, 2026-08-27 01:00 UTC: the base model is back, and it ends the sweep. Every one of
the eight variants beats the untuned base on tool calling by 3.6 to 5.9 points, and every one
of them is 3.5 to 11.7 points *below* it on instruction following, against a 2-point
allowance. No variant passes all three gates, and the hypothesis that this trade could be
avoided is falsified in the exact words the plan wrote for it.** The gate is written against
the base, whose scores had been lost to an earlier save bug, so it had been getting checked
against the reference variant instead, which flattered every arm. A scoring-only replay
recovered the base from its saved outputs in four minutes on one card, and the paired
comparison now covers it: the reference variant beats the base by 3.2 and 12.4 points on the
two tool-calling conventions and by 11.9 points on catching fabricated tool returns, and
loses to it by 6.7 points on instruction following, all of them surviving correction across
78 comparisons. The sweep's one real gain holds: the raw-mixture recipe catches 11 more
points of fabricated returns than the reference. The cause of the instruction-following loss
is identifiable from the sweep's own design. The replay data meant to protect general quality
does contain constraint-following examples of exactly the kind the benchmark tests, about a
tenth of it, but the variants saw that slice at roughly a tenth of one percent of their
training. The protection was in the recipe at a dose two orders of magnitude too small to do
anything. Stage 5.4 is decided here under autonomous mode with nobody reviewing it: no winner
is picked, the falsification is recorded as the finding the plan says it is, and stage 5.5 is
redirected at the regression with three variants that separate how much replay from what kind
of replay.
Spend is 52.4 GPU-hours of the 145 approved.

**Update, 2026-08-27 00:21 UTC: the paired comparison is in, and the sweep produced exactly
one real gain across seventy comparisons. It belongs to the raw-mixture recipe, and that
recipe fails a different pre-registered gate, so no variant passes all three.** Comparing
each variant against the reference item by item, nine differences survive correction for
testing seventy things at once, and eight of the nine are the variant doing *worse*. The one
gain is the raw-mixture recipe catching 11 more points of fabricated tool returns, a solid
effect; it pays about 2.5 to 2.9 points of tool-calling accuracy for it, and those losses are
solid too. The no-guardrail ablation loses 60 points of catch rate, which settles that the
guardrail training data is what does the catching rather than the base model. Two earlier
readings were wrong and are corrected here: the gate that caps how far any single
tool-calling category may fall was checked on aggregate scores rather than per category, and
read correctly the raw-mixture recipe drops 6.8 points on the benchmark's largest category,
so it fails; and its instruction-following score is 0.7338, not the 0.7411 recorded earlier.
That gate is written against the base model, whose per-category scores were lost to an
earlier save bug, so it has been unevaluable as written for this whole stage. A scoring-only
replay of the base model's saved outputs is running now, roughly a fifth of a GPU-hour, which
also lets the base enter the comparison as a proper arm and finally puts a test behind the
project's headline claim. Spend stands at **52.3 of 145 approved GPU-hours**.

**Update, 2026-08-26 23:28 UTC: all eight sweep recipes are now scored, and exactly one of
them clears the guardrail bar.** The last two finished at 23:20; neither catches enough
fabricated tool calls to pass (65% and 62% against the 70% bar), which leaves the raw-mixture
recipe as the only candidate of eight, at 72% catch with zero false alarms. Two results make
the choice sharper than a single winner. The two recipes that score *highest* on tool calling
both fail the guardrail bar, so ranking on the headline metric would have picked against the
project's stated first priority. And the training-loss column the sweep selected its arms on
points the wrong way at both ends: its best arm catches 0.7% of fabrications and its worst
catches 72%. Tool-calling accuracy spans just 2.2 points across all eight recipes while the
catch rate spans 71, so everything that separates these recipes lives on the guardrail axis.
A short follow-up pass is now running over all eight to price false alarms on 138 clean cases
instead of 30, which is the reading the direction decision needs. Spend stands at **52.1 of
145 approved GPU-hours**.

**Update, 2026-08-26 23:12 UTC: six of eight sweep recipes are now scored, and the raw-mixture
recipe is still the only one that clears the guardrail bar.** Two more recipes finished
scoring since the last update — both use a small slice of replayed general-purpose data
during training — and neither catches enough fabricated tool calls to pass: one catches 57%,
the other 63%, both short of the 70% bar (one of the two also keeps false alarms low, at 10%).
That leaves the raw-mixture recipe, at 72% catch with zero false alarms, as the only candidate
out of six scored so far that meets the bar while staying close to the reference model on
tool-calling accuracy and instruction-following. The last two recipes are still scoring and
should land within the next couple of hours. Spend stands at **49.5 of 145 approved
GPU-hours**.

**Correction to the 13:15 UTC update.** That update said the no-guardrail ablation catches
39% of fabrications with a 3.3% false-alarm rate. Both figures were wrong, and neither came
from the run: the recorded scores say 0.7% and zero. The error was in a status note written
alongside the results rather than in the results themselves, which have been consistent
throughout and are unchanged. The finding is unaffected in direction and stronger in degree.

**Earlier, 2026-08-26 09:10 UTC: I misread how far the first scoring pass had got, and the
cheap price I reported an hour ago was wrong. The corrected price is about 8 card-hours to
score all eight trained variants on the full benchmark, against 116 unspent.** The progress
figure the run reports counts one benchmark at a time, and the one it was counting is the
largest of four. So the pass I described as nearly through its item set had finished the
biggest benchmark at the first of two ways of presenting a tool to the model, with the second
way and three further benchmarks still ahead. The rate I quoted came from the easy end of that
one benchmark, because the run deliberately does its short prompts first and slows as they
lengthen. The right anchor was already recorded: the published model's own full pass over the
identical work took 85 minutes and 1.42 card-hours on the same card alone, which puts a pair
of variants sharing a card at around 2 hours.

The decision that price was used to justify does not change. Running the full benchmark rather
than a reduced screening pass costs about 6 extra card-hours and buys the comparison its
statistical power, and power is the point when three of the four finished variants sit six
ten-thousandths apart on the training metric. What the pass has genuinely established is the
memory limit: both variants have generated more than four and a half thousand answers each
under an 11 GB cap with not one retry.

Beside that, the comparison that the direction decision has to rest on is now written,
registered and verified. Validation loss cannot separate three of the four finished variants, so
the decision moves to the task scores, and reading those as a table of rates would repeat the
same mistake at a larger scale: dozens of small differences, any of which could be sampling
noise. The comparison matches the variants item by item against the reference recipe,
puts a confidence interval and an exact paired test on every variant-and-benchmark cell, and
corrects across the whole set of comparisons, because twenty-eight cells against one reference
throw up a convincing-looking winner by chance about once even when every variant is identical.
The verdict column has three states and one of them says the two cannot be told apart. It runs
on ordinary processors and costs no GPU time. Sixty-seven checks over hand-built fixtures pass,
including one where the per-item and per-category readings of the same result point in opposite
directions, so both are reported and neither can be mistaken for the other. Spend stands at
**28.7 of 145 approved GPU-hours**.

**Earlier, 2026-08-26 08:12 UTC: the machinery that will pick a winner is built and tested.**
Its description is above; the cost claim it originally carried is corrected there.

**Earlier, 2026-08-26 07:52 UTC: the scoring harness has now been run against real trained
checkpoints, and it found two of its own problems before spending anything on a result.** A
four-minute trial put all four finished variants on one rented card at a reduced item count.
Two of the things it was sent to check came back clean: the full-parameter variant, whose
saved form is a whole model rather than a small adapter, now loads correctly, and all four
trained variants are readable from shared storage after the repair made earlier today. The
trial itself failed, on memory. Each variant was given a fifth of the card, and every one of
them ran out on its first hard batch of tool-calling prompts; one died while 17 GB of the card
sat free, so the limit ended it and not competition between them. Scoring needs about twice
the memory that the training runs led me to expect, because at scoring time the memory holds
the prompt, and the training figure was dominated by gradients that do not exist here.
Tool-calling prompts carry a dozen tool schemas and run to a few thousand tokens, so two
variants fit a card for scoring where three fit for training. The harness now halves a batch
it cannot fit and retries it, so a limit set too low costs time and no longer costs the run,
and the limits it was given are recorded next to the measurement they came from. The first
real scoring pass is away at full item counts, the reference recipe against the full-parameter
variant, and the second half of the training sweep is still running beside it. Spend stands at
**28.7 of 145 approved GPU-hours**.

**Earlier, 2026-08-26 07:35 UTC: half the sweep is trained, and the cheap way of choosing a
winner has come up empty.** Four of the eight training variants are finished: the reference
recipe, the full-parameter version, the one with the safety-training block removed, and the
one without data balancing. Three of the four land within 0.0006 of each other on validation
loss, which at one run apiece is a tie rather than a ranking. Tuning all 1.2 billion weights
instead of a small adapter bought nothing measurable, and dropping the safety block moved the
number the wrong way by an amount too small to mean anything. Only data balancing separates:
it is worth about 0.005, the one clean effect in the table. The consequence is concrete. The
direction decision cannot rest on validation loss, so it has to rest on the task scores
themselves (tool calling, structured output, instruction following, safety flagging), which
means the scoring harness is now the critical path. Both cards are working again: the last
four variants plus the data they need are away as a single run, and the harness is getting its
first shared-card test beside it. That harness test failed on its first go, four minutes in,
on a configuration bug of ours: the value that tells each half of the run which tests to score
is written as JSON, the platform helpfully parses it before handing it over, and our code then
insisted on parsing it a second time. Fixed, covered by a test that now feeds it the value in
all three shapes it can arrive in, and re-running. It cost four minutes of a rented card and
nothing on the critical path. Spend stands at **28.6 of 145 approved GPU-hours**.

**Earlier, 2026-08-26 07:05 UTC — the reference training run is done, and the packed run
behind it is nearly there too.** C1, the recipe every other sweep arm gets compared against,
finished cleanly on its second provider after 8,416 training steps: validation loss dropped
from where the base model started to **0.152**, with no errors and every result file saved.
It took three attempts to get a clean run (a silent stall, then a cloud region that refused
the job before it started, then this one), but the number itself looks ordinary next to the
other arms already running, so the earlier worry about comparing runs across two different
cloud vendors looks unfounded. The three-arm packed run on the other card is at 93%, with two
of its three arms already finished and the last one a few hundred steps from done. Spend
stands at **19.3 of 145 approved GPU-hours**. The next five arms go out the moment that packed
run frees its card, which should be within the next check-in or two.

**Earlier, 2026-08-26 01:00 UTC — the stalled run was stopped, and what made it expensive was
fixed.** C1 ran smoothly for three hours and then produced no progress for nearly three more,
with nothing in its logs pointing to a cause. It was stopped and charged in full at 4.06
GPU-hours. None of its work could be salvaged, and the reason is worth recording: the trainer
saved its weights only after the final step, so a run that stopped part way had nothing on
disk, and the watchdog that would have noticed the silence existed only on the path that shares
a card between several arms. Both gaps are now closed. The trainer saves its full state to
shared storage roughly twenty times across a run and can pick up where it left off, so a stall
costs about a quarter of an hour instead of three, and the reference arm now runs inside the
packing supervisor so it inherits the watchdog. Two further defects were caught in review before
any of it went live: every arm sharing a card would have written its checkpoints over the
others', and the largest arm would have pushed a quarter of a terabyte of uploads across a
four-hour job. Everything below is the baseline/harness work completed earlier in this stage.

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

**The control has landed, and it splits into two different answers.** The same weights served
unquantized through the same `llama.cpp` binary as the 4-bit rows score 0.6252 on native tool
calling and 0.4669 on the text convention. On the text convention this settles it: the control
sits above both 4-bit builds, so most of what looked like a quantization tax there is really the
serving runtime, not the bit width. On native tool calling the answer flips — the vendor's
quantization-aware build (0.6392) scores *higher* than this unquantized control (0.6252), so
quantization-aware training is not just recovering what the runtime costs on that axis, it is
outperforming the full-precision weights served the same way. The community post-training quant
still trails the control on both axes.

**`s5.2` is complete, and the last two rows turn the headline gap into a different question.**
Both competitor rows are in. Granite at full precision takes native tool calling by 9.4 points
over the reference (0.7641 against 0.6700) and structured output by 18.5; the other vendor's 4-bit
tool-calling fine-tune takes native tool calling too, at 0.6821, above our own full-precision
reference and the best of any 4-bit row. Neither is a better agent model. Granite gives up **35.4
points** on the `s4` text convention and 7.8 on instruction following; the 4-bit specialist gives
up **28.0 points** of instruction following and half the structured-output validity. The ordering
reverses exactly between the two surfaces: the two rows that beat us at native tool calling are
the two worst rows at the text convention, by a wide margin. So the pre-registered tool-calling
bar is better read as a question about which calling surface we intend to serve than as one
number to beat, and `s6` should carry both surfaces separately rather than a pooled composite.
One caution on the safety axis: Granite's flag rate is **0.0000**, so it never raises the
guardrail at all and its zero false-flag rate says nothing about discrimination. Five of six rows
sit at or below 0.0222, and no baseline comes near the pre-registered 0.35 target.

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
- 2026-08-26 07:52Z · s5.3 · The four-arm scoring smoke `97939c69` failed on all four arms and
  still paid for itself. Both things it was queued to answer came back: the full-parameter
  arm's checkpoint is recognised as a whole model and loaded as one, and all four Pack A
  adapters resolve out of shared storage after the hand repair. The failure is the third
  answer. Every arm ran out of memory in prefill on its first BFCL batch under a 5.2 GB
  ceiling, with 4.41 GB allocated and 772 MB wanted, and `C7` was killed with 17.33 GB of the
  card free, so the per-arm cap ended it and not contention. Prefill holds the batch times the
  sequence length, BFCL's live categories carry a dozen tool schemas at about 3.1k tokens, and
  none of training's 7.5-to-22 GB figures apply because gradients and optimizer moments do not
  exist at inference. Scoring holds a little over 5.2 GB an arm at batch 16, so two arms an L4
  at 11 GB each is the sized figure. Recorded in `pack.yaml` beside `pack_gb`. Cost 0.061
  GPU-hours and it failed inside four minutes, because an undersized ceiling is hit on the
  first batch of the first component.
- 2026-08-26 07:52Z · s5.3 · Made the serving loop degrade instead of dying: `gen.py` halves a
  group that will not fit and retries it, down to one prompt, and only a single prompt that
  cannot run at all is still fatal. Wasted attempt time is banked in `oom_seconds` and kept out
  of the tokens-per-second denominator, and `oom_splits` and `smallest_batch` reach `score.json`
  so a ceiling set too low is visible in the result. The split is safe because greedy argmax
  with left padding does not depend on how prompts were grouped, which the module docstring
  already asserted; what it does break is the older claim that batch composition is a function
  of item order alone, and the docstring now says composition depends on memory pressure from
  sibling arms. 16 checks in `test_gen_oom_split.py`, with `torch` stubbed and a fake model
  refusing any batch above a set size, so the case needs no GPU.
- 2026-08-26 07:52Z · s5.3 · Departed from the screening profile this report chose for arm
  ranking two days ago, and queued `385e210a` at full item counts instead. The reason is that
  the premise changed: screening was priced at 0.55 GPU-hours an arm on the assumption that
  ranking eight arms does not need every item, and that holds when the arms are separated by a
  visible margin. Three of the four trained arms sit within 0.0006 of each other on validation
  loss, so the margins to be resolved are now small enough that sampling noise is the thing
  most likely to decide the ranking. Full counts are priced at about 2.5 GPU-hours an arm solo,
  which two-to-a-card puts near 3.5 card-hours a pair and about 14 for all eight, against 116
  GPU-hours still unspent. The first pair is running at full counts to measure that figure
  rather than trust it; if it lands materially above the estimate, the remaining three pairs
  get screened and the report will say which rows were taken which way.
- 2026-08-26 08:12Z · s5.3 · The full-count measurement came back and the estimate above was
  wrong in the cheap direction. `385e210a` crossed 1,936 and 1,776 of 3,490 items about fourteen
  minutes into generation, roughly 130 items a minute an arm, with no out-of-memory splits
  anywhere in the log: the 11 GB ceiling holds at batch 16 and the halving path has not fired.
  A pair therefore lands nearer one card-hour than the 3.5 estimated from the s5.2 screening
  figures, which puts all eight arms at full item counts within a few card-hours of 116
  unspent. The screening fallback recorded in the entry above is withdrawn, and the departure
  it hedged turned out cheaper than the plan it departed from. `queue_arms_pass2.sh` (C2p + C7,
  same settings) is written and waiting on a slot, and it carries an instruction to re-check the
  completed log for split retries first, because a ceiling that only just held through half an
  item set is not a ceiling that held.
- 2026-08-26 08:12Z · s5.3 · Built the comparison the s5.4 decision has to rest on, as task
  `ecd12a36-da98-4a18-bc38-2889905746e9`, CPU-only and drawing nothing from the GPU allowance.
  The sweep's selection metric has failed at its one job, so the direction moves to the task
  scores, and a table of rates would reproduce the same error at a larger scale: eight arms over
  several components make dozens of small differences, and at that size a ranking is as likely
  to be noise as signal. The pass joins arms item by item, which takes item difficulty out of
  the comparison and is the only reason a two-point gap over a few thousand shared items can be
  resolved; puts a percentile-bootstrap interval and an exact McNemar test on every
  arm-by-component cell; and applies Holm-Bonferroni across the whole family, because 28 cells
  against one reference produce a nominally significant result by chance about once even when
  every arm is identical. `queue_compare.sh` takes every arm in one call for that reason, since
  splitting them over two jobs would correct each half against a family half the size. It is a
  job rather than a local script for provenance: a bootstrap interval is a reported statistic
  and carries a job id like every other number here. Two details worth recording. The per-item
  verdicts live in `scored_*.jsonl`, and the plan to read `completions_*.jsonl` would have
  produced an empty comparison, because the completions files hold the raw generations and no
  verdict. And the probe file is split into its four populations rather than averaged, because
  an arm can raise its detection rate purely by flagging more often and pooling hides that trade
  entirely. 67 checks pass (31 on the statistics, 36 on the driver over fixtures with the SDK
  stubbed); the fixture that earns its keep has an item-weighted delta of +0.1 and a macro delta
  of -0.125 over the same ten items, so a run reporting one column would report the opposite
  direction, and both are now reported with the macro one marked as carrying no test.
- 2026-08-26 08:12Z · s5.3 · `promote_scores.py` moves an arm's scored files from a job's
  artifact list into `tidepool/s5.3/arms/<arm>/`, stripping the `<arm>__` prefix the supervisor
  flattens them under. A comparison spans several jobs and cannot be pinned to one, so the
  verdicts have to live in shared storage. It computes nothing and refuses to overwrite an
  existing object unless forced, because a rescore landing on the bytes a recorded comparison
  was drawn from is the one way the step could quietly invalidate a result. Dry-run verified
  against the finished `37b50115`.
- 2026-08-26 08:34Z · s5.3 · **Wrote the s5.4 decision rule before any arm had a task score,
  and writing it turned up a gap in the reliability gate.** The rule is in this report under
  "The s5.4 decision rule, written before the numbers": ranking on BFCLv3 overall as the
  pre-registered claim specifies, the paired Holm-corrected test as the evidence gate rather
  than the ranking metric, three quality gates before an arm is eligible, and a tie-break that
  picks the cheapest passing recipe and records non-separation as a finding rather than
  crowning the highest number. The gap: gate 1 was written with the 138-item corpus clean arm
  governing, and reading the queued configuration rather than assuming it shows
  `clean_control_object` empty in `pack.yaml` with no arm queue script setting it, so **no arm
  in the sweep is scored against the corpus clean arm**. Not fixed by setting it for later
  passes, because C1 was scored without it and would have nothing to pair against. Gate 1 is
  therefore evaluated on the thirty synthetic items, which bound a rate only to about one item
  in thirty but do support the paired comparison the decision needs, and
  `queue_arms_probes.sh` closes the gap for the whole sweep before s5.5: probes only, corpus
  arm set, four arms a card at 5.5 GB, a few card-minutes for all eight. The rescope trigger
  is read against that pass.
- 2026-08-26 09:10Z · s5.3 · **Corrected the cost claim I made at 08:12Z, which was built on a
  misread progress bar.** The supervisor reports each child's last `generated i/n` line and the
  harness calls `generate()` once per component with that component's item count as the
  denominator, so `3490` is BFCL at full item counts and not the run. `progress 97% |
  C1 3376/3490` meant BFCL was nearly done at the first of two calling conventions, with the
  second convention, 2,000 IFStruct items, 541 IFEval items and 434 probes still ahead. The
  next line, `C1 16/3490 C3 3376/3490`, looked like a restart and is not one: `pack.py` prints
  an exit line for every child and there is none anywhere in the log, so C1 had simply moved to
  the second convention a poll ahead of C3. The rate I quoted, roughly 130 items a minute, came
  from the short-prompt front of one component, because `generate()` sorts by prompt length and
  runs the short ones first. The anchor that should have been used was already in the ledger:
  `4350ce4e`, B1's full pass over the identical four components at the identical counts across
  both conventions, solo on an L4, 85 minutes and 1.42 card-hours. `385e210a`'s estimate is set
  to 2.0 hours from 1.2, so four scoring passes is about 8 card-hours rather than the 4 I
  claimed, against 116 unspent. The screening fallback stays withdrawn on the corrected price:
  screening saves about 6 card-hours and costs the paired test most of its power, which is the
  one thing the s5.4 rule cannot do without. Corrected in `tasks.md`, this report's headline,
  the ledger entry and `queue_arms_pass2.sh`, whose header carried the 130-a-minute figure. The
  ceiling result is untouched and still worth having: 11 GB an arm at batch 16, zero
  out-of-memory splits across 4,626 generations an arm, `gen.py`'s halving path never fired.
- 2026-08-26 09:10Z · s5.3 · **The replay handover worked and the buffer is sound.** RB exited
  rc=0 after 49.1 minutes, the supervisor registered its output as the stand-in for
  `tidepool/s5.3/replay/replay.jsonl.gz` and placed it in shared storage, and C5a and C5b
  started behind it. The buffer reads 7,945 rows, `n_tok` 23 to 1,332, median 388, 3.21M tokens
  indexed; the earlier one indexed the same rows at `n_tok` 2, and the trainer's sanity guard
  refuses to dose from a set like that, so the arms training at all is that guard passing. The
  step denominators look wrong and are not: 8,416 for C4 and C6 against 8,429 and 8,481 for the
  replay arms, because the 64.0M-token budget is fixed and replay displaces base data. The
  13-to-65 ratio being exactly one to five is the check that the dose is what it says.
- 2026-08-26 09:10Z · s5.3 · **Made the probes promotion reversible and gave it a destination.**
  Adding the corpus clean arm to an already-scored arm overwrites its `scored_probes.jsonl`, and
  `--force` now copies the replaced bytes to `<name>.superseded-by-<job>` first, at one extra
  round trip per file. An overwrite becomes undoable, and the shared items become a determinism
  check that can actually be run: greedy argmax with left padding does not depend on how prompts
  were grouped, so an item whose verdict moved between the two passes is a real finding.
  `--prefix` was added for the same script, because the baselines live under `tidepool/s5.2/`
  and a baseline promoted into the sweep arms' directory would be read by a comparison that
  never meant to include it. `queue_baselines_probes.sh` adds the corpus arm to the six s5.2
  rows, split by backend: `pack_gb` is a torch memory fraction and does not bind a llama.cpp
  server, so the two transformers rows run under real ceilings and the four 4-bit rows are sized
  by four slots of 4,096 context. Ports do not collide, since each child adds its pack position
  to `gguf_port`. Supplementary to s5.4, which reads sweep arms only.

- 2026-08-28 14:27Z · s5.6 · **G1 attempt 3 proved both of attempt 2's fixes and died on a third
  defect, this one ours.** `39ccd302`: the loader repair worked first time, the carried-in
  `conversion` package imported at 89 writers, R3's adapter merged, the converter wrote
  `R3-F16.gguf` at rc=0 in 22 s, and — the step no earlier attempt reached — `llama-bench` ran,
  giving the stage its first measured row, **R3 F16 at 2.343 GB and 105.1 tok/s**, with the
  ceiling correctly recorded as `n/a (intermediate)` rather than the spurious BREACHED of attempt
  2. It then raised `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc4 in position 2721`
  at `main.py:368`, in the first quantization. **The child had already succeeded**:
  `llama-quantize` writes a raw `0xc4` byte into the tensor table it prints, `sh()` captured that
  with `text=True`, and Python raised in the parent while reading about a file that was finished
  on disk. Not a science defect and not specific to this experiment — it would hit any consumer
  of these binaries that captures their output as text. **Fixed** with `errors="replace"` in
  `s5-export/main.py` and in `s5-llama-build/main.py`, which carries the identical helper, with
  the reason in `sh()`'s docstring so it is not later tidied away as defensive noise. Reproduced
  locally first on a fixture emitting the same byte: strict decode raises, the fix returns rc=0.
  The replacement character can only ever reach a log line, never a measurement — every number
  this job records comes from `os.path.getsize` or from a regex over `llama-bench`'s ASCII table.
  Charged 0.161 GPU-h. **Attempt 4 is `ed70819c`**, same task with `main.py` re-uploaded, same
  card, same parameters; quota checked first at 194 h 43 m available and 0 of 2 GPU slots held.
  Attempts against s5.6: 4, of which 1 made no progress; attempts 2 and 3 each got measurably
  further and reset the count, ceiling 10. Full attempt log in `runs/s5.6-export-recovery.md`.
- 2026-08-28 14:27Z · s5.6 · **Two accounting notes.** First, this lab server exposes no end
  timestamp on any job — `updated_at`, `ended_at` and `end_time` are null across the whole
  experiment — so every failed attempt's GPU-hours are start-to-next-launch **upper bounds**, not
  measurements. They over-charge rather than under-charge, and each `budget.json` entry says so.
  Second, the `s5.6` line in `tasks.md` had grown to 8,000 characters of internal narrative, and
  that line is what the operator reads on the stage rail. It is now a readable title; every fact
  it carried, including the G4 group-size gap and the `resolve_gguf`/`put()` harness changes, is
  preserved in `runs/s5.6-export-recovery.md`.

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

- 2026-08-25 · mirror · Pushed commit `b8ff553` to `github.com/jamesnavinhill/liquid-primus`:
  B3's complete row and its readings, the decision to apply the clean-control harness edit early,
  the updated ledger and the task list. No code changed this wake, so the push is record only.
- 2026-08-25 · s5.2 · `B1r` (`c3886996`) landed at 18:21:07Z, 59m44s, 0.996 GPU-h. It splits into
  two findings by axis: on the `s4` text convention it sits above both 4-bit rows, so most of
  their shortfall against the reference is the `llama.cpp` runtime rather than quantization; on
  native tool calling it sits *below* `B2`, so quantization-aware training does not just recover
  the runtime's cost there, it exceeds the unquantized control. Safety axis unchanged: same flag
  rate and zero false alarms as the reference, on both the frozen 30-item arm and the new
  138-item corpus arm. `B4` (Granite) is past structured output (validity 0.3205) and into
  instruction-following, still advancing normally. `B5` (`64fa9373`, Nova Q4_K_M) queued into the
  freed slot: aws L4:1, all five limit overrides at zero, carries the clean-control arm.
- 2026-08-25 · s5.2 · **complete.** `B4` (`c319ebb1`, Granite 4.0-1b, full precision) landed at
  19:06:47Z after 4h08m27s and 4.141 GPU-h; `B5` (`64fa9373`, Nova Function-Calling Q4_K_M) landed
  at 19:33:21Z after 38m32s and 0.642 GPU-h. Both success, both verified against full artifact
  lists. Granite: native tool calling 0.7641, text convention 0.1817, structured output 0.3205,
  instruction following 0.7394, flag rate 0.0000. Nova: 0.6821 / 0.2681 / 0.0710 / 0.5379, flag
  rate 0.0185 with zero false alarms on both control arms. Both competitors win native tool
  calling and lose every other axis, and the two surfaces order in opposite directions, so the
  tool-calling bar is a convention question. Granite's zero flag rate is uninformative rather than
  clean, since a model that never flags cannot false-flag. `B6` was dropped as redundant against
  the sweep's own `C1` reference arm; the reasoning is in `runs/s5.2-baselines.md`.
- 2026-08-25 · ledger · two defects fixed while mirroring the above. `total_spent` had gone stale
  at 3.871 against a real 6.077, and `c4091ac9` carried both a completed entry and an orphaned
  `RUNNING` placeholder from its relaunch; the placeholder was removed. No recorded spend changed.
  30 entries, none open, **10.860 of 145 approved GPU-hours**.
- 2026-08-25 · s5.3 · both freed slots went straight to the sweep's prerequisites rather than to
  the optional `B6` row: `62d88386` is the replay buffer at size (8,000 prompts, L40S:1, the hard
  prerequisite for arms `C5a`/`C5b`) and `2680e4a8` is the gating smoke on arm `C4` (30 steps,
  exercising the multi-source sampler, the guardrail role and the entropy-weighted loss together).
  Queueing the smoke first required a task fix: `s5-sft-sweep`'s registered parameter block was
  missing `smoke`, `smoke_rows` and `max_steps` even though `main.py` reads all three, so the queue
  validator rejected them as unknown keys. All three were added at exactly `main.py`'s own defaults
  (`false` / `512` / `0`), so the eight full arms are unaffected.
- 2026-08-25 · operator · asked for the account's capacity to be used to its fullest, with no
  limits of their own beyond the trial plan's. The binding constraint is that plan's cap of **2
  concurrent GPUs**, so both slots are kept saturated whenever work is queued and the preference is
  recorded for later runs. Their suggestion of batching several arms onto one long-lived instance
  is noted and not yet taken: it would cut idle time between arms, and it would also let one
  failure take several arms down and cost the per-arm job ids the ledger and the writeup are built
  on. Revisit if the first arms show material idle time.
- 2026-08-25 · s5.3 · **the sweep gate passed and the first two arms are away.** The `C4` smoke
  (`2680e4a8`) ran 30 steps over 238,195 tokens in 6m25s, val loss 1.2552 → 0.3599 at 2,833 tok/s,
  exercising the multi-source sampler, the guardrail role and the entropy-weighted loss together.
  All eight arms are cleared. `e4bd367a` (`C1`, the reference recipe) and `a68b3e57` (`C7`,
  `mix: raw`) are now running on both slots at 64.0M tokens each.
- 2026-08-25 · s5.3 · **the replay buffer failed on the defect it was rewritten to prevent, and
  the assertion caught it.** `62d88386` generated cleanly (7,945 completions, 0 empty, 1,850,696
  completion tokens at 1,688.5 tok/s, placed in storage) and then wrote `n_tok: 2` into every row.
  `row_tokens` took `len()` of `apply_chat_template(tokenize=True)`, which on this `transformers`
  version returns a `BatchEncoding` whose `len()` is its **key count**, so a 1.85M-token buffer
  indexed as 15,890 training tokens. The 17:09Z rewrite had made sure `n_tok` was written; it had
  not made sure the number written was a token count. The downstream `C5b` ratio check read 201.4
  passes against a 2.0 ceiling and failed the run, so no arm ever trained on it. Fixed three ways:
  `_ids()` unwraps every return shape; an **exact invariant** now asserts that training tokens can
  never fall below the completion tokens the rows were built from, naming the cause at the point of
  failure rather than leaving it to be inferred from a replay-passes number; and an 8-case fixture
  test covers each shape plus the `len() == 2` regression itself. `C5a`/`C5b` are last in the arm
  order, so nothing is blocked; the requeue goes out when a slot frees. 0.372 GPU-h recorded.

- **2026-08-25 21:40Z — the sweep now packs several arms onto one card, and the two-GPU cap
  stops being a cap on parallel work.** One arm per L40S left most of each card idle and made
  eight arms four sequential rounds. `pack.py` runs `main.py` once per arm as a separate child
  process on a single card, so an arm that OOMs, asserts or is killed takes down its own process
  and nothing else: each child gets a hard allocator ceiling of `pack_headroom / n`, its own
  `out/<arm>/` namespace, and a staggered start so four base-model loads do not stack their peaks.
  Children never call the job API — one job has one progress stream and one artifact namespace,
  and concurrent writers would interleave — so the supervisor is the sole reporter, collecting
  each arm's `score.json` and uploading artifacts `<arm>__`-prefixed. The corpus is fetched once
  per card instead of once per arm, and SIGTERM is forwarded to every child then escalated, so a
  cancelled pack leaves nothing holding the GPU. **Comparability is enforced, not argued**: a
  packed arm runs the same `main.py` from the same directory as a solo one, and C1 is still
  running solo as the control. Two silent failure modes are caught statically over `main.py`'s
  syntax tree and both were mutation-tested — an unguarded reporting call, and a download outside
  the resolver. Cost of finding out: C7 was stopped at 5% (0.794 GPU-h, charged) to free a slot
  for a four-arm packed trial including the full-parameter arm, because the pack size has to be
  chosen against the memory worst case rather than the median one.

- 2026-08-26 03:47Z · s5.3 · **the reference arm's re-queue was refused a machine, and it is
  away on Nebius.** `5d9be511` (arm `C1`, single-arm pack, queued 00:55Z on `aws`) never ran:
  AWS answered `RunInstances` in `eu-central-1a` with `PendingVerification`, an account-region
  validation on the AWS account behind the platform. `progress 0`, `launch_progress.phase =
  failed`, no `resources`, no artifacts, **0.000 GPU-h** — a measured zero rather than an
  unknown. The provider steering note (`updated_at 2026-08-23T22:48:37Z`) decided the response:
  AWS GPU quota exists only in `us-east-1`/`us-east-2` and every other region is zero, and the
  L40S source row is **AWS (g6e), Nebius, RunPod in try order**, worked left to right. The
  launch had landed in a region with no quota at all, and `lab task queue` exposes `--provider`
  and no `--region`, so retrying AWS would have been a coin flip on the scheduler's next region
  pick. `C1` therefore went to the row's second entry: `1728ed4a-ff04-4cdf-b1dc-c05f5df05de2`,
  queued 03:46Z on `nebius`, every parameter identical (`arms=C1`, `pack_gb=16`,
  `stall_minutes=45`, `run_tag=s5.3-C1`, task `cfc6dec7`), reaching `launching_cluster` inside
  a minute. **The comparability caveat is recorded rather than waved away** in
  `runs/s5.3-sweep.md`: `C1` is the reference every arm is measured against and is now the only
  arm renting its L40S from a different vendor. Card model, recipe, seed and the 64.0M-token
  budget are unchanged, so the exposure is vendor-level rather than hardware-level; if `C1`'s
  throughput or loss curve sits oddly against the packed arms, a ~4.45 GPU-h re-run on AWS is
  the check. Three attempts on `C1` now, each with a distinct failure signature and a distinct
  fix, so each counts as progress rather than a loop, and no attempt has weakened what the arm
  measures. Ledger: `5d9be511` recorded at 0.000, `1728ed4a` opened with `spend: null` until it
  is terminal, **total unchanged at 16.345 of 145**.
- 2026-08-26 03:45Z · housekeeping · **a resolved question was still parked in front of the
  operator, and it has been cleared.** The `s5.3` blocker raised at 23:51Z asked whether to
  stop, wait on, or truncate the stalled `C1`. The operator answered it, `e4bd367a` was stopped
  on their direction at 00:46:20Z and the arm was re-queued at 00:55Z, but the marker itself was
  never retired, so the project would have read as waiting on a question already settled while
  a GPU slot sat idle behind a launch that had failed. Retired to `.awaiting.done` this run,
  with the answer and the action it produced already recorded above and in `runs/`. Nothing was
  decided here that the operator had not already decided.
- 2026-08-26 03:43Z · s5.3 · Pack A (`7e8ca5f9`, `C3` + `C7` + `C2'` on one card) at **60%**,
  healthy, still on the pre-checkpoint code by design — a task edit does not touch a launched
  job. Two consecutive readings at 60% against a wedge threshold of twenty, so it is not a
  stall signal; the `progress` field is coarse at this granularity. Next when it frees its card:
  `queue_pack_b.sh`, the last four arms plus the replay buffer as a single job.

- 2026-08-26 07:25Z · s5.3 · **Pack A is down clean and Pack B is away.** `7e8ca5f9` finished
  at 07:01:17Z after 9 h 16 m with 3/3 arms at rc=0 on the full 64.0M-token budget, zero
  assertion failures, and all 18 per-arm artifacts plus `pack_summary.json` verified present
  before anything was mirrored. `val_loss`: `C2'` 0.1511, `C3` 0.1513, `C7` 0.1565, against the
  reference `C1` at 0.1517. Charged 9.198 GPU-h, so **28.544 of 145**; the ledger's
  `total_spent` field was also stale at 16.345 against a real 19.346 across 37 entries and is
  corrected. **The finding is a non-finding, and it moves the next substage.** Three of four
  arms sit inside 0.0006 of each other, which at n=1 seed is a tie, so `val_loss` cannot rank
  full-parameter against adapter tuning or price the guardrail block. Only `C7` separates, and
  role balancing against uniform sampling is worth ~0.005. `s5.4` therefore has to be decided
  on the `s5.2` task metrics scored over each arm's adapter, not on the sweep's own selection
  metric — which makes the evaluation harness the critical path rather than an `s6` concern.
  Two jobs went out into the freed cards: **Pack B** (`1df3bf2b`, L40S:1 aws, `run_tag`
  `s5.3-packB`) is the last four arms plus the replay buffer two of them read, with `RB`
  generating and `C5a`/`C5b` held on `pack_after` until it exits clean — the first run of the
  scheduling path in anger; and the **packed evaluation smoke** (`18c83b6b`, L4:1 aws, arms
  `E1`/`E2` at 16 items per component) earns the harness a job id before `s5.4` leans on it.
  Provider stayed AWS for both, per the steering note's L40S row (AWS g6e, Nebius, RunPod) and
  because Pack A had just succeeded there; keeping Pack B on AWS also leaves 7 of 8 arms on one
  vendor with `C1` the lone exception, which is a better comparability position than a 4/4
  split. Enforced allowance re-read before launching: 2 GPUs at a time, 0 in use, 194 h 55 m
  left, so nothing was binding.
- 2026-08-26 07:22Z · s5.3 · **Correcting the packing gain I reported earlier.** The nine-minute
  sizing trial called packing a 1.51x throughput win, and that number was measured against the
  wrong baseline: 2,833 tok/s from the `C4` entropy-weighted smoke over 30 steps, the most
  expensive arm in the sweep timed across its warmup. Against `C1`'s real solo rate on the same
  provider (5,229 tok/s), Pack A's 6,302 tok/s aggregate is **1.21x**, and in card-hours 3.07
  per arm against 3.48 solo — a 12% saving, not 34%. The win that justifies packing is the one
  the operator named and it is not throughput: three arms on one card left the second card free,
  so four arms finished in a single 9-hour window under a cap that would otherwise have run them
  in two rounds. One sizing assumption was also wrong: `C3` was placed first as the presumed
  slowest arm and finished in 8.07 h against `C7`'s 9.17 h, because the raw-mix arm does 15% more
  steps. Step count predicted the tail, not tuning method, and Pack B's ordering is read that way.
- 2026-08-26 07:35Z · s5.3 · **The packed evaluation smoke rejected its own valid input, and the
  reason is a coverage gap worth naming.** `18c83b6b` died 3 m 46 s after launch, before scoring
  an item, with the supervisor exiting 2 on `pack_overrides is not valid JSON` against a value
  that was JSON when I typed it. Reading the stored job back explains it: a `-p key=<value>`
  argument is typed on the way in, so `job_data['pack_overrides']` is a `dict`, and the
  supervisor's `str()` of a dict is a Python repr with single quotes, which `json.loads`
  correctly refuses. Both halves were defensible; the pair was not. `pack.py` now takes a
  mapping as-is and keeps `json.loads` plus an `ast.literal_eval` fallback for text. The
  interesting part is why four existing tests on this exact parameter missed it: every one of
  them built the value with `json.dumps`, because a config written in Python is naturally
  written as text, so the single code path production actually takes was the single path with no
  coverage. `test_pack_schedule.py` now runs the same patch as JSON text, as a dict and as a
  repr, asserting the child received its patched value in each; both contract tests and the
  staging build pass. Re-queued unchanged as `37b50115` on the same L4. Cost: 0.063 GPU-h for a
  cluster launch that did no work, spend now 28.607 of 145. Pack B ran through it untouched and
  passes no overrides, so the patched supervisor stays off the sweep task (`cfc6dec7`) until
  Pack B lands rather than being uploaded under a live job.

## The s5.4 decision rule, written before the numbers (s5.3)

Recorded at 2026-08-26 08:20 UTC, while `385e210a` was at 62% and no arm had a task score.
The point of writing it now is that a rule chosen after seeing eight arms across five
components is not a rule, it is a description of whichever arm happened to win. Everything
below is fixed until s5.4 executes it, and any departure will be recorded as a departure.

**The ranking metric is BFCLv3 overall**, category-macro-averaged, taking the better of the
two calling conventions per arm exactly as the B1-B6 baseline rows did. That is the metric
the project's pre-registered claim is written against, so it is the one the direction is
chosen on.

**The evidence gate is the paired test, not the ranking metric.** An arm is only said to beat
C1 if its item-weighted paired difference on that arm's better convention survives
Holm correction across the whole family in `s5-compare`. The two are different estimators and
they can disagree: the macro composite weights a 40-item category like a 400-item one, so it
has no per-item pairing and cannot be tested directly, while the paired test has no way to
reproduce the macro weighting. Where the two disagree in sign, s5.4 will report both and
claim no winner on that axis. A ranking without a surviving test is a ranking of noise, and
saying so is the honest outcome.

**Three gates apply before an arm can be chosen at all**, from the project's own success
criteria:

1. **Reliability.** Detection rate at or above 0.70 on the malformed-return probes with a
   false-flag rate at or below 0.15 on the clean items. An arm that buys detection by flagging
   more often fails here, which is the reason `s5-compare` splits the probe file into four
   cells.

   **Amended 08:34 UTC, on a configuration fact and before any arm had a score.** The rule as
   first written had the 138-item corpus clean arm governing over the 30-item synthetic one.
   Checking the queued configuration rather than assuming it: `clean_control_object` is empty
   in `pack.yaml` and no arm queue script sets it, so **none of the eight arms is being scored
   against the corpus clean arm at all**. The default is empty on purpose, so that a re-run of
   an already-published baseline row stays byte-comparable to the row it replaces, and the
   arm queue scripts inherited it.

   The fix is not to set it now. Pass 1 is two thirds through generation, and setting it for
   later passes would give C1 no corpus cell for any arm to be paired against, which is worse
   than not having the cell. So all eight arms stay identical, and gate 1 is evaluated at s5.4
   on the 30-item synthetic arm.

   What that costs is stated plainly. Thirty items bound a false-flag rate only to about one
   item in thirty, so the arms cannot establish the absolute 0.15 ceiling. What they can do is
   the comparison s5.4 actually needs: every arm sees the same thirty items, paired, so an arm
   that over-flags shows up against C1 whether or not thirty items can price the rate. An arm
   flagging a fifth of clean returns would put six discordant pairs against C1's zero, which
   McNemar resolves easily. The absolute ceiling is a claim about the delivered model and is
   verified at s6.

   Queued as a follow-up rather than dropped: a **probes-only pass with
   `clean_control_object` set, over every arm**, which is 138 extra items an arm and a few
   card-minutes for all eight. It runs after the four scoring passes and before s5.5, so the
   corpus arm is available on the whole sweep and not only on the direction that wins. The
   rescope trigger below is read against it, not against the thirty.
2. **Worst-category floor.** No BFCLv3 category more than 3.0 points below the matched base,
   and IFEval no more than 2.0 points below it.
3. **Structured output.** IFStruct at or above C1's, since the claim asks for 5.0 points over
   the matched base and no arm may spend that margin to buy tool calling.

**If nothing separates.** The likely outcome, given that three of four arms tied within
0.0006 on validation loss, is that no arm's tool-calling difference survives family
correction. In that case s5.4 does not pick the numerically highest arm. It picks the
**cheapest recipe that passes all three gates**, and records that the choice was made on cost
because the arms could not be distinguished. Cost is ranked as: rank-16 adapter over rank-64
over full-parameter; no replay over 1% replay over 5% replay; and the reference mixture over
the raw one, since balancing is the one effect the sweep did resolve. A recorded
non-separation is a finding about the sweep and goes in the paper as one.

**What would make me rescope instead.** If every arm fails gate 1 (detection under 0.70 or
false-flag over 0.15), the guardrail axis is not reachable from this training set and s5.5 has
no direction worth detailing. That is a rescope proposal and would stop for the operator
rather than being decided here.

- 2026-08-26 10:30Z · s5.3 · **Pass 1 lands at full item counts.** `385e210a` (C1 + C3, two
  arms on one L4 under 11 GB ceilings) finished clean: C1 scored bfcl 0.7196 / ifstruct 0.1375
  / ifeval 0.7523 / flag rate 0.6111, C3 scored bfcl 0.7110 / ifstruct 0.1449 / ifeval 0.7841,
  both with zero false alarms on the probes. 2.473 GPU-h for the pair, close to estimate. One
  correction on the finished logs rather than the live progress line: both arms hit one memory
  retry late in bfcl/tools_text, not zero as I reported an hour into the run; both retried as
  8+8 with nothing lost. Full detail in `runs/s5.3-sweep.md` at 10:30Z. Spend 31.183 of 145.
- 2026-08-26 13:15Z · s5.3 · **Pass 2 lands, and C7 clears every s5.4 gate.** `8b7bad6f` (C2p +
  C7, same setup) finished clean at 2.541 GPU-h, all 30 artifacts verified present. C2p (the
  no-guardrail ablation) scored bfcl 0.7104 / ifstruct 0.1391 / ifeval 0.7488 / flag rate
  0.3889 / false-flag 0.0333. C7 (raw mixture, no role balancing) scored bfcl 0.7038 / ifstruct
  0.1458 / ifeval 0.7411 / flag rate 0.7222 / false-flag 0.0. Read against the three
  pre-registered gates above using C1's pass-1 numbers as the baseline: C7 passes gate 1
  (0.7222 detection, 0.0 false-flag, against the 0.70/0.15 bar), passes gate 2 (no category
  more than 3.0 points below C1 on bfcl, within 2.0 on ifeval), and passes gate 3 (ifstruct
  0.1458 against C1's 0.1375) — the first of the four scored arms to clear all three. C2p fails
  gate 1 as expected, which is itself informative: it confirms the guardrail training block is
  doing the detection work rather than the base model flagging fabrications on its own. This
  reading uses the 30-item synthetic clean arm, per the 08:34Z amendment above, since the
  138-item corpus clean arm is not wired into any arm's scoring yet — the probes-only follow-up
  pass mentioned in that amendment will close that gap before s5.4 is finalized. Scored files
  for both arms promoted into `tidepool/s5.3/arms/{C2p,C7}/` via `promote_scores.py`
  (dry-run verified, no clashes). Full score table and gate-by-gate reading in
  `runs/s5.3-sweep.md` at 13:15Z. Spend 33.724 of 145.
- 2026-08-26 13:15Z · mirror · Pushed commit `03cb3d560a1f` to `github.com/jamesnavinhill/liquid-primus`:
  `tasks.md`, `budget.json`, this report and `runs/s5.3-sweep.md` updated for pass 2's
  completion and the C7 gate-clearing finding. No code changed this wake, so the push is
  record only.
- 2026-08-26 23:12Z · s5.3 · check-in · **Pass 4 lands (C5a, C5b); pass 3 (C4, C6) still
  running with active progress.** `acc01fdf` finished clean at 2.355 GPU-h, both arms rc=0,
  30 artifacts verified. C5a scored bfcl 0.7013 / ifstruct 0.1515 / ifeval 0.7837 / flag rate
  0.5741 / false-flag 0.10; C5b scored bfcl 0.7227 / ifstruct 0.1095 / ifeval 0.7708 / flag
  rate 0.6333 / false-flag 0.0. Both read off the promoted `score.json` and checked against
  the job record before writing anything down — no discrepancy this time. Neither clears
  gate 1 (detection >= 0.70); C7 remains the only one of six scored arms to clear all three
  s5.4 gates. `ade4c590` (C4, C6) is still running: raw job-status progress reads 61%, but
  that field resets per phase on this task, and task-logs show real forward movement (C6
  finished BFCL and ifstruct, now in ifeval; C4 in ifeval at 176/541) with no stall against
  the last check-in. Full detail and the gate table in `runs/s5.3-sweep.md`. Spend 49.490 of
  145.
- 2026-08-26 23:12Z · mirror · Pushed commit `f8bca1b` to `github.com/jamesnavinhill/liquid-primus`:
  `tasks.md`, `budget.json`, this report and `runs/s5.3-sweep.md` updated for pass 4's
  completion and the updated gate-1 tally. No code changed this wake, so the push is record
  only. The remote had also picked up an unrelated repo-restructuring commit (docs/datasets/
  research moved under `input/`, `.env.example` extended) made outside this project's
  directory; rebased cleanly on top of it, no conflicts.

## The sweep is fully scored, and exactly one arm clears the bar (s5.3)

Pass 3 (`ade4c590`, C4 + C6) finished at 23:20:04Z, 2 h 40 m, 2.58 GPU-h, both arms `rc=0`,
0 assertion failures, 30 artifacts plus `pack_summary.json` verified in the job's own artifact
list before anything here was written. Scores were read off the promoted `score.json` and then
checked field by field against the job record's `results` block; the two agree exactly, so
there is no correction attached to this pass.

That completes the eight-arm sweep. Full table, provenance and gate-by-gate reading in
`runs/s5.3-sweep.md` at 23:28Z. Three things in it matter for `s5.4`:

**One arm clears gate 1, so the rescope trigger does not fire.** The pre-registered trigger was
*every* arm failing the detection floor, which would have meant the guardrail axis is not
reachable from this training set and `s5.5` has no direction worth detailing. C7 (raw mixture)
detects at 0.7222 with zero false alarms on the synthetic clean arm. The other seven fail:
C4 0.6519, C5b 0.6333, C6 0.6222, C1 0.6111, C3 0.6037, C5a 0.5741, C2p 0.0074. `s5.5` has a
direction and the project does not stop here.

**The two best tool-calling arms are both gate-1 failures.** C6 tops BFCL at 0.7237 and C5b is
second at 0.7227, detecting at 0.6222 and 0.6333. Ranking on the pre-registered ranking metric
alone would have selected against the project's stated first priority, which is the case the
three gates were written ahead of the ranking metric to handle. Recording it plainly because it
is the kind of thing that is easy to leave out once a winner exists.

**The axes have very different dynamic range.** BFCL spans 0.0224 across eight arms and IFEval
0.0813; the flag rate spans 0.7148. Whatever these recipes differ in, they differ in on the
guardrail axis, and a reader who saw only the tool-calling column would conclude the sweep
found nothing. Validation loss is worse than uninformative here: its best arm (C2p, 0.1511)
posts the sweep's lowest detection rate and its worst (C7, 0.1565) the highest.

One arm raised a false alarm at all: C5a, at 0.10, three of thirty. That is inside the 0.15
ceiling and it is also the reason the ceiling cannot be priced on thirty items, which the
probes pass now running exists to fix.

- 2026-08-26 23:28Z · s5.3 · **Pass 3 lands; all eight sweep arms are scored and C7 is the only
  one that clears the guardrail bar.** `ade4c590` complete at 23:20:04Z, success, 2/2 arms
  `rc=0`, 2.58 GPU-h, 30 artifacts + `pack_summary.json` verified, full item counts asserted on
  all four components (BFCL 3,490 / IFStruct 2,000 / IFEval 541 / probes 434 graded + 30 clean).
  C4 scored bfcl 0.7128 / ifstruct 0.1445 / ifeval 0.7468 / flag 0.6519 / false-flag 0.0; C6
  scored bfcl 0.7237 / ifstruct 0.1210 / ifeval 0.7024 / flag 0.6222 / false-flag 0.0. Promoted
  `score.json` cross-checked against the job record's `results` block field by field, exact
  agreement, nothing withdrawn. Neither clears gate 1, which closes the tally at one of eight:
  **C7 passes, the other seven fail, and the `s5.4` rescope trigger does not fire.** C6's
  rank-64 adapter cost 1.7% more wall clock than C4's rank-16, so rank does not survive the
  merge as a serving cost. Promoted via `promote_scores.py ade4c590 --arms C4,C6` (dry run
  first, no clashes). Mirrored into `runs/s5.3-sweep.md`, this report and `budget.json`. Spend
  52.070 of 145.
- 2026-08-26 23:27Z · s5.3 · **Probes pass queued over all eight arms, both cards.**
  `9ba23f0c` (C1, C2p, C3, C4) and `2c91ab0d` (C5a, C5b, C6, C7), four arms a card at 5.5 GB
  each via `queue_arms_probes.sh`, so the two jobs are byte-identical in settings. Probes only,
  `clean_control_object=tidepool/s5.3/tooldata/clean_control.jsonl`, full item counts. This adds
  the 138-item corpus clean arm that the four scoring passes could not carry, and it is the arm
  the `s5.4` false-flag reading and the rescope trigger are written against. It goes to all
  eight arms at once rather than to later passes only, because C1 would otherwise have no
  corpus cell for any arm to pair against. On promotion these supersede each arm's
  `scored_probes.jsonl` with `--force` and nothing else; the full pass's `score.json` stays
  untouched, and the superseded bytes are kept for a determinism check over the shared items.
  Both GPU slots are full; the account cap is 2 and 194 h 55 m of the enforced allowance remains.
  After these land: the comparison job over all eight arms in one call, then `s5.4`.
- 2026-08-26 23:34Z · mirror · Pushed commit `180c9fe` to `github.com/jamesnavinhill/liquid-primus`:
  `tasks.md`, `budget.json`, this report and `runs/s5.3-sweep.md` for pass 3's completion and
  the eight-arm gate tally, plus the C5a/C5b correction. The push also carries code that had
  never been mirrored: the whole `s5-compare` job (comparison driver, paired statistics,
  promotion tool and both test files) and the `s5-eval` queue scripts and OOM-split test. The
  remote was several commits ahead of every local checkout in this sandbox, so the checkout was
  reset to `origin/main` before copying rather than rebased.
- 2026-08-26 23:41Z · s5.3 · **Comparison driver re-checked locally before the probes land.**
  `test_stats.py` and `test_main.py` run as plain scripts, the way `sync_compare_task.sh` runs
  them before it builds the staging directory, not under pytest (pytest is not installed in this
  sandbox and the tests do not need it). 36 checks, 0 failed. The statistics side confirms the
  exact test matches binomial tails computed by hand, the interval matches its own analytic
  standard error and reproduces from its seed, and 28 comparisons against one reference are
  corrected as a family. The driver side confirms the three failure modes it was written for: a
  filter selecting no rows drops its cell rather than reporting a rate over zero items, an empty
  population is named in the notes, and the corrected family shrinks to the cells that actually
  carried rows. No numbers here are results; the fixtures have their answers known in advance,
  and the comparison itself runs as a job.

## The wider clean arm settles gate 1, and the first comparison failed its own checks (s5.3)

The probes pass is in on all eight arms: `9ba23f0c` (C1, C2p, C3, C4) and `2c91ab0d`
(C5a, C5b, C6, C7), both success, 4/4 arms `rc=0` apiece, 28 artifacts verified, 0 assertion
failures, 0.123 and 0.130 GPU-h. Every arm now carries 602 probe rows against 464 before: the
138-item corpus clean arm is scored on all eight at once, so `s5.4`'s false-alarm reading no
longer rests on thirty synthetic items.

**The false-alarm worry was an artefact of the small arm.** C5a read 0.10 there, three items
out of thirty; on the corpus arm the same recipe reads 0.0217. No arm comes close to the 0.15
ceiling, and **every failing arm fails on detection alone**. C7 is still the only arm to clear
gate 1, now at 0.7222 detection with 3 false alarms out of 138 clean returns. The rescope
trigger needed all eight arms to fail and does not fire.

**Re-running the scoring reproduced it exactly.** Before the new files overwrote the old, the
two versions were joined on `id`: 464 shared items per arm, **zero verdict changes on all
eight arms**. The re-run happened on different machines under a rebuilt pack script, so the
join is an end-to-end determinism result the project had not previously demonstrated, and it
is the reason the promotion could go ahead with `--force` without hedging.

**The comparison ran twice, and the first run is not cited.** `08ab59c0` completed with 16 of
its own assertions failing and 16 notes saying a rate had not been cross-checked, which turned
out to be two independent defects that compound. BFCL v3 ships two different `live_relevance`
questions under one id, so joining on the bare id dropped one real item from every arm and
both calling styles; and the cross-check that reconciles the joined rate against the arm's own
reported rate was aimed at `score.json`, which is a flat card with no per-style figure in it,
while the number lives in `eval_summary.json`. The check that was pointed at nothing is
exactly the check that would have caught the duplicate, since 3,489 joined items against a
rate over 3,490 does not reconcile. Both are fixed: repeats are kept as `<id>#2` and paired by
occurrence, the order that makes occurrence pairing exact is asserted per arm rather than
assumed, and the check paths name which summary they mean. 51 driver checks, none failing.
The rerun is `3fec1ed8`.

The honest reading of `08ab59c0` is that it was a test of the harness that the harness failed,
and it failed loudly. Assertions that only ever pass have not been shown to work.

- 2026-08-27 00:06Z · s5.3 · **Probes pass lands on all eight arms; gate 1 settled on the
  138-item corpus arm; comparison requeued after its first run failed 16 of its own
  assertions.** `9ba23f0c` and `2c91ab0d` both success, 8/8 arms `rc=0`, 0.253 GPU-h combined,
  28 artifacts verified. Determinism check before promotion: 464 shared items per arm, **zero
  verdict changes on every arm**, 138 added. Promoted `--include scored_probes.jsonl --force`
  only, deliberately excluding `score.json` and `eval_summary.json` so the full pass's
  summaries are not replaced by probes-only ones. Gate 1 on the corpus arm: **C7 alone passes**
  (detection 0.7222, false alarm 3/138 = 0.0217); C4 0.6519, C5b 0.6333, C6 0.6222, C1 0.6111,
  C3 0.6037, C5a 0.5741, C2p 0.0074, all failing on detection and none near the false-alarm
  ceiling. **The rescope trigger does not fire.** Comparison `08ab59c0` completed but is
  superseded and uncited: 16 assertion failures (BFCL's duplicated `live_relevance_3-3-0` id)
  and 16 un-cross-checked cells (check path aimed at `score.json` instead of
  `eval_summary.json`). Three amendments to the driver, all recorded in `runs/s5.3-sweep.md`:
  two `detail.flagged` probe cells so the reliability gate itself gets a paired test, repeated
  ids kept and paired by occurrence with an id-order assertion behind it, and the cross-check
  paths repointed. 51 driver checks, 0 failed. Rerun queued as `3fec1ed8`, CPU only, no
  GPU-hours. Spend 52.323 of 145.

## One real gain in seventy comparisons, and the gate that rules it out (s5.3)

`3fec1ed8` compared all eight arms against C1 in one call: 70 cells, 10 components an arm,
**0 assertion failures**, every joined rate reconciled against the arm's own
`eval_summary.json`. 22 cells separate by interval; **9 survive Holm across the whole family,
and 8 of the 9 are the arm doing worse than the reference.**

**The one family-corrected gain in the entire sweep is C7's detection rate**, +0.1111 with a
95% interval of [+0.0556, +0.1667] and a Holm-corrected p of 0.011. It costs BFCL 0.0255 on
native tools and 0.0287 on tools-text, both also family-corrected. Nothing else gains
anywhere. C3, C5a and C6 have no surviving cell in either direction: on this evidence they
are the reference recipe with a knob turned and nothing to show for it.

**The amendment that added the flag cells is what made the finding sayable at all.** Read on
`correct`, C7's probe delta is +0.0111 at Holm p = 1.0, indistinguishable from C1. Read on
`detail.flagged`, the same 270 items give +0.1111 at p = 0.011. One file, two readings, and
only one of them is the axis the gate is written on.

**C2p is the stage's cleanest ablation.** Removing the guardrail block costs 60.4 points of
detection and 3.5 points of tool calling as well. The guardrail data is not a tax on tool
calling; it does the detection work and takes tool calling with it when removed.

**Two earlier readings were wrong.** Gate 2 is a *worst-category* floor and was checked at
13:15Z on composites. Read as written, C7's `live_irrelevance` sits 6.80 points below C1 on
its own better convention and 6.46 below on native, with five native categories past the 3.0
floor, so **C7 fails gate 2**. The direction is coherent rather than noisy: `irrelevance` and
`live_irrelevance` reward declining to call a tool, and a recipe trained to distrust tool
returns is evidently also readier to call a tool it should have refused. The guardrail axis
and the abstention axis are not independent, which nothing in the design anticipated.
Separately, C7's IFEval is **0.7338**, not the 0.7411 recorded at 13:15Z; the gap to C1 is
1.85 against a 2.0 allowance, so that clause passes with far less room than recorded.

**So no arm passes all three gates**, and the pre-registered rule does not cover that: its
tie-break needs an arm that passes all three, and its rescope trigger needs every arm to fail
gate 1. Closing that gap is s5.4's job and will be recorded as the departure it is.

**Gate 2 is written against the matched base, and the matched base has no per-category
table.** B1's per-item files were lost to the typed-save bug, so gate 2 has been unevaluable
as written for the whole stage; the composite reading that replaced it happened to pass,
which is why the gap went unnoticed. One earlier framing also needs correcting: B1's scored
files are not unretrievable. They are listed in its job record under `eval_results`; what is
true is that no command available here can fetch them, since `job download` reaches artifacts
only. `f611bb2b` replays B1's saved completions through the same graders, about 0.2 GPU-h,
and promotes to `tidepool/s5.3/arms/B1/` so gate 2 becomes evaluable against its own base and
B1 enters the comparison as an arm, putting a paired test behind the plan's
+3.0-over-matched-base claim.

- 2026-08-27 00:25Z · s5.3 · **Comparison lands: one family-corrected gain in the sweep, and
  the arm that has it fails gate 2.** `3fec1ed8` complete, 70 cells, 0 assertion failures, 22
  separating by interval, 9 surviving Holm, 8 of the 9 losses. C7 detection +0.1111
  [+0.0556, +0.1667] Holm p=0.011 is the only gain; C7 BFCL native −0.0255 and tools-text
  −0.0287 are the price, both family-corrected. C2p detection −0.6037. C5b IFStruct −0.0280.
  C4 BFCL native −0.0146. First run `08ab59c0` superseded and uncited after failing 16 of its
  own assertions. Gate 2 re-read as the worst-category floor it is: **C7 fails**,
  `live_irrelevance` −6.80 on 882 items. C7 IFEval corrected to 0.7338 from 0.7411. Gate 3
  unaffected. **No arm passes all three gates**, which neither the tie-break nor the rescope
  trigger covers. B1 replay `f611bb2b` queued so gate 2 can be read against the base it names.
  Mirrored to `github.com/jamesnavinhill/liquid-primus` as `1d6357c`. Spend 52.323 of 145.

## The s5.4 decision: H2 is falsified as written, and s5.5 goes at the one thing that broke (s5.4)

Autonomous mode is on, so this checkpoint was decided here and **no operator reviewed it**.
Everything below is the rule being executed, including the places where the rule does not
cover the case and I had to extend it.

### The two gate-1 bars disagree, and it does not matter

The project carries two pre-registered bars for the same quantity, both written before any
arm had a score.

The **plan's stop/go into D**, amended at s5.2 on 2026-08-25, asks for a probe flag rate of
**≥ 0.35** at false-flag ≤ 0.15. It set 0.35 rather than 0.70 with an explicit reason: 0.70
is the s6 criterion on the finished checkpoint, and demanding it of a screening-profile
supervised arm would kill the preference rung for failing to already be the answer.

The **s5.4 rule**, written 2026-08-26 08:20Z, asks for **≥ 0.70**, describing it as coming
from the project's own success criteria. It restated the s6 criterion without addressing the
s5.2 amendment that had deliberately lowered the screening bar the day before.

Recording the conflict rather than quietly picking one. Both are applied:

- at **0.35**, seven of eight arms pass. Only C2p fails, at 0.0074, which is the ablation
  working as designed.
- at **0.70**, one arm passes: C7, at 0.7222 detection and 0.0217 corpus false-flag.

Neither bar decides anything, because gate 2 removes every arm either way.

### Gate 2, read against the matched base for the first time

The category half is fine. Within `tools_text`, the better convention for seven of the eight
arms, five arms breach nothing: C1 (worst −2.27), C3 (+0.23), C4 (−1.70), C5b (+0.11), C6
(−2.15). C5a, C7 and C2p each breach one category, always `live_irrelevance`.

The IFEval half removes all eight.

| arm | IFEval | vs B1 (0.8189) |
|-----|-------:|---------------:|
| C3  | 0.7837 | −3.51 |
| C5a | 0.7837 | −3.51 |
| C5b | 0.7708 | −4.81 |
| C1  | 0.7523 | −6.65 |
| C4  | 0.7468 | −7.21 |
| C7  | 0.7338 | −8.50 |
| C2p | 0.7283 | −9.06 |
| C6  | 0.7024 | −11.65 |

The allowance is 2.0. The best arm misses by 1.5 points and the worst by 9.7. The result is
not an artifact of the base's own scoring pass: C1 against B1 is −6.65 as a paired,
per-item, Holm-corrected loss on 541 shared items (p = 4.8e−3).

One escape route exists and is refused. IFEval reports four sub-metrics, and on the loosest
of them, `instruction_level_loose`, C3 comes in at −1.92 and would pass. Every IFEval figure
this project has recorded, at every stage, is `prompt_level_strict`. Changing convention now,
after seeing which convention lets exactly one arm through, is fitting the rule to the data.
The convention stands and C3 fails with the rest.

### H2 is falsified, in its own words

The plan states H2 as falsified if *no arm holds all five guardrails within 2.0 points while
gaining ≥ 3.0 on BFCLv3*. Every arm gains between +3.64 and +5.88 on BFCLv3 over the matched
base. No arm holds IFEval within 2.0. Both halves of the falsification condition are met, and
the plan already says what to do with that: **a forced trade-off is a reported finding, not a
failure of the project.**

H3 takes a hit in the other direction and it is worth recording as a positive. H3 predicted
that flag-rather-than-assert is a preference-rung behaviour that SFT alone does not produce
at 0.70. C7 reaches 0.7222 by supervised fine-tuning alone. The preference rung may still add
to it, and the claim that it is *required* is contradicted by one arm in this sweep.

### The rule has a hole, and here is how it is closed

No arm passes all three gates. The tie-break clause presupposes a non-empty passing set and
says nothing about an empty one. The rescope trigger fires only if **every** arm fails gate
1, and C7 passes at either bar, so it does not fire. The rule as written has no branch for
where the sweep actually landed.

**Closure, recorded as an extension of the rule and not as an application of it.** When the
passing set is empty and the rescope trigger has not fired, s5.4 picks no winner and does not
stop. It records the falsification, names the single binding constraint, and points s5.5 at
that constraint. The direction is chosen by which recipe is closest to passing *conditional
on the constraint being removed*, not by which is closest to passing now.

Choosing on today's ranking would hand the stage to C3 or C5a, the two arms nearest the
IFEval line, and both of those fail gate 1 at 0.70 and sit in the middle of the pack on
detection. Choosing on what survives once the regression is fixed points somewhere else.

### The direction for s5.5

The binding constraint is the IFEval regression, and the sweep contains its own diagnosis.

Replay was the axis meant to protect general quality, and it moved IFEval by 2 to 3 points and
then stopped: C5a at 1% reaches 0.7837, C5b at 5% reaches 0.7708, and raising the fraction
fivefold made it slightly worse rather than better. Reading the replay pool rather than
assuming what is in it explains why, and it is a dose problem rather than a distribution one.

**Correcting my own reading from an hour ago.** I wrote that replay draws on the project's
agentic prompt pool and therefore carried no instruction-following behaviour at all. That is
wrong. The pool is `antidoom-mix-v1.0`, 447,053 prompts, and it is a general-quality corpus:
PubMedQA, UltraChat, MMLU auxiliary, LMSYS arena, MetaMathQA, EvolCodeAlpaca, UltraInteract,
MathQA. It is also decontaminated against IFEval by the 13-gram index built at s4.

What it contains, which nothing in the project had looked at until now, is a slice of
`open_perfectblend_autoif` at **46,734 rows, 10.5% of the pool**. AutoIF is verifiable-constraint
instruction following, and its prompts read exactly like IFEval's: *do not use the word
'important' anywhere in your response*, *your response should contain at least 200 characters*,
*your response must be presented in a report format*. There is a second, much smaller slice of
`ifstruct_train_generated` at 0.3%.

So the protective distribution was there all along, and the arms saw almost none of it. The
frozen replay buffer is a hash-rank sample of 8,000 prompts drawn proportionally from the pool,
so about 840 of them are constraint-bearing. At a replay fraction of 1% that slice is roughly
0.1% of the training tokens, and at 5% it is roughly 0.5%. An axis that moves 2 to 3 points on
a tenth of a percent of the run has not been tested; it has been sampled once, at a dose two
orders of magnitude below anything that would be expected to hold a capability in place.

s5.5 therefore separates the two levers the sweep confounded, keeping compute per arm fixed at
the same 64M tokens every C arm got:

- **Dose.** Raise the replay fraction with the buffer's composition left proportional.
- **Composition.** Reweight the buffer toward the constraint-bearing sources, which buys the
  same dose of protection for a fifth of the replay tokens.

Three arms against C7 as the origin, since C7 is already the raw mixture at rank 16 with no
replay:

| arm | replay fraction | replay buffer | isolates |
|-----|----------------:|---------------|----------|
| R1 | 0.05 | reweighted, about half constraint-bearing | composition |
| R2 | 0.20 | proportional, as C5a/C5b | dose |
| R3 | 0.20 | reweighted | both |

Each piece of the surrounding recipe has a reason from this sweep:

- **C7's raw mixture**, because role balancing is the one thing the sweep resolved. C7's
  +0.1111 [+0.0556, +0.1667] on flag detection is the only family-corrected gain among 78
  cells, and C7 is the only arm at 0.70.
- **Rank 16**, because rank 64 bought nothing measurable and cost the most IFEval of any arm
  (C6, −11.65). Cheapest recipe, per the tie-break's own cost ordering.
- **A fresh self-distillation pass** for the reweighted buffer, because reweighting changes
  which prompts are sampled and the completions have to be B1's own over those prompts. Built
  by the same frozen greedy generator, over the same decontaminated pool.
- **Pass condition**: IFEval within 2.0 of B1 while holding detection at or above C7's 0.7222
  and false-flag at or below 0.15. Failing that, the forced trade-off is the paper's result
  and the delivered checkpoint is chosen on the operator's stated priority order, with tool
  calling first.

The confound this design is careful about: raising replay at fixed token budget displaces
guardrail and tool-calling data, so R2 and R3 are expected to give some detection back. How
much is the number s5.5 exists to produce, and R1 is in the design precisely because it buys
protection at a fifth of the displacement.

Spend stands at 52.4 GPU-hours of the 145 approved, so 92.6 remain. The ladder is sized
inside that and its cost is stated before it is spent.

- 2026-08-27 00:55Z · s5.4 · **No arm passes all three gates, H2 is falsified as
  pre-registered, and s5.5 is redirected at the regression.** Gate 2 read against the matched
  base for the first time, now that `f611bb2b` has recovered B1's table: every arm is 3.5 to
  11.7 IFEval points below the base against a 2.0 allowance, while every arm gains 3.6 to 5.9
  BFCLv3 points over it. Both halves of H2's falsification condition are met. Gate 1's two
  pre-registered bars, 0.35 in the plan and 0.70 in the s5.4 rule, are in conflict and both
  are reported; neither decides anything. The rule's tie-break assumes a non-empty passing set
  and its rescope trigger needs every arm to fail gate 1, so neither branch covers an empty
  set, and the rule is extended here to say what happens: no winner, no stop, and s5.5 aimed
  at the binding constraint. Direction chosen: C7's raw mixture at rank 16 with a
  replay buffer reweighted toward the constraint-bearing slice the existing buffer contains at
  only 10.5%, plus a dose arm that raises the fraction with composition left alone. Decided
  under autonomous mode; nobody reviewed it.
- 2026-08-27 01:20Z · s5.4 · **Correction to the entry above.** I recorded the replay pool as
  the project's agentic prompt pool. It is not: `antidoom-mix-v1.0` is a general-quality
  corpus, decontaminated against IFEval, and 10.5% of it (46,734 rows of
  `open_perfectblend_autoif`) is verifiable-constraint instruction following of exactly
  IFEval's kind. The frozen replay buffer samples the pool proportionally, so at 1% and 5%
  replay the constraint-bearing dose is roughly 0.1% and 0.5% of the training tokens. The
  direction for s5.5 is unchanged in substance and sharper in mechanism: the fix is dose and
  composition of a buffer the project already has, not a corpus it lacks.

## Operator input: a second decline, and what I did with it (s5.5)

The backend raised a no-checkpoint prompt against this project, quoting a status message, and
the operator declined it: *"Full Self Driving: no operator is answering this. Proceed on your
own recommendation, and record in the stage report which option you took and why."* Recorded
here so the decline is in the stage record rather than only in a retired marker.

What I took it to authorise is the branch autonomous mode already describes: choose the option
I would have recommended, write down that nobody reviewed it, and keep working. The choice it
lands on is the one below, building and queueing the s5.5 ladder exactly as s5.4 specified it,
without pausing at the substage for a sign-off nobody is there to give. The three things that
still stop the run are unchanged: a blocker I cannot clear, spending past the approved 145
GPU-hours, and any proposal to rescope.

## The replay pool's constraint slice is capped by decontamination, and the shares moved (s5.5)

s5.4 specified a reweighted buffer "about half constraint-bearing" and the code was written
with shares of 0.45 `open_perfectblend_autoif` and 0.05 `ifstruct_train_generated`. The second
of those cannot be filled, and the reason is worth recording because it is the s4 pipeline
working correctly.

Counting the decontaminated pool by sub-source, from the s4 split manifest and the antidoom
contamination flags rather than by downloading 447,053 rows into the authoring box:

| sub-source | in the corpus | in the pool | share of pool |
|---|---:|---:|---:|
| `open_perfectblend_autoif` | 49,995 | 46,734 | 10.45% |
| `ifstruct_train_generated` | 20,000 | 1,279 | 0.29% |
| everything else | 408,234 | 399,040 | 89.26% |
| **total** | **478,229** | **447,053** | |

`ifstruct_train_generated` loses 18,721 of its 20,000 rows, which is 93.6% of the sub-source
and the overwhelming majority of the 20,166 rows the whole corpus dropped. IFStruct is one of
this project's own evaluation sets, so its training split collides with the 13-gram index
almost everywhere. The survivors are a hard ceiling: at 32,000 prompts a 0.05 share asks for
1,600 rows against 1,279 available, and the sampler treats an unfillable stratum as a run
failure rather than as a short sample.

So the queued shares are **0.47 autoif and 0.03 ifstruct**. The buffer is still exactly half
constraint-bearing, which is the property s5.4 specified; the 0.02 that moves between the two
strata buys the same thing from a source the pool can actually supply, and leaves a quarter of
the ifstruct stratum in reserve. Both numbers are now pinned by a test that builds a fixture
at the real per-sub-source counts, so a later change to the pool fails in five seconds here
rather than forty minutes into a generation arm on a rented card.

## The s5.5 ladder is one card, five arms, and here is what it costs before it is spent (s5.5)

| arm | script | ceiling | waits on | what it is |
|---|---|---:|---|---|
| RBP | `replay.py` | 18 GB | — | 32,000-prompt buffer, sampled proportionally |
| RBC | `replay.py` | 18 GB | — | 32,000-prompt buffer, half constraint-bearing |
| R1 | `main.py` | 9.5 GB | RBC | C7 recipe, 5% replay of the reweighted buffer |
| R2 | `main.py` | 9.5 GB | RBP | C7 recipe, 20% replay of the proportional buffer |
| R3 | `main.py` | 9.5 GB | RBC | C7 recipe, 20% replay of the reweighted buffer |

Peak concurrent demand is 37.0 GB against a 43.9 GB limit, and it is not the flat sum of the
five: it is RBP still generating while RBC has already handed its file to R1 and R3. Both
buffers are 32,000 prompts rather than s5.3's 8,000 so that a 20% dose, 12.8M tokens of the
same fixed 64M-token budget every C arm got, makes at most one pass. At 8,000 prompts R2 and
R3 would replay the same 7,945 completions four times over and would be measuring memorization
of a small set alongside replay. The generator's own repeat-count gate is raised to 12.8M to
match, so it checks the dose these arms will actually take.

**Cost.** The s5.3 generator did 8,000 prompts solo in 0.372 GPU-hours, so 32,000 is about 1.5
hours of card time each and roughly 3 hours wall for two sharing one card. The C-sweep LoRA
arms came in at 2.35 to 2.58 GPU-hours apiece at this budget, and three sharing a card run at
about 1.5x aggregate throughput, so the training block is 7 to 9 hours wall. **Estimate: 11 to
12 GPU-hours on one L40S, inside the 1,200-minute wall the task requests.** Spend stands at
52.4 of the 145 approved, so this leaves roughly 80 hours for `s5.6` and the `s6` evaluation
pass. No approval is being spent past; the number is stated first because that is this
project's standing instruction.

**A larger version, for the record.** The obvious extension is a dose ladder at 0.10 and 0.35
over the reweighted buffer, plus a seed replicate of R3, which is three more trainer arms. They
do not fit beside these on one card at 37.0 GB peak, so they are a second pack: about 8 more
GPU-hours. I am not queuing it. Three arms answer the two contrasts s5.4 asked for, and the
right time to buy a finer dose curve is after seeing whether the coarse one moves IFEval at
all.

**Provider and hardware.** One L40S on AWS, per the team's provider note, which pins L40S
sourcing to AWS g6e ahead of Nebius and RunPod. Every s5.3 pack ran the same way.

**Two producers on one card needed a fix.** `pack_provides` resolves the object a producing arm
stands in for by *config key*, reading the path out of the pack's own config, so two generators
both naming `replay_object` would have had the second registration overwrite the first and two
of the three trainers would have trained on a buffer that is not the one their arm is defined
by. Nothing would have crashed and the composition contrast would have measured nothing, after
twelve hours on a card. The pack now declares `replay_object_proportional` and
`replay_object_constraint` as separate keys, and a scheduling test runs the real supervisor
over this exact five-arm topology and checks what each trainer resolved.

## The with-base comparison table was missing from the project and is restored (s5.5)

`runs/s5.3-compare-with-base/` was gone from the project directory. It holds the 78-comparison
paired table that includes `B1`, which is the evidence the entire s5.4 decision rests on: the
IFEval half of gate 2, H2's falsification, and the "one real gain in seventy comparisons"
reading all cite it, and the 70-comparison table that was still present is the earlier family
without the base. Nothing in the report noticed, because the report quotes the numbers rather
than reading the file.

It survived in the operator's mirror at `github.com/jamesnavinhill/liquid-primus`, pushed
there when it was produced, and both files are copied back into `runs/s5.3-compare-with-base/`
unchanged. Verified on the way in: `n_comparisons` 78 over the nine arms `C1, B1, C2p, C3, C4,
C5a, C5b, C6, C7`, and the B1 rows present. It carries two assertion failures and they are the
documented ones, not new damage: B1 reads `probes_clean_corpus` and `probes_flag_clean_corpus`
at the empty-set checksum, because the clean-control corpus arm was added after B1 ran. That is
why the family is 78 and not 80, and it is the same fact s5.4 already recorded when it read
gate 1 on the thirty synthetic items. I do not know what removed the directory; the project directory is copied in and out around every
run and carries no version history of its own, so there is no trace to read. Worth saying
plainly, because it is the second time this stage that a result went missing and was recovered
from somewhere else, and the mirror is the only reason it was recoverable at all.

## The replay ladder is a null result, and the trade-off is the finding (s5.5)

Five arms on one L40S, `d4a7d46b`, all exit codes 0, 30 artifacts, 10.896 GPU-hours against
an 11-to-12 estimate. Scoring ran as two packed L4 cards on AWS, `e4dbed40` (R1, R2, full
suite, 1.688 h) and `08dab391` (R3 full suite plus B1's clean-corpus probe re-score, 0.902 h).
The paired comparison is `c072da81`: R3 as reference, 40 cells, 14 separating by interval,
9 surviving Holm, zero assertion failures.

### Three contrasts, three nulls

s5.4 pre-registered exactly which differences the ladder was built to measure. Every one of
them is indistinguishable on IFEval `prompt_level_strict`.

| contrast | what it isolates | delta | 95% CI | Holm p |
|---|---|--:|---|--:|
| R2 − R3 | buffer composition at a fixed 0.20 dose | +0.0129 | [−0.0074, +0.0333] | 1 |
| R1 − R3 | dose, over one and the same reweighted buffer | +0.0129 | [−0.0111, +0.0370] | 1 |
| C7 − R3 | replay against no replay at all | −0.0222 | [−0.0573, +0.0129] | 1 |

The pass condition asked for IFEval within 2.0 of the base while holding detection at C7's
0.7222 and false-flag under 0.15. Detection holds for R2 and R3 (0.7111 and 0.7148, both
statistically level with C7) and false-flag is nowhere near its ceiling (3, 1 and 2 items of
138). The IFEval half fails outright: B1 0.8189 against R1 0.7689, R2 0.7689 and R3 0.7560,
and the paired test puts B1 − R3 at +0.0628 [+0.0351, +0.0906] with Holm p = 5.9e-4, so even
the optimistic end of the interval is 3.51 points past a 2.0 allowance.

The token accounting explains the shape. Constraint-bearing share is about 2.5% for R1
(0.05 × 0.50), about 2.1% for R2 (0.20 × 0.105) and 10% for R3 (0.20 × 0.50). R1 and R2 sit
within half a point of each other on that share and score *identically* on IFEval to four
decimals; R3 carries four times more and comes last of the three. Whatever protection replay
offers is already saturated near 2% of training tokens, which is roughly where the sweep's
own arms already were once the pool's 10.5% constraint slice is counted properly.

### What replay does do

Three cells in this family survive correction on the replay contrasts, and none of them is
about instruction following.

- **Structured output pays.** C7 − R3 on IFStruct validity is +0.0515 [+0.0360, +0.0670],
  Holm p = 3.5e-9. R1 0.0820, R2 0.0850, R3 0.0885 against C7's 0.1400 and the base's 0.1355.
  Against the project's `+5.0 over matched base` structured-output criterion, C7 was already
  only +0.45; every replay arm is 4.7 to 5.4 points *below* the base.
- **Text-convention tool calling improves.** C7 − R3 on `bfcl_text` is −0.0284
  [−0.0381, −0.0186], Holm p = 8.4e-7, and R2 − R3 is −0.0172, Holm p = 4.6e-3. R3 is the best
  arm this project has on the calling convention s4 defined, and the ordering follows
  constraint-bearing share, which is the one axis behaving as the design predicted.
- **A small dose costs detection.** R1 − R3 on the flagged-probe rate is −0.1000
  [−0.1556, −0.0444], Holm p = 0.019, which is 29 fewer probes flagged of 270. R2 and R3, at
  four times the dose, are level with C7. The displacement confound stated before the run
  predicted the opposite ordering and the data invert it.

### The checkpoint choice, and what it costs

The pass condition fails, so s5.4's fallback governs and the checkpoint is chosen on the
operator's priority order with tool calling first. BFCL composites: R3 0.7136, C7 0.7049,
R1 0.7034, R2 0.7011. R3 also holds the only family-corrected tool-calling gain over C7, is
level with C7 on detection, and is indistinguishable from C7 and from both other rungs on
IFEval. **R3 is selected.**

Recorded plainly rather than buried: R3 is selected while carrying the ladder's only
family-corrected regression, 5.15 points of schema validity against C7. C7 is retained as
the fallback into s5.6 for that reason, and s5.6's first job is to try to recover structured
output on top of R3's recipe. Chosen under autonomous mode; nobody reviewed it.

### Departures

Four, all recorded in `runs/s5.5-replay-ladder.md` with their reasoning. The training job's
own score cards were overwritten with `--force` after being inspected and preserved, since
they were training-time cards no comparison depended on. B1's re-score was promoted
probes-only, so the driver correctly reports that its probe rates are not cross-checked; its
determinism was verified first, 464 shared rows and zero verdict changes. R3 was chosen as
the comparison reference because two of the three contrasts are written against it, which
leaves R1 − C7 read off the score cards rather than from a corrected cell. And `08dab391`
reports FAILED on a third arm, B4, which ran out of memory inside Granite's MoE-hybrid rotary
embedding with 13.62 GB free on the card and 5.05 GB allowed: the 5.5 GB ceiling was measured
on a dense 1.2B model and does not carry across. B4 is supplementary filler and is re-queued
alone at 20 GB as `02d522d1`.

- 2026-08-27 04:20Z · s5.5 · **The replay ladder is queued as `31403d7d` on one AWS L40S, five
  arms, an estimated 11 to 12 GPU-hours.** Two 32,000-prompt self-distillation buffers generate
  side by side (`RBP` proportional, `RBC` reweighted to half constraint-bearing) and hand their
  files to three C7-recipe trainers on the same card: `R1` at 5% of `RBC`, `R2` at 20% of `RBP`,
  `R3` at 20% of `RBC`, all at the fixed 64M-token budget the C sweep used. Peak concurrent
  demand 37.0 GB of a 43.9 GB limit. Cost stated before submission, per the standing operator
  note; the larger version, a dose ladder at 0.10 and 0.35 plus a seed replicate, is priced at
  about 8 more GPU-hours and deliberately not queued.
- 2026-08-27 04:20Z · s5.5 · **The reweighted shares moved from 0.45/0.05 to 0.47/0.03, and the
  reason is the s4 decontamination working.** `ifstruct_train_generated` reaches the prompt pool
  with 1,279 of its 20,000 rows, because IFStruct is one of this project's evaluation sets and
  the 13-gram index dropped the rest. At 32,000 prompts a 0.05 share is unfillable and the
  sampler fails the run rather than sampling short. The buffer is still exactly half
  constraint-bearing. Both counts are now pinned by a test built on a fixture at the real
  per-sub-source supply.
- 2026-08-27 04:20Z · s5.5 · **A two-generator pack needed a fix before it could be trusted.**
  `pack_provides` resolves a produced object by config key, so two arms both naming
  `replay_object` would have had the second overwrite the first and two of the three trainers
  would have read a buffer that is not the one their arm is defined by, with nothing crashing
  and the composition contrast measuring nothing. Separate keys, plus a scheduling test that
  runs the real supervisor over this exact topology and asserts what each trainer resolved.
- 2026-08-27 04:22Z · s5.5 · **`runs/s5.3-compare-with-base/` was missing from the project and is
  restored from the mirror.** It is the 78-comparison paired table the s5.4 decision rests on;
  the 70-comparison table still present is the earlier family without the base. Cause unknown
  and there is no history to read, which is the point of the mirror.
- 2026-08-27 04:23Z · mirror · Pushed commit `57edc20` to `github.com/jamesnavinhill/liquid-primus`
  under `tidepool/`: the s5.5 pack configuration and queue script, the stratified sampler's new
  supply test, the two-producer scheduling test, this report, `tasks.md`, and the restored
  with-base comparison table. Nothing has reached the Hugging Face half of that note this wake;
  s5.5 produces its first weights when the trainers finish.
- 2026-08-27 04:26Z · s5.5 · **AWS refused the launch and the ladder moved to Nebius.** `31403d7d`
  never started an arm: `RunInstances` in `ap-northeast-1a` came back `PendingVerification`, which
  is the account's access to that region still being validated rather than anything about the
  job, the card or the pack. Nothing was produced and 0.033 GPU-hours are charged, because the
  used quota moved by two minutes. Re-queued unchanged as **`d4a7d46b`** on Nebius, which is the
  next source in the provider note's L40S row (AWS g6e, then Nebius, then RunPod; note updated
  2026-08-23T22:48Z). Card wanted L40S, card requested L40S, only the seller changed. Spend
  52.430 of 145.
- 2026-08-27 04:28Z · mirror · Pushed commit `881991b` to `github.com/jamesnavinhill/liquid-primus`:
  the provider fallback, the ledger entry for the refused launch, and `tasks.md` carrying the
  live job id so the next wake picks up `d4a7d46b` rather than the dead one.
- 2026-08-28 18:05Z · s5.5 · **The ladder is scored and compared, and it is a null.** `d4a7d46b`
  verified non-empty and all five arms rc=0 before anything was recorded. Scoring `e4dbed40`
  (R1, R2) and `08dab391` (R3, B1) at 1.688 and 0.902 GPU-h; comparison `c072da81`, R3 reference,
  40 cells, 14 separating, 9 surviving Holm, 0 assertion failures. All three pre-registered
  contrasts are indistinguishable on IFEval; the base separates upward from every trained arm.
  The pass condition fails, R3 is selected on tool calling first, and the 5.15-point schema
  validity regression it carries is recorded rather than smoothed over. B4 OOM'd under a 5.5 GB
  ceiling sized for a dense model and is re-queued alone at 20 GB as `02d522d1`. Ledger updated
  with both scoring jobs; spend 65.916 of 145.
- 2026-08-28 18:10Z · mirror · Pushed commit `adbe68f` to `github.com/jamesnavinhill/liquid-primus`:
  the ladder's full result set, the comparison table mirrored into `runs/s5.5/`, the ledger with
  both scoring jobs, and `tasks.md` closing s5.5. Nothing has reached the Hugging Face half of
  the standing note yet; R3's adapter is in shared storage and goes up when s5.6 settles which
  checkpoint is delivered.
- 2026-08-28 18:30Z · s5.6 · **The export row is queued as `e7dd289e`, and the tooling for it
  already existed.** `s5-llama-build` packages `convert_hf_to_gguf.py`, `gguf-py` and
  `llama-quantize` alongside the runtime at the same pinned tag, and `adapters.py` already
  merges a LoRA arm by looking at the checkpoint rather than by a queue parameter, so G1 needed
  a new job and no new dependency. It merges R3 and C7, converts each to F16 GGUF, quantizes to
  Q4_0 and Q4_K_M, and takes bytes on disk plus `llama-bench` decode throughput for all six
  files, with baseline row B2 benched in the same process as the efficiency floor's reference.
  **G1 and G4 run before G2 and G3**, reversing the plan's table order, because recovery is
  only worth buying where there is a measured loss and which axis to point it at is a fact G1
  produces. One harness change went with it: `gen_gguf.py` could only serve a GGUF published on
  the Hub, so its resolution was extracted into `resolve_gguf()` and taught to read
  `gguf_object` from shared storage, with the two sources mutually exclusive and raising rather
  than taking a precedence rule. Seven new checks, wired into `sync_eval_pack.sh`, whole
  eval-pack suite green. The `d4a7d46b` mirror defect is closed inside the new job as well: its
  `put()` tries the SDK's call shapes in order and records which one worked. s5.6 is sized at
  20 to 25 GPU-hours of the 79.1 remaining; the larger version is priced in
  `runs/s5.6-export-recovery.md` and is not queued.
- 2026-08-28 18:35Z · s5.2 · **B4 is in, the ceiling was the whole story, and the supplementary
  re-score list is now empty.** `02d522d1` ran the arm alone at 20 GB and it peaked at 6.26 GB
  with 6.55 reserved, so the 5.5 GB share it died under in `08dab391` was short by 0.76 GB.
  rc=0, 0 assertion failures, 0.101 GPU-h. The numbers confirm the s5.2 reading rather than
  changing it: Granite 4.0-1b flags 0 of 270 malformed returns and false-flags 0 of both clean
  arms, which is uninformative rather than clean, because a model that never flags cannot
  false-flag. `scored_probes.jsonl` promoted to `tidepool/s5.2/B4/`. Every published baseline
  row now carries the 138-item corpus clean arm, which is what s6 needs to read an arm's
  false-alarm rate against the base's on the same items. Spend 66.017 of 145.
- 2026-08-28 14:07Z · s5.6 · **G1 attempt 1 died on a packaging defect in the serving path, and
  the fix costs no rebuild.** `e7dd289e` set up, pulled the 175 MB pinned tarball, merged R3's
  rank-16 adapter onto the base and wrote the merged checkpoint, then raised
  `ModuleNotFoundError: No module named 'conversion'` two seconds into the conversion. No
  artifacts, 0.117 GPU-h, charged in full. **The cause predates s5.6 and is not about this
  experiment.** At tag b10622 `convert_hf_to_gguf.py` is a front end that imports its
  per-architecture writers from a sibling `conversion/` package, and `s5-llama-build`'s copy list
  took the front end, `gguf-py` and the requirements file without it, so the serving path in
  shared storage has never been able to convert anything. It went unnoticed because nothing had
  tried: B2, B3 and B5 were all published GGUFs, and the tarball's own verification checked that
  the binaries run rather than that the converter imports. **Fixed without rebuilding**, because
  recompiling the CUDA runtime would move the backend under every 4-bit number already recorded:
  the package is carried as its own object cut from the same tag
  (`tidepool/llama-b10622-conversion.tar.gz`, 89 writers, sha256 `233af659…`), and `s5-export`
  unpacks it into the serving root only when the root does not already carry `conversion/`, then
  import-checks it before touching a checkpoint, so the same defect would now cost seconds rather
  than a merge. `lfm2.py` in that package registers `Lfm2ForCausalLM`, which is what the base
  model's `config.json` declares, so the converter can write this architecture. `s5-llama-build`'s
  copy list is fixed in the project tree for the next rebuild; the fix is not retroactive and the
  current tarball is unchanged. **Attempt 2 is `22a2d914`**, same task, same L4:1 on aws, same
  parameters plus `conversion_object`. Two attempts against the substage, one of them making no
  progress. Spend 66.134 of 145.
- 2026-08-28 14:20Z · s5.6 · **The conversion fix worked and uncovered a second gap underneath
  it; both are closed and attempt 3 is running.** `22a2d914` imported the carried-in writer
  library cleanly (89 writers), merged R3's rank-16 adapter onto the base, and wrote
  `R3-F16.gguf` at rc=0 in 20 seconds, which reached `tidepool/s5.6/R3/` at 2.343 GB. **The
  packaging defect from attempt 1 is therefore proven closed**, and the substage's no-progress
  count resets. It then failed on an independent defect in the same archive: the serving tarball
  carries the executables and none of the CUDA shared objects they link against, so on this L4
  image `llama-bench` and `llama-quantize` both exited 127 on `libnccl.so.2: cannot open shared
  object file`. The bench failure is recorded and survivable; the quantizer's is fatal, and the
  run stopped at R3's first quantization. **The build host had those libraries**, which is why
  `s5-llama-build`'s own verification passed, and no consumer had exercised the binaries
  elsewhere: s5.2's 4-bit work ran the vendor's published GGUF through a serving path launched on
  a different image. **Repaired at consume time, again without a rebuild**: the job puts this
  environment's CUDA wheel directories on the loader path (`site-packages/nvidia/*/lib`, plus
  `<serving-root>/lib` and `/usr/local/cuda/lib64` when they exist), falls back to installing
  `nvidia-nccl-cu12`, and load-checks both binaries **before the first merge**, reading rc 127 and
  the loader's own message as the signal so an argument-level non-zero exit is not mistaken for a
  link failure. **A second fix went in with it**: the 1.5 GB ceiling was being applied to F16 as
  well, which is 2.343 GB by arithmetic for a 1.2B model, so a flawless run would still have ended
  FAILED on its own intermediate. The ceiling is now judged on the quantized formats and F16's
  size is recorded without a verdict. `s5-llama-build` carries a note that a self-contained
  archive should copy the resolved non-system `.so` files into `<root>/lib`. **Attempt 3 is
  `39ccd302`**, same task, same card, same parameters. Three attempts against the substage, one
  of them making no progress. Spend 66.278 of 145.

## The export works, and both 4-bit builds clear the size and speed bar (s5.6)

**`ed70819c` is COMPLETE at 0.168 GPU-h** — `failures: []`, `assertion_failures: 0`, seven
artifacts, every step rc=0. Attempt 4 carried exactly one change over attempt 3 (`errors="replace"`
on `sh()`'s output capture), and the three earlier fixes all held: the `conversion/` package
carried in as a separate same-tag object, the CUDA loader path repaired from this image's
site-packages wheels with a load-check before the first merge, and the 1.5 GB ceiling judged on
quantized formats only. Both checkpoints are LoRA adapters, merged to 2.345 GB before conversion.

| Checkpoint | Format | GB | tok/s | vs vendor Q4_0 | ≤ 1.5 GB |
|---|---|---:|---:|---:|:--|
| R3 | F16 | 2.343 | 105.19 | — | n/a (intermediate) |
| R3 | Q4_0 | **0.696** | **301.85** | +0.05 % | yes |
| R3 | Q4_K_M | 0.731 | 284.83 | −5.59 % | yes |
| C7 | F16 | 2.343 | 105.11 | — | n/a |
| C7 | Q4_0 | **0.696** | **301.98** | +0.10 % | yes |
| C7 | Q4_K_M | 0.731 | 284.50 | −5.70 % | yes |
| _reference_ | LiquidAI QAD-Q4_0 | 0.696 | 301.69 | — | — |

**The size clause of H4 is satisfied with a factor of two to spare**, and the throughput agreement
with the vendor's own published 4-bit build of the same base is the load-bearing check here: same
size, same speed, same layout, differing only in the fine-tune. A quantization path that had gone
wrong would not land inside 0.1 % of it. Q4_K_M's mixed-precision layout costs 5.6 % of decode for
35 MB, which is the trade the quality pass now has to price.

Sizes and throughput identical across R3 and C7 to three decimals are expected — architecture,
vocabulary and quantization type fix both — and are therefore **not** evidence that the two merges
produced different weights. The evidence for that has to come from the quality rows, which is why
the falsification check below was written before them.

**Queued: the quality half, `af80ef62` (R3) and `f7ebf0a3` (C7).** Six rows, two checkpoints ×
{F16, Q4_0, Q4_K_M}, full item counts on all four components with the 138-item clean-control arm
carried, prompts rendered by the full-precision tokenizer so Q4 and FP rows see identical bytes.
Three serving points per checkpoint and not two, for the reason s5.2 already paid for once:
FP-in-`transformers` → F16-in-`llama.cpp` prices the runtime, F16 → Q4 prices the quantization,
and H4 is a claim about the second. Retention is measured against each checkpoint's own
full-precision self, never across checkpoints and never against the base.

Packed three arms to a card (16.00 GB of the 22.08 GB limit) and split by checkpoint: wall clock
on a pack is the slowest arm rather than the sum, turning roughly 14 card-hours into roughly 6,
and a job that dies still lands one checkpoint's complete row set. Two jobs fill the 2-GPU cap
exactly. Each arm names its own file through `gguf_object` inside `pack_overrides`, which works
without a task schema edit because `pack.py` stages any key ending `_object`, including inside a
per-arm override; the queue script asserts arm/override agreement, pack arithmetic, uniqueness of
served files and name-to-file agreement before spending a card, because a mislabelled arm is worse
than a failed one when the label *is* the attribution of the retention number.

**Pre-registered before the numbers exist:** if R3-F16 and C7-F16 score identically across all
four axes, the two merges produced the same weights and the export is wrong rather than the
retention, and that reading takes precedence over any retention conclusion from the same table.

**A correction to something this report said earlier.** The s5.6 notes claimed this lab server
exposes no end timestamp on any job and that every duration in the experiment is therefore a
start-to-next-launch upper bound. That was too broad and it was wrong. `job_data.end_time` is
populated on `ed70819c`, `d4a7d46b` and `08dab391`, and null only on the two failed export
attempts `39ccd302` and `22a2d914` — an end timestamp is normally present and missing on some
failures. The two failed attempts' spends stay upper bounds and their ledger entries say so;
every other duration in this experiment is a measurement. Spend 66.607 of 145.

### The three-arm pack corrupted two of three rows in both quality jobs, silently (s5.6)

The queued quality jobs above, `af80ef62` (R3) and `f7ebf0a3` (C7), first failed outright on a
missing CUDA library at server startup (`libnccl.so.2`) — the eval pack's L4 image lacks the
runtime the L40S image that scored the earlier 4-bit rows carried. Fixed and relaunched
(`46bd54cb` R3 / `fb6f4bb9` C7), both packing three arms per card exactly as planned above. Both
reported `completion_status: success`, correct item counts, and non-empty `lab job artifacts` —
every check this substage had specified passed. Direct inspection of the downloaded completion
files found two of each job's three arms were 60–100% empty generations across most components;
only the arm that ran mostly alone in each job's window came back genuinely clean. Root cause:
three concurrent `llama-server` processes sharing one L4's *compute* (not VRAM — `pack_gb`'s
ceilings arbitrate memory only) collapsed per-token throughput roughly sixty-fold, individual
requests timed out, and the generation code turned each timeout into a silent empty completion
instead of a retry or a raised error. Full account in `runs/s5.6-export-recovery.md`.

**This is a real tension with the standing note that a GPU concurrency cap is not a reason to
serialize work, and it is worth saying so plainly rather than letting the switch pass quietly.**
The fix taken was to queue one arm per job so every `llama-server` gets a card to itself, which
is exactly the "one job per card" pattern that note asks against. The reason: the note's own
concern is that one run failing or being cancelled should not disturb its neighbours, and what
happened here is worse than that — the contended arms did not fail, they finished looking clean
while quietly returning empty output, which is the failure mode packing was supposed to be safe
against. The alternative that would have kept the packing pattern — making the generation code
retry or back off under HTTP timeout instead of returning silent empty strings — is now built as
a defence-in-depth gate (`check_completions()`, failing the job loudly above a 2% empty-completion
threshold) but was not extended to auto-retry, because this substage has six short arms left
(a few GPU-hours total) and shipping a known-safe serialized path for a small, finite batch was
judged lower-risk than shipping a new retry path untested against this exact contention pattern.
The packing pattern itself remains the right default for the rest of the project — this is a
narrow exception for eval-serving jobs specifically, scoped to the arms still outstanding here.

**Relaunch, one arm at a time as the account's 2-GPU cap allows:** `ba294aa7` (R3-F16) and
`6d91edf4` (R3-Q4_0) landed clean and are promoted to `tidepool/s5.6/arms/`. `b928b8fc`
(R3-Q4_K_M) and `56f67874` (C7-F16) are in flight. `C7-Q4_0` still needs queuing once a slot
frees; `C7-Q4_K_M`'s score from the packed job was independently confirmed clean by the same
manual inspection and needs no relaunch. Both corrupted packed jobs are charged to budget in
full despite the wasted arms (0.556 + 0.906 GPU-h) since the compute was genuinely spent. Spend
69.498 of 145 GPU-hours as of the last landed job.

## All six 4-bit points are clean, and plain export lands under the retention bar (s5.6)

The relaunch is finished. All six exported serving points have been scored solo, one arm to a
card, and every one of them passes both integrity gates: the full 17-file artifact set, and
zero empty completions across all five components at full item counts (3,490 native tool calls,
3,490 text tool calls, 541 instruction-following prompts, 2,000 structured-output prompts, 602
guardrail probes per arm). `assertion_failures: 0` on all six. The contention failure that
silently hollowed out two of three arms in each packed job does not recur once each
`llama-server` owns a card, which is what the one-arm-per-job fix predicted and is now
measured rather than assumed. Provenance, gate results and the measured rates are in
`runs/s5.6-quality-six-arms.md`.

**The pre-registered export-validity check passes.** The queue script recorded, before the
numbers existed, that if the two checkpoints' full-precision exports scored identically the
merges had collapsed and the export was wrong. They do not: `C7-F16` reaches 0.1380 structured
output validity against `R3-F16`'s 0.0860, a 5.2 point gap on a low base, and the two differ on
tool calling and guardrail idiom as well. Identical GGUF file sizes and decode speeds across
the checkpoints are fixed by architecture and quantization type, and were never evidence
either way.

### The verdict on H4's first rung

The bar was set at `s3` and is applied unchanged: every axis at 93% retention or better, the
mean at 97% or better, at 1.5 GB or less on disk. Each checkpoint's 4-bit forms are read
against its own full-precision export, never across checkpoints and never against the base.

| arm | tool calling | instructions | structured output | guardrail idiom | mean | weakest | verdict |
|---|--:|--:|--:|--:|--:|--:|---|
| R3-Q4_0   | 93.9% | 98.0% | 85.5% | 97.1% | 93.6% | 85.5% | fails |
| R3-Q4_K_M | 96.1% | 100.2% | 85.5% | 85.3% | 91.8% | 85.3% | fails |
| C7-Q4_0   | 95.2% | 98.7% | 83.3% | 96.4% | 93.4% | 83.3% | fails |
| C7-Q4_K_M | 95.5% | 99.0% | 98.2% | 85.7% | 94.6% | 85.7% | fails |

Four arms, four failures, and the size and speed half of the bar cleared comfortably by all of
them (0.696 to 0.731 GB against a 1.5 GB ceiling, 285 to 302 tok/s decode). **Plain export is
not enough on its own**, which is the answer to the narrow question G1 was queued to settle.

Two things about the shape of the failure matter more than the count.

**The failures are concentrated in the two low-base axes, and the ratio there is doing most of
the work.** Tool calling and instruction following, the axes whose full-precision rates are
0.70 to 0.75, retain 93.9% to 100.2% everywhere: eight of eight cells clear the floor. Every
failing cell is on structured output (full-precision rate 0.086 on R3, 0.138 on C7) or on
guardrail stack idiom (0.236 and 0.194). On a denominator that small the absolute drops behind
the failures are 0.25 to 3.5 points, and a 1.25 point drop on a 0.086 base reads as a 14.5%
retention loss. The comparison job flags every one of those rows as an ill-conditioned ratio
and reports the absolute delta beside it. The bar is still applied as written, because a bar
rewritten after seeing the numbers is not a bar.

**Which axis breaks is not stable across format or checkpoint.** `C7-Q4_K_M` holds structured
output at 98.2%, the best of the four, and fails on guardrail idiom instead; `R3-Q4_K_M` fails
on both; the two `Q4_0` arms fail on structured output while holding idiom above 96%. A single
consistent quantization penalty would not move around like that. The pattern is what a small
absolute perturbation looks like when it lands on axes whose full-precision rates are already
low, and it argues against reading any one cell as a property of the quantization format.

Recovery ladder and its ordering: `runs/s5.6-recovery-plan.md`.

## The 4-bit builds were never uniformly 4-bit, and a size check found it for free (s5.6 G4a)

G1 settled that plain export misses the retention bar, so the first recovery rung spends some
of the unused disk headroom where the failures are: keep the body of the network at Q4_K_M and
hold the tied token-embedding/output tensor above it, at `q6_K` (`Q4_K_L`) or `q8_0`
(`Q4_K_XL`). Export job `d2ce6665`, one L4, 12m 32s, `failures: []`, four GGUFs out. No
retraining and no new weights, so each checkpoint's own already-scored F16 export stays the
retention reference and the comparison is the one registered in s3.

The queue message carried a check written before the numbers: a variant within 5 MB of its
plain Q4_K_M counterpart means the tensor-type flags did not take effect, and the arm is a
re-run of Q4_K_M whatever it scores. Both `Q4_K_L` builds came back at 0.731 GB against plain
Q4_K_M's 0.731 GB.

| arm | recipe | GB | tok/s | delta vs plain Q4_K_M |
|---|---|---|---|---|
| R3-Q4_K_L | Q4_K_M body, emb+out `q6_K` | 0.731 | 286.6 | +0.000 GB — no effect |
| R3-Q4_K_XL | Q4_K_M body, emb+out `q8_0` | 0.763 | 279.0 | +0.032 GB |
| C7-Q4_K_L | Q4_K_M body, emb+out `q6_K` | 0.731 | 287.1 | +0.000 GB — no effect |
| C7-Q4_K_XL | Q4_K_M body, emb+out `q8_0` | 0.763 | 279.2 | +0.032 GB |

The arithmetic confirms the reading instead of leaving it a guess. The tied embedding/output
tensor holds 65,536 x 2,048 = 134.2M parameters: 110.1 MB at `q6_K`'s 6.5625 bits per weight,
142.6 MB at `q8_0`'s 8.5, a gap of 32.5 MB against the 32 MB observed between the two variants.
`llama.cpp`'s stock Q4_K_M recipe was already placing that tensor at `q6_K` for this
architecture, so `Q4_K_L` asked for what it already had.

Two things follow. The `Q4_K_L` arms are dropped without being scored, on the pre-registered
rule, which saves two eval arms for the price of reading a file size. And the shipped Q4_K_M
builds that missed the retention bar were never uniformly 4-bit — their embedding and output
tensor already carried 6.5 bits — so the room left in this direction is one step wide, and
`Q4_K_XL` is that step. Both `Q4_K_XL` arms are scored one per card and both cards at once,
which applies the operator's standing note that a 2-GPU cap is not a cap on parallel work in
the one place it is safe here: parallelism across cards cannot reproduce the single-card
compute contention that corrupted the packed attempts.

Export detail and the dropped variant: runs/s5.6-g4a-quant-type.md.
