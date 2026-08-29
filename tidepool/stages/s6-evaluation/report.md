# Stage 6: Evaluation

_Status: `s6.1`, `s6.2` and `s6.3` complete. `s6.5` seed replication is training. `s6.4` is
under way and the story checkpoint follows it._

## Headline

**The guardrail is the result, and it is large: on tool returns that are corrupted,
contradicted or silently empty, the base model raises a flag on 0.7% of cases and both
fine-tuned candidates raise it on 73 to 78%, for 1.5 to 5 points of lost restraint on a clean
corpus.** On the text calling convention the stack actually uses, the candidates gain 12.5
points while IBM Granite 4.0 1B loses 22.9 and the Nova function-calling build loses 29.7; on
the JSON-native convention the candidates are flat and Granite leads everyone by 5.1, and that
flat cell turns out to be a cancellation: the candidates lose 8.6 to 10.5 points where a tool
call is warranted and gain 15.9 to 16.8 where declining is the right answer, so the fine-tune
shifted how readily the model reaches for a tool. Two
regressions are real and unhedged: instruction following falls 7.2 points against a 2-point
allowance, and structured-output validity falls 5.0 points for the primary candidate, though
the second candidate holds flat and Granite is 18.5 points ahead of all of us there. **Calibration then changed what the guardrail result means: the untrained base separates
defective tool returns from clean ones at 0.920 AUROC while raising a warning on 6 of 458 items, so most of what the fine-tune added on the two semantic defect types is the willingness to say so, and the near-zero flag rate on stale dates and wrong entities is a threshold rather than blindness.** Every
number is a paired item-level test against our own full-precision model, Holm-corrected across
all 74 comparisons in the family; 29 survive.

## Work log

**2026-08-29 · s6.1 · every baseline row put back on the current instrument.** `s5.3` had found
that the graders moved after `s5.2` and replayed the matched base against them, but only the
base: the four competitor and 4-bit-base rows were left on the old instrument, and `s6` is the
first place a finalist is differenced against a *competitor*. All four were re-scored from their
recorded completions, which costs no generation (job `651c0193`, 0.02 GPU-h, four arms packed on
one L4, 8 to 9 seconds each, 0 assertion failures). One row moved, by the same −0.0051 of native
tool-calling composite the base had moved; every other cell is byte-identical, so the competitor
claims the paper leans on are now *stated* as unaffected rather than assumed to be. The replay
also recovered the 11-category breakdowns, IFStruct partial credit and the stack-idiom probe,
none of which the `s5.2` table carried. Written up in `runs/s6.1-baseline-replay.md`; verdicts
under `tidepool/s6.1/baselines/<arm>/`.

**2026-08-29 · s6.1 · the held-out comparison family, one pass, 74 comparisons** (job
`d08ff6cb`, CPU only). Eight arms against the matched full-precision base: both finalists at F16
and Q4_K_M, the two 4-bit builds of the base, and both competitors. 45 cells separate by
interval, 29 survive the Holm correction, 0 assertion failures. The comparison harness was
extended to read each arm from its own storage prefix, because the family spans three of them
and copying roughly half a gigabyte of per-item verdicts into a fourth directory would have put
a second copy of every verdict in storage and made "which bytes was this drawn from" ambiguous;
the resolved prefix per arm is recorded in the summary, and a prefix naming an arm not in the
comparison is refused rather than silently ignored.

**2026-08-29 · s6.5 · seed replication, two failures diagnosed and a third attempt away.** The
project's headline rests on one training draw. Two further draws of the named recipe were
queued, varying only the seed that sets LoRA initialization and dropout masks — row order is
deterministic and unchanged by construction, so these are three draws of the initialization and
the report will not call them "three seeds" and leave it. The first attempt (`1d7ee70d`,
0.019 GPU-h) died in a minute: the pack supervisor stages shared inputs by scanning config keys
that end in `_object`, and the replay buffer's path lived only in the recipe table inside the
trainer, somewhere the supervisor never reads. It had been invisible until now because at
`s5.5` a sibling arm *generated* that buffer inside the same pack. The second (`f1d0b3df`,
0.04 GPU-h) staged correctly, reached training, and both arms died at about a minute inside
`multiprocessing`'s forkserver handshake, with the DataLoader worker resetting the connection
during the authkey exchange; two wandb runs in a single arm's console is the tell that the
worker was re-executing module scope. The cause was not isolated further and did not need to
be — workers only tokenize and collate, the loader runs with shuffle off, and the sampler is
deterministic, so the worker count cannot move a gradient. The worker count is now a parameter
and this run sets it to zero (`386f2ad7`).

**`s6.2` — decomposing every headline delta, twice.** The per-group breakdown was built into
the comparison driver rather than run as a separate analysis, since `macro_delta` already
walked the same structure. It carries its own Holm family, its own rendered table, and tests
asserting that switching it on leaves every headline delta and Holm p untouched. The first
pass (`9739c5ce`) split only the two BFCL cells, because the breakdown rode on `group_field`
and BFCL's `category` was the only one in the component table. It answered its question and
raised a bigger one, so a second pass (`792fab04`) gave the probes and structured output their
own `subgroup_field`: `mode` on tool-return probes (nine corruption types), `arm` on
stack-idiom probes (the six technologies this project was told to specialize along),
`output_format` on structured output. The two keys are separate on purpose, because
`group_field` also switches on the macro column and every comparison recorded before `s6.2`
rendered those cells without one. Three new tests hold them apart; 69 checks green. 389
subgroup cells, 108 significant in their own family, against 74 headline cells and 29. Both
passes CPU only, 0 GPU-hours, 0 assertion failures, headline cells identical across all three
artifacts. Full tables in `runs/s6.2-subgroups.md`.

**`s6.3` — giving the guardrail a score, because the flag does not have one.** The deployed
guardrail is a frozen regex family over free-form completions: one operating point, and no
confidence behind it. `s6.2` made that a live problem rather than a stylistic one, since the
flag rate is at or near ceiling on the seven corruptions that leave the payload malformed and
near zero on `wrong_entity` and `stale_as_of`, the two that do not, and a single yes/no cannot
distinguish a threshold sitting too high from a signal that is not there. The design is not
pre-registered in the plan, so it was chosen here and is recorded as such: each tool-return
probe's own messages are replayed with one frozen auditor turn appended, and the first token's
probability mass on `no` against `yes` becomes a suspicion score. One forward pass per item,
nothing sampled and nothing decoded, so the number cannot depend on a stopping rule. It yields
AUROC over the 270 defective items against the 138-item clean corpus, the same statistic per
corruption mode against that shared clean pool, detection rate at four false-alarm budgets,
ECE with reliability bins, and the scalar's AUROC against each arm's own recorded regex flags,
which is what separates "cannot see it" from "sees it and does not say it". The frozen
30-item synthetic clean arm is scored alongside but excluded from the rank statistic, so its
quartiles are available as a check on whether the corpus negatives stand in for clean traffic.
Three arms on one L4 (`0296212c`): B1 the untrained base, so a high AUROC on R3 can be
attributed to the training rather than to the base already carrying the signal, plus the two
finalists at full precision. Only the transformers backend exposes token probabilities and it
is by plan the one every full-precision number here comes from, so a 4-bit arm raises rather
than quietly reporting a curve on different numerics. Each arm reads the scored probe file the
sweep already wrote, so the scalar is ranked against the flags `s6.1` and `s6.2` published
instead of a fresh sampled pass. 28 fixture checks on the statistics and 12 on the forward
pass; the latter caught a real defect, a build without `logits_to_keep` retrying the
unsupported keyword on every batch. **The interpretive limit was written down before the
numbers landed: a teacher-forced verdict on a direct question is an easier task than
volunteering a warning mid-answer, so a high AUROC cannot license a claim about the shipped
guardrail. The inference runs one way. Chance-level AUROC would have closed the threshold
hypothesis outright and sent the remaining work to the probe corpus.** It came back the other
way, on every arm including the untrained one, in 0.018 GPU-hours and one minute of card time:
the base separates defective tool returns from clean ones at 0.9198 while flagging 6 of 458
in free text, and both blind-spot modes are recoverable at a 5% false-alarm budget. The
finding is in `runs/s6.3-calibration.md` and summarized under Results.

**2026-08-29 · s6.4 · the contamination headline was an artifact, and the corrected figure is
0.02%** (job `705a9d7d`, CPU only, fourth pass, 0 assertion failures). The third pass reported
that 11.47% of shipped training rows share a 13-word span with a scored eval item, and I passed
that on as a contamination rate. It is not one. The gram index keeps a single owner per gram, so
a span two eval sets share is credited to whichever was indexed first; re-scanned against
single-set indexes, the benchmark carrying the headline metric touches 99 of 494,341 training
rows (0.0200%), and 56,689 of the original 56,707 belong to our own probe bank on the strength
of one shared instruction sentence that sits in every tool-calling system prompt. The probe bank
contributes nothing on the target side, which is where a benchmark answer would have to appear
to be learned. **Every target-side item is now quoted rather than counted**, and none of the
three is a leaked answer: one run of consecutive integers inside a `valid_values` list, and two
English phrases inside prose code answers. The `s4.3` decontamination held. Written up in
`runs/s6.4-decon.md`. An n-gram index cannot see a paraphrased answer, so the behavioural half
runs separately as job `4bab8d12`.

## Results

Paired item-level deltas against `B1`, our own LFM2.5-1.2B-Instruct at full precision, replayed
through the current graders. **HOLM** = survives correction across all 74 cells; **ci** = the
95% bootstrap interval excludes zero but the cell does not survive the family correction.

| Arm | native tools | stack-text tools | ifeval | ifstruct | guardrail detect | clean-corpus restraint |
|---|--:|--:|--:|--:|--:|--:|
| B1 (base, absolute) | 0.6822 | 0.5504 | 0.8189 | 0.1355 | 0.0074 | 1.0000 |
| R3-F16 | −0.0060 | **+0.1252** | **−0.0721** | **−0.0495** | **+0.7333** | −0.0145 |
| R3-Q4_K_M | −0.0155 ci | **+0.1192** | **−0.0702** | **−0.0620** | **+0.7741** | −0.0362 ci |
| C7-F16 | +0.0052 | **+0.0977** | **−0.0795** | +0.0025 | **+0.7370** | −0.0145 |
| C7-Q4_K_M | −0.0201 ci | **+0.0711** | **−0.0869** | +0.0000 | **+0.7296** | −0.0507 ci |
| B2 (base, Q4_0) | −0.0163 ci | **−0.0894** | −0.0111 | +0.0015 | +0.0148 | — |
| B3 (base, Q4_K_XL) | **−0.0542** | **−0.0834** | −0.0166 | −0.0160 ci | +0.0000 | — |
| B4 (Granite 4.0 1B) | **+0.0510** | **−0.2292** | **−0.0795** | **+0.1850** | −0.0074 | — |
| B5 (Nova FC, Q4_K_M) | −0.0143 | **−0.2971** | **−0.2810** | **−0.0645** | +0.0111 | +0.0000 |

**The guardrail is the study's largest effect by an order of magnitude.** 0.0074 to 0.7333 is
not a shifted rate, it is a capability the base model does not have: nine of ten unreliable tool
returns pass without comment, and after training three of four are caught. Neither competitor
moves it — Granite *never* flags, on any probe, so its zero false-alarm rate says nothing about
discrimination. The cost is bounded and small: 1.45 points of clean-corpus restraint at full
precision, 3.6 to 5.1 quantized, against 73 points of detection.

**Tool calling splits by convention, and the split is the specialization argument.** The two
models tuned hardest for function calling are the two worst on the stack's text convention, by
22.9 and 29.7 points, while our candidates gain 12.5. On the JSON-native convention our
candidates are flat against the base and Granite leads by 5.1. Nothing here beats Granite at
native calling; what the work bought is the convention the stack speaks, and the pre-registered
tool-calling bar has to be read as two numbers rather than one.

**Two regressions, stated plainly.** Instruction following falls 7.2 points against an allowance
of 2.0, on every finalist, at both precisions — the specialization tax `s5.4` first measured and
the `s5.5` ladder reduced without eliminating. Structured-output validity falls 5.0 points for
R3; C7 holds flat, which is the whole reason it was carried, and Granite is 18.5 points ahead of
every arm in the study. Both regressions are Holm-significant and neither is a measurement
artefact.

**Quantization costs about two points of native tool calling and nothing else that separates.**
R3-F16 to R3-Q4_K_M is −0.0095 native, +0.0060 stack-text, −0.0125 ifstruct, and the guardrail
detection rate does not move. The direct paired test of that step is the second comparison
family, running.

**Which finalist ships, tested directly** (job `7725b157`, 30 comparisons, its own Holm family,
reference `R3-Q4_K_M`). Two deltas against a common base are not a test of the difference
between them, so the four finalist arms were compared against the named configuration on their
own. The trade is clean and it is the only place they differ:

| Against R3-Q4_K_M | stack-text tools | ifstruct | everything else |
|---|--:|--:|---|
| C7-Q4_K_M | **−0.0481** | **+0.0620** | indistinguishable |
| C7-F16 | **−0.0215** | **+0.0645** | native tools +0.0206 (ci only) |
| R3-F16 | +0.0060 | +0.0125 ci | indistinguishable |

**R3 stays the named configuration.** C7 buys 6.2 points of structured-output validity and pays
4.8 points on the calling convention the stack uses, and the project's own priority order puts
tool-calling and agentic reliability first. The structured-output gap is also a gap between two
failing options: 0.0735 against 0.1355 first-attempt schema validity, where a usable model would
be an order of magnitude higher, and Granite reaches 0.3205. Carrying C7 was the right call at
`s5.6` — the difference was real and it took a direct paired test to price it — and the price is
now known.

**Quantization does not damage the guardrail; it moves its operating point.** R3-F16 against
R3-Q4_K_M is indistinguishable on tool calling in both conventions and on instruction following.
On the probes the 4-bit build flags *more*: detection 0.7815 against 0.7407 and clean-corpus
restraint 0.9638 against 0.9855, both separating by interval and neither surviving correction.
A 4-bit build that is slightly more willing to raise the guardrail than its own full-precision
self is a different finding from quantization damage, and the report should not merge them.

### What the breakdown changes (`s6.2`)

**The flat native tool-calling cell is a cancellation across a decision boundary.** BFCL's
eleven categories divide into eight where a call is warranted and three where declining is
correct. Pooled within each half against the base: R3-F16 is −0.0860 / +0.1588, R3-Q4_K_M
−0.1047 / +0.1684, C7-F16 −0.0234 / +0.0640. The largest single cell is `live_irrelevance`,
n=882, at **+0.2120** for R3-Q4_K_M, and the losses concentrate in the smaller curated
categories (`simple` n=400 at −0.2200). So the finalists are less willing to reach for a tool,
which is the same direction as the guardrail and plausibly the same behaviour. **A large part
of the effect is not ours**: B2 and B3 are the base model's own 4-bit builds and show the same
shape, B3 at −0.157 / +0.157, deeper than either R3 arm. R3-F16 is full precision, so its
−0.086 / +0.159 is the training's own share, and it is most of the total.

The stack's text convention shows no such trade. **All eleven categories move up** for R3-F16,
with the biggest gains on the hardest categories: `parallel` +0.3600 and `parallel_multiple`
+0.3300, both Holm, against `live_irrelevance` +0.0204. The two conventions differ in kind, and
the improvement on the one the stack uses is general.

**Granite's text-convention score is an artifact of never calling.** It scores exactly 0.0000
on all eight call-warranted categories and 0.9989 / 1.0000 on the irrelevance ones. Its pooled
−0.2292 therefore understates the gap: 1140 of 3490 items are free points for a model that
never tries. On call-warranted items R3-F16 goes 0.5566 → 0.7115 while Granite goes to zero.

**The guardrail is a structural-integrity detector.** Across the nine corruption types it is at
0.80 to 1.00 on the seven that produce a malformed payload, and at 0.30 (`wrong_entity`) and
0.067 (`stale_as_of`) on the two that produce a well-formed one. Nothing is broken in those
returns, so a detector keyed on structure has nothing to fire on. The deployable claim is that
it catches broken returns and will not tell you a fresh-looking answer is about last quarter.

**The structured-output regression is entirely JSON.** `ifstruct` is 1000 JSON items and 1000
YAML. R3-F16 is −0.0950 on JSON (Holm) and −0.0040 on YAML; R3-Q4_K_M is −0.1080 and −0.0160.
Nova shows the same JSON-only shape. The training corpus is dense in JSON-shaped tool-call text,
which is the obvious hypothesis and one this run does not test.

**Negative finding: the stack-idiom probe cannot support its own question.** Split by
technology it gives cells of 16 to 40 items; no finalist cell separates on any of the six, and
`mcp` sits at 0.0000 for the base and for six of the eight arms. The probe was built to measure
the axis this project was told to specialize along, and at 144 items it is not powered to say
anything in either direction. Recorded as a limitation of the instrument, not as evidence that
specialization failed.


### What calibration changes (`s6.3`)

**The base model already perceives the defect.** B1 raises a free-form flag on 6 of 458
tool-return probes and ranks the 270 defective ones above the 138 clean ones at 0.9198 AUROC.
R3 reaches 0.9867 and C7 0.9762. The guardrail result stands as a behavioural result and has
to be restated as one: on the semantic corruptions the fine-tune moved the reporting, and its
perceptual gains are concentrated on the structural ones (silently_truncated 0.8355 to 0.9990,
truncated_json 0.8693 to 0.9993, type_swapped 0.9297 to 1.0000, unit_swapped 0.8512 to 0.9978).

**The two modes the flag misses are the two the base was best at.** `stale_as_of` sits at
0.9287 for the base and 0.9766 for R3 against a 6.7% flag rate; `wrong_entity` at 0.9452 and
0.9271 against 30%. At a 5% false-alarm budget the scalar detects 83% of each for R3. The
threshold hypothesis `s6.2` left open is confirmed, which retires the item `s6.2` had ranked
highest for future data work: the remaining gap on those two modes is a deployment decision.

**Ranking is good and calibration is not.** ECE is 0.4036 for B1, 0.3198 for R3, 0.2429 for
C7. B1's lowest reliability bin holds 259 of 458 items at a mean score of 0.0077 against an
observed defect rate of 0.4788. The 5%-budget cut is 0.0140 for B1, 0.1396 for R3 and 0.4687
for C7, so the useful threshold is arm-specific and nowhere near a half. No arm's raw
probability may be read as a probability; all three may be thresholded.

**The flag is a coarse read of the same evidence.** AUROC of the scalar against each arm's own
recorded regex flags is 0.8344, 0.8516 and 0.8389. High enough to identify the signals with
each other, short enough of 1.0 that the free-form pass adds noise of its own.

**The limit was fixed before the numbers landed and still holds.** Answering a direct question
about a tool result is an easier task than volunteering a warning mid-answer, so none of this
licenses a claim about the shipped guardrail on those two modes. It licenses the deployment
option, the restatement of what the training bought, and an operating-point table the model
card can carry. There is no calibration number for the named Q4_K_M configuration, because
only the transformers path exposes token probabilities and the component raises rather than
reporting a curve on different numerics.

## Outputs

- `runs/s6.1-baseline-replay.md` — why every baseline row had to be re-scored, what moved, what did not.
- `tidepool/s6.1/baselines/<arm>/` — the four replayed competitor and 4-bit-base rows, per-item verdicts.
- Job `651c0193` — the four-arm score-only replay.
- Job `d08ff6cb` — the 74-comparison held-out family.
- Job `7725b157` — the finalists against the named configuration, its own Holm family.
- `runs/s6.2-subgroups.md` — where every headline delta comes from, and what the breakdown
  changes about how three of them read.
- Job `9739c5ce` — the s6.2 subgroup breakdown, first pass, BFCL only. Superseded.
- Job `792fab04` — the s6.2 subgroup breakdown over the whole suite, own Holm family.
- Job `386f2ad7` — the seed replication, third attempt.
- `runs/s6.3-calibration.md` — the scalar behind the guardrail, per corruption mode, with
  operating points and the limits on reading them.
- Job `0296212c` — the s6.3 guardrail calibration, three arms on one card.
- `tidepool/s5.3/tooldata/clean_control.jsonl` — the 138 clean-corpus items that are the
  calibration's negatives, unchanged since `s5.3`.

## Next steps

`s6.4` error analysis, which now has four jobs: name the items the grader edit reclassified,
read the failing parses behind the JSON-only structured-output regression (the scored files
already point at a formatting-envelope failure rather than a schema one, and a queued job has
to establish that before it can be cited), look at what the model emits on the call-warranted
native items it lost, and run the decontamination re-check together with the black-box
peakedness test. `s6.5` closes when the seed replication lands. Then the story checkpoint,
which `s6.3` has changed the input to: the guardrail is still the headline behavioural result,
and the claim underneath it is now about reporting on the semantic modes and perception on
the structural ones.

A deliverable `s6.3` created and `s7` should carry: a scalar-thresholded guardrail, with the
per-arm operating points already tabulated. It costs one forward pass per tool return and
needs no new data. The threshold is arm-specific, so the model card has to publish the number
alongside the checkpoint.

Carried forward as instrument limitations rather than results: the stack-idiom probe is 144
items over six technologies and cannot resolve a per-technology claim in either direction, and
`ifeval` has no grouping field so it is the one component with no breakdown.
