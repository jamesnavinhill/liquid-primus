# Research plan — tidepool, tier 1

_Written at `s3.1`–`s3.3`. Grounded in `../s2-research-summary/report.md`; the scope it
plans against is `../../overview.md` at `literature-grounded (revised s2.3)`._

Target: a stack-specialized derivative of `LFM2.5-1.2B-Instruct` that reaches BFCLv3 overall
≥ 52.43 and ≥ +3.0 over a matched base rerun, holds every guardrail within 2.0 points, and
ships a 4-bit GGUF ≤ 1.5 GB retaining ≥ 97% of its own full-precision average with no axis
below 93%.

## s3.1 Hypotheses and approach families

Five hypotheses, each falsifiable by a named experiment and a named metric.

**H1 — Supervised specialization is where most of the win is.**
A LoRA supervised fine-tune on a repaired tool-calling and structured-output mix moves
BFCLv3 and IFStruct validity substantially without touching architecture.
*Why it might work:* the closest published precedent takes a 1B model from 9.6% to 88.9%
average schema accuracy with LoRA SFT alone and no constrained decoding
(`2505.04016v1`), and a 270M model went from 87% parse failure to under 1% through dataset
repair plus full-parameter SFT (`2603.16901v1`).
*Why it might not:* tool-use SFT gains shrink sharply below 3B (`2508.12685v3`), and one
matched study moved BFCLv3 only 60.5 → 61.4 (`2601.02151v1`).
*Measured by:* BFCLv3 overall and per-category, IFStruct first-attempt validity.
*Falsified if:* the best SFT arm gains < 1.5 points over the matched base on BFCLv3.

**H2 — The forgetting tax is controllable, and three cheap controls beat one expensive one.**
Entropy-weighted loss, a small replay fraction, and parameter-space merging back toward the
base together hold IFEval, IFBench, Multi-IF, MMLU-Pro and GPQA within 2.0 points while H1's
gain is retained.
*Why it might work:* entropy-weighted loss halved the general-average collapse in a matched
setting (`2601.02151v1`: 74.8 → 77.5 recovered against 81.1 baseline); a 0.1% replay buffer
lifted retention from 45% to 65% (`2512.18934v1`); two-way merging is the vendor's own
documented post-training step (`2511.23404v1` §4.4, `2407.08699v2`).
*Why it might not:* every one of those papers still records a residual tax, and the same
matched study breached a 2-point IFEval guardrail even after mitigation.
*Measured by:* guardrail deltas against the matched base rerun; worst-category floor.
*Falsified if:* no arm holds all five guardrails within 2.0 points while gaining ≥ 3.0 on
BFCLv3. **A forced trade-off is a reported finding, not a failure of the project.**

**H3 — Flag-rather-than-assert is a preference-rung behaviour, not an SFT behaviour.**
Two-sided preference pairs (honest abstention chosen; a fabricated call *and* a needless
refusal both rejected) lift probe flag rate to ≥ 0.70 at false-flag ≤ 0.15, where SFT alone
does not.
*Why it might work:* the same pair construction cut fabricated-call rate 90.2% → 55.8%
(`2510.22977v2`), and preference optimization beat SFT for refusal by 3.4× at 7B
(`2510.10390v1`).
*Why it might not:* SFT-taught abstention is largely lexical pattern-matching
(`2512.04597v1`), and the two rates trade off at r = −0.78 (`2510.10390v1`).
*Measured by:* probe flag / false-flag / functional-accuracy triple, plus the
corrupted-evidence and text-only control arms.
*Falsified if:* the control arms score within noise of the real arm, meaning the behaviour is
lexical rather than evidential.

**H4 — 4-bit retention is buyable after the fact, and cheaply.**
Post-training quantization as a warm start, then distillation from this project's own
full-precision checkpoint, then an error-targeted preference pass of a few hundred examples,
holds every axis at ≥ 93% and the average at ≥ 97%.
*Why it might work:* PTQ warm-start plus generalized-JSD distillation recovered IFEval
20.51 → 45.19 where naive quantization-aware training degenerated (`2506.09104v1`);
error-targeted stepwise DPO recovered a collapsed 0.5B model from 332 examples in 3–5 minutes
per GPU (`2505.11574v4`); 4-bit is the recoverable regime (`2604.19884v1`).
*Why it might not:* nine papers show structured output degrading first, and one in this exact
architecture family measured the Q4_0 penalty *growing* with capability, 2.5% → ~6%
(`2608.20210v1`).
*Measured by:* per-axis retention of the 4-bit GGUF against its own full-precision parent.
*Falsified if:* any axis stays below 93% after the full recovery rung.

**H5 — Verifiable-reward RL adds a further point or two, or it collapses; either is
informative.**
A decomposed graded reward from a supervised warm start improves BFCLv3 beyond the best
SFT+preference checkpoint without raising fabrication rate.
*Why it might work:* graded, schema-aware, correctly-weighted rewards produced real gains at
8B (`2605.16790v1`) and the recipe for surviving small scale is documented
(`2510.07737v1`, `2605.27954v1`, `2608.03092v1`).
*Why it might not:* plain GRPO at 1.5B "reliably collapses" (`2510.07737v1`), scored
00.00/00.00 at 1B under a format reward (`2605.27954v1`), and equal-weight multi-reward left
format compliance below 9% at 1.5B (`2608.03092v1`).
*Measured by:* BFCLv3 delta over the best non-RL checkpoint; fabrication rates; entropy,
valid-action ratio and duplication ratio as live health signals.
*Falsified / killed if:* any of the three health signals crosses its stop line (below), at
which point the arm is abandoned and the budget returns to H4.

Two approach families are deliberately **not** hypotheses here and are recorded as excluded
with reasons in `overview.md`: constrained decoding (a measured 43.5-point accuracy tax at
sub-3B, `2605.26128v1`) and any ensemble or multi-teacher method (does not fit an 8 GB
single-GPU envelope, and needs M≥2 trained teachers).

## s3.2 Experiment matrix

Notation: **FP** = bf16 parent; **Q4** = exported GGUF. Every row is one queued job or one
sweep of a queued job. Screening suite = BFCLv3 + IFStruct + IFEval + both probes
(~0.3 GPU-h). Full suite = screening plus BFCLv4, ToolSandbox, IFBench, Multi-IF, MMLU-Pro,
GPQA Diamond, HumanEval, MBPP (~1.0 GPU-h).

### Protocol frozen for every row

One serving backend for all FP comparisons and one for all Q4 comparisons, greedy decoding,
one frozen prompt and reasoning template, full generation config recorded per run, GGUF
conversion counted as part of the backend. Paired significance (McNemar / Wilcoxon,
Benjamini-Hochberg) with Cohen's κ and error-set Jaccard reported alongside. Rationale and
citations: `../s2-research-summary/report.md`, "Metrics and evaluation choices".

### B — Baselines (`s5.2`), all reruns, none quoted

| Row | Model | Precision | Suite |
| --- | ----- | --------- | ----- |
| B1 | `LFM2.5-1.2B-Instruct` | FP | full |
| B2 | `LFM2.5-1.2B-Instruct` | Q4 (vendor QAD Q4_0) | full |
| B3 | `LFM2.5-1.2B-Instruct-GGUF` unsloth UD-Q4_K_XL | Q4 | full |
| B4 | Granite 4.0-1B | FP | full |
| B5 | `NovachronoAI/LFM2.5-1.2B-Nova-Function-Calling-GGUF` | Q4 | full |
| B6 | base (non-instruct) LFM2.5-1.2B, **if published** | FP | screening |

B1 is the reference for every delta. B2 and B3 together answer "does our 4-bit beat the best
cheap quant". B6 is the optional starting-checkpoint ablation and nothing depends on it.

### C — Supervised sweep (`s5.3`), H1 and H2

| Row | Arm | Varies | Metric it moves | Suite |
| --- | --- | ------ | --------------- | ----- |
| C1 | LoRA SFT, r16 / α32 / LR 1e-4, full mix | reference arm | BFCLv3, IFStruct | screening |
| C2 | LoRA SFT, **public data only** | data provenance | isolates stack specialization | screening |
| C3 | Full-parameter SFT | adapter vs full | BFCLv3 ceiling | screening |
| C4 | LoRA SFT + entropy-weighted loss | loss shaping | guardrail deltas | screening |
| C5 | Replay fraction ∈ {0%, 1%, 5%} | replay | guardrail deltas | screening ×3 |
| C6 | Rank ∈ {16, 64} on the winning recipe | capacity | BFCLv3 vs guardrails | screening ×1 extra |
| C7 | **Raw mix, uniform sampling** | mix weighting | isolates what role balancing buys | screening |

**Amended at `s4.4`, on measured data.** Two rows above rested on assumptions the rendered
corpus contradicts, and both amendments are recorded in
`../s4-data-preparation/report.md`, "s4.4 Preprocessing — measured".

`C1` now trains on a **role-balanced sample**, not the raw mix. The rendered corpus puts 79%
of its 201.5M tokens on SQL and code, 20% on tool calling and 0.7% on structured output, so
uniform sampling would spend the training budget inversely to the project's priority order.
Per-role sampling weights are calibrated once at `s5.1` against a single epoch's token count,
with tool calling and structured output together taking at least half the budget. `C7` keeps
the raw mix as an explicit comparison, so the balancing is a measured choice rather than a
silent one.

`C5`'s replay buffer draws on the **base model's own greedy completions over a sample of the
`s4.4` prompt pool**, generated once as a job at `s5.1` and frozen for the whole sweep. The
general-quality corpus carries no assistant responses at all — all 447,053 renderable rows end
on a user turn — so there is no shipped general-instruction data to replay. Self-distillation
replay is what a prompt set with no responses is designed for, and it keeps the replay
distribution on-policy for the checkpoint being protected. Pulling a fourth-party instruction
corpus in at `s5` instead would add a licence surface and an unaudited contamination surface.

Hyperparameters are taken from the closest matched precedents rather than invented: LoRA
r16 / α32 / LR 1e-4 / batch 48–64 / 3 epochs (`2409.00920v2`, `2508.12685v3`), entropy
weighting over top-20 logits (`2601.02151v1`).

**Stop/go into D:** the best of C1–C6 must gain ≥ 1.5 points on BFCLv3 over B1 with no
guardrail worse than −3.0. If not, H1 is falsified and the project reports that rather than
proceeding to spend on preference optimization.

### D — Preference rung (`s5.3`), H3

| Row | Objective | Varies | Suite |
| --- | --------- | ------ | ----- |
| D1 | Regularized DPO (λ·(log-ratio-sum)², λ=2×10⁻⁴) | the safe default (`2502.17507v2`) | screening |
| D2 | Truncated-importance-sampling offline objective | `2607.19450v1` recipe | screening |
| D3 | Length-normalized margin loss, β=5.0, cosine 8e-7→8e-8 | the vendor's own shipped loss (`2511.23404v1`) | screening |
| D4 | Winner of D1–D3 with soft tail-loss capping off | noise robustness of auto-generated pairs (`2603.07211v2`) | screening |

Pairs: honest abstention chosen against both a fabricated call and a needless refusal
(`2510.22977v2`). Prompts from `antidoom-mix-v1.0` plus generated malformed-return items.
**Stop/go into E:** probe flag rate ≥ 0.70 at false-flag ≤ 0.15, with the corrupted-evidence
control arm scoring materially below the real arm.

### E — Merging (`s5.3`), H2

| Row | Method | Varies | Suite |
| --- | ------ | ------ | ----- |
| E1 | Slerp c=0.5, specialized ↔ base | two-way only (`2407.08699v2`) | screening |
| E2 | TIES-Merging | sparsification (`2511.23404v1` App. B) | screening |
| E3 | DARE | drop rate | screening |

A merged adapter is never the shipped artifact: the winning mixture gets a final pass
(`2605.15220v1`).

### F — Reinforcement-learning arm (`s5.5`), H5, conditional

Runs only if D produced a checkpoint clearing its stop/go, and only from that warm start.
Reward: `R_format + R_parse(name, param, dtype, graded clip(1−0.25p,0,1)) + R_exec + R_answer`
with answer correctness weighted 5 against a [0,4] structural budget (`2605.16790v1`), sparse
dimensions up-weighted rather than equal-weighted (`2608.03092v1`), degenerate all-pass and
all-fail groups dropped (`2602.03452v2`), no pass-rate densification (`2601.03525v3`), a flat
penalty on detected shortcut behaviour (`2603.07084v2`), no KL penalty, exponential LR decay
(`2510.07737v1`). No model-scored reward at any point.

| Row | Arm |
| --- | --- |
| F1 | Full decomposed reward |
| F2 | Ablation: schema term removed (tests the "safe tool routing" failure, `2605.16790v1`) |
| F3 | Ablation: equal weights (tests the sub-9% format-compliance failure, `2608.03092v1`) |

**Hard kill lines, checked every 25 steps:** policy entropy falling below 60% of its value at
step 0; valid-action ratio falling below its warm-start value; sentence-duplication ratio
rising above 1.5× warm start (all three from `2605.27954v1`). Two consecutive breaches ends
the arm and returns its remaining budget to G.

### G — Export and recovery (`s5.6`), H4

| Row | Step | Varies |
| --- | ---- | ------ |
| G1 | GGUF export, Q4_0 and Q4_K_M | format; both measured on disk size and throughput |
| G2 | PTQ warm start + KL(T=1) distillation from our own FP checkpoint, attention layers and preceding recurrent layers at BF16, KV-cache FP8, LR 1e-5→1e-6 | `2601.20088v3` |
| G3 | Error-targeted stepwise preference pass, a few hundred examples | `2505.11574v4` |
| G4 | Group size 32 vs 64 in the quantizer | `2604.07888v1`, `2407.11062v3` |

**Stop/go into s6:** ≥ 97% average retention with no axis below 93%, at ≤ 1.5 GB on disk.

### H — Final evaluation (`s6`)

Full suite on the three finalists at FP and Q4; three seeds on the single winner; the
robustness set (corrupted-evidence and text-only probe controls, tool-return serialization
ablation, nesting-depth stratification, BFCL AST-only reported separately); fabrication rates
on every finalist; decontamination re-check including the black-box peakedness test
(`2402.15938v3`).

## s3.3 Compute and time budget

**Enforced allowance, read from `lab team quota me` on 2026-08-25:** 12,000 GPU-min
(200 GPU-h) available, **2 GPUs at a time**, allowed types A10, A10G, L4, L40, L40S, T4,
RTX6000, RTXA6000, A100-40GB. Nothing in this matrix asks for more than 2 GPUs at once or an
accelerator type outside that list, so no row needed reshaping for the cap.

**Hardware choice, and the steering note that shaped it.** The team's provider note
(`updated_at` 2026-08-23T22:48:37) sets the card list, the try order, and the selection rule
(memory first, then throughput; never escalate into T4 because it is the only allowed card
without bf16; A100-40GB is the bandwidth pick and *not* an upgrade from a 48 GB card). Taking
that together with the enforced type list:

- **Default card: L40S, 48 GB** (three sources in the note's row: AWS g6e, Nebius, RunPod).
  Comfortably holds full-parameter fine-tuning of a 1.2B model in bf16 with optimizer state,
  and holds the batch sizes the precedents used.
- **Bandwidth-bound rows (G2 distillation, F1–F3 rollout generation): A100-40GB**
  (GCP, Lambda, RunPod), per the note's guidance that it is the throughput pick.
- **Evaluation-only rows: L4, 24 GB** (three sources), which is ample for 1.2B inference and
  leaves the larger cards free.
- **T4 is not used for any training row**, per the note's bf16 warning.
- One protocol constraint overrides card convenience: every number that enters a paired
  comparison is measured on the same card as its counterpart, and if a card becomes
  unavailable mid-study the earlier side is re-measured on the new card rather than compared
  across hardware. That rule is also what the note prescribes for paired numbers.

**Estimation basis.** Throughput assumed at 6,000 training tokens/s for a 1.2B model on one
L40S (a deliberately conservative fraction of the card's dense bf16 peak, since the hybrid
short-convolution blocks are not the shape these throughput figures are usually quoted for).
An SFT arm is ~60k examples × ~1,000 tokens × 2 epochs ≈ 120M tokens ≈ 5.5 GPU-h. Screening
suite ≈ 0.3 GPU-h, full suite ≈ 1.0 GPU-h, both at batch-parallel greedy inference on L4.

| Phase | Rows | GPU-h |
| ----- | ---- | ----- |
| Smoke (`s5.1`) | fixture runs of every task type | 2 |
| B baselines | 5 full suites + 1 screening | 10 |
| C supervised sweep | 8 training runs + 8 screenings | 38 |
| D preference rung | 4 runs + 4 screenings | 10 |
| E merging | 3 merges + 3 screenings | 2 |
| F RL arm (conditional) | 3 arms, kill lines enforced | 30 |
| G export + recovery | 2 formats × (export, distil, DPO pass) + group-size ablation + 4 full suites | 12 |
| H final evaluation | 6 full suites + 3 seeds + robustness set | 12 |
| **Subtotal** | | **116** |
| Contingency at 25% | re-runs, failed launches, re-measurement after a card substitution | 29 |
| **Planned total** | | **145 GPU-h = 8,700 GPU-min** |

The ledger in `../../budget.json` tracks the same figure, in `GPU-hours`, with `planned: 145`.

That leaves 3,300 GPU-min (55 GPU-h) of the enforced allowance unspent, which is the margin
for the one thing most likely to need it: re-measuring a baseline on a different card if the
default becomes unavailable mid-study.

**What a larger version would cost, stated before anything is spent** (the operator's
standing preference is to size the first sweep to what is available and price the bigger one
up front):

- **Three seeds on every sweep arm instead of one** (the multi-seed discipline several papers
  in the corpus lack): +2× phases C, D, E ≈ **+100 GPU-h**.
- **Transferring the winning recipe to tier 2 (`LFM2.5-2.6B`)**, which `overview.md` names as
  the first thing to promote: roughly 2.2× the per-token cost, over the winning path only
  (one SFT, one preference run, export and recovery, full evals) ≈ **+45 GPU-h**.
- **A full RL treatment rather than a kill-line-guarded arm** (the published recipes at this
  scale use rollout budgets several times larger than F allows): ≈ **+80 GPU-h**.

All three together is ≈ 370 GPU-h against a 200 GPU-h allowance, so any of them needs a quota
raise and none is started without one.

**Overrun risks.** Three, with responses. Row F is the largest single line item and the most
likely to waste budget, which is why it carries hard kill lines rather than a fixed step
count. The MMLU-Pro pass dominates the full-suite cost, so intermediate checkpoints get the
screening suite only and the full suite is reserved for baselines and finalists. And a card
substitution forces re-measurement of one side of any paired comparison, which the
contingency line exists to absorb.
