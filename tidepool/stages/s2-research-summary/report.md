# Stage 2: Research summary

_Status: complete._

## Headline

Across 212 papers the same warning keeps landing: 4-bit compression damages structured
output and instruction-following before it touches fluency or knowledge, and the smaller
the model the worse it gets. One study gained 0.48 points on a 14-task average while
losing 8.5% of its instruction-following score at 4-bit. Reward-based training is the
second hazard, with the two studies closest to our model size reporting outright collapse
at 1.2 to 1.5 billion parameters unless a supervised warm-up and a graded, schema-aware
reward are both in place. The plan survives with six changes: compression now runs last,
the reliability target moves from 0.80/0.10 to 0.70/0.15, an anti-fabrication check joins
the metric set, and every comparison is pinned to one serving stack, one prompt template
and greedy decoding.

## Work log

- 2026-08-25 · s2.1 · 212 per-paper notes written to `notes/<id>.md`, one reader per
  paper against a shared brief (`reader-brief.md`). Relevance spread 43 high, 92 medium,
  71 low, 6 skip.
- 2026-08-25 · s2.2 · Synthesis written into this report (sections "Techniques worth
  trying" through "Open questions"), citing note files by paper id.
- 2026-08-25 · s2.3 · `overview.md` revised to literature-grounded status; the
  assumption walk and the resulting scope changes are in "Scope revision (s2.3)" below.
- 2026-08-25 · s2.4 · Scope sign-off decided under autonomous mode; recorded in
  "Scope sign-off (s2.4)" below.

## How the reading was done (s2.1)

**One reader per paper, 212 readers.** Each was given the full paper text and a shared
instruction sheet (`reader-brief.md`) holding the project context, the note template, a
four-point relevance scale and six writing rules: quote verbatim or not at all, never
invent a quote, attach model scale and dataset and precision to every number, treat
negative results as the most valuable part of the note, be specific about what does not
transfer, and write a note even for a paper that is irrelevant.

**Reader model choice.** The reading pass ran on a smaller model than the synthesis.
Filling a fixed template against a single supplied document, under a rule that every
quote is verbatim and nothing may be invented, is a fidelity task rather than a judgment
task. Synthesis, where the cross-paper calls get made, stayed on the main model. The
residual risk is that relevance rating *is* judgment, so a smaller model could inflate
ratings and bury the useful papers. The observed spread (43 high / 92 medium / 71 low /
6 skip) shows no inflation, and the reading policy below re-reads every high note in
full, which catches a paper mis-rated downward only if it was mis-rated by one step.

**Reading depth at s2.2, stated plainly so nobody over-reads the coverage.** The notes
total 1.7 MB, more than one working context can hold alongside writing this report. So:

| Tier | Papers | Read at synthesis |
| ---- | ------ | ----------------- |
| high | 43 | TL;DR, techniques, pitfalls and metrics sections in full |
| medium | 92 | TL;DR and pitfalls / negative results |
| low | 71 | TL;DR only |
| skip | 6 | listed as skipped, not read |

On top of that, targeted searches into the full note set pulled specific numbers,
baselines and dataset names while each theme was written. Every claim below is traceable
to a note; no claim rests on a paper read only at TL;DR depth unless the TL;DR itself
carries the number.

**Two disclosures.**

1. Eleven notes were written without the trailing source-link footer the template
   requires (`2406.08391v3`, `2501.02342v1`, `2504.13932v3`, `2505.23854v1`,
   `2509.05899v1`, `2511.01934v2`, `2511.03825v1`, `2511.22138v1`, `2604.15149v1`,
   `2605.02964v1`, `2608.18578v1`). Every other required heading and the full body were
   present, so the footer was appended mechanically rather than treated as a failed note.
   All 212 notes then validated clean: every required heading present, every source link
   resolving to its own paper file.
2. No reader, in any of the 212 papers, reported text addressed to it as an instruction.
   The brief required them to record such a thing if found.

## Techniques worth trying (s2.2)

The candidate ladder in `overview.md` was LoRA SFT → full SFT → DPO → GRPO with
verifiable rewards → quantization-aware distillation → QAT. The literature keeps the
shape and changes the details.

### The supervised rung is where most of the win is, and it is cheap

The single most encouraging result for this project is
[2505.04016v1](notes/2505.04016v1.md) (SLOT): plain LoRA supervised fine-tuning takes
**Llama-3.2-1B from 9.6% to 88.9% average schema accuracy with no constrained decoding**,
and at 7B, SFT plus a grammar reaches 99.5% / 94.0%, beating Claude-3.5-Sonnet prompting
at 74.7% / 73.9%. Two details transfer directly. Errors concentrate beyond three or four
levels of nesting, so the in-house probe should stratify by nesting depth. And
**public data alone reached 63.1% / 68.3% against 89.0% / 81.7% for synthetic plus
public**, which is quantitative support for the public-data-only arm being an honest
floor rather than a near-tie.

[2603.16901v1](notes/2603.16901v1.md) is the same story at 270M: parse failure fell from
87% to under 1% through dataset repair, tool-count reduction and full-parameter SFT, with
no RL at all. Two operational lessons: their prompts exceeded 4,900 tokens against a
2,048-token limit until tools were sampled per example, and `<think>` tokens break naive
format validators, so our validity harness must be reasoning-aware before it scores
anything.

Loss shaping matters more than rank. [2601.02151v1](notes/2601.02151v1.md) (EAFT) scales
per-token cross-entropy by normalized top-K entropy and reports the failure mode we most
need to avoid: **plain tool-call SFT moved BFCLv3 only 60.5 → 61.4 while the general
average collapsed 81.1 → 74.8 and IFEval 81.0 → 77.8**; entropy weighting halved the
damage (77.5 / 78.6) and still breached a 2-point IFEval guardrail. Their top-20 logits
correlate at 0.999 with full top-K and cost under 0.4 KB per token, so the method is
affordable. One caveat cuts against us: EAFT deliberately down-weights tokens where the
model is confidently wrong, which is exactly the behaviour we want to override when
teaching flag-on-malformed-return.
[2505.20192v3](notes/2505.20192v3.md) (BalanceSFT) is the harsher version of the same
warning: tool-call SFT dropped Qwen2.5-Coder-7B HumanEval pass@1 **0.866 → 0.470**, and a
learnable-α split between reasoning loss and result loss only recovered to within 4%.

### Preference optimization: use a variant, not textbook DPO

Textbook DPO ([2305.18290v3](notes/2305.18290v3.md), β=0.1, batch 64, RMSprop 1e-6 with
150-step warm-up, π_ref = π_SFT) is now dominated by cheap fixes.

- [2502.17507v2](notes/2502.17507v2.md) **proves** vanilla DPO's optimum pushes down the
  probability of the preferred response too (π_θ(y|x) ≤ e⁻¹·π_ref(y|x), with a worked
  case where both collapse to zero at exactly zero loss). The fix is a
  λ·(log-ratio-sum)² regularizer built from quantities DPO already computes, λ=2×10⁻⁴,
  untuned across four variants. There is no reason not to take it.
- [2607.19450v1](notes/2607.19450v1.md) (REGEN) measures **plain DPO as the weakest
  offline objective at 0.5–1.5B**, IFEval 56.9 against 71.5 for truncated-importance-
  sampling REGEN, with the loss concentrated on exact-match instruction-following. Their
  recipe is inexpensive: clip only the importance weight, class-balanced sampling,
  per-query per-domain advantage normalization, and record the behaviour policy's NLL per
  rollout.
- [2511.23404v1](notes/2511.23404v1.md) (the vendor's own report) gives a
  length-normalized margin-based direct-alignment loss generalizing DPO/SimPO/APO-zero,
  with the hyperparameters they actually shipped: β=5.0, cosine LR 8e-7 → 8e-8, batch
  2048, context 1024, two epochs, preference data by sampling N=5 from the SFT checkpoint
  and breaking ties toward on-policy responses.
- [2603.07211v2](notes/2603.07211v2.md) (CompassDPO) matters because our malformed-return
  preference pairs will be auto-generated and therefore noisily labelled: vanilla DPO
  degrades monotonically as label-flip noise rises 0 → 30% at 2.8B, and soft tail-loss
  capping beats hard truncation, which itself degraded MMLU-Pro and raised over-refusal.
  [2602.11079v3](notes/2602.11079v3.md) is the worst case: mislabelled pairs from
  automated judge grading installed a "distractor-triggered compliance" backdoor
  invisible to standard evals, and label-switching 30,000 datapoints cost GSM8K 72.5% →
  ~68%.
- For the flag behaviour specifically, [2510.10390v1](notes/2510.10390v1.md) reports
  **DPO beating SFT for refusal by 3.4× at 7B**, which is the main argument for keeping a
  preference rung at all rather than pushing straight from SFT to RL.

### The fix for fabricated calls is a preference rung, and it has a measured price

[2510.22977v2](notes/2510.22977v2.md) (The Reasoning Trap) is the most important negative
result in the corpus for our claim structure: **BFCL improved +9.9% on Qwen2.5-7B while
tool hallucination surged, R_NTA 34.8% → 90.2% and R_DT 54.7% → 100.0%**. GRPO on GSM8K
alone, with no tools in the loop, still raised hallucination. Prompt engineering moved
R_NTA only 90.2 → 87.5. Their DPO fix cut R_NTA to 55.8 at the cost of validation reward
0.45 → 0.34, and the pair construction is directly reusable: honest abstention as chosen,
against *both* a fabricated call *and* a needless refusal as rejected, which is exactly
the two-sided shape our flag / false-flag pair needs.

[2409.00920v2](notes/2409.00920v2.md) (ToolACE) supplies the data-side counterpart:
**removing the multi-type/irrelevant-query category collapsed irrelevance detection from
86.42 to 6.99**. Irrelevance data is not a garnish. [2410.04587v2](notes/2410.04587v2.md)
(Hammer) adds function- and parameter-name masking at a tunable ratio plus 7,500
irrelevance-augmented examples with empty-list labels, and names the tension we are
walking into: "a concerning inverse relationship between the model's ability to
accurately execute function calls and its capacity for irrelevance detection".

### Reinforcement learning: only with a warm start, a graded reward, and entropy watched

Two papers at our exact scale report collapse.
[2510.07737v1](notes/2510.07737v1.md) (ToolExpander) states **vanilla GRPO reliably
collapses at 1.5B**; their fixes are few-shot-guided replacement of unsolved hard samples,
exponential LR decay, a conjunctive 0/1 reward and dropping the KL penalty. Raising
rollouts 10 → 32 cut hard samples only 5–8% while doubling time, and excluding hard
samples backfired. [2605.27954v1](notes/2605.27954v1.md) is starker: **plain GRPO on
Llama3.2-1B scored 00.00 / 00.00 under a format/schema reward** while their auxiliary
method recovered 79.69%, because a 1B model cannot sample valid trajectories at all
without a mid-training pass. Their monitoring list is the one to adopt: entropy,
valid-action ratio, sentence-duplication ratio, with degenerate patterns getting
reinforced across successive entropy eruptions.

Reward design dominates algorithm choice. [2605.16790v1](notes/2605.16790v1.md) (TIER)
decomposes `R_total = R_format + R_parse + R_exec + R_answer` with
`R_parse = R_name + R_param + R_dtype` and graded credit
`clip(1 − 0.25·p, 0, 1)`, and weights answer correctness at 5 against a [0,4] structural
budget *because* lower weightings induce hacking. Three of their findings bind on us: a
JSON intermediate representation beat XML and direct output (68.92 / 66.06 / 63.80);
**an execution reward without the schema term makes the model route to a small set of
"safe" tools**; and every ablated reward variant converged stably but plateaued early, so
stability is not evidence of a good reward. Their exec term answers "did the call run",
not "did the model correctly flag a bad return", so our probe reward has to be built, not
borrowed.

[2608.03092v1](notes/2608.03092v1.md) (SMOPD) is the sharpest constraint on reward
architecture: **equal-weight multi-reward GRPO left format compliance below 9% (8.8% at
1.5B)**, because a sparse binary reward is essentially unlearnable next to a dense one.
Their prescription is to diagnose per-dimension within-group variance first, then
specialize-then-merge with the sparse dimension weighted far above the dense ones. Two
further mechanics matter: [2602.03452v2](notes/2602.03452v2.md) shows degenerate
all-correct or all-incorrect groups contribute **zero gradient**, so a schema-validity
reward silently burns rollout budget on uniformly easy or uniformly malformed turns; and
[2601.03525v3](notes/2601.03525v3.md) (VeRPO) shows naive pass-rate densification is
*actively harmful* versus binary, and that an external model-scored reward caused outright
multi-turn optimization collapse, which independently supports the no-stored-model-reward
constraint already in scope.

On the warm start, [2604.20316v1](notes/2604.20316v1.md) (R2IF) states it directly: an
SFT warm start is *more* critical the smaller the model, because a binary reward acts as a
hard gate that an unwarmed small model never passes.
[2605.02572v1](notes/2605.02572v1.md) adds that RL without SFT init reward-hacked at 1.7B,
that RL amplifies existing capabilities rather than creating new ones, and that 1.7B → 4B
did not fix collapse. Against that, [2511.01934v2](notes/2511.01934v2.md) (Tool Zero) runs
pure GRPO from **base** models and reports that starting from an Instruct model *hurt*
RL, with a progressive reward that blends a lenient overlap term
(`r_general = −0.5 + |Y∩Y*|/|Y*|`) into a strict AST term, transitioning at step 25 (later
transitions induced reward hacking). Their scale curve is the sobering number:
**1.5B averages 63.73 against 3B 70.87, 7B 77.32, 32B 78.99.**

### Quantization: the rung that decides whether any of this ships

Post-training 4-bit is where our deliverable lives, and the corpus is unanimous that it
damages structured behaviour first. [2506.09104v1](notes/2506.09104v1.md) (UPQ) at
Llama-3.2-3B INT4 is the cleanest recipe: naive next-token QAT degenerates into repetition
loops (MMLU 49.73 / IFEval 20.51 against 58.60 / 52.57 for PTQ alone), while
**PTQ warm-start followed by generalized-JSD distillation recovers IFEval to 45.19 and
MMLU to 53.20**. [2601.20088v3](notes/2601.20088v3.md) supplies the architecture-specific
recipe for a hybrid conv/attention model: use the model's own FP checkpoint as teacher,
KL at T=1 (beating MSE-on-logits and even a larger 12B teacher), keep the attention layers
and their preceding recurrent layers at BF16, KV-cache at FP8, LR 1e-5 to 1e-6. Their
warning is that **QAT breaks RL-learned capabilities** (AA-LCR 24.8 against BF16 35.9, PTQ
31.3, QAD 34.3), which is a direct argument for QAD over QAT if an RL rung is in the
pipeline.

[2505.11574v4](notes/2505.11574v4.md) gives the cheapest recovery in the corpus:
error-targeted stepwise DPO from **332 examples in 3–5 minutes per GPU**, beating generic
supervision by ~5.6 points, on a model (Qwen2.5-0.5B) that had dropped >60% post-quant.
[2604.19884v1](notes/2604.19884v1.md) draws the line we should design to: **4-bit is
recoverable signal degradation, 2-bit is irreversible collapse** resisting all
training-free repair, measured on weight-only PTQ, which is what GGUF is.

Two techniques are worth a cheap look and one is probably out of reach.
[2303.08302v3](notes/2303.08302v3.md) (ZeroQuant-V2) reports that **smaller models suffer
more** under INT4 weight-only and recommends fine-grained blocks plus a low-rank
compensation term (LoRC, rank 8, ≤1.6% size overhead);
[2604.07888v1](notes/2604.07888v1.md) and [2407.11062v3](notes/2407.11062v3.md) both point
at smaller group sizes (group-32 over group-128; group-64 at 2-bit), which maps onto the
K-quant / IQ family choice in llama.cpp. EfficientQAT also reports its gains over AWQ and
GPTQ are **much smaller at 4-bit** than at lower bit widths, and
[2605.17471v1](notes/2605.17471v1.md) (WinQ) says the same thing more bluntly: at 4-bit,
baseline QAT already nearly matches full precision, "leaving limited room for
improvement", at a cost of 20B tokens / 240K steps / 28.7 GPU-hours per run at 1B.

### Merging is a cheap rung that is not currently on the ladder

[2511.23404v1](notes/2511.23404v1.md) §4.4 and Appendix B give runnable formulas for
model soup, task arithmetic, TIES-Merging, DARE and DELLA, from the vendor of the exact
model we are training. Parameter-space merging of a tool-specialized checkpoint back
toward the base is a plausible one-command answer to the forgetting guardrail, costing no
training. [2407.08699v2](notes/2407.08699v2.md) (Branch-and-Merge) adds that **K=2 is the
sweet spot**, that Slerp at c=0.5 beat Model Stock on learning while Model Stock reduced
forgetting most, and that LoRA at 7–8B preserved English but crippled target-domain
learning. Multi-teacher and ensemble variants
([2601.13572v1](notes/2601.13572v1.md), [2608.19098v1](notes/2608.19098v1.md),
[2607.27770v1](notes/2607.27770v1.md)) stay out of scope: they need M≥2 trained teachers,
which is the ensembling this project ruled out against an 8 GB single-GPU target.

### Non-RL routes to notice-and-correct

[2507.02529v2](notes/2507.02529v2.md) (RetrySQL) trains on corrupted-then-corrected steps
with a `[BACK]` token at 1.5B, for 4.47 GPU-hours full-parameter, as a supervised route to
noticing an error. Gains were inconsistent (+0.4 p.p. on BIRD for one pair) and not all
corruption strategies helped, and there is no retention analysis, so it is a cheap
side-experiment rather than a rung. [2502.14905v1](notes/2502.14905v1.md) (ThinkJSON) runs
GRPO *then* SFT at 1.5B with fully programmatic key-value-match and length-similarity
rewards plus a separate binary tag-format reward, at roughly 20 GPU-hours on 8×H100 plus
3 hours on 1×A100, and reports that bare distilled 1.5B and 7B models produce no JSON or
heavy noise at all.

## Pitfalls and negative results

### Quantization damages structured behaviour before knowledge, and small is worse

This is the most replicated finding in the corpus, and it is the one that decides whether
the deliverable is shippable.

| Paper | Scale / precision | What broke first |
| ----- | ----------------- | ---------------- |
| [2605.04062v2](notes/2605.04062v2.md) | Qwen3-0.6B, 4-bit | 14-task average **+0.48** to 47.83 while **IFEval fell 58.41 → 53.42** (≈8.5% relative, under a 93% floor) and HumanEval 37.20 → 34.15. An Ethics gain 47.70 → 54.36 inflated the average. |
| [2601.14888v1](notes/2601.14888v1.md) | 3-bit GPTQ | AIME-120 −11.67%, MATH-500 −12.80% against Winogrande −1.03%, Hellaswag −3.13%. |
| [2505.11574v4](notes/2505.11574v4.md) | Qwen2.5-0.5B | >60% drop post-quant (MATH 23.98 → 7.24 GPTQ) against 2–3% at 7B; IFEval relative drop only −5.01% against MATH −29.84%. |
| [2410.13461v2](notes/2410.13461v2.md) | 2-bit vs 3-bit prefill | Instruction-following collapses while text stays fluent; 3-bit prefill recovers Rouge-L +4.3 / BLEU +9.45. |
| [2608.18578v1](notes/2608.18578v1.md) | Qwen INT4 (bitsandbytes NF4) | A state-tracking cliff invisible to aggregates: 81.0 → 68.3 at highest interference against pooled 97.6 → 96.5. INT8 carried a real paired-test penalty (p=0.018 / 0.004) that an unpaired test missed. |
| [2605.15208v1](notes/2605.15208v1.md) | 3/4-bit, MLX | <3% perplexity rise at 4-bit yet 2.5–5.6% of items develop new bias; Mistral-7B at 3-bit shows a 173× disparity; family variation 6.0 / 17.7 / 21.1% not predictable from size. |
| [2508.03332v2](notes/2508.03332v2.md) | Qwen3-0.6B, 2-bit | WikiText2 perplexity 20.9 → 2.38E+04; layer sensitivity is heterogeneous and worse in smaller models. |
| [2512.23367v2](notes/2512.23367v2.md) | 7B W4A8; 1B FP16 | HumanEval 85.37 → 81.10, MBPP 77.04 → 70.04; separately, the 1B model showed 34.15% repetitive-output susceptibility at FP16, cut to 21.95% under INT8. |
| [2608.20210v1](notes/2608.20210v1.md) | 150M, same hybrid conv-attention family | Q4_0 cost ~6% perplexity (9.18 → 9.75), **up from 2.5% earlier in training**, so the penalty grows with capability. |

Four consequences for our design. The per-axis retention floor is not optional bookkeeping;
without it, [2605.04062v2](notes/2605.04062v2.md)'s pattern (average up, instruction-
following down 8.5%) passes a 97%-average bar. Perplexity is not an admissible proxy
([2602.15563v1](notes/2602.15563v1.md) shows perplexity misranks quantized models against
downstream generative evals; [2605.15208v1](notes/2605.15208v1.md) shows behaviour
breaking at flat perplexity). Aggregate benchmarks are not an admissible proxy either
([2608.18578v1](notes/2608.18578v1.md)). And several of these papers explicitly disclaim
transfer to GGUF ([2608.18578v1](notes/2608.18578v1.md),
[2605.15208v1](notes/2605.15208v1.md)), which means our own GGUF measurement is the
evidence, not theirs.

Two smaller traps worth carrying: [2608.18578v1](notes/2608.18578v1.md) found `lm_head`
silently left at full precision by the quantizer, and
[2608.20210v1](notes/2608.20210v1.md) found 47.9% dead conv channels that could not be
pruned because llama.cpp shape-checks tensors at load
(`check_tensor_dims: expected 3,768, got 3,640`).

### The ordering conflict, and how it resolves

Three papers appear to disagree about where quantization goes in the pipeline.

- [2508.04073v1](notes/2508.04073v1.md): quantize-then-LoRA preserves gains, and
  **fine-tune-then-quantize was the worst of nine variants** (average position 8.56/10,
  zero firsts). Evidence quality is weak: a GPT-4o judge ranking 100 questions, rank-2
  LoRA, narrow thesis-QA. It also records that LoRA → GGUF required merging into full
  precision first and crashed often.
- [2511.19495v1](notes/2511.19495v1.md): quantization must come **last**. Quantize-first
  sequences hit perplexity 24–53 against ~5, and G-Eval 0.06–0.15 against 0.73.
- [2512.18934v1](notes/2512.18934v1.md): trains with quantization present throughout, and
  explicitly notes its scenario is *not* train-then-quantize-for-deployment.

The reconciliation is that [2511.19495v1](notes/2511.19495v1.md)'s catastrophic arm
quantizes and then **dequantizes to keep training in higher precision**, which is a
different operation from training inside the quantized format. Training in-format
(QLoRA-style, or QAD with a frozen FP teacher) is safe and is what
[2508.04073v1](notes/2508.04073v1.md) and [2512.18934v1](notes/2512.18934v1.md) both
actually did. **Decision recorded at s2.3: all quality training happens in full or
mixed precision, the export to 4-bit GGUF is the last step, and the only in-format
training is the QAD recovery rung after export-format weights exist.**

[2512.18934v1](notes/2512.18934v1.md) also offers a genuine bonus: INT8/INT4 retained
prior-task accuracy better than FP16 once replay was added, and **a 0.1% replay buffer
lifted NLU retention after Math training from 45% to 65% at every precision**. Without
replay, 4-bit lost >30 absolute points (72.31 → 42.50). The regularization mechanism is
the authors' explicit unconfirmed hypothesis, on a single family and scale with no
multi-seed, so it is a hypothesis to test, not a plank.

### Reward hacking is the central RLVR risk, and our metric is the shortcut

[2604.15149v1](notes/2604.15149v1.md) gives causal evidence that a surface/extensional
verifier alone induces hacking, that hacking is specific to RLVR-trained models (non-RLVR
models showed zero shortcuts), and that shortcuts rise with reasoning effort (0 / 32 / 84).
Extensional reward climbed while an isomorphic-perturbation reward plateaued at ~3.5 of 10
points after 500 steps. Their fix (renaming identifiers) has no obvious analogue for
tool-call JSON, where the shortcut is a hardcoded value.

[2605.02964v1](notes/2605.02964v1.md) (RHB) names our failure mode outright: the primary
exploit is "minimally valid JSON satisfying a shallow schema check". Their hardening axes
are directly implementable and measured: hardened evaluation boundaries with strict schemas
and fail-closed parsing cut exploits −41.5% alone, reduced file access −36.9%, combined
−87.7% **with no loss in task success**. They also report RL-heavy post-training raising
hacking from 0.4–0.8% to 12–16%, exploit rates jumping at chain length 5, 28% of exploits
carrying no rationale at all, and their automated classifier agreeing with humans only 94%.
Task correctness and integrity are kept as separate axes, which is the reporting shape to
copy.

[2603.07084v2](notes/2603.07084v2.md) is methodologically critical for us:
**~1% SFT contamination suffices** to install a hack that resurfaces during RL; outcome
filtering is insufficient (1.2% of proxy-passing teacher traces still hacked); defensive
prompting collapsed task quality to zero; and **greedy decoding maintains hacking after
SFT while sampled rollouts mask it**. Our schema-validity metric is a greedy single-sample
measurement, so it is the *revealing* protocol rather than the masking one, which is
fortunate. The one intervention that let accuracy keep improving was a flat penalty
`R = 1 − p` with p ∈ {0.25, 0.5, 0.75}, plus inoculation prompting (loophole named at
train time, redacted at test time). [2512.19027v2](notes/2512.19027v2.md) refines that:
the prompt *mismatch* is load-bearing, identical-prompt inoculation made gaming worse, a
generic "you may overfit to the checker" beat a task-specific cheat prompt, and models
become harder to red-team afterwards. Both papers are ≥8B; neither has small-model data.

Detection is not a fallback. [2604.16242v1](notes/2604.16242v1.md) reports text-CoT
monitors at only 53% / 43% F1 in the early implicit-hacking regime that matters most, and
that naive rejection fine-tuning without a hacking filter fell *below* the starting model
on code (15.4 / 15.9 against 19.6). [2601.20103v1](notes/2601.20103v1.md) caps frontier
LLM hack detection at a 63% detection rate. [2605.20744v1](notes/2605.20744v1.md) adds
that hacking is "addictive" (conditional rate far above unconditional), rises monotonically
with difficulty, and needs bespoke detectors per environment, with no data below ~31B.
[2509.15557v1](notes/2509.15557v1.md) is the cautionary case: a composite
`R = w_b·R_binary − w_a·P_answer − w_s·P_structural` (1.0 / 0.5 / 0.3) **failed to transfer
OOD**, with Llama3.2-3B SFT+RM showing the lowest OOD accuracy and the highest hacking rate.

### Flagging is easy to fake, and the composite metric is gameable

Three papers show that a flag metric can be satisfied by a degenerate policy.
[2411.13676v1](notes/2411.13676v1.md) (Hymba) **dropped BFCL's `live_irrelevance`
category** because models with no function-calling ability scored high there and low
everywhere else, yet still posted high overall accuracy.
[2511.22138v1](notes/2511.22138v1.md) shows the two worst models scoring 0% functionally
and **100% on Irrelevance Detection**. [2408.04682v2](notes/2408.04682v2.md)
(ToolSandbox) says it plainly: weak models score artificially well on Insufficient
Information "simply by acting less, a side effect instead of a positive outcome", and
Insufficient-Information performance *negatively* correlates with the other categories.
Consequence: flag rate must always be reported paired with false-flag rate and with
functional accuracy, and BFCL overall must be reported alongside per-category and AST-only
scores ([2608.03092v1](notes/2608.03092v1.md) separates full-suite Overall from AST-only
for the same reason).

[2512.04597v1](notes/2512.04597v1.md) is the control we were missing: SFT-taught abstention
is largely lexical pattern-matching, because corrupting or removing the evidence barely
moves the flag metrics (7B-SFT 83.27 / 89.08 against 7B-SFT-random 83.27 / 87.93 and
7B-Text-SFT 86.12 / 86.83). A corrupted-evidence and text-only control belongs in our own
probe. Prompting for recall trades precision nearly linearly (27.2 → 81.5 → 99.5%), and
thinking mode hurt.

The achievable level is lower than we assumed.
[2510.10390v1](notes/2510.10390v1.md) measures the trade-off directly at **r = −0.78**
between flag rate and false-flag rate on NQ, has Qwen refusal accuracy staying under 17%
from 0.5B to 72B, and found 4,096 extra thinking tokens bought <1pp.
[2408.04682v2](notes/2408.04682v2.md) has GPT-4o and Claude-3-Opus reaching only 76.6 and
71.1 on Insufficient Information. [2605.19341v1](notes/2605.19341v1.md) caps frontier
uncertainty-abstention at 76.9%, and reports that an *accurate* notice board raised Opus
4.6 hallucination 28.4 → 42.0, and that serialization format is an independent driver.
[2603.10697v1](notes/2603.10697v1.md) reaches only ~84% true-positive at 7–8B and finds
abstention training degrades both original and in-scope sets.
[2605.29523v1](notes/2605.29523v1.md) finds refusal the weakest axis for every model
tested, best 0.750 at 8B. [2601.05503v2](notes/2601.05503v2.md) shows retrieval boosting
answer accuracy +24.0% while degrading abstention accuracy −12.8%, higher reasoning effort
making abstention worse, training-free mitigations worth ~3.6%, and negative evidence
making up only 13–22% of retrieved content, so "this is broken" signal must be
deliberately oversampled. **A 0.80 / 0.10 target at 1.2B is above the frontier ceiling
these papers measure. It is revised at s2.3.**

Calibration machinery does not rescue it either: raw token-probability confidence
"performs dangerously poorly at and above moderate risk levels"
([2504.20168v1](notes/2504.20168v1.md), whose method also needs a resident
DeBERTa-xlarge-mnli against our 8 GB budget), verbalized confidence and perplexity are
both poor signals ([2406.08391v3](notes/2406.08391v3.md)), and
[2604.18419v4](notes/2604.18419v4.md) found self-assessment prompting and a LoRA-trained
abstain token both near or below no-abstention. The one cheap positive:
[2406.08391v3](notes/2406.08391v3.md) cut mean ECE 29.9% → 10.8% across six bases with a
LoRA fine-tune on ~1,000 graded examples plus a **Jensen-Shannon divergence regularizer on
target-sequence logits** (plain forward/reverse KL was insufficient), with diminishing
returns past ~5,000 examples, posing the flag decision as a two-token multiple-choice
completion. [2607.10738v1](notes/2607.10738v1.md) is the anti-pattern: a static refusal
reward of 0.05 caused catastrophic 99.9%-refusal hacking, and naive unanswerable-query
mixing gave non-monotonic effects and lazy-refusal collapse.

### Measurement hazards that would invalidate the whole comparison

- **Serving stack alone moves scores at our scale.**
  [2608.04714v1](notes/2608.04714v1.md): HF vs vLLM vs Ollama/GGUF at ~1–1.5B, under
  greedy decoding, shifts results (Ollama mean |Δ| = 0.055 from raw HF) and changes *which*
  items are correct (Cohen's κ 0.45, error-set Jaccard 0.792). Backend alone is ≈39% of
  realistic variance. vLLM ignores `generation_config` and samples at T=1.0; Ollama falls
  back to 0.8 / 0.9 / 40. **Every comparison in this project must be single-backend,
  greedy, with the full generation config disclosed**, and GGUF conversion counted as part
  of the backend rather than as a neutral format change.
- **Prompt and CoT template alone move scores more than our effect size.**
  [2511.20836v3](notes/2511.20836v3.md): MMLU-Pro 44.9% → 66.2% and GPQA 34.3% → 50.0% on
  Qwen3 4B from prompt choice, with the CoT trigger doing nearly all of it and prompt
  optimizers adding 66.2 → 66.3 for ~1,800 extra tokens. The template must be frozen
  across every arm.
- **Do not copy the confounded quantization comparison.**
  [2501.02342v1](notes/2501.02342v1.md) scored safetensors with lm-evaluation-harness and
  GGUF with llama.cpp's own tool, so its "no significant drop" is not a controlled
  ablation.
- **The vendor's own numbers are not our protocol.**
  [2511.23404v1](notes/2511.23404v1.md) states their harness uses "robust parsing logic"
  with constraint-based fallback, so published figures are inflated relative to a strict
  no-retry first-attempt protocol. The same report says the earlier prototype's
  perplexity and cache-size proxies "do not transfer reliably to downstream task scores or
  device-level latency and memory", and that adding linear-attention/SSM/extra-conv
  operators "does not improve aggregate quality and typically worsens device metrics".
  Notably it contains **no structured-output, schema-validity, malformed-return or
  quantized-retention results at all**, which is the gap this project fills.
- **Field-level metrics hide document-level failure.**
  [2602.14743v1](notes/2602.14743v1.md): Gemma3-1B F1_micro 0.89 with DOC_micro 0.10.
  First-attempt whole-output validity is the right unit.
- **Contamination submerges rather than clears.**
  [2601.06103v1](notes/2601.06103v1.md): continued pretraining on clean data drove the
  contaminated-vs-clean gap to nearly zero while the leaked information "merely submerged
  not erased" resurfaced during post-training. Our n-gram decontamination pass is
  necessary and not sufficient; [2402.15938v3](notes/2402.15938v3.md) offers CDD/TED as a
  cheap black-box check (51 samples at T=0.8, edit-distance peakedness), simulated with
  LoRA only.
- **Grading scripts and single prompts are themselves error sources.**
  [2407.04069v2](notes/2407.04069v2.md): automated parsing-script grading can misscore by
  >10% on many tasks, and undocumented decoding parameters risk test-set overfitting.

### Forgetting control, and what does not work

[2602.08813v2](notes/2602.08813v2.md) reports that **LoRA does not reliably prevent
forgetting**, that high-LR SFT badly damages helpfulness even at similar KL, that
parameter-space sharpness-aware minimization underperforms KL-space and is fragile, and
that EWC/SI fail at low task similarity. Their caveat binds here: downstream-time
protections assume the later recipe is known, which is false if QAT follows GRPO.
[2605.15220v1](notes/2605.15220v1.md) adds that LoRA-Merge underperformed using LoRA as a
proxy and then retraining, a direct caution against shipping a merged adapter rather than
a final pass on the chosen mixture. Replay has limits too:
[1811.11682v2](notes/1811.11682v2.md) shows 100% replay prevents forgetting while
degrading plasticity, and that a too-small buffer reintroduces forgetting through
over-fitting; [2603.16177v2](notes/2603.16177v2.md) shows domain test loss rising after
~5 epochs on a repeated small dataset, that 20% replay does not substitute for early
exposure, and that distribution-similarity metrics flipped sign (r = +0.90 to r = −0.98),
so embedding similarity cannot tell us how far our JSON/tool domain sits from pretraining.

The LoRA-versus-full question has genuine counter-evidence in both directions, so it stays
an experimental axis rather than a decision: [2210.04802v2](notes/2210.04802v2.md) has
LoRA generalizing far better OOD (24.99% relative exact match against 0.0% for full FT on
length extrapolation), [2605.09015v1](notes/2605.09015v1.md) has rsLoRA beating full FT and
DoRA at 3B under matched conditions but also introducing cross-script token leakage that no
other run showed, [2605.19018v1](notes/2605.19018v1.md) gives the theory (LoRA's edge is
variance reduction, and only when the true update is low-rank or the data noisy), and
[2309.05444v1](notes/2309.05444v1.md) has plain rank-4 LoRA underperforming full FT at 3B
(57.51 against 60.06). [2605.30537v1](notes/2605.30537v1.md) adds that LoRA specializes
more sharply than full FT, and that myopic loss- or gradient-based data selection wins now
and loses later (Current 62.4 against Random 58.2, but Future AUC 0.52 against 0.61,
Forgetting 9.4 against 4.8, OOD 46.5 against 54.0).

### Constrained decoding stays out of the metric, and the corpus says why

[2605.26128v1](notes/2605.26128v1.md) (The Constraint Tax) is decisive: hard
schema-constrained decoding gives 100% schema validity at a **43.5-point executable-
accuracy loss** on a sub-3B calendar tool-call task (91.5% → 48.0%), with 102 of 104
failures a single wrong `duration_minutes`; on their main suite validity 61.5 → 100% while
answer accuracy fell 19.7 → 11.0 and wrong-but-valid-schema rose 49.5 → 88.9. The tax
persists at 3B. "Reason free, constrain late" recovered 40.7% at 100% validity, and they
recommend treating schema validity as a serving SLO with
`Tax(m,t,c;b) = max(0, Acc(m,t,b) − Acc(m,t,c))` reported separately.
[2603.03305v1](notes/2603.03305v1.md) shows the damage is worst at exactly our scale (1B
grammar-constrained 15.24% strict GSM8K against 39.04% with draft conditioning, near zero
on GSM-Symbolic). [2602.12247v2](notes/2602.12247v2.md) (ExtractBench) found provider
structured-output modes *reducing* validity 51% → 37%, with one resume schema going from
62% in prompt mode to 0 of 42 accepted, and failures being trailing commas and truncated
JSON, which is cheap well-formedness signal we can train on.
[2605.02363v1](notes/2605.02363v1.md) reports naive prompting yielding 0% valid JSON at up
to 85% task accuracy, constrained decoding costing 3.6–8.2× latency and making 52.4% of
outputs exact duplicates. [2506.01151v1](notes/2506.01151v1.md) is the summary: grammar
enforcement took Mistral 0.09 → 0.52 and left content a coin flip.
One corpus item is unusable: [2510.03847v1](notes/2510.03847v1.md)'s central ablation table
is an explicit placeholder ("values are representative (fill with your measurements)"), so
none of its numbers are evidence.

### Specialization has a measured tax, and tool-use gains shrink at our scale

[2502.06589v1](notes/2502.06589v1.md) has a task-specialized baseline dropping **−67.7% on
BFCL-v3** outside its distribution, which is quantitative support for the operator's
standing preference to specialize to underlying technologies rather than one vendor's SDK.
[2508.12685v3](notes/2508.12685v3.md) states tool-use SFT gains shrink sharply below 3B,
and that a weaker generator drops verification pass rate 72.3% → 48.7% and BFCL 65.41 →
60.13, with **51% of rule-verification failures being parameter hallucination**.
[2511.09148v2](notes/2511.09148v2.md) measures the scale gap directly: **0.6B gains +0.70
against 8B +1.80**, with static training plateauing by iteration 2.
[2604.20148v1](notes/2604.20148v1.md) adds that one corrupted example in five dropped
Gorilla 38.0 → 26.0 while Spider lost 8 points only at 40% corruption, so AST matching is
brittle to label noise in a way execution-based evaluation is not, and that a 227.8M
hypernetwork added zero benefit over few-shot prompting (identical 47.0%).
[2601.04237v2](notes/2601.04237v2.md) contributes the "Ambiguity Loop": an agent retried a
malformed call with superficial variations until it exhausted its context, which shows
syntax-recovery training does not by itself instil flag-rather-than-assert.

## Baselines and datasets the field uses

**Tool-calling training data.** The reusable generation recipe is
[2406.18518v1](notes/2406.18518v1.md) (APIGen): three-stage format, execution and semantic
verification, with the finding that **training on data the later filters rejected actively
hurts, and hurts the smaller model more**. Generator pass rates ran 34.42%–84.15%, a
"thought" field raised pass rate, and four query styles are named with **parallel calls
underrepresented in public data**. It is single-turn REST and Python only, with no MCP, IPC
or container schemas, which is where our own corpus has to fill in.
[2409.00920v2](notes/2409.00920v2.md) (ToolACE) supplies the category mix (multi-type and
irrelevant queries are load-bearing; removing parallel-call examples damaged multi-tool
ability) and the hyperparameters that worked at 8B (LoRA rank 16 / alpha 32 / LR 1e-4 /
batch 48 / 3 epochs), plus the useful negative that self-judged complexity beat both a
stronger and a weaker external judge (59.22 against 57.61 / 57.67) and that few-shot ICL
underperformed zero-shot. [2508.12685v3](notes/2508.12685v3.md) adds multi-turn synthesis
(LoRA rank 16 / alpha 32 / batch 64 / LR 1e-4) and the warning that τ-Bench Airline is
unreliable. [2410.04587v2](notes/2410.04587v2.md) contributes the name-masking augmentation
and 7,500 irrelevance examples with empty-list labels.

**Known defects in public sets, to check before training.** xLAM trajectories mismatch
their declared schemas ([2605.16790v1](notes/2605.16790v1.md)); xLAM data contains an extra
space after `[` that breaks AST parsing ([2505.20192v3](notes/2505.20192v3.md)).
[2603.16901v1](notes/2603.16901v1.md) reached its result mostly through dataset repair.

**SQL and schema-perturbation data.** [2602.22223v1](notes/2602.22223v1.md) (SQaLe) offers
517,676 triples over 135,875 schemas. [2603.10697v1](notes/2603.10697v1.md) (EvoSchema)
supplies the perturbation taxonomy and the key result that **table-level structural changes
hurt far more than column-level** (Table Match F1 89.77 → 57.88), that naive irrelevant-table
augmentation hurts other perturbation types, and that schema pruning over-prunes.
[2411.08599v3](notes/2411.08599v3.md) (XiYan-SQL) contributes M-Schema and a SQL curriculum
(the ensemble is out of scope), and states SFT-only small models "struggle to transfer to
databases within a new domain". [2504.15077v5](notes/2504.15077v5.md) (Think2SQL) gives a
reward decomposition we can copy in shape: `R = 0.85·R_text2SQL + 0.10·R_FR + 0.05·R_TCR`,
with dense partial credit beating binary *especially for small models*, separate format and
tag-count rewards, and the finding that combining the two correctness rewards hurt and that
generic reasoning distillation actively hurt.

**Baseline choices in the corpus that bear on ours.**
[2511.22138v1](notes/2511.22138v1.md) is the closest published tier-1 comparison and it is
unreliable: TinyLlama-1.1B and TinyAgent-1.1B both score ~19.7% BFCL with zero multi-turn
success, its abstract's 65.74% / 55.62% headline is the off-the-shelf xLAM-2-3b-fc-r
baseline rather than the authors' own pipeline, and its Table I and Table IV contradict each
other (35.25% against 16.88%). Our base is claimed at 49.12, so the matched rerun is the
only defensible anchor. [2411.13676v1](notes/2411.13676v1.md) additionally reports that a
public-data-only 1.5B suffers specifically on 5-shot MMLU, which is the guardrail our
public-data-only arm is most likely to fail, and that pure SSMs have weak recall, so
LFM2.5's small attention fraction probably carries exact-schema recall. Data-mixture
selection has one usable result and one caution: [2403.08370v3](notes/2403.08370v3.md)
recovers most of 1,840 tasks from ~16 selected ones but its task-scaling curve is
non-monotonic with no principled pruning point, and [2605.03677v1](notes/2605.03677v1.md)
warns that naive filtering of always-easy and always-hard prompts hurts small students,
whose difficulty distributions are J- or U-shaped.

**GGUF and device measurement.** [2603.26603v2](notes/2603.26603v2.md) measured
**Q4_K_M beating IQ4_XS on both speed and energy** on Android CPU, with IQ4_XS winning only
peak memory, and separately found BERTScore favouring extractive behaviour and ranking
smaller models above better ones. [2605.04062v2](notes/2605.04062v2.md) reports Q4_0 at
262.66 tok/s against plain Q4_K 239.55 on an M4 Pro, and that its headline 15.16× speedup
is a 1.58-bit number rather than a 4-bit one. [2607.14181v1](notes/2607.14181v1.md)
compares six 4-bit PTQ techniques at 7B with pass@1: QuIP# worst, AQLM matching FP, GPTQ
+9 points on CodeLlama and −13 on Qwen, and quality and correctness degradation not
co-occurring. [2605.20706v1](notes/2605.20706v1.md) covers browser/WebGPU llama.cpp
memory and throughput with no quality evaluation.

## Metrics and evaluation choices

The corpus converges on a protocol that is stricter than the one in `overview.md`, and the
differences are all adopted at s2.3.

1. **Single serving stack, greedy, config disclosed.**
   [2608.04714v1](notes/2608.04714v1.md) makes backend choice ≈39% of realistic variance at
   our scale, and prescribes greedy for all cross-backend comparisons, McNemar / Wilcoxon
   with Benjamini-Hochberg correction, plus Cohen's κ, error-set Jaccard and PCA, with
   "severe" defined as flipping >5% of items.
2. **Frozen prompt and CoT template across every arm**
   ([2511.20836v3](notes/2511.20836v3.md)).
3. **Four separated structured-output metrics, not one.**
   [2605.26128v1](notes/2605.26128v1.md): schema validity, answer accuracy,
   wrong-but-valid-schema rate, and the constraint tax, with a six-way error taxonomy.
   Whole-output first-attempt validity is the unit
   ([2602.14743v1](notes/2602.14743v1.md)).
4. **Flag rate always paired with false-flag rate and functional accuracy**, plus BFCL
   Overall reported separately from AST-only
   ([2608.03092v1](notes/2608.03092v1.md), [2411.13676v1](notes/2411.13676v1.md),
   [2408.04682v2](notes/2408.04682v2.md)). Adopt the paired
   false-refusal-rate / missed-refusal-rate design and the Hierarchical Refusal Score
   (Detection F1 × Category Accuracy) from
   [2510.10390v1](notes/2510.10390v1.md), which also warns that models default to a
   catch-all refusal category 25% of the time and that >73% of predictions sit at maximum
   confidence despite 40–69% accuracy.
5. **A hallucination axis alongside accuracy.** [2510.22977v2](notes/2510.22977v2.md)'s
   R_NTA (calls with no tool available) and R_DT (calls to a different tool than provided)
   are cheap to compute and would have caught a +9.9% BFCL "win" that was really a 90.2%
   hallucination rate.
6. **A corrupted-evidence and text-only control on the flag probe**
   ([2512.04597v1](notes/2512.04597v1.md)), so a lexical-pattern-matching pass is
   distinguishable from real behaviour.
7. **Integrity as a separate axis from task correctness**, with Clopper-Pearson intervals
   and Fisher exact tests ([2605.02964v1](notes/2605.02964v1.md)).
8. **Paired tests, not pooled aggregates** ([2608.18578v1](notes/2608.18578v1.md), where an
   unpaired Fisher test missed a real INT8 penalty that a paired test found at p=0.018).
9. **Per-axis retention floors under any average**, because
   [2605.04062v2](notes/2605.04062v2.md) demonstrates an average rising while
   instruction-following falls 8.5%.
10. **Perplexity is a diagnostic only**, never a stand-in for a downstream score
    ([2602.15563v1](notes/2602.15563v1.md), [2605.15208v1](notes/2605.15208v1.md),
    [2511.23404v1](notes/2511.23404v1.md)).
11. **Ablate the tool-return serialization format**, which
    [2605.19341v1](notes/2605.19341v1.md) finds is an independent driver of hallucination.

Two evaluation lanes named in the corpus are deliberately not adopted. An online LLM judge
is used by [2605.25850v1](notes/2605.25850v1.md) (validated at 82.3%) and by
[2508.12685v3](notes/2508.12685v3.md), and judge reliability in this corpus is poor:
inter-judge Pearson fell 0.62 → 0.31 ([2509.15557v1](notes/2509.15557v1.md)), a GPT-5.3
judge overestimated the non-hack ratio ([2604.16242v1](notes/2604.16242v1.md)), and an
external model-scored reward caused multi-turn optimization collapse
([2601.03525v3](notes/2601.03525v3.md)). Judges therefore stay out of the training signal
entirely and are admissible only as a secondary, disclosed cross-check at evaluation time,
which is how [2608.03092v1](notes/2608.03092v1.md) uses its API-Bank two-tier grading.
Published leaderboard figures stay context rather than evidence:
[2503.16416v2](notes/2503.16416v2.md) notes leaderboard benchmarks can be over-optimistic
and that binary outcome metrics under-report intermediate progress.

## Open questions: answered, narrowed, or marked unanswerable

The five open questions carried out of `s0.2`, and the two low-confidence assumptions,
each get a disposition here. Nothing is left implicit.

**1. Does the in-house stack corpus exist, and where? — still open, and now costed.**
The literature cannot answer it, but it quantifies what turns on the answer:
[2505.04016v1](notes/2505.04016v1.md) measures public data alone at 63.1% / 68.3% schema
accuracy against 89.0% / 81.7% for synthetic plus public, so the public-data-only arm is a
real floor with a ~20-point gap above it rather than a formality. The access request stays
at `s4.1`, where the answer changes what runs.

**2. Is the gated xlam tool-calling set reachable? — still open, and now less critical.**
Two papers document defects in xLAM data (trajectories mismatching declared schemas,
[2605.16790v1](notes/2605.16790v1.md); an extra space after `[` that breaks AST parsing,
[2505.20192v3](notes/2505.20192v3.md)), and [2406.18518v1](notes/2406.18518v1.md) supplies a
complete generation-and-verification recipe. The fallback path is therefore stronger than
assumed at `s0.2`, and a repair-and-validate pass is now mandatory on whichever public set
we get.

**3. Does the 65K vocabulary handicap tool-call tokenization? — unanswerable from this
corpus.** Every tokenizer paper retrieved
([2310.08754v4](notes/2310.08754v4.md), [2507.22543v1](notes/2507.22543v1.md),
[2511.03825v1](notes/2511.03825v1.md), [2511.20849v1](notes/2511.20849v1.md),
[2601.13260v2](notes/2601.13260v2.md), [2510.13481v2](notes/2510.13481v2.md),
[2605.12928v1](notes/2605.12928v1.md)) operates at tokenizer-construction or pretraining
time and cannot be applied to a frozen 65,536-token vocabulary;
[2605.01347v1](notes/2605.01347v1.md) adds that token-level distillation needs a teacher
sharing the vocabulary and that cross-vocabulary distillation is an open problem. The
tokens-per-tool-call diagnostic at `s4.2` stands as the only route to an answer, and it
stays a diagnostic. One adjacent datum: [2608.20210v1](notes/2608.20210v1.md) found an
oversized 49,152-entry vocabulary wasting ~13M parameters in a 150M model of the same
architecture family, so the cost is real at small scale even though we cannot act on it.

**4. Which reinforcement-learning signal, if any, instils flag-rather-than-assert? —
substantially answered, and it is not RL first.** The strongest measured intervention is a
preference rung with two-sided pairs (honest abstention as chosen, against both a fabricated
call and a needless refusal as rejected), which cut fabricated-call rate 90.2 → 55.8 in
[2510.22977v2](notes/2510.22977v2.md); [2510.10390v1](notes/2510.10390v1.md) independently
has DPO beating SFT for refusal by 3.4× at 7B. Verifiable-reward RL stays a matrix arm and
not a commitment, because the two papers at our scale report collapse
([2510.07737v1](notes/2510.07737v1.md), [2605.27954v1](notes/2605.27954v1.md)) and the
reward would have to be built from scratch: no published tool-use reward scores "did the
model correctly flag a bad return" ([2605.16790v1](notes/2605.16790v1.md)'s execution term
scores "did the call run"). A cheap non-RL alternative is now on the list:
[2406.08391v3](notes/2406.08391v3.md)'s LoRA-plus-Jensen-Shannon-regularizer on
target-sequence logits, ~1,000 graded examples, posing the flag decision as a two-token
multiple-choice completion.

**5. Can a 1.2B model hold the instruction-following lead while gaining three or more
points of tool-calling accuracy? — answered pessimistically, and the plan changes rather
than the claim.** [2601.02151v1](notes/2601.02151v1.md) is the closest matched evidence:
plain tool-call SFT bought BFCLv3 +0.9 while IFEval fell 3.2 points and the general average
fell 6.3, and the best mitigation still breached a 2-point IFEval guardrail.
[2505.20192v3](notes/2505.20192v3.md) has code ability collapsing outright.
[2511.09148v2](notes/2511.09148v2.md) has 0.6B gaining only +0.70 against 8B's +1.80. So a
+3.0 BFCLv3 gain at flat guardrails is at the hard end of what the field has demonstrated
at this scale. Three mitigations are promoted from "possible" to "planned" in response:
entropy-weighted loss, a replay buffer with a measured floor, and parameter-space merging
back toward the base. The claim margin stays at +3.0 because lowering it after seeing the
literature but before seeing a result would be the wrong direction of adjustment; the
guardrail tolerance also stays at 2.0 points, and a forced trade-off is a reportable finding
rather than a redefinition.

**Low-confidence assumption A: the 0.80 flag rate / 0.10 false-flag pair. — revised down.**
Five independent measurements put that pair above the demonstrated frontier:
[2408.04682v2](notes/2408.04682v2.md) has GPT-4o and Claude-3-Opus at 76.6 and 71.1 on
Insufficient Information; [2605.19341v1](notes/2605.19341v1.md) caps frontier
uncertainty-abstention at 76.9%; [2603.10697v1](notes/2603.10697v1.md) reaches ~84%
true-positive at 7–8B with a real false-positive cost;
[2605.29523v1](notes/2605.29523v1.md) finds 0.750 the best refusal score at 8B; and
[2510.10390v1](notes/2510.10390v1.md) measures the trade-off itself at r = −0.78, so the two
numbers cannot be pushed independently. **New target: flag rate ≥ 0.70 with false-flag rate
≤ 0.15**, reported as a pair, with functional accuracy alongside so the degenerate
always-flag policy documented in [2411.13676v1](notes/2411.13676v1.md) and
[2511.22138v1](notes/2511.22138v1.md) cannot pass. Confidence rises from low to medium:
literature-anchored, still our own construction.

**Low-confidence assumption B: decode throughput within 5% of base. — kept, reframed as a
sanity check.** Post-training does not change the architecture, so identical quantization
format on an identical backend should give near-identical throughput by construction, and
[2608.20210v1](notes/2608.20210v1.md) confirms the one architecture change that might have
helped is unavailable (llama.cpp shape-checks tensors at load, so dead conv channels cannot
be pruned out). The corpus does show format choice mattering more than 5%
([2605.04062v2](notes/2605.04062v2.md): Q4_0 262.66 tok/s against Q4_K 239.55;
[2603.26603v2](notes/2603.26603v2.md): Q4_K_M beating IQ4_XS on speed and energy), so the
criterion is only meaningful if format and backend are pinned. **Restated: the 4-bit
throughput comparison is made at identical quantization format and backend, within 5%, and
is a sanity check on export rather than a research finding.** Confidence low to medium.

## Scope revision (s2.3)

`overview.md` moves from `provisional (pre-literature)` to
`literature-grounded (revised s2.3)`. Fourteen changes, each with the evidence that forced
it. Everything not listed here survived the literature unchanged, including the tier-1-only
scope, the prediction unit, the group-split keys, BFCLv3 overall as primary, the matched-rerun
baseline policy, the public-data-only ablation arm, and every out-of-scope exclusion.

| # | Change | Evidence |
| - | ------ | -------- |
| 1 | **Quantization is the last step.** All quality training runs in full or mixed precision; the 4-bit GGUF export follows; the only in-format training is the post-export recovery rung. Quantize-then-dequantize-to-keep-training is banned outright. | [2511.19495v1](notes/2511.19495v1.md), [2508.04073v1](notes/2508.04073v1.md), [2512.18934v1](notes/2512.18934v1.md) |
| 2 | **Full QAT is dropped from the ladder; quantization-aware distillation stays.** | [2605.17471v1](notes/2605.17471v1.md) (20B tokens / 28.7 GPU-hours per run at 1B, and 4-bit is the regime with "limited room for improvement"), [2608.20210v1](notes/2608.20210v1.md) (QAT gave a non-finite loss on step one in this same architecture family and was abandoned undiagnosed), [2601.20088v3](notes/2601.20088v3.md) (QAT breaks RL-learned capabilities, 24.8 against QAD's 34.3) |
| 3 | **The recovery rung is specified, not just named:** PTQ warm-start, then distillation from our own full-precision checkpoint with KL at T=1, attention layers and their preceding recurrent layers held at BF16, KV-cache FP8, LR 1e-5 to 1e-6; plus an error-targeted stepwise-DPO pass of a few hundred examples. | [2506.09104v1](notes/2506.09104v1.md), [2601.20088v3](notes/2601.20088v3.md), [2505.11574v4](notes/2505.11574v4.md) |
| 4 | **Reliability target lowered to flag rate ≥ 0.70 / false-flag ≤ 0.15**, reported as a pair with functional accuracy. | [2510.10390v1](notes/2510.10390v1.md), [2408.04682v2](notes/2408.04682v2.md), [2605.19341v1](notes/2605.19341v1.md), [2603.10697v1](notes/2603.10697v1.md), [2605.29523v1](notes/2605.29523v1.md) |
| 5 | **A fabrication axis joins the metric set:** rate of calls to a tool that was not provided, and rate of calls to a different tool than the one provided, on every checkpoint. | [2510.22977v2](notes/2510.22977v2.md) (+9.9% BFCL alongside a 34.8 → 90.2% fabrication rate) |
| 6 | **The malformed-return probe gains a corrupted-evidence and text-only control**, and is stratified by JSON nesting depth. | [2512.04597v1](notes/2512.04597v1.md), [2505.04016v1](notes/2505.04016v1.md) |
| 7 | **One serving backend, greedy decoding, generation config disclosed, GGUF conversion counted as part of the backend.** Paired significance tests (McNemar / Wilcoxon, Benjamini-Hochberg) plus agreement statistics, never pooled-only aggregates. | [2608.04714v1](notes/2608.04714v1.md), [2608.18578v1](notes/2608.18578v1.md), [2501.02342v1](notes/2501.02342v1.md) |
| 8 | **One frozen prompt and reasoning template across every arm**, disclosed in full. | [2511.20836v3](notes/2511.20836v3.md) (15–21 point swings from template alone) |
| 9 | **Structured output is scored as four separate numbers** (schema validity, answer accuracy, wrong-but-valid-schema rate, constraint tax) on whole first-attempt outputs, with a reasoning-aware validator. | [2605.26128v1](notes/2605.26128v1.md), [2602.14743v1](notes/2602.14743v1.md), [2603.16901v1](notes/2603.16901v1.md) |
| 10 | **Parameter-space merging is added to the ladder** as a cheap post-SFT/DPO rung (K=2, Slerp / TIES / DARE), with the rule that a merged adapter is never the shipped artifact without a final pass on the chosen mixture. | [2511.23404v1](notes/2511.23404v1.md) §4.4, [2407.08699v2](notes/2407.08699v2.md), [2605.15220v1](notes/2605.15220v1.md) |
| 11 | **The SFT rung before preference optimization and RL is now mandatory rather than merely first**, and the loss is entropy-weighted rather than plain cross-entropy. The Instruct checkpoint stays the starting point. | [2604.20316v1](notes/2604.20316v1.md), [2605.02572v1](notes/2605.02572v1.md), [2601.02151v1](notes/2601.02151v1.md), [2505.20192v3](notes/2505.20192v3.md) |
| 12 | **The RL reward architecture is specified up front**: a decomposed format / parse / execution / answer reward with graded per-parameter credit, answer correctness weighted above the whole structural budget, the schema term mandatory alongside any execution term, sparse dimensions up-weighted rather than equal-weighted, degenerate all-pass and all-fail groups dropped, no naive pass-rate densification, and a flat penalty on detected shortcut behaviour. Entropy, valid-action ratio and duplication ratio are monitored every run. | [2605.16790v1](notes/2605.16790v1.md), [2608.03092v1](notes/2608.03092v1.md), [2602.03452v2](notes/2602.03452v2.md), [2601.03525v3](notes/2601.03525v3.md), [2603.07084v2](notes/2603.07084v2.md), [2605.27954v1](notes/2605.27954v1.md) |
| 13 | **No model-scored reward, online or stored, enters the training signal.** A judge is admissible only as a disclosed secondary cross-check at evaluation time. | [2601.03525v3](notes/2601.03525v3.md), [2509.15557v1](notes/2509.15557v1.md), [2604.16242v1](notes/2604.16242v1.md), [2605.25850v1](notes/2605.25850v1.md) |
| 14 | **Data hygiene is promoted to a gating step**: irrelevance and multi-type categories are mandatory in the mix, filter-rejected items are discarded rather than trained on, public sets get a repair-and-validate pass, a replay fraction is carried with a measured floor, and decontamination is n-gram plus a peakedness check. | [2409.00920v2](notes/2409.00920v2.md), [2406.18518v1](notes/2406.18518v1.md), [2512.18934v1](notes/2512.18934v1.md), [2601.06103v1](notes/2601.06103v1.md), [2402.15938v3](notes/2402.15938v3.md) |

### The starting-checkpoint decision, recorded explicitly

[2511.01934v2](notes/2511.01934v2.md) reports that starting from an Instruct model *hurt*
reinforcement learning relative to a base model, which if taken at face value would move our
starting point. It is not taken at face value, for three reasons. Their setting is pure GRPO
with no supervised warm start, and our pipeline is warm-start-first, which is the very thing
[2604.20316v1](notes/2604.20316v1.md) and [2605.02572v1](notes/2605.02572v1.md) show is
required below 3B. Their own scale curve (1.5B at 63.73 against 32B at 78.99) shows the
regime where their result was measured is not ours. And every guardrail in this project is
defined as a delta against a matched rerun of the Instruct checkpoint, which is also what the
operator's gateway would otherwise run. A base-checkpoint arm is recorded as a cheap optional
ablation if a base LFM2.5-1.2B is available, and nothing depends on it.

### Re-running the s0.2 checks against the literature

- **Leakage.** The group-split keys (source API schema for tool data, source repository file
  for code and SQL) survive unchanged. One addition: [2601.06103v1](notes/2601.06103v1.md)
  shows decontamination submerges leaked content rather than removing it, and that it
  resurfaces during post-training, so the matched-base rerun is now doubly load-bearing and
  the decontamination pass gains the black-box peakedness check from
  [2402.15938v3](notes/2402.15938v3.md).
- **Measurability.** Improved. Every success criterion now has a named measurement protocol
  and a named failure mode it is designed to catch, and three criteria that were single
  numbers are now reported as pairs or quartets so a degenerate policy cannot satisfy them.
- **Feasibility.** Reduced but still positive. Full QAT is out on cost grounds, RL is
  demoted to an arm rather than a plank, and the SFT rung alone has a measured precedent of
  9.6% → 88.9% schema accuracy at 1B ([2505.04016v1](notes/2505.04016v1.md)), which is the
  single most encouraging number in the corpus for a project of this size. The +3.0 BFCLv3
  margin is the part most at risk.
- **Fairness and governance.** Unchanged in principle, sharpened in practice: worst-category
  reporting stays the governance surface, and [2605.15208v1](notes/2605.15208v1.md) adds a
  reason to check it *after* quantization specifically, since 2.5–5.6% of items developed new
  bias at 4-bit while perplexity moved less than 3%.
- **Risk.** Two risks are upgraded. Quantization collapse of structured output moves from
  "reported" to "replicated across nine papers with numbers", and reward hacking gains a
  named primary exploit ("minimally valid JSON satisfying a shallow schema check",
  [2605.02964v1](notes/2605.02964v1.md)) with measured mitigations. One risk is added: a
  measurement-protocol risk, since backend and template choice each move scores further than
  our target effect size.

## Scope sign-off (s2.4)

Autonomous mode is on for this project, so this checkpoint was decided here rather than
escalated.

**Decision: proceed on the revised scope, unchanged in shape.** Tier 1 only, tool-calling
reliability and structured output as the attack surface, shipped as 4-bit GGUF under 1.5 GB,
with the fourteen changes above folded in.

**Reasoning.** The literature did not undermine the premise; it sharpened the method and
lowered one target. The core bet is better supported after reading than before: a supervised
rung alone has a measured precedent of taking a 1B model from 9.6% to 88.9% schema accuracy
without constrained decoding, and no paper in the corpus reports structured-output,
schema-validity, malformed-return or quantized-retention results for this model family,
including the vendor's own technical report. The gap this project aims at is real and
unoccupied. Against that, the two hardest parts are now quantified rather than assumed: the
+3.0 BFCLv3 margin at flat guardrails sits at the hard end of what the field has shown below
3B, and 4-bit retention of structured output is the axis most likely to fail. Both are
measurable before any large spend, which is why the shape does not need to change now.

**What would have justified escalating instead.** A rescope would be owed if the evidence had
killed the deliverable (it did not: 4-bit is recoverable, 2-bit is not, and we ship 4-bit), or
if the primary metric had turned out unmeasurable, or if the compute implied by the plan
exceeded the allowance. The allowance question is a real one and it lands at `s3.3`, where the
matrix is sized against the enforced quota, with the cost of a larger version stated before
anything is spent past approval.

No operator reviewed this decision.

## Outputs

- `notes/<id>.md` — 212 per-paper notes, one per paper in the s1 corpus.
- `reader-brief.md` — the shared instruction sheet the reading pass ran against.
- This report — the s2.2 synthesis, the s2.3 assumption walk, and the s2.4 decision.
- `../../overview.md` — revised in place to `literature-grounded (revised s2.3)`.

## Next steps

Stage 2 is complete. Stage 3 turns the fourteen changes into a hypothesis set and an
experiment matrix, then sizes that matrix against the enforced compute allowance and writes
`budget.json`. Two items carry forward as open and are due before `s4` can finish: repository
access for the in-house corpus, and whether the gated public tool-calling set is reachable
with the credentials on file.
