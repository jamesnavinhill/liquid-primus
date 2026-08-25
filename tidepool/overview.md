# Tidepool — a stack-specialized on-device LFM2.5 fleet, starting with reliable tool calling

Status: literature-grounded (revised s2.3)

_Revised against 212 papers at `s2.3`. The evidence for every change, and for every
assumption that survived unchanged, is in `stages/s2-research-summary/report.md`._

## Project description

The operator runs an agent gateway (litellm/langfuse routing, with a voice/video layer on
top) through which every unit of work flows: research, planning, implementation, auditing,
tool calls, and code execution. Today that work is billed per API call to frontier models.
The goal is a small, fast, private model fleet that the gateway can route to per task,
running on a single 8 GB consumer GPU or on CPU, that does the same work at a fraction of
the cost. The starting material is LiquidAI's LFM2.5 family: a linear/hybrid architecture
(gated short-convolution blocks plus a few grouped-query attention blocks) that is fast to
prefill and decode on consumer hardware, published under a license that permits derivatives,
and shipped across every size tier the fleet needs.

The full ask spans four device tiers, a vision-language lane, a hybrid mixture-of-experts
family, and a quantization lane. That is more than one research project can execute
honestly, and the operator's own directive says how to sequence it: start in the smallest
tier where a win is cheapest to demonstrate, prove it holds, then scale up, and name a
specific weakness to improve. **This project takes that instruction literally.** Its primary
lane is tier 1, the sub-1.5 GB micro-orchestrator built from `LFM2.5-1.2B-Instruct`, and the
weakness it attacks is the one the operator ranked first: reliable tool and function calling,
including the behaviour of flagging rather than asserting when a tool returns garbage. Tiers
2 through 4, the vision-language models, and the mixture-of-experts lanes are sequenced
behind that result, and the boundary is stated in "Out of scope" below.

The prediction unit is **one agent turn**: a system prompt, a conversation history, a set of
tool schemas, and any tool results returned so far, in; one model response out, which is
either a structured tool call, a structured refusal or flag, or a natural-language answer.
The consumer is the operator's gateway, which parses that response programmatically. A
response that is well-written prose but structurally invalid is a failure at this interface,
which is why structural validity on the first attempt is a headline metric here rather than
a footnote.

Two things follow from the operator's standing preferences and shape the work throughout.
First, specialization aims at the **underlying technology** (SQL dialects, JSON Schema, MCP,
inter-process communication, containers, OpenTelemetry tracing semantics) and not at any one
vendor's SDK, so the fleet survives the stack shuffling underneath it; that is a stated owner
decision and it governs both the training mix and the in-house evaluation. Second, a quality
win that costs on-device efficiency is not a win: every improved checkpoint must ship a
4-bit GGUF that holds the full-precision quality bar inside the tier's on-disk footprint, and
that quantized form is what gets measured, not only the BF16 parent.

### Dataset: a public post-training mix plus an in-house stack corpus

There is no single dataset. The training side is a mix of open, license-checked, published
sets covering the capability axes, and the evaluation side is a suite of published benchmarks
plus two probes built here. All public identifiers below were verified by the operator against
Hugging Face on 2026-08-25 and are re-verified at `s4.1`; the in-house corpus does not exist
yet and its availability is an open question recorded below.

| Source | Role | License / access | Scale | Teaches or measures |
| ------ | ---- | ---------------- | ----- | ------------------- |
| `Salesforce/xlam-function-calling-60k` | ~~train~~ **not used** | CC-BY-4.0, **gated, access refused** | 60k | superseded at `s4.1` by ToolACE + Synth-APIGen + Hermes (70,387 rows, all Apache-2.0). The terms have not been accepted on the account whose token we hold |
| `Team-ACE/ToolACE` | train | Apache-2.0 | 11,300 | multi-turn tool calling with returns; added at `s4.1` |
| `argilla/Synth-APIGen-v0.1` | train | Apache-2.0 | 49,402 | execution-verified single-call tool data; added at `s4.1` |
| `NousResearch/hermes-function-calling-v1` | train | Apache-2.0 | 11,578 across 5 configs | multi-turn tool calling and the only JSON-schema-conditioned training signal; added at `s4.1` |
| `LiquidAI/ifstruct-v1.0` | **eval only** | Apache-2.0 | 2,000 test prompts | valid JSON/YAML output under varied phrasing. Corrected at `s4.1`: the repository ships a single test split with no gold responses, so it cannot be trained on. The structured-output training signal comes from the Hermes JSON-mode configs (2,583 rows) |
| `LiquidAI/antidoom-mix-v1.0` | train (prompts only) | Apache-2.0 | 100k–1M | prompt set for preference pairs against looping and doom behaviour |
| `m-a-p/CodeFeedback-Filtered-Instruction` | train | Apache-2.0 | 100k+ | iterative code generation with execution feedback |
| `b-mc2/sql-create-context` | train | CC-BY-4.0 | 78,577 | schema-grounded text-to-SQL, built to suppress invented column names |
| `Clinton/Text-to-SQL-v1` | train | Apache-2.0 | 100k+ | text-to-SQL scale filler |
| In-house stack corpus | train | operator-owned | 11 public non-forked repositories, ~270 MB | the operator's repo idioms, schema and query patterns. Gateway trajectories are **not** reachable: they live in a running service no credential on file reaches (`s4.1`) |
| BFCLv3 / BFCLv4, ToolSandbox | eval | public benchmark | — | tool-calling accuracy and reliability, the primary axis |
| IFEval, IFBench, Multi-IF | eval | public benchmark | — | instruction-following guardrails |
| MMLU-Pro | eval | public benchmark | — | general-knowledge guardrail. GPQA Diamond is gated and access was refused at `s4.1`, so MMLU-Pro carries this axis alone |
| Malformed-tool-return probe | eval | built here (`s4`) | ~300 items | flag-rather-than-assert behaviour on broken tool output |
| Stack-idiom code probe | eval | built here (`s4`) | ~150 items | idiomatic output for the operator's underlying technologies |

**Split strategy: group split, with the grouping key being the source API schema for
tool-calling data and the source repository file for code and SQL data.** The temporal
ordering that would motivate a time split is absent here, and a random split would let the
same tool schema or the same repository file appear on both sides, which inflates a
tool-calling score exactly the way this project must not be inflated. Published benchmark
test sets stay physically separate from every training mix at the file level, and a
decontamination pass at `s4.3` checks the training mix against benchmark prompts before any
run. The two probes built here are held out entirely and never enter a training mix.

### Target & label window

There is no learned label in the supervised sense. What is scored is a **programmatic
judgement on a single model response**, computed after generation and using only the prompt,
the tool schemas, the conversation history, and a reference answer fixed before the model ran:

- **Tool calling.** The emitted call is parsed and compared against the reference by abstract
  syntax tree match on function name and arguments, and, where the benchmark supports it, by
  executing the call and comparing the result. Scoring is per BFCL category (simple, multiple,
  parallel, parallel-multiple, multi-turn, irrelevance, live), and the per-category scores are
  reported alongside the aggregate.
- **Structured output.** The response is validated against the requested JSON or YAML schema
  on the **first** attempt, with no retry, no repair pass, and no constrained decoding. First
  attempt is the contract because the gateway does not retry.
- **Flag rather than assert.** On the probe, a tool return is deliberately empty, truncated,
  type-wrong, or an error payload. A response counts as correct when it names the problem and
  declines to supply the missing value, and incorrect when it asserts a value the tool never
  returned. The complementary false-flag rate is measured on matched well-formed returns.

No post-prediction information enters any of these. Reference answers, schemas, and probe
labels are fixed before the model is run, the model sees no benchmark answer key, and no
metric consults anything the gateway would not have at inference time.

### Evaluation

- **Primary axis, tool calling:** BFCLv3 overall and per-category, BFCLv4, ToolSandbox
  success rate, and IFStruct first-attempt schema validity. Reported as absolute scores and
  as deltas against a matched rerun of every baseline in the same harness with the same
  seeds, decoding parameters, and prompt templates.
  - **Neither BFCLv3 nor IFStruct has a public anchor at tier 1, and no published figure may
    be quoted beside either.** Recorded at `s5.2`, once both had run at full item counts. For
    BFCLv3 the reason is scale: this harness reports an unweighted mean over 11 named AST and
    restraint categories, and the published 52.43 is a weighted average over a wider benchmark
    including categories not run here. For IFStruct the reason is that **the LFM2.5-1.2B-Instruct
    card publishes no IFStruct score at all** — its seven columns are GPQA, MMLU-Pro, IFEval,
    IFBench, Multi-IF, AIME25 and BFCLv3. The 85.49 in the source material belongs to LFM2.5-2.6B,
    a separately post-trained checkpoint of twice the size. On both metrics the operative bar is
    therefore the `B1` reference row's own figure from this harness at full counts, which is what
    the plan specified: a matched rerun, never a quoted number.
- **Reliability axis:** flag rate and false-flag rate on the malformed-tool-return probe,
  reported as a pair, with functional accuracy alongside. Either number alone is gameable: a
  model that flags everything scores perfectly on one and catastrophically on the other, and
  two papers in the corpus record models scoring 100% on irrelevance detection while scoring
  0% functionally. The probe carries a corrupted-evidence and a text-only control arm, so a
  model pattern-matching on wording is distinguishable from one actually reading the return,
  and it is stratified by JSON nesting depth.
- **Fabrication axis:** rate of calls to a tool that was not provided, and rate of calls to a
  different tool than the one provided, measured on every checkpoint. Added because a +9.9%
  benchmark gain has been published alongside a fabrication rate rising from 34.8% to 90.2%,
  and neither BFCL nor IFStruct would have caught it.
- **Structured output, four numbers not one:** first-attempt whole-output schema validity,
  answer accuracy, wrong-but-valid-schema rate, and (where a constrained mode is measured at
  all) the accuracy tax it charges. Field-level scores are not reported alone: one paper in
  the corpus records per-field F1 of 0.89 against document-level accuracy of 0.10. The
  validator is reasoning-aware, since thinking tokens break naive format checks.
- **Guardrails, no regression:** IFEval, IFBench, Multi-IF for instruction following;
  MMLU-Pro and GPQA Diamond for general knowledge. These are held, not maximized.
- **Stack code:** the in-house idiom probe, plus HumanEval and MBPP as public anchors, so a
  claim about the operator's stack does not rest only on a probe built here.
- **Subgroup reporting:** by BFCL category, by tool-schema complexity (argument count and
  nesting depth), and by language on Multi-IF. Worst-category performance is reported
  alongside the mean in every results table, and the worst-category floor below is a
  gating criterion rather than a diagnostic.
- **Efficiency:** on-disk GGUF size, prefill and decode throughput, and peak resident memory,
  all measured at 4-bit on one fixed hardware configuration for every model compared.
- **Quantization retention:** every axis above, measured on the 4-bit GGUF and on its own
  full-precision parent in the same harness, reported per axis and not only as an average.
  Perplexity is a diagnostic and never a stand-in for a downstream score.
- **Measurement protocol, fixed before the first run and identical across every arm:** one
  serving backend, greedy decoding, the full generation config disclosed, one frozen prompt
  and reasoning template, and GGUF conversion counted as part of the backend rather than as a
  neutral format change. Backend choice alone accounts for roughly 39% of realistic score
  variance at this model size, and template choice alone has moved published scores by 15 to
  21 points, both of which exceed this project's target effect size. Significance is tested
  paired (McNemar or Wilcoxon with Benjamini-Hochberg correction) and reported with agreement
  statistics, because a pooled aggregate has been shown to hide a real per-item penalty.

### Baselines and model candidates

Baselines, each rerun in this project's harness under matched conditions rather than quoted
from a card:

1. `LiquidAI/LFM2.5-1.2B-Instruct` — the vendor post-trained checkpoint this project starts
   from and the reference for every "no regression" claim. Published tier-1 scores: BFCLv3
   49.12, IFEval 86.23, IFBench 47.33, Multi-IF 60.98, MMLU-Pro 44.35, GPQA 38.89.
2. Granite 4.0-1B — the only sub-2B competitor with a stronger published tool-calling number
   (BFCLv3 52.43), and therefore the tier's incumbent on the primary axis.
3. `NovachronoAI/LFM2.5-1.2B-Nova-Function-Calling-GGUF` — an existing community
   function-calling finetune of the same base, and the direct test of whether this project
   beats work already done.
4. `unsloth/LFM2.5-1.2B-Instruct-GGUF` (UD-Q4_K_XL) and the vendor QAD Q4_0 inside
   `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` — the quantization baselines. A quantized result here
   is measured against the best cheap quant, not against a naive one.
5. **Public-data-only LoRA SFT** — the same recipe trained without any in-house stack data.
   This is the ablation that isolates whether stack specialization, rather than tool-calling
   SFT in general, is what buys the win.

Model candidates, simplest first, all starting from LFM2.5-1.2B checkpoints:

1. LoRA supervised fine-tuning on a tool-calling and structured-output mix.
2. Higher-rank or full supervised fine-tuning, with a data-mix ratio ablation across the
   capability axes.
3. Preference optimization over generated malformed-tool-return pairs, targeting the
   flag-rather-than-assert behaviour that supervised fine-tuning alone is reported not to
   instil in small models. Pairs are two-sided: honest abstention as chosen, against both a
   fabricated call and a needless refusal as rejected, which is the construction measured to
   cut fabrication rate from 90.2% to 55.8%. The objective is a regularized or
   importance-sampled variant rather than textbook DPO, since textbook DPO is proven to push
   down the preferred response's probability too and measures as the weakest offline objective
   at 0.5 to 1.5 billion parameters. Auto-generated pair labels are noisy, so tail-loss
   capping is on by default.
4. Parameter-space merging of the specialized checkpoint back toward the base as a cheap
   forgetting-control rung, with two-way merges only. A merged adapter is never the shipped
   artifact without a final pass on the chosen mixture.
5. Reinforcement learning with verifiable rewards (GRPO) as a matrix arm rather than a plank,
   since the two studies closest to this scale report outright collapse. It runs only from a
   supervised warm start, with a decomposed format / parse / execution / answer reward that
   gives graded per-parameter credit, weights answer correctness above the whole structural
   budget, keeps the schema term mandatory alongside any execution term, up-weights the sparse
   dimensions rather than equal-weighting them, drops degenerate all-pass and all-fail groups,
   and carries a flat penalty on detected shortcut behaviour. Entropy, valid-action ratio and
   duplication ratio are monitored every run. No model-scored reward, stored or online, enters
   the training signal.
6. Post-export recovery in the quantized format: post-training quantization as a warm start,
   then distillation from this project's own full-precision checkpoint with KL at temperature
   1, the attention layers and their preceding recurrent layers held at BF16, KV-cache at FP8,
   learning rate 1e-5 to 1e-6, followed by an error-targeted stepwise preference pass of a few
   hundred examples. Full quantization-aware training is **not** on the ladder: it costs
   roughly 20 billion tokens per run at this scale for the regime where it has the least to
   offer, it failed outright in a published run on this same architecture family, and it is
   measured to break reinforcement-learned capabilities that distillation preserves.

All quality training happens in full or mixed precision and the 4-bit export is the last step.
Quantizing and then dequantizing to continue training in higher precision is banned: it is the
one ordering the literature shows collapsing outright.

Ensembling is out of scope: the deliverable is a single checkpoint per tier that the gateway
routes to, and an ensemble does not fit the memory envelope.

### Constraints

- **Compute:** the training budget is open at this stage. Nothing in the operator's material
  states one, and the operator asked explicitly for the first sweep to be sized to what is
  actually available with the cost of a larger version stated before anything is spent past
  approval. `s3.3` reads the enforced allowance and `s3.4` puts the number and the matrix in
  front of the operator.
- **Inference:** a single 8 GB consumer GPU plus a modern CPU is the reference device, with an
  RTX 2080 Super (8 GB) / 32 GB RAM / i7-10875H machine named as the operator's own. Anything
  that needs a workstation to run is a miss regardless of its score.
- **Footprint:** the tier-1 deliverable is at most 1.5 GB on disk at 4-bit. The base's Q4_0
  form is roughly 0.85 GB, so the specialization has real headroom but no licence to grow a
  tier.
- **Latency:** 4-bit decode throughput within 5% of the base's 4-bit throughput on the same
  hardware, measured at a fixed 1K-token prefill.
- **License:** LFM Open License v1.0 permits derivatives, fine-tuning, and redistribution, with
  a commercial-use threshold of $10M annual revenue per legal entity that this use sits below.
  Derivative artifacts carry the license, a change notice, and a clear statement that they
  derive from LFM2.5. Training material stays open-weight and license-checked; the gated xlam
  set requires an access request before `s4`.
- **Fairness and governance:** no demographic attributes exist in this data and none enter any
  input. The governance surface here is capability coverage rather than demography, so
  worst-category and per-language reporting is on by default and a category regression beyond
  the floor below fails the claim even when the mean improves.
- **Reproducibility:** every run records its seed, its data hash, its decoding parameters, and
  its exact eval command. Every baseline comparison is a matched rerun, and every published
  number quoted from a vendor card is labelled as such.

### Success criteria

Pre-registration: pre-registered claim

**Claim:** a stack-specialized LFM2.5-1.2B derivative reaches **BFCLv3 overall at or above
52.43 and at least 3.0 points above a matched rerun of `LFM2.5-1.2B-Instruct`**, and beats a
matched rerun of the NovachronoAI community function-calling finetune on the same axis, while
holding every guardrail metric within tolerance; and its Q4_0 GGUF, at **1.5 GB or less on
disk**, retains **97% or more** of that model's own full-precision average across the axis
suite.

- **Structured output:** IFStruct first-attempt schema validity at least 5.0 points above the
  matched base.
- **Reliability:** on the malformed-tool-return probe, flag rate at or above 0.70 with a
  false-flag rate at or below 0.15 on matched well-formed returns, reported together with
  functional accuracy on the same probe. Revised down from 0.80 / 0.10 at `s2.3`: five
  independent measurements put that pair above the demonstrated frontier, including frontier
  models reaching only 76.6 and 71.1 on the closest public category, and the two rates are
  measured to trade off against each other at r = −0.78.
- **No fabrication regression:** neither the no-tool-available call rate nor the wrong-tool
  call rate rises above the matched base's.
- **Worst-category floor:** no BFCLv3 category falls more than 3.0 points below the matched
  base, and no guardrail metric (IFEval, IFBench, Multi-IF, MMLU-Pro, GPQA) falls more than
  2.0 points below it.
- **Quantization floor:** no single axis retains less than 93% of the full-precision parent,
  so the 97% average cannot be bought by averaging a collapsed structured-output score against
  intact knowledge scores.
- **Efficiency floor:** 4-bit decode throughput within 5% of the base's 4-bit throughput,
  measured at identical quantization format and identical backend on the same hardware. Since
  post-training does not alter the architecture, and the one architecture change that might
  have helped is blocked by llama.cpp's tensor shape check at load, this is a sanity check on
  the export rather than a research finding. Format choice matters more than the tolerance
  does, which is why format is pinned.

The tolerances above are this project's own defaults, chosen so the margins are auditable
rather than adjustable after the fact. The literature review revisited them at `s2.3`: the
reliability pair was lowered, the efficiency floor was reframed, and everything else was left
alone. They are frozen from here and any later change is a recorded finding, not an edit. The
+3.0 BFCLv3 margin at flat guardrails is knowingly at the hard end of what the field has
demonstrated below 3 billion parameters, and it was not lowered, because relaxing a target
after reading the literature and before seeing a result is the wrong direction of adjustment.

### Risks

- **Benchmark contamination.** Public tool-calling training sets overlap the benchmarks that
  score them, and a contaminated win is indistinguishable from a real one at the top line.
  Mitigation: file-level separation of eval from train, an n-gram decontamination pass at
  `s4.3` against every benchmark prompt set, and the two held-out probes built here.
- **Vendor numbers not reproducible.** Every baseline figure in the operator's material is
  self-reported by its vendor under an unstated harness. Mitigation: matched reruns are the
  primary evidence and published numbers are a secondary cross-check, with any gap between
  them reported rather than reconciled.
- **Catastrophic forgetting.** A 1.2B model specialized hard on tool calling can lose the
  instruction-following lead that makes it worth using. Mitigation: guardrail tolerances are
  gating criteria, LoRA is tried before full fine-tuning, and a mixed replay ratio is an
  explicit axis in the matrix.
- **Reward hacking under verifiable rewards.** A schema-validity reward is trivially satisfied
  by an empty but well-formed call, and the literature names exactly that as the primary
  observed exploit: "minimally valid JSON satisfying a shallow schema check". Reinforcement-
  heavy post-training has been measured to raise exploit rates from under 1% to 12–16%, and
  automated detectors reach only 53% F1 in the early regime that matters. Mitigation: the
  reward includes execution or semantic correctness alongside a mandatory schema term, the
  flag/false-flag and fabrication pairs are checked on every RL checkpoint, the evaluation
  boundary is hardened with strict schemas and fail-closed parsing (measured at −41.5%
  exploits alone, −87.7% combined with reduced file access, at no cost in task success),
  integrity is reported as an axis separate from task correctness, and detected shortcut
  behaviour carries a flat penalty rather than a filter.
- **Tokenizer mismatch.** The 1.2B carries the pre-expansion 65K vocabulary, and long
  identifiers and dense JSON may tokenize inefficiently against a 128K-vocabulary sibling.
  Mitigation: tokens-per-tool-call is measured as a diagnostic at `s4.2`; in-place tokenizer
  expansion is a named fallback and is out of scope for this project.
- **Quantization collapse of structured output.** Structured output and instruction-following
  degrade first under 4-bit, ahead of fluency and knowledge, and smaller models suffer more.
  Nine papers in the corpus replicate it with numbers, including one where a 14-task average
  rose 0.48 points while instruction-following fell 8.5%. Mitigation: per-axis retention with
  its own floor so an average cannot hide it, quantization-aware distillation rather than
  naive post-training quantization as the default path, no reliance on perplexity or pooled
  aggregates as proxies, and a design bounded at 4-bit, which the literature reports as
  recoverable, never 2-bit, which it reports as irreversible.
- **In-house data unavailable.** The "specialized to our stack" claim rests on a corpus that
  does not exist yet. Mitigation: the public-data-only ablation is a baseline either way, and
  the claim narrows honestly if the corpus does not materialize.
- **Compute allowance too small for the matrix.** Mitigation: the matrix is sized against the
  enforced allowance at `s3.3` and the operator sees the cost of a larger version at `s3.4`.
- **Measurement protocol swamping the effect.** Serving backend alone accounts for roughly
  39% of realistic score variance at this model size even under greedy decoding, and it
  changes which items are correct rather than only how many; prompt and reasoning-template
  choice alone has moved published scores by 15 to 21 points. Both exceed this project's
  target effect size, and one paper in the corpus draws a quantization conclusion that is
  invalid because it scored two formats with two different harnesses. Mitigation: one backend,
  greedy decoding, one frozen template, the full generation config disclosed, GGUF conversion
  counted as part of the backend, and paired significance testing.
- **Reward inflation in any self-improving loop.** Stored model-scored rewards inflate on wrong
  episodes, which would corrupt an agentic RL lane that scored its own past trajectories.
  Mitigation: rewards stay programmatic and verifiable; no model-scored stored reward is reused
  as training signal.

### Out of scope

- **Tiers 2, 3, and 4 as primary claims.** The 2.6B dev assistant, the VL multimodal sensor,
  and the 8B-A1B reasoning brain are sequenced behind the tier-1 result. A 2.6B transfer of the
  winning recipe is the first candidate to promote and is costed at `s3.3`.
- **The mixture-of-experts lanes.** Improving the native `lfm2_moe` 8B-A1B, router-only
  fine-tuning of it, and weight-merging improved dense specialists into its fabric are all
  named goals and none of them are executed here. The composable core is the 128K-vocabulary
  fabric shared by the 2.6B, the 8B-A1B, and the VL-3B, and the 1.2B does not sit in it, so a
  tier-1 result is a prerequisite for that work rather than a component of it.
- **Pre-training or continued pre-training from scratch.** This project post-trains. The base
  models cost 28T to 38T tokens and reproducing that is out of reach by physics.
- **Speculative-decoding drafter retraining and the llama.cpp Metal mixture-of-experts gap.**
  Both are real efficiency wins and neither moves the quality bar, since speculative decoding
  is exact.
- **In-place tokenizer expansion of the 1.2B.** Measured as a diagnostic, not executed.
- **Audio models, MLX and ONNX runtime lanes, and the retriever and encoder families.**
- **Beating a leaderboard in the abstract.** The target is the best available tier-1 model,
  vendor or community, on the axes named above, under matched conditions.

## Assumptions

Revised at `s2.3`. `Basis` is `prompt` (stated by the operator), `document` (in the operator's
attached material), `literature [id]` (grounded in a paper read at `s2.1`, cited by id), or
`default` (this project's own call, still unevidenced).

| Decision | Basis | Confidence |
| -------- | ----- | ---------- |
| Scope this project to tier 1 (`LFM2.5-1.2B-Instruct`, sub-1.5 GB at 4-bit) rather than all four tiers at once | document | high |
| Attack tool-calling reliability and structured output as the named weakness, ahead of code and general quality | document + literature [2511.23404v1] | high |
| Sequence the mixture-of-experts family behind the tier-1 result rather than running it in parallel | document | medium |
| Prediction unit is one agent turn consumed programmatically by the operator's gateway | prompt | high |
| Group split keyed on source API schema (tool data) and source repository file (code and SQL data) | default + literature [2601.06103v1] | high |
| BFCLv3 overall as the primary metric, with per-category **and AST-only** scores reported alongside | document + literature [2608.03092v1, 2411.13676v1] | high |
| A held-out malformed-tool-return probe as the measure of flag-rather-than-assert behaviour, since no public benchmark scores it directly | default + literature [2511.23404v1, 2605.16790v1] | high |
| The probe carries a corrupted-evidence and text-only control arm and is stratified by nesting depth | literature [2512.04597v1, 2505.04016v1] | high |
| A held-out in-house idiom probe as the measure of "our stack", anchored by HumanEval and MBPP | default + literature [2502.06589v1] | medium |
| Flag rate at or above **0.70** with false-flag rate at or below **0.15**, reported with functional accuracy (was 0.80 / 0.10) | literature [2510.10390v1, 2408.04682v2, 2605.19341v1, 2603.10697v1, 2605.29523v1] | medium |
| A fabrication axis (no-tool-available and wrong-tool call rates) on every checkpoint | literature [2510.22977v2] | high |
| A 3.0-point margin over the matched base on BFCLv3, and a 5.0-point margin on IFStruct validity | default; known hard per literature [2601.02151v1, 2511.09148v2] | medium |
| A 2.0-point guardrail tolerance and a 3.0-point worst-category floor | default; known at risk per literature [2601.02151v1] | medium |
| A 93% per-axis quantization floor underneath the 97% average bar | literature [2605.04062v2, 2608.18578v1] | high |
| Decode throughput within 5% of base, at identical quantization format and backend, as a sanity check rather than a finding | default + literature [2605.04062v2, 2608.20210v1] | medium |
| Matched reruns of every baseline as primary evidence, with vendor-published numbers as a secondary cross-check | document + literature [2511.23404v1, 2511.22138v1] | high |
| One backend, greedy decoding, one frozen prompt template, full generation config disclosed, paired significance tests | literature [2608.04714v1, 2511.20836v3, 2608.18578v1, 2501.02342v1] | high |
| A public-data-only LoRA arm as the ablation isolating stack specialization from tool-calling SFT in general | default + literature [2505.04016v1] | high |
| Supervised fine-tuning is mandatory before preference optimization and reinforcement learning, not merely first | literature [2604.20316v1, 2605.02572v1] | high |
| LoRA before full fine-tuning, kept as an experimental axis rather than a decision | literature [2210.04802v2, 2605.09015v1, 2605.19018v1, 2309.05444v1] | medium |
| Entropy-weighted supervised loss, a replay fraction, and parameter-space merging as the three forgetting controls | literature [2601.02151v1, 2512.18934v1, 2511.23404v1, 2407.08699v2] | medium |
| All quality training in full or mixed precision, 4-bit export last, in-format training only after export | literature [2511.19495v1, 2508.04073v1, 2512.18934v1] | high |
| Quantization-aware distillation as the recovery rung; full quantization-aware training dropped | literature [2605.17471v1, 2608.20210v1, 2601.20088v3] | high |
| Verifiable-reward RL is a matrix arm, not a plank, and only from a warm start with a graded decomposed reward | literature [2510.07737v1, 2605.27954v1, 2605.16790v1, 2608.03092v1] | high |
| The Instruct checkpoint stays the starting point, with a base-checkpoint arm as an optional cheap ablation | literature [2511.01934v2, 2604.20316v1] | medium |
| No model-scored reward, stored or online, in the training signal; a judge only as a disclosed secondary eval cross-check | prompt + literature [2601.03525v3, 2509.15557v1, 2604.16242v1] | high |
| Irrelevance and multi-type categories mandatory in the mix; filter-rejected items discarded rather than trained on; public sets repaired and validated first | literature [2409.00920v2, 2406.18518v1, 2605.16790v1, 2505.20192v3] | high |
| Constrained decoding stays out of the metric and out of the deliverable | literature [2605.26128v1, 2603.03305v1, 2602.12247v2, 2605.02363v1] | high |
| Capability coverage, not demography, as the governance surface, with worst-category and per-language reporting on by default | default + literature [2605.15208v1] | high |
| Deliverables are a model card, a reproducibility package, a paper, and GGUF artifacts mirrored to the operator's own repositories | prompt | high |
| Specialization targets underlying technologies rather than any one vendor's SDK, shaping both the training mix and the in-house probe | document + literature [2502.06589v1] | high |

## Open questions

Each was checked against the 212-paper corpus at `s2.3`. Two remain open on access grounds,
one is unanswerable from this literature, and two are answered.

- **Does the in-house stack corpus exist, and where? — open, and now costed.** The literature
  cannot answer it, but it prices the answer: public data alone measured 63.1% / 68.3% schema
  accuracy against 89.0% / 81.7% for synthetic plus public in the closest study, so the
  public-data-only arm is a real floor roughly 20 points below the target rather than a
  formality. Proceeding on public data plus synthesized stack-flavoured items, and raising a
  request for repository access at `s4.1` where the answer changes what runs.
- **Is the gated xlam tool-calling set reachable with the credentials on file? — answered at
  `s4.1`: no.** It is `gated: auto` and the terms have not been accepted on the account whose
  token we hold, so every request returns `403`. Accepting a licence on someone's behalf is a
  legal act, not a configuration step, so no attempt was made to route around it. ToolACE plus
  Synth-APIGen plus Hermes replaces it at 70,387 rows, all Apache-2.0, built by the same
  execution-verified recipes. Two papers document defects in the xlam data anyway (trajectories
  that mismatch their declared schemas, and an extra space after `[` that breaks
  abstract-syntax-tree parsing), and the repair-and-validate pass they motivate is what `s4.2`
  and `s4.3` carried out on the substitute mix. If the terms are accepted later the rows are
  additive and invalidate nothing already measured.
- **Does the 65K vocabulary materially handicap tool-call tokenization? — unanswerable from
  this literature.** Every tokenizer paper retrieved operates at tokenizer-construction or
  pretraining time and cannot be applied to a frozen 65,536-token vocabulary, and
  cross-vocabulary distillation is itself an open problem. Tokens-per-tool-call at `s4.2`
  remains the only route to an answer here, and it stays a diagnostic. One adjacent datum: an
  oversized vocabulary was measured wasting roughly 13M parameters in a 150M model of this same
  architecture family, so the cost is real at small scale even though acting on it is out of
  scope.
- **Which reinforcement-learning signal, if any, instils the flag-rather-than-assert habit? —
  answered: preference optimization first, reinforcement learning as an arm.** The strongest
  measured intervention is a two-sided preference rung, which cut fabricated-call rate from
  90.2% to 55.8%, with preference optimization independently beating supervised fine-tuning for
  refusal behaviour by 3.4× at 7B. Verifiable-reward reinforcement learning stays a matrix arm,
  because the two studies at this scale report collapse and no published tool-use reward scores
  "did the model correctly flag a bad return", so that term has to be built. A cheap non-RL
  alternative also enters the plan: a small calibration fine-tune with a Jensen-Shannon
  regularizer on target-sequence logits, measured to cut expected calibration error from 29.9%
  to 10.8% on roughly 1,000 graded examples.
- **Can a 1.2B model hold the instruction-following lead while gaining three or more points of
  tool-calling accuracy? — answered pessimistically; the plan changed and the claim did not.**
  The closest matched evidence has plain tool-call supervised fine-tuning buying +0.9 on the
  primary axis while instruction-following fell 3.2 points and the general average fell 6.3,
  with the best published mitigation still breaching a 2-point guardrail; another study has
  code ability collapsing outright. So this is the hardest part of the project. Three
  mitigations are promoted from possible to planned in response: entropy-weighted loss, a
  replay fraction with a measured floor, and parameter-space merging back toward the base. A
  forced trade-off remains a reportable finding rather than a redefinition.
