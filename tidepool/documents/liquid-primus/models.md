# Models — the LiquidAI LFM2.5 target family

What we are improving and building from. All IDs and facts below were verified against
Hugging Face and Liquid's own blogs on **2026-08-25**. This is the family, not the method —
Primus decides how to work with it.

> Scope note: the working outline says "LiquidAI models ONLY, not training other models in
> this session." The competitor models in `docs/benchmarks.md` are **baselines to be tested
> against**, not co-targets. The improvement work starts from LFM2.5 checkpoints.

## Why this family is the target (our read)

- **On-device first**: linear/hybrid architecture (gated short convolution blocks + a small
  number of grouped-query attention blocks) gives fast prefill AND decode on CPU+GPU, and a
  small memory footprint — the property our fleet depends on.
- **Covers our size tiers**: 0.23B → 8B-A1B, dense and MoE, plus VL and Audio variants —
  less community fragmentation than class-equivalent families (good: fewer conflicting
  "improvements" to sort through, and a clean baseline).
- **First-party recipes are published**: post-training, quantization (QAD), speculative
  decoding (DSpark), tokenizer expansion, and fine-tuning notebooks are all documented —
  a strong starting prior, not a mandate.
- **Permissive-enough license** for derivatives (see License below).

## Architecture (verified)

- **Dense (lfm2)** — verified per-model from `config.json`:
  - 1.2B: 16 layers (6 GQA attention + 10 gated conv), hidden 2048, **65K vocab**, 128K
    pos-emb (card: 32K context), rope_theta 1e6
  - 2.6B: **30 layers (8 GQA attention + 22 conv)**, hidden 2048, **128K vocab**, 128K ctx,
    rope_theta 1e7
- **MoE (lfm2_moe)** — `LFM2.5-8B-A1B`: total **8.47B**, active **~1.5B** ("A1B"),
  24 layers (4 attention + 20 conv), 32 experts, `num_experts_per_tok` 4,
  `moe_intermediate_size` 1792, `num_dense_layers` 2, expert bias enabled,
  hidden 2048, **128K vocab**, 128K ctx, rope_theta 5e6
- **VL** — vision-language: LFM2.5-VL-3B pairs a **SigLIP2 400M NaFlex** encoder with the
  LFM2.5-2.6B text backbone; 128K vocab (doubled via in-place tokenizer expansion);
  ~34T pre-train tokens, 4× more vision data than the prior VL.
- **Tokenizer expansion (in-place)**: the LFM2→LFM2.5 8B-A1B move continued BPE merges on a
  multilingual corpus so most tokens are identity-mapped, new tokens decompose into source
  sub-tokens, and new embedding rows init to the mean of their decomposition; quality
  recovered by embedding-only training then brief full-model continued pre-training. This is
  a published, reproducible technique we can reuse.

## The family (verified HF IDs, updated 2026-08-24)

Text generation (dense `lfm2` / MoE `lfm2_moe`):

- `LiquidAI/LFM2.5-230M` (+ `-GGUF`) — 0.23B
- `LiquidAI/LFM2.5-350M` (+ `-GGUF`) — 0.35B
- `LiquidAI/LFM2.5-1.2B-Base` (+ `-GGUF`, `-ONNX`) — 1.2B raw
- `LiquidAI/LFM2.5-1.2B-Instruct` (+ `-GGUF`) — 1.2B post-trained
- `LiquidAI/LFM2.5-1.2B-Thinking` — 1.2B reasoning variant
- `LiquidAI/LFM2.5-2.6B-Base` and `LiquidAI/LFM2.5-2.6B` (+ `-GGUF`) — 2.7B
- `LiquidAI/LFM2.5-8B-A1B-Base` and `LiquidAI/LFM2.5-8B-A1B` (+ `-GGUF`) — 8.47B MoE / 1.5B active

Vision-language (image-text-to-text):

- `LiquidAI/LFM2.5-VL-450M` — 0.45B
- `LiquidAI/LFM2.5-VL-1.6B` (+ `-GGUF`) — 1.6B
- `LiquidAI/LFM2.5-VL-3B` (+ `-GGUF`) — 3B, the current flagship edge VLM

Audio:

- `LiquidAI/LFM2.5-Audio-1.5B` (+ `-GGUF`) — native speech-to-speech, QAT INT4 detokenizer

Utility variants:

- `LiquidAI/LFM2.5-Encoder-230M`, `LFM2.5-Encoder-350M` — bidirectional encoders
  (classification/routing/PII/policy), 8K context, very fast on CPU
- `LiquidAI/LFM2.5-Retrievers` — multilingual search encoders
- Liquid "Nanos" collection (LFM2-350M/1.2B): task-specific Extract/RAG/translation micro-models

Speculative-decoding drafters (exact: quality parity, faster decode):

- `LiquidAI/LFM2.5-1.2B-Instruct-DSpark`, `LFM2.5-2.6B-DSpark`, `LFM2.5-8B-A1B-DSpark`
  (+ GGUF). Up to ~3.2× throughput (2.6B on M4/M5: 323→864 tok/s on CPU; 18% on-device for
  the 8B-A1B MoE today — Metal MoE path is the noted gap). Day-one llama.cpp + SGLang support.

Task-specific datasets Liquid publishes (see `datasets/`): `LiquidAI/ifstruct-v1.0`,
`LiquidAI/antidoom-mix-v1.0`, `LiquidAI/nanobeir-multilingual-extended`.

## Tier → model mapping (our device classes)

Verified on-disk GGUF sizes for the 8B-A1B (repo listing): Q4_0 **4.84 GB**, Q4_K_M 5.16 GB,
Q5_K_M 6.03 GB, Q6_K 6.96 GB, Q8_0 9.01 GB, F16/BF16 16.95 GB.

| Tier (on-disk, ~4-bit) | Model | Verified/est. footprint | Role |
| --- | --- | --- | --- |
| < 1.5 GB | LFM2.5-1.2B-Instruct / 350M | ~0.8–0.9 GB (Q4_0) | micro-orchestrator, routing, intent |
| 1.5 – 3 GB | LFM2.5-2.6B | ~1.8 GB (Q4_K_M) | intermediate dev assistant |
| 3 – 5 GB | LFM2.5-VL-3B / VL-1.6B | VL-3B ~3.1 GB (Q4); VL-1.6B ~1.5 GB | multimodal sensor (screens, docs, UI) |
| 5 – 8 GB | LFM2.5-8B-A1B | 4.84 GB (Q4_0, verified) | the "brain", MoE, reasoning |

Reference device: **RTX 2080 Super 8 GB, 32 GB RAM, i7-10875H** (this machine) and
consumer CPUs (Ryzen AI Max+ 395 / Apple M-class as published references).

## First-party recipes (prior knowledge, not requirements)

- **2.6B 4-stage agentic post-training**: (1) 2× SFT weighted toward agentic data (tool use,
  web search, harness trajectories); (2) per-domain specialist teachers (math/code/tool);
  (3) multi-domain **on-policy distillation** (MOPD) into one student; (4) **Agentic RL**
  inside real agent harnesses (OpenClaw / Hermes) with a black-box harness proxy capturing
  token-level trajectories.
- **VL-3B recipe**: SFT with knowledge distillation from a larger teacher + **antidoom**
  training, then **multi-reward RL**.
- **8B-A1B recipe**: reasoning-only (explicit CoT), scaled pretraining 12T→38T tokens,
  context extended 32K→128K via RoPE base change + long-doc/long-trajectory mid-training,
  a targeted **preference-optimization stage to kill doom loops** (+ an RL shaping reward on
  loop-trigger words), and an **avg@k knowledge-boundary RL stage to cut hallucination**
  (reinforce abstention beyond reliable knowledge).
- **QAD (quantization-aware distillation)**: a high-precision teacher distilled into a
  Q4_0 student; recovers ~97% of BF16 avg accuracy vs. plain PTQ Q4_0.
- **DSpark speculative decoding**: DFlash-style parallel backbone + Markov token head +
  confidence-scheduled verifier; exact (greedy output unchanged), up to 3.2×.
- **Fine-tuning**: official `Liquid4All/cookbook` notebooks — SFT, VLM-SFT, DPO, GRPO
  (verifiable tasks + unsloth), continued pretraining; VL-3B supported as of ~2 weeks ago.

## License (verified from the repo `LICENSE` file)

**LFM Open License v1.0** (hf tag `license: other` / `license_name: lfm1.0`). Permissive
like Apache-2.0 in form: grants copyright + patent rights to **reproduce, prepare Derivative
Works, sublicense, and redistribute** (with standard conditions: ship the license, mark
modified files, retain notices). The **only substantive limit** is a **commercial-use
threshold of $10M annual revenue per legal entity** — commercial use above that is not
licensed; qualified non-profits are exempt for non-commercial/research use.

Implication for us: improving the models, fine-tuning, merging a hybrid MoE, and
redistributing our derivatives are all permitted for our (sub-threshold, personal) use.
Keep derivative artifacts clearly marked as derived from LFM2.5, and carry the license +
change notices with anything we distribute.

## Inference ecosystem (day-one, verified)

llama.cpp (GGUF), MLX, vLLM, SGLang, ONNX, and Liquid's LEAP (iOS/Android). vLLM `0.26.0`
used for the VL-3B benchmark numbers. Transformers `>=5.9.0` for the `lfm2`/`lfm2_moe`/VL
model classes.

## Open questions to answer with Primus (not prescribed)

- Which tier to win first (cheapest demonstrable win) and the matched conditions.
- How much first-party recipe to reuse vs. replace (the 2.6B MOPD + agentic-RL pipeline is a
  strong prior, but our goal is *our* stack, not theirs).
- Whether a hybrid MoE from the *improved* dense experts beats simply improving the
  first-party 8B-A1B MoE — a genuine open question, see `docs/moe.md`.
- How far QAD / speculative decoding vs. our own quantization strategy helps hold the bar.
