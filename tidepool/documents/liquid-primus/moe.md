# MoE — the hybrid family: goal, findings, options

Goal: from our **improved** versions of the LFM2.5 models, build a **hybrid mixture-of-experts
family** that can follow instructions, reason, call tools, receive images, and produce code —
in our workflows — even at a larger on-disk cost, as long as *active* parameters per token stay
small and it runs on consumer hardware. This doc is findings + options; the process is
Primus's to choose.

## Key finding 1 — "hybrid MoE" already exists in the family

The outline assumed we'd need to hand-roll a custom gating wrapper over non-transformer
blocks. **We don't.** Liquid ships a native `lfm2_moe` layer type (verified from
`LFM2.5-8B-A1B/config.json`):

- 24 layers: **4 full_attention (GQA) + 20 gated short-conv blocks** (the "hybrid" part)
- FFN position is **MoE routing to SwiGLU experts**: 32 experts, **top-4** active per
  token, `moe_intermediate_size` 1792, expert bias on, 2 dense layers
- 8.47B total / ~1.5B active; hidden 2048; **128K context; 128K vocab**
- Released base + post-trained; GGUFs; llama.cpp/MLX/vLLM/SGLang day-one

So the first deliverable of the "hybrid family" can be **the improved 8B-A1B itself**, and
the open research is how far we push the routing/experts for *our* tasks — not whether the
arch class works (Liquid proved it at 1.5B active, 128K ctx, under 6 GB).

## Key finding 2 — Liquid's own "fusion" is distillation, not weight-merge

The outline proposed "freeze expert blocks, train only the router." Liquid's actual recipe
for combining specialized capabilities is **multi-domain on-policy distillation (MOPD)** from
the 2.6B: train one specialist teacher per domain (math, code, tool use, …), then distill all
of them into a single student with on-policy rollouts, then agentic RL. That is their
"post-training fusion" — and it works inside one model without any weight-merge trickery.
It is the strongest prior for "one model that handles all our modalities-of-work."

## Key finding 3 — partial architecture alignment (verified from config.json)

Cross-family expert surgery depends on shape alignment. Verified:

- **2.6B ↔ 8B-A1B are composable**: both `hidden_size` 2048, both **128K vocab with
  tied word embeddings**, both 32 heads / 8 KV heads, same BOS/EOS IDs (124894/124900).
  Differences to manage: depth (30 vs 24 layers), FFN dim (10752 dense vs 1792×32-expert
  MoE), attention placement pattern (8 vs 4 attention layers), and **RoPE θ (1e7 vs 5e6)**.
- **1.2B is misaligned**: 65K vocab (pre-expansion), BOS/EOS 1/7, RoPE θ 1e6,
  16 layers, FFN 12288. It does NOT slot into the 128K/2048 fabric without embedding and
  RoPE surgery. If the hybrid family wants the 1.2B-class expert, the options are
  (i) build it back from `LFM2.5-1.2B-Base` under a continued tokenizer-expansion recipe
  (publishable technique, see `research/2607.15232v1.pdf`), or (ii) keep it as a stand-alone
  fleet member routed by option D.

Implication: the composable core for a "hybrid from our improved versions" is
**{2.6B-class dense, 8B-A1B MoE, VL-3B (2.6B backbone + SigLIP2)}** — same fabric, same
vocab. That is a materially better-posed problem than merging arbitrary same-size
transformer families, and it's the fabric option C should target.

## Options (ranked, not mandated)

**A. Improve the native MoE directly** (highest expected value, lowest risk)
Post-train `LFM2.5-8B-A1B(-Base)` with our stack data (SFT → specialist teachers/MOPD →
agentic RL → QAD to 4-bit). "Hybrid family" = improved dense 1.2B/2.6B + VL 3B + this
improved MoE brain. No new architecture risk; the yardstick is `docs/benchmarks.md` tier 4.

**B. Router-only / expert-additive fine-tune on the MoE** (the outline's idea, made
concrete)
Keep base expert weights frozen; train the router (+ small LoRA on attention/conv) on
our-domain trajectories so routing learns to send our token distributions (screens/UI,
our-stack code, tool-call JSON) to the experts that serve them best. Cheap (router +
adapters only), reversible, and a clean controlled experiment: "does routing specialize
without hurting general quality?" Natural follow-up if (A) shows task imbalance across
experts.

**C. Weight-merge our improved *dense* specialists into the MoE fabric** (the real
"hybrid from our improved versions")
Since our goal names "a hybrid MoE family **from those improved versions**," the
architecture-faithful reading is: take our improved 1.2B/2.6B/VL experts and compose them
into a shared 128K-vocab, 2048-hidden expert network — either (i) as same-arch expert
columns inside the `lfm2_moe` scaffold (pending finding-3 shape check), or (ii) via
dense-merge methods (task arithmetic / TIES / DARE / mergekit-style interpolation —
mergekit targets transformers, so this needs custom code against the transformers
`Lfm2MoeForCausalLM`) across *same-architecture* improved variants of one base (e.g.
merge our code-specialist 2.6B + tool-specialist 2.6B before routing them as experts).
This is where the actual research risk lives: router quality after merge, and whether
merged experts survive 4-bit. Treat as the ambitious lane; (A)/(B) are the guaranteed ones.

**D. Inter-model routing as the "family" glue** (zero weight-merge, immediate)
A "hybrid family" can also be read as *our fleet*: a tiny always-on router (a fine-tuned
`LFM2.5-Encoder-230M/350M` — Liquid ships a zero-shot **prompt routing** demo on it, 8K
context, CPU-only) dispatches to the 1.2B (routing/extract) / 2.6B (dev task) / VL-3B
(screens) / 8B-A1B (heavy reason) models, exactly how our gateway already works. This is
the complement to intra-model MoE — it is what our harness does today at the API level,
and doing it with 230M local encoders makes the switch free. Worth doing regardless of
which of A–C wins; it is also the fallback that keeps the project productive while C is
risky. Note the related research `research/2608.00017v1.pdf` (memory reward inflation /
echo gap in self-improving agents) on how retrieved-memory routing degrades when rewards
are correlated — relevant to how the router's learned preferences should be scored.

## What is out of scope by physics (be honest about it)

- **Pre-training a new MoE from scratch** at this quality: Liquid spent 12T→38T tokens on
  8B-A1B. We are a post-trainer, not a pre-trainer; everything here builds on their CPT.
- **A single model that does native speech-to-speech AND vision AND deep reasoning at
  1.5B active**: the family solves that as *separate* models (Audio-1.5B, VL-3B, text
  MoE). If the goal is "one model handles all," the realistic reading is intra-model
  multimodal (VL + tools + code in the 8B-A1B-class), not audio-included; the voice agent
  stays on the audio model per the existing architecture.

## Candidate yardsticks for the MoE lane

- Same axis suite as tier 4 (`docs/benchmarks.md`), plus a **routing-quality probe**:
  expert-usage distribution per task class (do tool-call turns actually route to the
  tool-trained experts?), and a **regression guard** on general quality (AA-Omniscience
  must not drop; that's the one axis where LFM2.5-8B-A1B currently shines).
- Efficiency: active params per token held at ~1.5B class; on-disk 4-bit ≤ our tier-4
  footprint (~4.8 GB Q4_0 is the verified reference).
- QAD-then-route vs route-then-QAD ordering is an open experiment (quantization can
  reshape router logits — measure it).
