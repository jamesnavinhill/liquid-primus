# Benchmarks — the baseline table to beat

The incumbents, per device-class tier, with their **published** post-training scores. Our job
is to surpass these on our capability axes under fair, matched conditions. Numbers below were
transcribed from the vendors' own model cards / blogs on **2026-08-25**; treat them as a
**floor to verify**, since the field moves fast — re-check the latest public cards before
claiming a win.

Reading guide:

- "Beat" = better on our axes (instruction-following, tool/function-calling reliability,
  agentic multi-step, our-stack code), with general quality not regressed.
- The 1.2B table uses the Liquid card's 7 columns (GPQA, MMLU-Pro, IFEval, IFBench, Multi-IF,
  AIME25, BFCLv3; GPQA/MMLU-Pro/IFBench/AIME25 per ArtificialAnalysis methodology).
- For 8B-A1B, Liquid reports an **AA-Omniscience Index** (higher = better; rewards correct,
  penalizes hallucinated; range -100..+100) plus IFEval/IFBench/Multi-IF/BFCL/τ². A strong
  hallucination-resistance story is part of the bar, not a bonus.

## Tier 1 — sub-1.5 GB (1.2B-class micro-orchestrator)

Incumbent we start from: `LiquidAI/LFM2.5-1.2B-Instruct` (1.17B, 16 layers = 10 conv + 6 GQA,
28T pretrain, 32K ctx, 65K vocab). Liquid's own stated caveat: "not recommended for
knowledge-intensive tasks and programming" — that's the specific weakness to fix here.

Scores (from the LFM2.5-1.2B-Instruct card / LFM2.5 launch blog):

| Model | GPQA | MMLU-Pro | IFEval | IFBench | Multi-IF | AIME25 | BFCLv3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **LFM2.5-1.2B-Instruct** | **38.89** | **44.35** | **86.23** | **47.33** | 60.98 | 14.00 | 49.12 |
| Qwen3-1.7B (instruct) | 34.85 | 42.91 | 73.68 | 21.33 | 56.48 | 9.33 | 46.30 |
| Granite 4.0-1B | 24.24 | 33.53 | 79.61 | 21.00 | 43.65 | 3.33 | 52.43 |
| Gemma 3 1B IT | 24.24 | 14.04 | 63.25 | 20.47 | 44.31 | 1.00 | 16.64 |
| Llama 3.2 1B Instruct | 16.57 | 20.80 | 52.37 | 15.93 | 30.16 | 0.33 | 21.44 |

Note on the incumbents: the *only* sub-2B competitor in Liquid's table with a competitive
tool-calling number is **Granite 4.0-1B** (BFCLv3 52.43, above LFM's 49.12). Granite 4.0 is
the model to beat for reliable tool-calling in this tier. LFM leads on instruction-following
by a wide margin; the gap to close is tool-call reliability + our-stack code.

## Tier 2 — 1.5–3 GB (2.6B-class dev assistant)

Incumbent: `LiquidAI/LFM2.5-2.6B` (2.7B, 128K ctx, post-trained for agentic work). Liquid
evaluated it **against models up to ~4× its size**. From the 2.6B blog (their 6-col slice):

| Model | AIME25 | LiveCodeBench v6 | IFBench | Multi-IF | IFStruct | BFCLv4 |
| --- | --- | --- | --- | --- | --- | --- |
| **LFM2.5-2.6B** | 51.87 | 59.41 | **59.17** | **80.07** | **85.49** | 56.88 |
| competitor (smaller Gemma) | 26.33 | 54.92 | 34.08 | 69.44 | 64.85 | 36.98 |
| competitor (mid Gemma) | 34.27 | 63.77 | 39.24 | 77.35 | 76.65 | 46.39 |
| competitor (~2B Qwen) | 49.33 | 60.85 | 48.40 | 55.67 | 36.25 | 50.56 |
| ~9.7B Qwen (larger) | 56.07 | 69.86 | 56.47 | 62.55 | 78.50 | **60.13** |

Agentic (2.6B blog): ToolSandbox **77.83**, Claw-Eval 62.85, PinchBench 68.22, τ³-Banking
5.67, BrowseComp+(OpenClaw) 26.89. Liquid's own read: LFM2.5-2.6B **tops every
instruction-following benchmark in the group and every tool-use benchmark except BFCLv4**,
where only the ~9.7B Qwen edges it; it beats both Gemmas and holds even with the Qwens on
agentic. **Coding is the one place the larger models keep a clear lead** — again the named
weakness for us.

## Tier 3 — 3–5 GB (VL 1.6B / 3B multimodal sensor)

Incumbents: `LFM2.5-VL-1.6B` and the current flagship `LFM2.5-VL-3B` (SigLIP2 400M NaFlex +
LFM2.5-2.6B backbone, ~34T tokens, 128K vocab, 4× more vision data).

**VL-1.6B** (launch blog; 14-col VL suite abridged):

| Model | MMStar | MM-IFEval | BLINK | MMMU(val) | OCRBench v1 | ScreenSpot-v2 avg |
| --- | --- | --- | --- | --- | --- | --- |
| **LFM2.5-VL-1.6B** | 50.67 | **52.29** | **48.82** | 40.56 | — | — |
| InternVL3.5-1B | 50.27 | 36.17 | 44.19 | **41.89** | — | — |
| LFM2-VL-1.6B (old) | 49.87 | 46.35 | 44.50 | 39.67 | — | — |

**VL-3B** (VL-3B blog; vLLM 0.26.0, non-reasoning, direct-answer):

| Model | MM-Star | MMMU | DocVQA | ChartQA | ScreenSpot-v2 avg | MM-IFEval | ToolSandbox | BFCLv4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **LFM2.5-VL-3B** | 63.3 | 48.4 | **91.1** | **81.3** | **78.7** | 60.6 | **59.5** | 32.5 |
| InternVL3.5 (1B) | 57.7 | 45.6 | 89.8 | 80.4 | 69.4 | 51.4 | 26.4 | 20.5 |
| InternVL3.5 (2B) | 45.3 | 41.1 | 85.7 | 43.2 | 57.2 | 65.6 | 56.5 | 33.2 |
| InternVL3.5 (4B) | 52.9 | 49.3 | 87.4 | 42.1 | 52.0 | 68.2 | 61.6 | 40.0 |
| FastVLM-1.5B | 57.7 | 52.0 | 73.6 | 81.7 | 65.5 | 47.1 | N/A | N/A |
| FastVLM-2B | 65.5 | 60.7 | 81.8 | 86.2 | 69.4 | 54.5 | N/A | N/A |
| Qwen3.5-2B (VL) | 59.3 | 44.1 | 94.8 | 84.2 | 70.1 | 63.1 | 65.0 | 53.6 |

VL-3B leads its size class on real-world image tasks and screen/document reading (ScreenSpot
v2 avg 78.7 — far above the 1.6B's class); Liquid states its tool use is "on par with
Gemma-4-E2B and Qwen3.5-2B." **Our target for this tier: the Qwen3.5-2B row** (DocVQA 94.8,
BFCLv4 53.6, ScreenSpot 70.1) — beat it on screen/UI grounding + tool-calling while holding
the on-device speed (VL-3B decodes 228 tok/s on M5 Max, 116 on Ryzen AI Max+, ~3 GB).

> The 8 numeric VL-3B comparison columns in the blog are not all labeled in plain text;
> the rows above use the values as printed against the four named comparators (InternVL3.5
> family, FastVLM, Qwen3.5-2B). Re-derive the exact column mapping from the blog's table
> image before publishing external-facing numbers.

## Tier 4 — 5–8 GB (8B-A1B MoE "brain")

Incumbent: `LiquidAI/LFM2.5-8B-A1B` (8.47B total / 1.5B active, 128K ctx, 128K vocab,
reasoning-only, 38T pretrain). Competitors span dense and MoE up to ~26B (8B-A1B blog,
May 2026).

**Knowledge / instruction-following (AA-Omniscience Index + IFEval/IFBench/Multi-IF):**

| Model | AA-Omniscience | IFEval | IFBench | Multi-IF |
| --- | --- | --- | --- | --- |
| **LFM2.5-8B-A1B** | **-24.70** | **91.84** | 56.47 | **79.93** |
| Granite-4.0-H-Tiny (7B/A1B) | -75.50 | 82.23 | 21.28 | 59.00 |
| Qwen3.5-4B | -51.53 | 87.80 | 50.38 | 67.43 |
| Qwen3-30B-A3B-Thinking-2507 | -51.31 | 90.82 | 51.11 | 79.04 |
| Gemma-4-E2B-IT (5.1B) | -72.00 | 82.93 | 33.53 | 69.70 |
| Gemma-4-E4B-IT (8B) | -50.67 | 87.74 | 39.48 | 77.58 |
| Gemma-4-26B-A4B-IT | -62.07 | **91.40** | 47.25 | **82.06** |
| gpt-oss-20b (21B/3.6B) | -49.17 | 86.73 | 58.65 | 76.64 |

**Math / agentic (MATH500, AIME25, GPQA, τ²-Telecom, τ²-Retail, BFCLv3, BFCLv4):**

| Model | MATH500 | AIME25 | BFCLv3 | BFCLv4 | τ²-Telecom | τ²-Retail |
| --- | --- | --- | --- | --- | --- | --- |
| **LFM2.5-8B-A1B** | 88.76 | 42.53 | 64.36 | 48.50 | **88.07** | 39.82 |
| Granite-4.0-H-Tiny | 59.20 | 4.93 | 56.89 | 28.52 | 16.67 | 18.42 |
| Qwen3.5-4B | 80.76 | **54.28** | 71.06 | **54.01** | **87.72** | **71.93** |
| Qwen3-30B-A3B-Thinking-2507 | 86.48 | 71.67 | 73.39 | 50.53 | 21.93 | 56.14 |
| Gemma-4-E2B-IT | 64.00 | 26.00 | 56.44 | 31.91 | 22.37 | 18.95 |
| Gemma-4-E4B-IT | 65.00 | 34.33 | 57.31 | 33.92 | 26.75 | 42.11 |
| Gemma-4-26B-A4B-IT | 94.20 | 68.67 | 68.87 | 55.87 | 42.11 | 55.26 |
| gpt-oss-20b | 92.40 | 68.53 | 62.52 | 49.88 | 57.24 | 53.51 |

Read: LFM2.5-8B-A1B **leads on hallucination-resistance (AA-Omniscience) and τ²-Telecom**,
and matches the 26B on IFEval/Multi-IF at a fraction of active params. The clear gaps to
close: **AIME25 (math) where Qwen3.5-4B leads even at smaller size, τ²-Retail (Qwen 71.93 vs
39.82), and BFCLv4 raw tool-call score (Qwen 54.01 vs 48.50)**. Those are the specific,
named weaknesses for this tier.

## Community baselines (also count as incumbents)

These are real, downloadable, and some already beat the vendor base — we must beat the best
of *both* vendor and community:

- **unsloth** LFM2.5 GGUFs (`unsloth/LFM2.5-1.2B-Instruct-GGUF`, `-8B-A1B-GGUF`) — includes
  the strong **UD-Q4_K_XL** post-training quant; per the QAD blog it is the external bar Liquid
  matched. Baseline any quant work against it.
- **NovachronoAI/LFM2.5-1.2B-Nova-Function-Calling-GGUF** — a community function-calling
  finetune; a direct "does our tool-calling win beat an existing community attempt" baseline.
- **lmstudio-community / mlx-community** LFM2.5 MLX 4/6/8-bit — the on-device quant ecosystem.
- The HF model tree shows **114 finetunes and 82 quantizations** from `LFM2.5-1.2B-Base` —
  a fast-moving community. Before claiming a tier win, re-scan the finetune/quant lists.

## Efficiency bar (the "optimize one dimension" axis)

Quality wins don't count if they cost on-device efficiency. Reference decode/prefill
(1K prefill, llama.cpp Q4_0, from the cards):

| Model | Device | Prefill tok/s | Decode tok/s | Memory |
| --- | --- | --- | --- | --- |
| LFM2.5-1.2B-Instruct | Ryzen AI 9 HX 370 (CPU) | 2975 | 116 | 856 MB |
| LFM2.5-1.2B-Instruct | Snapdragon Gen4 NPU | 4391 | 82 | 0.9 GB |
| Qwen3-1.7B | Ryzen AI 9 HX 370 (CPU) | 2008 | 62 | 1.46 GB |
| LFM2.5-2.6B | M5 Max | 220 (decode) | — | <2.5 GB |
| LFM2.5-VL-3B | M5 Max / Ryzen AI Max+ | 228 / 116 (decode) | — | ~3 GB |
| LFM2.5-8B-A1B | M5 Max / Ryzen AI Max+ | 253 / 146 (decode) | — | <6 GB |

DSpark speculative decoding (exact, quality parity) lifts 2.6B to ~864 tok/s on M-class CPU
and 1.2B-Instruct to ~1384 tok/s; 8B-A1B only ~18% on-device today (Metal MoE gap) — that
gap is a candidate efficiency win.

## How to turn these into a win (the yardsticks, our pick)

Pick 2–3 axes per tier to move and hold the rest as guardrails:

- **Tier 1**: tool-calling reliability (BFCLv3 / ToolSandbox) to at/above Granite 4.0-1B +
  our-stack code, without dropping IFEval.
- **Tier 2**: our-stack code (LiveCodeBench + a small in-house Next/React/Tauri eval) to
  close the "larger models keep a clear lead" gap, hold IF/ToolSandbox.
- **Tier 3**: screen/UI grounding (ScreenSpot-v2) + BFCLv4 to at/above Qwen3.5-2B, hold
  DocVQA/ChartQA and on-device speed.
- **Tier 4**: AIME25 math + τ²-Retail + BFCLv4 to at/above Qwen3.5-4B, hold AA-Omniscience
  and τ²-Telecom.
- **Quant axis**: every improved model must ship a 4-bit GGUF that holds the
  `docs/quants.md` quality bar (QAD ~97% of BF16 is the reference).

All numbers above are transcribed vendor/com self-reported. An independent, matched-rerun on
our reference hardware is part of proving a win — "the vendor says X" is a starting baseline,
not the finish line.
