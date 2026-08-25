# Quants — options and the quality bar

Goal from the outline: produce **similar-sized or smaller, and outperforming**, GGUF quants
of the (improved) models. This doc catalogs the quantization *options* and sets the quality
bar; Primus decides which to run and in what order. Verified against Liquid's QAD blog
(2026-08-19) and the current GGUF repos (2026-08-24).

## The quality bar (non-negotiable)

- A shipped 4-bit GGUF must hold **≥ ~97% of its full-precision average accuracy** on the
  axis suite (the reference point is Liquid's QAD result, below). Plain PTQ Q4_0 that loses
  more than that is a miss.
- The bar is measured **per capability axis** (tool-calling, IF, code, vision), not just
  one aggregate — a quant can hide 4-bit damage on knowledge while collapsing structured
  output, which is exactly what our harness needs most.
- Every improved model must ship at least a Q4_0 and a Q4_K_M GGUF in its tier footprint,
  plus whatever imatrix/UD variant wins on our evals.
- Where a vendor QAD checkpoint already exists (LFM2.5-230M/350M/1.2B/2.6B), it **is** the
  baseline: our quant of our improved model must beat *that*, not just the PTQ GGUF.

## Option 1 — Post-training quantization (PTQ), the default path

Quantize a finished checkpoint with llama.cpp / GGUF tools. Cheap, fast, reversible.
Known limitations for this family: the QAD blog explicitly states vanilla PTQ Q4_0 loses
meaningful accuracy (which is why Liquid released QAD), and structured-output/tasks
degrade first. Variants to compare:

- `Q4_0`, `Q4_K_M` (imatrix), `Q5_K_M`, `Q6_K`, `Q8_0` — the standard scale.
- **unsloth UD-Q4_K_XL** (dynamic imatrix) — the external reference that matches Liquid's
  own QAD on 230M/1.2B; a strong cheap baseline to beat.
- Per-tensor type mixing (e.g. keep router/embedding/LM-head at Q8, experts at Q4) —
  especially relevant for the 8B-A1B MoE, where the routing layer is cheap to keep full-precision
  and the experts dominate the bytes.

Verified 8B-A1B GGUF on-disk sizes (repo listing): Q4_0 **4.84 GB**, Q4_K_M 5.16 GB,
Q5_K_M 6.03 GB, Q6_K 6.96 GB, Q8_0 9.01 GB, F16 16.95 GB.

## Option 2 — Quantization-Aware Distillation (QAD), the first-party recipe — the reference

Liquid's released recipe (QAD blog, 2026-08-19): a high-precision teacher distills into a
quantized student *while training*, so the student adapts to the quantization. Results
(reported, mean over 5 repeats):

- QAD Q4_0 retains **97.1% / 96.5% / 97.4% / 96.6%** of BF16 baseline on 230M / 350M /
  1.2B-Instruct / 2.6B respectively.
- Matches Q5_K_M quality (230M/350M) or Q4_K_M quality (1.2B/2.6B) **at 3–33% higher decode
  throughput** than the K_M quants.
- Eval suite used: GPQA Diamond, MMLU-Pro, IFEval, IFBench, Multi-IF, BFCLv4, +
  scale-appropriate math (GSM8K for ≤350M, AIME25 for 1.2B/2.6B). That is the axis suite
  we adopt for our quant bar (plus our-stack code + a VL screen-reading axis for VL models).
- Day-one llama.cpp-compatible Q4_0 artifacts. Released repos: `LFM2.5-230M-GGUF`,
  `LFM2.5-350M-GGUF`, `LFM2.5-1.2B-Instruct-GGUF`, `LFM2.5-2.6B-GGUF` (QAD files inside).

Why it matters for us: **QAD changes what "4-bit" can mean** — if the student is trained
against the quantized weights, our improved models can live at Q4_0 memory/speed without the
usual quality cliff, which directly serves the "similar sized or smaller AND outperforming"
goal. The open question: does QAD (or a close variant) hold up for the *8B-A1B MoE* and for
the *VL* models, where Liquid has not (yet) released QAD?

## Option 3 — QAT-style training and mixed-precision architecture choices

Options that push the quality frontier further, at the cost of training budget:

- **Quantization-Aware Training (QAT)** of the fine-tune itself (train the LoRA/full SFT at
  low precision with fake-quant) — the LFM2.5-Audio model was QAT'd to INT4 with "virtually
  zero quality loss" (INT4 0.89 WER vs FP32 0.89 in Liquid's table), so the family is
  QAT-friendly.
- **Mixed precision within one GGUF**: router + attention + LM head at higher precision,
  expert FFNs at 4-bit (MoE) / conv kernels at 4-bit, GQA at higher. Candidate for the
  4.84→≤4.5 GB footprint at equal-or-better quality.
- **Speculative decoding (DSpark) as a speed axis, not a quality axis**: exact (greedy output
  identical), so it never moves the quality bar — it only makes the 4-bit experience feel
  larger. Pairing a Q4_0 target with its DSpark drafter is nearly free latency improvement;
  the known gap is the 8B-A1B MoE on-device (only ~18% today) — fixing the llama.cpp Metal
  MoE verification path is a concrete, well-scoped efficiency experiment.

## Option 4 — Format/runtime axes (not weight quant, but on-disk/latency)

- **MLX** quant (4/6/8-bit) for the Apple-silicon lane; **ONNX** int8 for CPU-only lanes
  (Liquid's encoders note that int8/ONNX paths are the gap for always-on CPU use).
- **GGUF vs EXL2/other runtimes** — compare on our reference hardware only if a real gap
  appears; don't add a runtime lane on speculation.

## Candidate experiments (options, ranked by expected value — our read, not a mandate)

1. **QAD our improved 2.6B and 8B-A1B** (QAD not yet released for 8B-A1B/VL by Liquid) —
   proves the biggest "outperforming AND smaller-or-equal" margin is available where it
   matters most to us.
2. **Mixed-precision 8B-A1B GGUF** (Q8 router/attention + Q4 experts) at ≤5 GB, measured
   against the 4.84 GB Q4_0 and the UD-Q4_K_XL — a clean "optimize one dimension" win if it
   holds the axis suite.
3. **UD-Q4_K_XL-style imatrix sweep** as the cheap baseline for every improved checkpoint,
   so "beats vendor QAD" is always measured against the best cheap quant, not just Liquid's.
4. **DSpark gap fix / drafter retrain for our improved models** — exact speedups, zero quality
   risk; high ROI on agent latency (function-calling latency -57% for 2.6B is the reference).
5. **QAT of the stack-specialized SFT** (train at INT4) for the 1.2B/2.6B — if QAD alone
   leaves a gap on our axes, this is the deeper fix; budget it only if (1)/(2) are
   insufficient.

## Guardrails for the quant work

- Always evaluate the quant **and** the full-precision improved model in the same harness,
  same tasks, same seeds — a quant regression is only real against a matched full-precision
  run.
- Structured output (JSON/tool-call validity) gets first-class eval attention: it is the most
  quant-fragile axis and the one our harness depends on.
- Keep an FP16/BF16 GGUF of each improved model as the in-format ceiling for comparison
  (that's Liquid's own convention in the QAD blog).
