# Sources — verified, Aug 2026

Every load-bearing claim in this project traces to one of these. All were fetched and read
**2026-08-25** (model cards / blogs / repos), not recalled. Re-verify before publishing any
external-facing number: the field moves fast (Liquid itself updated the LFM2.5 cards
~2026-08-24, one day before this snapshot).

## Liquid AI — primary

- LFM2 Technical Report — <https://arxiv.org/abs/2511.23404> (local: `research/2511.23404v1.pdf`)
- In-Place Tokenizer Expansion (LFM2→LFM2.5 128K vocab) — <https://arxiv.org/abs/2607.15232>
  (local: `research/2607.15232v1.pdf`)
- LFM2.5 launch (Jan 2026): 1.2B family, VL-1.6B, Audio-1.5B, 1.2B bench table, inference
  speeds — <https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai>
- LFM2.5-8B-A1B (May 2026): MoE release, 128K ctx, 38T tokens, reasoning-only, full bench
  tables, training highlights (tokenizer expansion, context extension, doom-loop +
  hallucination RL) — <https://www.liquid.ai/blog/lfm2-5-8b-a1b>
- LFM2.5-2.6B (Aug 2026): 4-stage agentic recipe, MOPD, agentic-RL-in-harness, bench table
  vs ~4×-larger models, community hallucination reports — <https://huggingface.co/blog/LiquidAI/lfm2-5-2-6b>
  (mirror: <https://www.liquid.ai/blog/lfm2-5-2-6b>)
- LFM2.5-VL-3B (Aug 2026): SigLIP2 400M NaFlex + 2.6B backbone, VL bench table vs
  InternVL3.5/FastVLM/Qwen3.5-2B, ScreenSpot-v2, speed — <https://www.liquid.ai/blog/lfm2-5-vl-3b>
  (also <https://huggingface.co/blog/LiquidAI/lfm2-5-vl-3b>)
- LFM2.5 QAD Q4_0 (2026-08-19): quantization-aware distillation, ~97% BF16 recovery, axis
  suite (GPQA-Diamond, MMLU-Pro, IFEval, IFBench, Multi-IF, BFCLv4, GSM8K/AIME25) —
  <https://huggingface.co/blog/LiquidAI/qad> (mirror <https://www.liquid.ai/blog/qad>)
- LFM2.5-DSpark (2026-08-20): speculative decoding, exact, up to 3.2×, llama.cpp PR #27383,
  SGLang PR #31041 — <https://huggingface.co/blog/LiquidAI/lfm25-dspark>
  (mirror <https://www.liquid.ai/blog/lfm25-dspark>; base paper <https://arxiv.org/abs/2607.05147>)
- LFM2.5-Encoders (2026-07-28): bidirectional 230M/350M, 8K ctx, CPU speed, prompt-routing
  demo — <https://huggingface.co/blog/LiquidAI/lfm2-5-encoders>
- LFM Open License v1.0 — verified from
  <https://huggingface.co/LiquidAI/LFM2.5-8B-A1B/blob/main/LICENSE> (raw: /raw/main/LICENSE)
- Fine-tuning notebooks (CPT/SFT/DPO/GRPO, VL-3B support) —
  <https://github.com/Liquid4All/cookbook/tree/main/finetuning/notebooks>
- Liquid HF org (all model + dataset IDs verified 2026-08-24/25) — <https://huggingface.co/LiquidAI>
- Docs index (for deeper dives) — <https://docs.liquid.ai/llms.txt> ; fine-tune:
  <https://docs.liquid.ai/lfm/fine-tuning/leap-finetune> ; LEAP finetune repo:
  <https://github.com/Liquid4All/leap-finetune>

## Datasets (verified 2026-08-25)

- `Salesforce/xlam-function-calling-60k` — gated, cc-by-4.0, 60k, execution-verified
- `m-a-p/CodeFeedback-Filtered-Instruction` — apache-2.0, OpenCodeInterpreter
- `b-mc2/sql-create-context` — cc-by-4.0, 78,577 rows, schema-context text-to-SQL
- `Clinton/Text-to-SQL-v1` — apache-2.0, 100k+
- `LiquidAI/ifstruct-v1.0` — apache-2.0 (<https://huggingface.co/datasets/LiquidAI/ifstruct-v1.0>)
- `LiquidAI/antidoom-mix-v1.0` — apache-2.0 (<https://huggingface.co/datasets/LiquidAI/antidoom-mix-v1.0>)

## Competitor baselines (cards to re-verify each run)

- Gemma 4: `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-26B-A4B-it`
  (also `-qat-q4_0-gguf` variants, `litert-community` LiteRT LM builds)
- Qwen 3.5: `Qwen/Qwen3.5-2B` (image-text-to-text!), `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`,
  - `unsloth/*-GGUF` for the quant lane
- gpt-oss-20b (`openai/gpt-oss-20b`), Granite 4.0 (`ibm-granite/granite-4.0-h-tiny` —
  confirm exact ID)
- Community LFM2.5 derivatives: `unsloth/LFM2.5-1.2B-Instruct-GGUF`,
  `unsloth/LFM2.5-8B-A1B-GGUF` (UD-Q4_K_XL), `NovachronoAI/LFM2.5-1.2B-Nova-Function-Calling-GGUF`,
  `lmstudio-community/LFM2.5-1.2B-Instruct-MLX-4bit`, `mlx-community/LFM2.5-1.2B-Base-4bit`.
  Model tree: 114 finetunes + 82 quantizations off `LFM2.5-1.2B-Base` (re-scan before claims).

## Agent-memory research (for the routing/RL lane)

- Memory Reward Inflation in Self-Improving LLM Agents (Zamanifar et al., Jun 2026) —
  <https://arxiv.org/abs/2608.00017> (local: `research/2608.00017v1.pdf`) — echo-gap: stored
  LLM-scored rewards inflate on wrong episodes; LUCID de-inflation lifts BIRD exec accuracy
  52.4→56.9%. Relevant to how our router/RL should score retrieved work.

## Methodology refs named in Liquid's tables

- ArtificialAnalysis methodology (GPQA/MMLU-Pro/IFBench/AIME25 scoring, AA-Omniscience
  Index) — <https://artificialanalysis.ai/methodology/intelligence-benchmarking>
- VLMEvalKit (VL-bench harness) ; vLLM 0.26.0 (Liquid's VL-3B eval harness)
