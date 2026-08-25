# Outline (source working-notes from the Google Notebook)

> **STATUS (2026-08-25):** worked up and broken out into the per-topic docs. Keep this as
> the raw discussion record only — do NOT source claims from it anymore. Where it conflicts
> with the topic docs, the topic doc is right (it was verified against primary sources).
>
> Moved out → where:
>
> - Goals + workflow/stack → `readme.md` (goal/directive) + `docs/fine-tune.md` (specialization map)
> - Architecture/benchmarks/citations → `docs/models.md` + `docs/benchmarks.md`
> - SFT datasets → `datasets/README.md` (verified)
> - MoE process/science → `docs/moe.md`
> - Quant plan → `docs/quants.md`
>
> **Corrections vs. verified sources (so nobody re-mines these mistakes):**
>
> - `2608.00017` is NOT a Liquid paper — it's "Memory Reward Inflation in Self-Improving LLM
>   Agents" (Zamanifar et al., Jun 2026). Kept in `research/`: relevant to agentic memory
>   scoring/routing. The two LFM papers are `2511.23404` (LFM2 tech report) and
>   `2607.15232` (in-place tokenizer expansion, LFM2-8B-A1B→LFM2.5-8B-A1B).
>
> **Ops note (2026-08-25):** first `.env` (5 API keys) was accidentally committed in the
> root commit; history was rewritten with git-filter-repo, old objects purged locally
> (reflog expire + `gc --prune=now`), and the remote `main` force-updated. Commit SHAs
> therefore differ from anything seen on 8/24-8/25. All 5 keys (Firecrawl, Tavily, Brave,
> Exa, HuggingFace) are to be treated as COMPROMISED and rotated; `.env.example` is the
> new template. The repo has two remotes configured (`origin` + `jamesnavinhill`) pointing
> at the same private repo.
> - The `Trelis/Function_Calling_Extended` dataset is 59 paid rows; the real tool-calling
>   SFT set is `Salesforce/xlam-function-calling-60k`.
> - `m-a-p/CodeFeedback-Filtered-Instruction` verified real (Apache-2.0).
>   `iamtarun/python_code_instructions_18k_alpaca` does NOT resolve on HF — drop it.
> - Baseline class has moved: `Qwen3-1.7B`/`Gemma 3 1B` tables are from the Jan-2026 LFM2.5
>   launch; the Aug-2026 competitor set is Gemma 4 (E2B/E4B/26B-A4B), Qwen3.5 (2B/4B/9B),
>   gpt-oss-20b, Granite 4.0-H-Tiny. See `docs/benchmarks.md`.
> - The outline's "custom gating wrapper" MoE plan is superseded: Liquid ships a native
>   `lfm2_moe` (8B-A1B) — see `docs/moe.md` findings 1–3.
> - 8B-A1B Q4_0 GGUF = 4.84 GB verified (fits tier 4 under 8 GB).
> - License `lfm1.0` = permissive derivatives/redistribution, $10M revenue commercial
>   threshold (see `docs/models.md`).

Goals:
Surpass LiquidAI's on-record post-training benchmarks with multi-modal agentic scores
produce similar sized or smaller, and outperforming gguf quants of those models

create a hybrid MoE? style model family from their VL + Sized Base Models
That can follow instructions, reason, call tools, receive images, produce code, etc.
Targeting a set of device-class sizes (<1.5,1.5-3,3-5,5-8) with further gguf quants from those

LiquidAI has post-trained models to test against
Community recognized finetunes or quants that claim improvement over base should be added to the baseline test/standards.

My workflow and Tech stack to optimize for --AIM for the underlaying or broader ecosystem that keeps the focus without losing the plot.
Windows/Linux, Docker, Nvidia GPUS
Node, Python, Next.js, React, Typescript, JavaScript...
Vite, electron, Tauri...
GitHub, Cloudflare, Oracle, AWS...
Supabase, Neon, MongoDB...
ElevenLabs, Anam, real-time voice/video...
Sentry, Posthog, ClickHouse, OpenTelemetry...

I have a voice/video agent that sits on-top/aside from my agent harnesses and connects through typical api,mcp, flows to delegate/route work across disciplines.

From a casual convo with my on-top agent we come up with a plan for a project and send it off to our gateway. That gateway (currently) then routes our requests thru litellm/langfuse to various model providers with some other connected parts obv, but importantly --this work, from researching, planning, implementing, auditing, tool calls, code execution, browser-use, etc. all routes through the harness. We're aiming to find and improve open-weight models that fit each one of these tasks, as well as think of ways to create a hybrid model that can handle all (even if at a larger on-disk cost). This will allow us to fill in our API costs with low-imprint models across our workflows trained to our techstack.

These models seem like a great target.
Current: just updated a few hours ago!
Variety: Covers all sizes for our use-cases
Multi-modal: Image input is crucial, understanding screenshots, etc..
Value: Less community variants and finetunes than other class-equivalent models.

---

### Technical Docket: On-Device Architectures, Datasets, and MoE Merging

This docket consolidates the core mathematical specifications, benchmarks, official documentation references, and technical pathways to execute your plan of building a custom, stack-specialized, multi-modal hybrid MoE from the **Liquid AI (LFM2.5)** and **Qwen3** on-device families.

---

### Liquid AI (LFM2.5) Architectural Specifications & Baselines

Liquid Foundation Models (LFMs) do not use the standard Transformer attention mechanism. Instead, they are built with **computational units rooted in dynamical systems, signal processing, and numerical linear algebra**, giving them linear scaling complexity (meaning sub-millisecond context ingestion/prefill).

#### 1. Available Open-Weight Model Core Sizes

- **LFM2.5-1.2B-Base**: The raw, unaligned pre-trained checkpoint. This is your primary candidate for heavy domain-specific SFT or initializing a custom MoE.
- **LFM2.5-1.2B-Instruct**: General-purpose instruction-tuned model, trained with SFT, preference alignment, and multi-stage reinforcement learning.
- **LFM2.5-VL-1.6B**: Vision-language model using the LFM2.5 base backbone. It is tuned for multi-image comprehension and multilingual vision understanding.
- **LFM2.5-Audio-1.5B**: Native speech-to-speech audio-language model. It runs a custom, LFM-based audio detokenizer that converts discrete model tokens into waveforms. This detokenizer is **8x faster than LFM2's Mimi detokenizer** on mobile CPUs and is quantization-aware trained (QAT) to run at INT4 precision with virtually zero quality loss.

#### Performance Benchmarks (General & Multi-Modal)

Below are the official records for the **LFM2.5** family against standard on-device models, including Qwen3, Google Gemma 3, Meta Llama 3.2, and IBM Granite:

**Text-Based Capabilities:**

| Model | GPQA | MMLU-Pro | IFEval | IFBench | BFCLv3 (Tool Calling) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LFM2.5-1.2B-Instruct** | **38.89** | **44.35** | **86.23** | **47.33** | **49.12** |
| **Qwen3-1.7B (Instruct)** | 34.85 | 42.91 | 73.68 | 21.33 | 46.30 |
| **Granite-4.0-1b** | 24.24 | 33.53 | 79.61 | 21.00 | **52.43** |
| **Gemma 3 1B IT** | 24.24 | 14.04 | 63.25 | 20.47 | 16.64 |
| **Llama 3.2 1B Instruct** | 16.57 | 20.80 | 52.37 | 15.93 | 21.44 |

**Vision-Language (VLM) Capabilities:**

| Model | MMStar | MM-IFEval | BLINK | MMMU (Val) | IFEval (Text) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LFM2.5-VL-1.6B** | **50.67** | **52.29** | **48.82** | 40.56 | **71.89** |
| **InternVL3.5-1B** | 50.27 | 36.17 | 44.19 | **41.89** | 68.29 |
| **LFM2-VL-1.6B** (Old) | 49.87 | 46.35 | 44.50 | 39.67 | 64.26 |

#### Inference Speeds (Prefill / Decode)

Because of their linear architecture, prefill speeds on local silicon are blistering (measured in tokens/second with 1K prefill context):

- **Qualcomm Snapdragon Gen4 (NPU)**: **4,391 prefill tok/s** / 82 decode tok/s (Memory footprint: **0.9 GB** in Q4_0).
- **AMD Ryzen AI 9 HX 370 (CPU)**: **2,975 prefill tok/s** / 116 decode tok/s (Memory footprint: **856 MB** in Q4_0).
- *(For comparison, Qwen3-1.7B on the same AMD CPU yields 2,008 prefill / 62 decode at a 1.46 GB footprint).*

#### Citations & Official Papers

<https://arxiv.org/abs/2608.00017>
"2608.00017v1.pdf"
<https://arxiv.org/abs/2607.15232>
"2607.15232v1.pdf"
<https://arxiv.org/abs/2511.23404>
"2511.23404v1.pdf"

- **Liquid AI Official Repository**: [Hugging Face/LiquidAI](https://huggingface.co/LiquidAI).

---

### Deconstructing the Target Suite & Device-Class Sizes

Your target list fits perfectly into the precise physical memory boundaries of local hardware. At standard 4-bit (Q4_K_M) or 8-bit (Q8) GGUF quantizations, here is how your chosen targets match your size tiers:

1. **Tier 1: < 1.5 GB — `LFM2.5-1.2B-Base`**
    - *Quantized footprint*: **~850MB** in Q4_0.
    - *Role*: Your micro-orchestrator. Running locally on Tauri or Electron, it handles instant routing, intent classification, and state updates.
2. **Tier 2: 1.5 – 3 GB — `LFM2.5-2.6B-Base`**
    - *Quantized footprint*: **~1.8GB** in Q4_K_M.
    - *Role*: The intermediate developer assistant. It has enough parameter capacity to understand your system's folder structures and file-system paths.
3. **Tier 3: 3 – 5 GB — `LFM2.5-VL-3B`** *(and LFM2.5-VL-1.6B)*
    - *Quantized footprint*: **~2.2GB - 3.2GB**.
    - *Role*: Your multimodal sensor. Crucial for "browser-use" subagents, parsing real-time screenshots of code editors, and tracking visual UI states.
4. **Tier 4: 5 – 8 GB — `LFM2.5-8B-A1B`**
    - *Quantized footprint*: **~4.5GB - 5.5GB**.
    - *Role*: The "Brain" of the swarm. This is the master expert that handles complex code execution, system architecture, and deep knowledge synthesis.

---

### Fine-Tuning Specifically to Your Tech Stack

If you want these models to replace your API costs, they must be trained as **highly opinionated Senior Software Engineers** for your exact environment:

- **Tauri/Electron & React**: The SFT dataset should be heavily populated with clean examples of React rendering cycles, IPC (Inter-Process Communication) bridges between Electron and Node, and Vite configuration files.
- **Database & Telemetry**: Instead of teaching the model general SQL, SFT-align it to write optimized **ClickHouse** queries for massive logging datasets and structured **Supabase/Neon** relational schemas.
- **The Sentry/OpenTelemetry Integration**: Teach your model how to read a raw **Sentry error payload** or **PostHog event stream** and immediately synthesize a debugging plan.

By feeding these specific architectures into the base models during your SFT phase, your on-device models will instantly output code tailored to your exact stack, easily surpassing generalist cloud APIs.

### SOTA Datasets (SFT & Alignment)

#### 1. Code Generation (Next.js, React, Node, Python, Tauri)

- **`m-a-p/CodeFeedback-Filtered-Instruction`**: A highly curated dataset focusing on interactive code correction, debugging, and iterative coding conversations. Perfect for aligning the "Implementer" and "Auditor" roles of your harness.
- **`iamtarun/python_code_instructions_18k_alpaca`**: Clean Python task executions.
- **`Artifacts / Local Custom Extraction`**: To train models on the Next.js/React/Tauri boundary, the most effective approach is to run a local script that parses premium open-source repositories (e.g., extracting files containing Tauri `invoke()` hooks, Vite configurations, and Electron main-to-renderer IPC setups). These are compiled into raw text files and formatted into SFT prompt-completion pairs.

#### 2. Tool Calling & Agentic MCP Orchestration

- **`Trelis/Function_Calling_Extended`**: Designed to train models to output strict JSON templates matching custom schemas and function definitions without dropping brackets.
- **`Salesforce/xlam-function-calling-60k`**: A large-scale, high-quality function-calling dataset that aligns base models for complex API routing.
- **Model Context Protocol (MCP)**:

#### 3. Databases & Logging (Supabase, Neon, PostgreSQL, ClickHouse)

- **`b-mc2/sql-create-context`**: Paired natural language questions with exact SQL schema definitions and corresponding target queries.
- **`Clinton/Text-to-SQL-v1`**: Excellent for training your models to read complex schemas (PostgreSQL / ClickHouse dialect structures) and synthesize raw queries.

---

### The MoE (Mixture of Experts) Process & Science

An MoE maps independent "expert" networks to a centralized routing gating layer, activating only a small percentage of total parameters per token to keep inference latency exceptionally low.

#### 1. The Mergekit Pathway

The open-source community primarily uses **`mergekit`** to construct custom MoEs.

- **Underlying Method**: In standard transformers, `mergekit` clones self-attention layers and feeds them into a newly initialized gating network.
- **The LFM Constraint**: Because LFMs do not use traditional transformer self-attention blocks, **traditional parameter interpolation on weight space must be handled differently**. You cannot merge non-transformer layers directly with transformer layers using standard mergekit configurations.
- **SST/LFM Gating**: To create a hybrid MoE from Liquid's family, you must build a custom PyTorch gating wrapper that routes tokens across the linear state-space operators of the different LFM expert models (e.g., routing visual tokens to the `LFM2.5-VL-1.6B` expert block and text tokens to the `LFM2.5-1.2B-Base` expert block).

#### 2. The Four-Stage Reasoning Post-Training Pipeline

When you transition from SFT to reinforcement learning to optimize your MoE, you should adopt the exact pipeline used to build Qwen3's reasoning capacities:

```
  Stage 1: Long CoT Cold Start (SFT) 
  └─ Fine-tune on long, step-by-step mathematical & coding reasoning traces.
  
  Stage 2: Reasoning-Based Reinforcement Learning (RL)
  └─ Scale up GRPO utilizing rule-based compilers or syntax validators for rewards.
  
  Stage 3: Thinking Mode Fusion
  └─ Blend long CoT and direct instruction-tuning data to integrate quick-response.
  
  Stage 4: General Domain RL
  └─ Run GRPO across 20+ general-domain tasks (format alignment, tool-calling syntax).
```

### Creating a Hybrid LFM-MoE Model Family

To create a single hybrid model family that can follow instructions, write React code, call tools, and receive images, you can use **mixture-of-experts (MoE) block-merging**.

#### How to build an LFM-MoE

Because LFMs are composed of linear dynamic operators and feed-forward state blocks (rather than standard multi-head attention), you can construct an MoE similarly to how Mistral or Qwen3-MoE are built:

1. **Block Cloning**: You take your trained, stack-specialized `LFM2.5-1.2B` (Expert A) and your trained `LFM2.5-VL-1.6B` or `LFM2.5-2.6B` (Expert B).
2. **Initialize a Router**: You initialize a tiny, lightweight gating router layer.
3. **Post-Training Fusion**: You freeze the heavy mathematical weights of the cloned LFM expert blocks and only train the routing layer. During training, the router learns to send visual tokens (like screenshots of your Next.js app) to the VL-experts, and Javascript/Node compilation errors to the code-expert blocks.

This gives you a model that has an **on-disk size of ~5GB**, but only **activates a fraction of its parameters per token** during inference. You get the multi-modal visual capability of a VL model and the raw coding speed of a base model wrapped into a single, blazing-fast GGUF package
---
