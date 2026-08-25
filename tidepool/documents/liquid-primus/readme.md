# Primus-Liquid — Goal & Directive

A direction for Primus to take from here. It points at a goal, a baseline to beat, the
constraints that hold, and the priority when they conflict — not a step-by-step method.
The research process (which recipe, which data, which order) is yours to design and to
vary; this is what "good" has to mean.

Full supporting evidence lives alongside this file and is pulled down on demand:

- `docs/models.md` — the LiquidAI target family, sizes, architecture, first-party recipes, license
- `docs/benchmarks.md` — the baseline table to beat: our family vs. the Aug-2026 competitor class
- `docs/quants.md` — quantization options and the quality bar for our GGUF targets
- `docs/moe.md` — the hybrid MoE goal, the first-party reference architecture, and the method options
- `docs/fine-tune.md` — our workflow, the stack we specialize for, and the dataset map
- `datasets/` — curated SFT + eval datasets (verified IDs, licenses, sizes, role)
- `research/` — primary papers (Liquid technical reports + tokenizer expansion + one agent-memory paper)

## The goal

Produce **improved versions of the LiquidAI LFM2.5 family** and **a hybrid mixture-of-experts
family built from them**, specialized to the way this organization actually works — its
workflows, its tooling, and its stack — at edge/on-device sizes.

Concretely, a "good" outcome looks like a set of models where, per device-class size tier,
ours **measurably beats the incumbent baseline** on the capability axes that matter to us
(instruction-following, reliable tool/function calling, agentic multi-step work, code
production for our stack, and — for the VL models — reading screens, documents, and UI),
**without** regressing general quality, and where the quantized GGUF forms we ship hold the
full-precision quality bar.

We are not optimizing for a single leaderboard score. We are optimizing for a small, fast,
private model fleet that can be dropped into our own agent harness (research → plan →
implement → audit, with tool calls and code execution) and do the work at a fraction of the
API cost.

## The baseline to beat (the incumbents)

The models whose published post-training results we are trying to surpass, in our size tiers:

1. **LiquidAI's own LFM2.5 post-trained checkpoints** (1.2B-Instruct, 2.6B, 8B-A1B, VL-1.6B,
   VL-3B) — our direct starting point, and the "on-record" scores we want to clear.
2. **The current open-weight competitor class at these sizes** — as of our Aug-2026 data
   snapshot this is Gemma 4 (E2B / E4B / 26B-A4B), Qwen 3.5 (2B / 4B / 9B), gpt-oss (20b),
   and Granite 4.0 (H-Tiny 7B-A1B). See `docs/benchmarks.md` for the exact published numbers
   and the per-tier mapping. Re-verify these against the latest public cards, since the field
   moves fast — the snapshot is a floor, not a ceiling.
3. **Community finetunes/quants that claim improvement over base** — e.g. the unsloth
   LFM2.5 GGUFs, NovachronoAI function-calling finetune, and any other recognized community
   variant in our tiers. These are fair baselines too, and some already beat the vendor base.

The point is not to beat "LiquidAI" in the abstract. It is to beat the **best available model,
vendor or community, in each size tier**, on the axes we care about, under fair/matched
conditions, and to show the win survives a better metric than a single score.

## Our size tiers (device-class)

Four on-disk tiers at standard 4-bit GGUF, each with a role in the fleet:

- **< 1.5 GB** — micro-orchestrator: routing, intent, state (1.2B-class)
- **1.5 – 3 GB** — intermediate developer assistant, understands our folder/file structure
  (2.6B-class)
- **3 – 5 GB** — multimodal sensor: reads screenshots, editor state, UI, documents
  (VL 1.6B / 3B-class)
- **5 – 8 GB** — the "brain": complex code execution, architecture, synthesis (8B-A1B-class)

The hybrid MoE family is allowed to be larger on disk in exchange for activating a small
fraction of parameters per token, but it must run on the same consumer hardware (a single
8 GB GPU, or CPU, is the reference device).

## Constraints (hard)

- **Compute**: reference device is a single 8 GB consumer GPU plus a modern CPU; target
  on-device inference, not a cluster. Keep each experiment within what a few GPUs can do;
  if a run needs more, flag the cost rather than assume it.
- **License**: the LFM2/LFM2.5 weights are under the **LFM Open License v1.0** — it permits
  derivative works, fine-tuning, and redistribution (modified files carry change notices).
  The one limitation is a commercial-use threshold of **$10M annual revenue per legal entity**
  (non-profts exempt). Our use is below the threshold and non-commercial/personal, so all
  of the above is in-scope. Keep derivative artifacts clearly marked as derived from LFM2.5.
- **Open-weights only**: everything we start from must be downloadable open-weight; no API-only
  models as targets or teachers beyond whatever we already pay for.

## Priority when goals conflict

When a trade-off comes up, resolve it in this order:

1. **Quality on our axes** (tool-calling reliability + agentic follow-through first, then
   our-stack code, then general) — do not trade tool-use reliability for raw benchmark score.
2. **On-device efficiency** (memory + latency at 4-bit) — the whole point is it runs on our
   hardware; a model that needs a workstation to run is a miss.
3. **General quality** (don't regress the base's strengths) — a specialist that forgot how to
   behave is not an improvement.

If two of these genuinely conflict, state the trade-off and let the result decide — do not
silently optimize one at the expense of another.

## What we can hand you (pull on demand)

- **Models**: the full LFM2.5 family (base + post-trained + VL + audio + encoders + the
  8B-A1B `lfm2_moe` MoE) and its GGUFs, on Hugging Face under `LiquidAI/...` — see
  `docs/models.md` for the exact IDs.
- **Data**: verified SFT + eval datasets for function-calling, code, text-to-SQL,
  instruction structure, and antidoom, with licenses and sizes — see `datasets/`.
- **Recipes to draw from**: Liquid's own post-training (2.6B 4-stage agentic recipe, VL
  SFT+KD+antidoom then multi-reward RL, 8B-A1B reasoning-only RL with anti-hallucination and
  anti-doom stages), QAD quantization, DSpark speculative decoding, and in-place tokenizer
  expansion — documented in `docs/models.md` and `docs/moe.md`. These are starting points,
  not requirements.
- **Papers**: the LFM2 technical report, the LFM tokenizer-expansion paper, and an
  agent-memory/reward-inflation paper relevant to the harness — in `research/`.

## Suggested shape (your call)

This maps most cleanly onto a **"beat the incumbent"** and an **"optimize one dimension
(efficiency at fixed quality)"** effort, with an open question underneath: *can a hybrid MoE
built from these on-device models cover all of our modalities at a small active-parameter
cost, and hold the quality bar after 4-bit quantization?* Start in the smallest tier where a
win is cheapest to demonstrate, prove it holds, then scale up. Name a specific weakness to
improve (our top candidate: reliable tool-calling + long-horizon follow-through on a
consumer GPU) and improve it without breaking the rest.

I am not sure of the exact compute budget I want to spend per experiment — size the first
sweep for what's available and tell me what a bigger version would cost before spending past
what's approved.
