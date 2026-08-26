# Datasets — SFT & eval catalog

Curated input data for the post-training, mapped to the capability axes in
`docs/fine-tune.md`. **Verified set (section 1)**: IDs, licenses, and sizes confirmed
against Hugging Face on 2026-08-25. **Eval set (section 2)**: the benchmark names used in
Liquid's own published tables and the ArtificialAnalysis methodology — the IDs get resolved
against HF at eval time; the names are the contract.

Ownership convention: public data below is *input* data. Our own repo-mined and
gateway-transcript data (the in-house set) is NOT in this folder — it is generated at
project time and owned by the work it informs; keep it out of version control until the
format is stable.

## 1. SFT / training data (verified 2026-08-25)

| Dataset | HF ID | License | Size | Teaches | Notes |
| --- | --- | --- | --- | --- | --- |
| APIGen function-calling | `Salesforce/xlam-function-calling-60k` | cc-by-4.0 | 60k | tool/function calling, structured JSON | 3-stage verified (format, execution, semantic). **Gated** — request access. This is the real tool-calling SFT set; the outline's `Trelis/Function_Calling_Extended` is 59 **paid** rows and was a red herring. |
| OpenCodeInterpreter instructions | `m-a-p/CodeFeedback-Filtered-Instruction` | apache-2.0 | 100k–1M | iterative code generation with execution feedback + refinement | Pairs with our "implementer/auditor" harness roles. |
| SQL with schema context | `b-mc2/sql-create-context` | cc-by-4.0 | 78,577 | NL + CREATE TABLE → correct SQL | Built to **prevent column/table-name hallucination** — directly our Postgres/ClickHouse need. WikiSQL+Spider lineage. |
| Large text-to-SQL | `Clinton/Text-to-SQL-v1` | apache-2.0 | 100k+ | NL → SQL | Scale filler; pair with the above. |
| Structured-output compliance | `LiquidAI/ifstruct-v1.0` | apache-2.0 | 1k–10k | valid JSON/YAML under varied phrasing | Liquid's own benchmark (Jul 2026); scoreable without constrained decoding — good as both training signal and eval axis. |
| Antidoom prompts | `LiquidAI/antidoom-mix-v1.0` | apache-2.0 | 100k–1M | prompts for looping/doom preference training | Prompts-only by design: responses are generated on this set and looping traces retained for preference pairs. Use with `dpo_with_trl.ipynb` / RL. |
| Multilingual retrieval | `LiquidAI/nanobeir-multilingual-extended` | see repo | ~692k rows | retrieval representations | For the encoder/retriever lane if we extend beyond generation. |

**Gaps we close in-house (no public equivalent):** our-framework code idioms
(Next/React/TS, Node, Python, Electron/Tauri IPC, Vite), our ClickHouse schema + query
patterns, gateway research→plan→implement→audit trajectories with real tool calls and
results, Sentry-style error payloads → debugging plans, PostHog-style event streams →
insight. See `docs/fine-tune.md` for the mining approach.

## 2. Eval axes (names from Liquid's published tables + AA methodology)

The yardsticks a result is "good" against — per tier, see `docs/benchmarks.md`.

- **Tool calling / function**: BFCLv3, BFCLv4, ToolSandbox, τ²-Bench (telecom/retail),
  τ³-Bench, IFStruct (structured output).
- **Instruction following**: IFEval, IFBench, Multi-IF.
- **Knowledge / honesty**: GPQA (Diamond), MMLU-Pro, AA-Omniscience Index (accuracy +
  non-hallucination, ArtificialAnalysis methodology).
- **Math / reasoning**: GSM8K, MATH500, AIME25.
- **Code**: LiveCodeBench v6, HumanEval, MBPP (+ our in-house stack eval).
- **Agentic end-to-end**: Claw-Eval, PinchBench, BrowseComp+, `sapbot/ask-my-agent-bench-2`
  (present in the 1.2B card's own eval results), BIRD (text-to-SQL under memory-based
  agentic conditions — the setup in `research/2608.00017v1.pdf`).
- **Vision (VL tier)**: MMStar, MMMU / MMMU-Pro, BLINK, MME, DocVQA, InfographicVQA,
  ChartQA, OCRBench v1/v2, TextVQA, ScreenSpot-v2 (desktop/mobile/web), RefCOCO, MuirBench,
  HallusionBench, POPE, RealWorldQA, SimpleVQA, SEED-Bench, MMBench, CountBenchQA,
  LogicVista, MathVista, MM-IFEval. (All appear in the VL-3B / LFM2.5 launch tables —
  VLMEvalKit or vLLM 0.26.0 as Liquid used.)

## 3. Handling rules

- Keep eval sets **separate from training mixes** at the file level so a "win" can't come
  from contamination.
- Every dataset we *produce* (in-house SFT/RL pairs, distilled teachers) must carry its
  recipe (source prompts, generator model, filter) in a sidecar — the same
  reproducibility standard the goal demands of the final models.
- Gated/licensed sets (xlam is CC-BY but gated) get access requested before the project
  kicks off, not mid-run.
