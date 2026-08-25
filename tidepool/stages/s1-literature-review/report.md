# Stage 1: Literature review

_Status: complete._

## Headline

The literature search came back full: 212 papers with complete text, covering every one of
the 22 questions the scope raised, with no sub-topic left thin enough to limit what the
project can claim. The two areas the plan was most exposed on are both well served, with
roughly a dozen papers each on whether 4-bit compression damages structured output more
than it damages knowledge, and on how reward-based training gets gamed when the reward is
a schema check.

## s1.1 — research queries

Derived from the provisional `overview.md`: one or more per distinct sub-topic, plus a
dedicated query for every low-confidence assumption and every open question, since those
are exactly what the scope revision needs evidence to settle.

### arXiv categories

`cs.CL` is the primary category for this project: tool calling, structured output,
instruction tuning, and benchmark contamination are natural-language-processing
literature, much of it conference-only. `cs.LG` covers the training-method and
quantization work, `cs.AI` the agentic and planning work, `cs.SE` the code-generation
work, `cs.DB` the text-to-SQL work, and `stat.ML` the calibration and abstention work.
Naming them this way matters for the next substage: a thin result on a `cs.CL`-native
sub-topic means the indexed corpus never held that literature, rather than that the
field is quiet.

Chosen: `cs.CL`, `cs.LG`, `cs.AI`, `cs.SE`, `cs.DB`, `stat.ML`.

### Queries

**The primary axis — tool calling in small models**

| # | Query | Sub-topic it covers |
| - | ----- | ------------------- |
| Q1 | reliable function calling and tool use in small language models under 3B parameters | the core claim: is a tier-1 tool-calling win a known, achievable thing |
| Q2 | fine-tuning language models for structured JSON output and schema compliance without constrained decoding | first-attempt schema validity, our structured-output metric |
| Q3 | benchmark design and evaluation methodology for LLM function calling, BFCL and tool-use benchmarks | whether our primary metric measures what we think it does |
| Q4 | multi-step agentic tool use and long-horizon task reliability in compact language models | the agentic follow-through half of the operator's named weakness |

**The behaviour with no benchmark — flag rather than assert**

| # | Query | Sub-topic it covers |
| - | ----- | ------------------- |
| Q5 | LLM abstention and refusal to answer when retrieved context or tool output is missing or corrupted | the flag-rather-than-assert probe, and whether anyone has scored it |
| Q6 | hallucination under tool failure: language models fabricating values absent from API responses | the failure mode the probe is built to catch |
| Q7 | calibration and confidence estimation for language model tool calls and structured predictions | our calibration axis, and the low-confidence 0.80/0.10 target |

**The four candidate recipes**

| # | Query | Sub-topic it covers |
| - | ----- | ------------------- |
| Q8 | LoRA versus full fine-tuning for capability specialization in sub-3B language models | candidate 1 versus candidate 2, and the order we try them in |
| Q9 | data mixture ratios and replay to prevent catastrophic forgetting during instruction tuning | the guardrail-tolerance risk: keeping the instruction-following lead |
| Q10 | direct preference optimization for instilling behavioral traits and refusal behavior in language models | candidate 3, the preference-optimization path to the flag behaviour |
| Q11 | reinforcement learning with verifiable rewards for code, schema validity, and executable tool calls | candidate 4, and the reward design that avoids trivially-valid calls |
| Q12 | reward hacking and specification gaming in verifiable-reward reinforcement learning for language models | the named risk under candidate 4 |
| Q13 | on-policy knowledge distillation to fuse domain specialist teachers into a single student model | the vendor's own recipe, and whether it is worth reproducing at this size |

**The quantization question**

| # | Query | Sub-topic it covers |
| - | ----- | ------------------- |
| Q14 | four-bit post-training quantization quality degradation across task types in small language models | the per-axis retention bar and whether 97% average is the right shape |
| Q15 | quantization-aware training and distillation for low-bit language models with quality recovery | the default path to the 4-bit deliverable |
| Q16 | does low-bit quantization degrade structured output and instruction following more than knowledge tasks | the specific risk the per-axis 93% floor exists to catch |
| Q17 | on-device language model inference efficiency, throughput and memory at low precision | the efficiency floor, currently a low-confidence 5% tolerance |

**Evidence hygiene**

| # | Query | Sub-topic it covers |
| - | ----- | ------------------- |
| Q18 | training data contamination detection and decontamination for language model benchmarks | the top risk in the scope: a contaminated tool-calling win |
| Q19 | reproducibility of language model benchmark scores across evaluation harnesses and prompt templates | why every baseline here is a matched rerun rather than a card quote |

**Adjacent techniques the scope depends on**

| # | Query | Sub-topic it covers |
| - | ----- | ------------------- |
| Q20 | tokenizer vocabulary size effects on code and JSON tokenization efficiency and downstream quality | the 65K-vocabulary open question |
| Q21 | schema-grounded text-to-SQL and suppressing hallucinated column and table names | the data-and-telemetry capability axis |
| Q22 | hybrid linear attention and convolution language model architectures for on-device inference | the family we are building on, and whether its recipes transfer |

### Coverage of the scope's soft spots

Every low-confidence assumption and open question in `overview.md` has a query pointed at
it: the flag-rate and false-flag targets (Q5, Q6, Q7), the throughput floor (Q17), the
65K-vocabulary question (Q20), which reinforcement-learning signal instils the flag habit
(Q10, Q11, Q12), and whether a 1.2B model can gain tool-calling accuracy without losing
its instruction-following lead (Q8, Q9). Two sub-topics are expected to be thin in an
indexed corpus built from machine-learning submissions: tool-calling benchmark design
(Q3) and abstention under tool failure (Q5, Q6), both of which are natural-language-
processing and conference-heavy. If they come back thin, the next substage records that
as a coverage gap rather than a quiet absence.

## s1.2 — retrieval results

**212 papers retrieved, all with full text on disk** (`papers/`, 17 MB). Every one of the
22 queries cleared the three-paper coverage threshold on the indexed corpus alone, so the
external escalation tier was never invoked. The index below reconciles exactly against
`papers/`: 212 table rows, 212 files, nothing in one and not the other.

### Coverage per query

| Query | Kept | Query | Kept |
|---|---|---|---|
| Q1 reliable function calling, sub-3B | 8 | Q12 reward hacking / spec gaming | 10 |
| Q2 structured JSON without constrained decoding | 7 | Q13 on-policy multi-teacher distillation | 8 |
| Q3 function-calling benchmark methodology | 13 | Q14 4-bit PTQ degradation by task type | 12 |
| Q4 multi-step / long-horizon agentic use | 10 | Q15 QAT and QAD quality recovery | 13 |
| Q5 abstention on missing or corrupt context | 9 | Q16 does low-bit hurt structure > knowledge | 10 |
| Q6 hallucination under tool failure | 7 | Q17 on-device inference at low precision | 10 |
| Q7 calibration of tool calls | 10 | Q18 contamination detection / decontamination | 8 |
| Q8 LoRA vs full fine-tuning | 10 | Q19 harness and template reproducibility | 5 |
| Q9 data mixture and forgetting | 8 | Q20 tokenizer vocabulary effects | 7 |
| Q10 DPO for behaviour and refusal | 12 | Q21 schema-grounded text-to-SQL | 11 |
| Q11 RL with verifiable rewards | 13 | Q22 hybrid linear-attention architectures | 11 |

**No coverage gaps.** The four queries pre-flagged at s1.1 as likely thin in a
`cs.LG`-built corpus all cleared comfortably: Q3 kept 13, Q5 kept 9, Q6 kept 7, Q19 kept
5, Q21 kept 11. Nothing has to be recorded as an unretrievable sub-topic, and s2.2 is not
limited in what it may claim on any of the 22 axes.

**Nothing dropped for missing full text.** Every selected hit downloaded and produced
real paper text; the smallest file is 9.4 KB and the median is well above that. No hit was
recorded as found-but-unreadable.

**The diminishing-returns stop rule did not fire.** All 22 queries ran to completion; no
two consecutive queries contributed zero new papers after cross-query dedup, and the
thinnest query still added 5. Returns had not flattened when the query list ran out, which
is worth noting as a limit: a longer query list would very likely have kept adding
material, so 212 is a floor on what this corpus holds for the project, not a ceiling.

**One retrieval note for the record.** The single most directly relevant document found is
the LFM2 technical report (`2511.23404v1`), the architecture family the project builds on.
Also present is a paper on the same lineage applied to drug discovery (`2603.03517v1`).
Both are first-party accounts and get read as evidence about the architecture, with the
usual discount applied to a vendor's own claims about its own model.

### Index

| file | title | query | source | relevance to this project |
|---|---|---|---|---|
| 2511.22138v1.md | TinyLLM: Evaluation and Optimization of Small Language Models for Agentic Tasks on Edge Devices | Q1 | corpus | sub-3B models on BFCL for edge tool-calling |
| 2604.20148v1.md | Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models | Q1 | corpus | hypernetwork adaptation for SLM tool use |
| 2605.26128v1.md | The Constraint Tax: Measuring Validity-Correctness Tradeoffs in Structured Outputs for Small LMs | Q1 | corpus | structured-output validity/correctness tradeoff at sub-3B scale |
| 2510.03847v1.md | Small Language Models for Agentic Systems: A Survey of Architectures, Capabilities, and Deployment Trade-offs | Q1 | corpus | survey of SLM function calling/JSON/LoRA/edge deployment |
| 2501.02342v1.md | Optimizing Small Language Models for In-Vehicle Function-Calling | Q1 | corpus | applied SLM function-calling fine-tuning case study |
| 2604.20316v1.md | R2IF: Aligning Reasoning with Decisions via Composite Rewards for Interpretable LLM Function Calling | Q1 | corpus | composite reward design for function-calling reasoning |
| 2603.16901v1.md | From Language to Action in Arabic: Reliable Structured Tool Calling via Data-Centric Fine-Tuning | Q1 | corpus | data-centric SFT recipe for reliable tool calling |
| 2410.04587v2.md | Hammer: Robust Function-Calling for On-Device Language Models via Function Masking | Q1 | corpus | on-device function-calling robustness technique |
| 2605.02363v1.md | When Correct Isn't Usable: Improving Structured Output Reliability in Small Language Models | Q2 | corpus | SLM structured-output reliability without constrained decoding |
| 2505.04016v1.md | SLOT: Structuring the Output of Large Language Models | Q2 | corpus | fine-tuning-based output structuring method |
| 2502.14905v1.md | Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherence | Q2 | corpus | RL for schema adherence |
| 2602.12247v2.md | ExtractBench: A Benchmark and Evaluation Methodology for Complex Structured Extraction | Q2 | corpus | benchmark for structured extraction quality |
| 2602.14743v1.md | LLMStructBench: Benchmarking Large Language Model Structured Data Extraction | Q2 | corpus | structured-extraction benchmark methodology |
| 2506.01151v1.md | Earley-Driven Dynamic Pruning for Efficient Structured Decoding | Q2 | corpus | grammar-constrained decoding baseline for comparison |
| 2603.03305v1.md | Draft-Conditioned Constrained Decoding | Q2 | corpus | training-free structured decoding alternative, tested on 1B model |
| 2503.16416v2.md | A Survey on Evaluation of LLM-based Agents | Q3 | corpus | agent/tool-use evaluation methodology survey |
| 2409.00920v2.md | ToolACE: Winning the Points of LLM Function Calling | Q3 | corpus | function-calling data synthesis and BFCL results |
| 2511.01934v2.md | Tool Zero: Training Tool-Augmented LLMs via Pure RL from Scratch | Q3 | corpus | RLVR for tool-augmented LLMs |
| 2605.01347v1.md | MAD-OPD: Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate | Q3 | corpus | on-policy distillation for tool-calling agents |
| 2507.03336v4.md | Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky | Q3 | corpus | SFT recipe for reliable enterprise tool calling |
| 2511.09148v2.md | LoopTool: Closing the Data-Training Loop for Robust LLM Tool Calls | Q3 | corpus | data-training loop for tool-call robustness |
| 2505.20192v3.md | BalanceSFT: Improving LLM Function Calling with Balanced Training Signals and Data Hardness | Q3 | corpus | data-mix/hardness balancing for function-calling SFT |
| 2510.07737v1.md | ToolExpander: Extending the Frontiers of Tool-Using RL to Weak LLMs | Q3 | corpus | RLVR for tool use in small/weak models |
| 2408.04682v2.md | ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use | Q3 | corpus | stateful tool-use benchmark design |
| 2601.13572v1.md | Behavior Knowledge Merge in Reinforced Agentic Models | Q3 | corpus | BFCL-evaluated RL agentic post-training |
| 2601.02151v1.md | Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate Forgetting | Q3 | corpus | forgetting mitigation evaluated on BFCL |
| 2502.06589v1.md | Hephaestus: Improving Fundamental Agent Capabilities via Continual Pre-Training | Q3 | corpus | continual pretraining for agent/tool capability |
| 2406.18518v1.md | APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets | Q3 | corpus | verifiable function-calling data generation pipeline |
| 2607.05378v1.md | CompactionRL: RL with Context Compaction for Long-Horizon Agents | Q4 | corpus | long-horizon agent RL efficiency |
| 2603.06713v1.md | Scaling Agentic Capabilities, Not Context: Efficient RL Finetuning for Large Toolspaces | Q4 | corpus | RL finetuning efficiency over large tool catalogs |
| 2511.19314v1.md | PRInTS: Reward Modeling for Long-Horizon Information Seeking | Q4 | corpus | reward modeling for multi-step agent tasks |
| 2602.21265v2.md | ToolMATH: A Diagnostic Benchmark for Long-Horizon Tool Use under Tool-Catalog Constraints | Q4 | corpus | long-horizon tool-use benchmark design |
| 2602.02619v2.md | daVinci-Agency: Unlocking Long-Horizon Agency Data-Efficiently | Q4 | corpus | data-efficient long-horizon agent training |
| 2512.07850v1.md | SABER: Small Actions, Big Errors - Safeguarding Mutating Steps in LLM Agents | Q4 | corpus | agent reliability under error-prone action steps |
| 2508.12685v3.md | ToolACE-MT: Non-Autoregressive Generation for Agentic Multi-Turn Interaction | Q4 | corpus | multi-turn tool-calling data generation |
| 2604.24964v1.md | Odysseys: Benchmarking Web Agents on Realistic Long Horizon Tasks | Q4 | corpus | long-horizon agent benchmark methodology |
| 2605.02572v1.md | On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length | Q4 | corpus | empirical study of training vs. horizon length |
| 2505.18135v2.md | Tool Preferences in Agentic LLMs are Unreliable | Q4 | corpus | tool-selection reliability issues in agentic LLMs |
| 2510.22977v2.md | The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination | Q5 | corpus | reasoning-induced tool hallucination |
| 2604.27283v1.md | Learning When to Remember: Risk-Sensitive Bandits for Abstention-Aware Memory Retrieval | Q5 | corpus | abstention-aware retrieval for coding agents |
| 2605.25850v1.md | TIAR: Trajectory-Informed Advantage Reweighting for LLM Abstention Learning | Q5 | corpus | RL method for training abstention behavior |
| 2607.10738v1.md | To Answer or to Abstain: Mitigating Search-Agent Hallucinations via Abstention-Aware RL | Q5 | corpus | RL-trained abstention vs. hallucination on broken retrieval |
| 2601.05503v2.md | Over-Searching in Search-Augmented Large Language Models | Q5 | corpus | failure mode analysis in tool/search-augmented LLMs |
| 2512.04597v1.md | When Robots Should Say "I Don't Know": Benchmarking Abstention in Embodied QA | Q5 | corpus | abstention benchmark methodology transferable to tool-output QA |
| 2604.18419v4.md | Knowing When to Quit: A Principled Framework for Dynamic Abstention in LLM Reasoning | Q5 | corpus | principled abstention framework |
| 2608.04286v1.md | Eliciting Intrinsic Hallucinations in LLMs via Semantically Equivalent Adversarial Attacks | Q5 | corpus | probing intrinsic hallucination triggers |
| 2605.29523v1.md | K-FinHallu: A Hallucination Detection Benchmark for Multi-Turn RAG in Korean Finance | Q5 | corpus | hallucination-detection benchmark methodology |
| 2604.04269v2.md | Beyond Fluency: Toward Reliable Trajectories in Agentic IR | Q6 | corpus | reliability of agent trajectories against retrieval/tool noise |
| 2602.19239v1.md | Attention Deficits in Language Models: Causal Explanations for Procedural Hallucinations | Q6 | corpus | causal mechanism of hallucination during procedures |
| 2607.18360v1.md | HALLMARK: Diagnosing Three Failure Modes in LLM Citation Verifiers | Q6 | corpus | failure-mode taxonomy for hallucination verifiers |
| 2605.01047v1.md | LLM Ghostbusters: Surgical Hallucination Suppression via Adaptive Unlearning | Q6 | corpus | targeted hallucination suppression method |
| 2605.19341v1.md | HalluWorld: A Controlled Benchmark for Hallucination via Reference World Models | Q6 | corpus | controlled hallucination benchmark |
| 2412.10246v2.md | Detecting LLM Hallucination Through Layer-wise Information Deficiency | Q6 | corpus | internal-signal hallucination detection on unanswerable inputs |
| 2507.23221v1.md | A Single Direction of Truth: Linear Residual Probe Exposes and Steers Contextual Hallucinations | Q6 | corpus | probe-based detection/steering of contextual hallucination |
| 2504.20168v1.md | MICE for CATs: Model-Internal Confidence Estimation for Calibrating Agents with Tools | Q7 | corpus | internal confidence signals for tool-calling agents |
| 2004.04361v2.md | Calibrating Structured Output Predictors for Natural Language Processing | Q7 | corpus | foundational calibration method for structured prediction |
| 2411.02454v2.md | Graph-based Confidence Calibration for Large Language Models | Q7 | corpus | calibration method for LLM predictions |
| 2403.05973v1.md | Calibrating Large Language Models Using Their Generations Only | Q7 | corpus | black-box calibration from generations |
| 2505.23854v1.md | Revisiting Uncertainty Estimation and Calibration of Large Language Models | Q7 | corpus | survey/analysis of LLM calibration methods |
| 2406.08391v3.md | Large Language Models Must Be Taught to Know What They Don't Know | Q7 | corpus | training models for calibrated abstention |
| 2404.04689v1.md | Multicalibration for Confidence Scoring in LLMs | Q7 | corpus | multicalibration technique for LLM confidence |
| 2604.12513v1.md | Agentic Control in Variational Language Models | Q7 | corpus | uncertainty/selective prediction for agentic control |
| 2501.17994v1.md | InnerThoughts: hidden-state calibration cheaper than PEFT | Q7 | corpus | lightweight calibration alternative to LoRA/QLoRA |
| 2509.20645v3.md | Anticipatory Evaluation of Language Models | Q7 | corpus | self-confidence estimation as part of eval methodology |
| 2605.19018v1.md | LoRA vs. Full Fine-Tuning: A Theoretical Perspective | Q8 | corpus | core theory of LoRA vs full FT capability specialization |
| 2605.30537v1.md | The Long-Term Effects of Data Selection in LLM Fine-Tuning | Q8 | corpus | data selection effects on FT outcomes |
| 2210.04802v2.md | Full fine-tuning vs. LoRA empirical comparison | Q8 | corpus | direct empirical LoRA/full-FT comparison |
| 2401.04151v1.md | Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning | Q8 | corpus | iterative LoRA method improving capability capture |
| 2406.16989v2.md | Retrieval-Augmented Mixture of LoRA Experts for Uploadable Machine Learning | Q8 | corpus | LoRA expert mixture for specialization |
| 2406.14956v1.md | Unlocking the Global Synergies in Low-Rank Adapters | Q8 | corpus | LoRA synergy/limitation analysis |
| 2602.05988v1.md | Layer-wise LoRA fine-tuning: a similarity metric approach | Q8 | corpus | layer-selective LoRA allocation method |
| 2508.04073v1.md | Efficient Strategy for Improving Large Language Model Capabilities | Q8 | corpus | LoRA-based capability improvement strategy |
| 2309.05444v1.md | Mixture of PEFT experts (MoLoRA / IA3 / MoV comparison) | Q8 | corpus | PEFT method comparison including LoRA mixtures |
| 2605.09015v1.md | LLiMba: Sardinian on a Single GPU - Adapting a 3B Language Model to a Vanishing Romance Language | Q8 | corpus | LoRA specialization case study at ~3B scale |
| 2407.08699v2.md | Mitigating Catastrophic Forgetting in Language Transfer via Model Merging | Q9 | corpus | forgetting mitigation via merging |
| 2603.16177v2.md | The Finetuner's Fallacy: When to Pretrain with Your Finetuning Data | Q9 | corpus | data-mixture timing for fine-tuning |
| 2508.09510v1.md | Enhancing Memory Recall in LLMs with Gauss-Tin: A Hybrid Instructional and Gaussian Replay Approach | Q9 | corpus | replay method against forgetting |
| 2602.08813v2.md | Robust Policy Optimization to Prevent Catastrophic Forgetting | Q9 | corpus | forgetting-resistant RL policy optimization |
| 2402.01364v2.md | Continual Learning for Large Language Models: A Survey | Q9 | corpus | core survey of continual learning/forgetting in LLMs |
| 2403.08370v3.md | SMART: Submodular Data Mixture Strategy for Instruction Tuning | Q9 | corpus | principled data-mixture selection for instruction tuning |
| 1811.11682v2.md | Experience Replay for Continual Learning | Q9 | corpus | foundational replay method |
| 2605.15220v1.md | Always Learning, Always Mixing: Efficient and Simple Data Mixing All The Time | Q9 | corpus | practical data-mixing recipe against forgetting |
| 2305.18290v3.md | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Q10 | corpus | core DPO paper |
| 2503.11701v1.md | A Survey of Direct Preference Optimization | Q10 | corpus | DPO methods survey |
| 2510.10390v1.md | RefusalBench: Generative Evaluation of Selective Refusal in Grounded Language Models | Q10 | corpus | refusal-behavior benchmark methodology |
| 2603.07211v2.md | Compass DPO: Dynamics-Controlled DPO for Robust Safety Alignment | Q10 | corpus | DPO variant for behavior/safety alignment |
| 2602.16835v1.md | NeST: Neuron Selective Tuning for LLM Safety | Q10 | corpus | targeted tuning for safety/refusal behavior |
| 2412.16339v2.md | Deliberative Alignment: Reasoning Enables Safer Language Models | Q10 | corpus | reasoning-based alignment for safe refusal |
| 2602.11079v3.md | Probe-Based Data Attribution: Discovering and Mitigating Undesirable Behaviors in LLM Post-Training | Q10 | corpus | data attribution for undesired post-training behaviors |
| 2505.01706v1.md | Inducing Robustness in a 2-dimensional Direct Preference Optimisation Paradigm | Q10 | corpus | DPO robustness variant |
| 2502.17507v2.md | C2-DPO: Constrained Controlled Direct Preference Optimization | Q10 | corpus | constrained DPO for controlled behavior |
| 2410.21597v3.md | Reducing the Scope of Language Models | Q10 | corpus | scoping/refusal behavior shaping |
| 2501.13677v3.md | HumorReject: Decoupling LLM Safety from Refusal Prefix via A Little Humor | Q10 | corpus | refusal-behavior mechanism analysis |
| 2410.06293v1.md | Accelerated Preference Optimization for Large Language Model Alignment | Q10 | corpus | faster DPO-family optimization |
| 2603.19220v2.md | Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation | Q11 | corpus | RLVR + on-policy distillation post-training recipe |
| 2602.00513v3.md | Minerva: RL with Verifiable Rewards for Cyber Threat Intelligence LLMs | Q11 | corpus | applied RLVR recipe |
| 2511.00066v4.md | Sharpness-Guided Group Relative Policy Optimization via Probability Shaping | Q11 | corpus | GRPO variant for RLVR training stability |
| 2511.17473v1.md | Masked-and-Reordered Self-Supervision for RL from Verifiable Rewards | Q11 | corpus | self-supervised augmentation for RLVR |
| 2504.15077v5.md | Think2SQL: Reinforce LLM Reasoning Capabilities for Text2SQL | Q11 | corpus | RLVR for schema-grounded SQL generation |
| 2601.03525v3.md | Beyond Binary: Turning Partial Success into Dense Verifiable Rewards for Code Generation | Q11 | corpus | dense reward shaping for verifiable code rewards |
| 2601.22230v1.md | DaJ: Data-Reweighted LLM Judge for Test-Time Scaling in Code Generation | Q11 | corpus | judge-based reward reweighting for code RL |
| 2510.23038v2.md | Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated RL | Q11 | corpus | tool-integrated RLVR for judge/agent reasoning |
| 2605.16790v1.md | TIER: Trajectory-Invariant Execution Rewards for Multi-Step Tool Composition | Q11 | corpus | verifiable execution rewards for tool-call chains |
| 2601.04237v2.md | SAGE-32B: Agentic Reasoning via Iterative Distillation | Q11 | corpus | iterative distillation for agentic RL capability |
| 2605.27954v1.md | Cyclical Entropy Eruption: Entropy Dynamics in Agent Reinforcement Learning | Q11 | corpus | training-dynamics analysis of agent RLVR |
| 2512.16144v1.md | INTELLECT-3: Technical Report | Q11 | corpus | end-to-end RLVR post-training report |
| 2602.02979v3.md | CPMobius: Iterative Coach-Player Reasoning for Data-Free Reinforcement Learning | Q11 | corpus | data-free RLVR training scheme |
| 2601.20103v1.md | Benchmarking Reward Hack Detection in Code Environments via Contrastive Analysis | Q12 | corpus | reward-hack detection benchmark for code RL |
| 2509.15557v1.md | Reward Hacking Mitigation using Verifiable Composite Rewards | Q12 | corpus | composite reward design against hacking |
| 2512.19027v2.md | Recontextualization Mitigates Specification Gaming without Modifying the Specification | Q12 | corpus | specification-gaming mitigation technique |
| 2605.02964v1.md | Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use | Q12 | corpus | tool-use reward-hacking benchmark, core |
| 2603.07084v2.md | Reward hacking emergence from confirmed-cheating synthetic SFT data | Q12 | corpus | link between synthetic SFT data quality and reward hacking |
| 2604.13602v1.md | Reward Hacking in the Era of Large Models: Mechanisms, Emergent Misalignment, Challenges | Q12 | corpus | survey of reward-hacking mechanisms |
| 2604.16242v1.md | Detecting and Suppressing Reward Hacking with Gradient Fingerprints | Q12 | corpus | gradient-based reward-hack detection |
| 2604.15149v1.md | LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking | Q12 | corpus | core empirical demonstration of verifier gaming |
| 2605.20744v1.md | Hack-Verifiable Environments: Towards Evaluating Reward Hacking at Scale | Q12 | corpus | scalable reward-hacking evaluation environments |
| 2602.03452v2.md | Beyond Variance: Prompt-Efficient RLVR via Rare-Event Amplification and Bidirectional Pairing | Q12 | corpus | RLVR sample-efficiency method |
| 2605.03677v1.md | Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe | Q13 | corpus | core on-policy distillation recipe |
| 2608.03092v1.md | SMOPD: Specialize-and-Merge Online Policy Distillation | Q13 | corpus | specialist-merge on-policy distillation |
| 2608.19098v1.md | Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation | Q13 | corpus | multi-teacher distillation imbalance diagnosis |
| 2607.27770v1.md | Beyond the Best Teacher: Expanding and Compressing the Reasoning Solution Manifold | Q13 | corpus | teacher-selection strategy for distillation |
| 2604.00626v3.md | A Survey of On-Policy Distillation for Large Language Models | Q13 | corpus | core survey of on-policy distillation |
| 2607.19450v1.md | REGEN: Replay-recycling for Expert-to-Generalist Distillation with Offline RL | Q13 | corpus | expert-to-generalist fusion via distillation |
| 2601.15657v1.md | Integrating Knowledge Distillation Methods: A Sequential Multi-Stage Framework | Q13 | corpus | multi-source/multi-teacher KD framework |
| 2504.14772v2.md | Multi-teacher knowledge distillation fusion for LLMs | Q13 | corpus | fusion mechanisms for multi-teacher KD in LLMs |
| 2604.19884v1.md | From Signal Degradation to Computation Collapse: Two Failure Modes of LLM Quantization | Q14 | corpus | taxonomy of quantization failure modes |
| 2605.15208v1.md | Quantization Undoes Alignment: Bias Emergence in Compressed LLMs Across Models and Precision | Q14 | corpus | per-axis (bias/alignment) quantization degradation |
| 2605.04062v2.md | EdgeRazor: A Lightweight Framework for LLMs via Mixed-Precision Quantization-Aware Distillation | Q14 | corpus | mixed-precision QAD framework |
| 2512.18934v1.md | When Less is More: 8-bit Quantization Improves Continual Learning in LLMs | Q14 | corpus | quantization interacting with forgetting |
| 2411.03350v2.md | A Comprehensive Survey of Small Language Models in the Era of Large Language Models | Q14 | corpus | SLM survey incl. quantization/efficiency background |
| 2403.12544v1.md | AffineQuant: Affine Transformation Quantization for Large Language Models | Q14 | corpus | PTQ method |
| 2507.11959v1.md | PoT-PTQ: A Two-step Power-of-Two Post-training for LLMs | Q14 | corpus | PTQ method |
| 2607.14181v1.md | Quantize with Confidence? An Empirical Study of Quantization for Code Generation | Q14 | corpus | per-task (code) PTQ degradation study |
| 2508.03332v2.md | Exploring Layer-wise Information Effectiveness for Post-Training Quantization in Small LMs | Q14 | corpus | layer-wise PTQ effects in small models |
| 2607.25451v1.md | Bits and Memories: Measuring Verbatim Extraction Across LLM Quantization | Q14 | corpus | quantization effect on memorization/verbatim recall |
| 2512.21651v3.md | Rethinking Output Alignment For 1-bit Post-Training Quantization of LLMs | Q14 | corpus | extreme-low-bit PTQ output alignment |
| 2608.18578v1.md | Compress and Forget: bitsandbytes Quantization Amplifies Proactive Interference in LLMs | Q14 | corpus | quantization amplifying forgetting/interference |
| 2602.22592v1.md | pQuant: Towards Effective Low-Bit Language Models via Decoupled Linear Quantization-Aware Training | Q15 | corpus | core QAT method |
| 2407.11062v3.md | EfficientQAT: Efficient Quantization-Aware Training for Large Language Models | Q15 | corpus | core QAT method |
| 2510.13998v1.md | BitNet Distillation | Q15 | corpus | distillation for extreme low-bit quality recovery |
| 2607.08643v1.md | BiSCo-LLM: Lookup-Free Binary Spherical Coding for Extreme Low-Bit LLM Compression | Q15 | corpus | extreme low-bit compression method |
| 2510.03267v2.md | PT2-LLM: Post-Training Ternarization for Large Language Models | Q15 | corpus | ternarization PTQ method |
| 2605.17471v1.md | WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points | Q15 | corpus | QAT training-dynamics acceleration |
| 2601.20088v3.md | Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery | Q15 | corpus | core QAD recipe for accuracy recovery |
| 2505.01043v1.md | Low-Precision Training of Large Language Models: Methods, Challenges, and Opportunities | Q15 | corpus | survey of low-precision training methods |
| 2303.08302v3.md | ZeroQuant-V2: Exploring Post-training Quantization from Comprehensive Study to Low Rank Compensation | Q15 | corpus | foundational PTQ + low-rank compensation |
| 2412.04787v3.md | Direct Quantized Training of Language Models with Stochastic Rounding | Q15 | corpus | quantized training method |
| 2307.00331v2.md | Quantization Variation: A New Perspective on Training Transformers with Low-Bit Precision | Q15 | corpus | low-bit training perspective |
| 2307.05972v1.md | Self-Distilled Quantization: Achieving High Compression Rates in Transformer-Based LMs | Q15 | corpus | self-distillation for quantization recovery |
| 2504.13932v3.md | Enhancing Ultra-Low-Bit Quantization of LLMs Through Saliency-Aware Partial Retraining | Q15 | corpus | saliency-aware partial retraining for ultra-low-bit |
| 2505.11574v4.md | Quantization Meets Reasoning: Exploring and Mitigating Degradation of Low-Bit LLMs in Math Reasoning | Q16 | corpus | per-task (reasoning) quantization degradation |
| 2512.23367v2.md | Post-Training Quantization of OpenPangu Models for Efficient Deployment on Atlas A2 | Q16 | corpus | applied PTQ deployment case with per-task results |
| 2601.07878v1.md | Sliced-Wasserstein Distribution Alignment Loss Improves Ultra-Low-Bit Quantization of LLMs | Q16 | corpus | distribution-alignment loss for ultra-low-bit quality |
| 2602.15563v1.md | 1-Bit Wonder: Improving QAT Performance in the Low-Bit Regime through K-Means Quantization | Q16 | corpus | K-means-based low-bit QAT |
| 2511.19495v1.md | A Systematic Study of Compression Ordering for Large Language Models | Q16 | corpus | ordering of compression stages vs. task-level quality |
| 2604.07888v1.md | Bit-by-Bit: Progressive QAT Strategy with Outlier Channel Splitting for Stable Low-Bit LLMs | Q16 | corpus | progressive QAT stabilization method |
| 2601.14888v1.md | What Makes Low-Bit Quantization-Aware Training Work for Reasoning LLMs? A Systematic Study | Q16 | corpus | core per-axis QAT study for reasoning vs. other tasks |
| 2511.22483v1.md | Enhancing Trustworthiness with Mixed Precision: Benchmarks, Opportunities, and Challenges | Q16 | corpus | per-axis trustworthiness effects of mixed precision |
| 2506.09104v1.md | Unifying Block-wise PTQ and Distillation-based QAT for Progressive Quantization toward 2-bit | Q16 | corpus | combined PTQ+QAT pipeline toward 2-bit instruction-tuned LLMs |
| 2505.18231v2.md | NSNQuant: A Double Normalization Approach for Calibration-Free Low-Bit KV Cache Quantization | Q16 | corpus | KV-cache low-bit quantization, adjacent quality effects |
| 2505.06461v1.md | Challenging GPU Dominance: When CPUs Outperform for On-Device LLM Inference | Q17 | corpus | on-device inference hardware trade-off study |
| 2503.09114v2.md | Sometimes Painful but Promising: Feasibility and Trade-offs of On-Device Language Model Inference | Q17 | corpus | on-device feasibility/trade-off study |
| 2502.14458v2.md | Llamba: Scaling Distilled Recurrent Models for Efficient Language Processing | Q17 | corpus | distilled recurrent model for efficient on-device-style inference |
| 2504.03664v2.md | PIPO: Pipelined Offloading for Efficient Inference on Consumer Devices | Q17 | corpus | consumer-device inference efficiency technique |
| 2410.13461v2.md | Progressive Mixed-Precision Decoding for Efficient LLM Inference | Q17 | corpus | mixed-precision decoding for efficiency |
| 2605.05819v1.md | HCInfer: An Efficient Inference System via Error Compensation for Resource-Constrained Devices | Q17 | corpus | error-compensated low-precision inference system |
| 2605.20706v1.md | Llamas on the Web: Memory-Efficient, Multi-Precision LLM Inference with WebGPU | Q17 | corpus | multi-precision on-device/browser inference |
| 2510.06126v1.md | lm-Meter: Unveiling Runtime Inference Latency for On-Device Language Models | Q17 | corpus | on-device latency measurement methodology |
| 2603.26603v2.md | Joint evaluation of energy, performance, resource use, and output quality for on-device LLM inference | Q17 | corpus | joint efficiency+quality evaluation for on-device LLMs |
| 2508.07329v1.md | Precision-heterogeneous CPU/GPU parameter management for efficient LLM deployment | Q17 | corpus | heterogeneous-precision deployment strategy |
| 2605.21543v1.md | Provable Joint Decontamination for Benchmarking Multiple Large Language Models | Q18 | corpus | core decontamination method for benchmark validity |
| 2311.06233v7.md | Data Contamination Quiz: A Tool to Detect and Estimate Contamination in LLMs | Q18 | corpus | contamination detection tool |
| 2505.13249v1.md | RN-F: A Novel Approach for Mitigating Contaminated Data in Large Language Models | Q18 | corpus | contamination mitigation method |
| 2605.19999v1.md | LLM Benchmark Datasets Should Be Contamination-Resistant | Q18 | corpus | benchmark design principles for contamination resistance |
| 2601.06103v1.md | The Impact of Post-training on Data Contamination | Q18 | corpus | how post-training (SFT/RL) interacts with contamination |
| 2402.15938v3.md | Generalization or Memorization: Data Contamination and Trustworthy Evaluation for LLMs | Q18 | corpus | core analysis of contamination vs. evaluation trust |
| 2401.06059v1.md | Investigating Data Contamination for Pre-training Language Models | Q18 | corpus | pretraining-stage contamination investigation |
| 2509.25531v5.md | MixtureVitae: Open Web-Scale Pretraining Dataset (with contamination scanning) | Q18 | corpus | dataset-level decontamination scan methodology |
| 2608.04714v1.md | What We Observe as LLM Behavior Can Be a Side-effect of Inference Backend | Q19 | corpus | core reproducibility issue: inference backend affects scores |
| 2407.04069v2.md | A Systematic Survey and Critical Review on Evaluating Large Language Models | Q19 | corpus | eval methodology reproducibility survey |
| 2511.20836v3.md | Structured Prompts Improve Evaluation of Language Models | Q19 | corpus | prompt-template effect on eval scores |
| 2408.12263v1.md | Toward the Evaluation of LLMs Considering Score Variance across Instruction Templates | Q19 | corpus | core study of template-induced score variance |
| 2410.14872v2.md | How to Evaluate Reward Models for RLHF | Q19 | corpus | reward-model eval reproducibility, adjacent methodology |
| 2511.03825v1.md | How Different Tokenization Algorithms Impact LLMs and Transformer Models for Binary Code Analysis | Q20 | corpus | tokenizer choice effects on code-adjacent tasks |
| 2510.13481v2.md | Vocabulary size and fertility effects on downstream accuracy (Arabic tokenizer study) | Q20 | corpus | vocab size vs. fertility vs. downstream quality |
| 2310.08754v4.md | Tokenizer Choice For LLM Training: Negligible or Crucial? | Q20 | corpus | core study of tokenizer choice impact |
| 2507.22543v1.md | Pre-trained Models Perform the Best When Token Distributions Follow Zipf's Law | Q20 | corpus | tokenization distribution effects on quality |
| 2511.20849v1.md | Length-MAX Tokenizer for Language Models | Q20 | corpus | tokenizer design for efficiency |
| 2601.13260v2.md | Stop Taking Tokenizers for Granted: They Are Core Design Decisions in LLMs | Q20 | corpus | core argument for tokenizer as design decision |
| 2605.12928v1.md | The Efficiency Gap in Byte Modeling | Q20 | corpus | tokenization/byte-level efficiency analysis |
| 2507.02529v2.md | RetrySQL: Text-to-SQL Training with Retry Data for Self-Correcting Query Generation | Q21 | corpus | self-correction training for text-to-SQL |
| 2412.19718v1.md | Schema hallucination reduction via dense retrieval for Text-to-SQL | Q21 | corpus | core hallucinated-schema-element suppression method |
| 2509.05899v1.md | X-SQL: Expert Schema Linking and Understanding of Text-to-SQL with Multi-LLMs | Q21 | corpus | schema linking method for text-to-SQL |
| 2410.07295v2.md | ITERGEN: Iterative Semantic-Aware Structured Output Generation | Q21 | corpus | grammar/parser-guided SQL generation |
| 2602.22223v1.md | SQaLe: A Large Text-to-SQL Corpus Grounded in Real Schemas | Q21 | corpus | schema-grounded text-to-SQL dataset |
| 2503.12730v5.md | TinySQL: A Progressive Text-to-SQL Dataset for Mechanistic Interpretability Research | Q21 | corpus | small-model text-to-SQL dataset/interpretability |
| 2402.08100v1.md | Investigating the Impact of Data Contamination in Text-to-SQL Translation | Q21 | corpus | contamination in text-to-SQL benchmarks |
| 2411.08599v3.md | A Preview of XIYAN-SQL: A Multi-Generator Ensemble Framework for Text-to-SQL | Q21 | corpus | ensemble text-to-SQL framework |
| 2603.10697v1.md | EVOSCHEMA: Towards Text-to-SQL Robustness Against Schema Evolution | Q21 | corpus | robustness to schema changes/hallucinated names |
| 2512.16083v1.md | Schema filtering and ranking for Text-to-SQL under context-length limits | Q21 | corpus | schema-selection method to reduce hallucination |
| 2310.18376v4.md | SQLformer: Deep Auto-Regressive Query Graph Generation for Text-to-SQL Translation | Q21 | corpus | structured generation architecture for SQL |
| 2508.15884v3.md | Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search | Q22 | corpus | hybrid-architecture NAS for efficient LMs |
| 2407.05591v1.md | Convolution Augments Attention: Solving Associative Recall with One Layer | Q22 | corpus | foundational conv+attention hybrid mechanism |
| 2603.03517v1.md | MMAI Gym for Science: Training Liquid Foundation Models for Drug Discovery | Q22 | corpus | LFM-family architecture training (same lineage as LFM2.5) |
| 2511.23404v1.md | LFM2 Technical Report | Q22 | corpus | direct technical report of the base model family |
| 2510.19338v2.md | Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning | Q22 | corpus | hybrid linear-attention architecture design |
| 2411.13676v1.md | Hymba: A Hybrid-head Architecture for Small Language Models | Q22 | corpus | core hybrid conv/attention small-LM architecture |
| 2601.01313v1.md | Spectral-Window Hybrid (SWH) sequence modelling | Q22 | corpus | hybrid sequence-modeling architecture |
| 2608.20210v1.md | Daedalus-150M: A Convolution-Attention Hybrid Designed for CPU Inference | Q22 | corpus | conv-attention hybrid tuned for CPU/on-device inference |
| 2605.17653v1.md | LLMForge: Multi-Backend Hardware-Aware NAS with Infinite-Head Attention for Edge LMs | Q22 | corpus | hardware-aware architecture search for edge LMs |
| 2507.12442v4.md | Characterizing State Space Model and Hybrid Language Model Performance with Long Context | Q22 | corpus | SSM/hybrid architecture performance characterization |
| 2603.15954v2.md | MobileLLM-Flash: Latency-Guided On-Device LLM Design for Industry Scale Deployment | Q22 | corpus | on-device architecture/latency co-design |

## Work log

- 2026-08-25 · s1.2 · Ran all 22 queries against the indexed corpus; kept 212 papers
  with full text → `papers/`; index and per-query counts → this file. No query fell
  under the three-paper threshold, so the external tier was not needed; no hit was
  dropped for missing full text; the stop rule did not fire.
- 2026-08-25 · s1.1 · Wrote 22 queries across six arXiv categories, with dedicated
  coverage for every low-confidence assumption and open question in the provisional
  scope → this file.

## Outputs

- `stages/s1-literature-review/report.md` (this file) — queries, per-query
  coverage, and the 212-row index
- `stages/s1-literature-review/papers/` — 212 full-text papers, 17 MB

## Next steps

Stage complete. s2.1 puts one reader on each of the 212 papers to produce a structured
note, then s2.2 synthesizes them and s2.3 rewrites the scope against the result.
