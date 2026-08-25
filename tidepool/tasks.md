# Tasks: tidepool — improving the LFM2.5 on-device family for our agent stack

Status: `[ ]` todo · `[~]` in progress · `[!]` blocked · `[x]` done · `[-]` cancelled (the operator closed the project)
Active stage: s5

> Project notes (from the operator's plan in `documents/liquid-primus/`): the goal is
> improved LFM2.5 checkpoints plus a hybrid MoE family, specialized to this
> organization's stack, across four on-disk device tiers, with 4-bit GGUF forms that hold
> the full-precision quality bar. Priority when goals conflict: tool-calling and agentic
> reliability first, then on-device efficiency, then no regression in general quality.
> Reference device is a single 8 GB consumer GPU. The operator wants artifacts mirrored
> to their GitHub and Hugging Face repos as they land.

## s0 - Intake & scoping
- [x] s0.1 Capture initial prompt → initial-prompt.md
- [x] s0.2 Self-scope → provisional overview.md + assumptions

## s1 - Literature review
- [x] s1.1 Formulate research queries
- [x] s1.2 Run literature searches (academic-research CLI); copy papers → stages/s1-literature-review/papers/

## s2 - Research summary
- [x] s2.1 Extract relevant learnings per paper (link back to papers/<id>.md)
- [x] s2.2 Synthesize focused research summary
- [x] s2.3 Revise overview.md against the literature
- [x] s2.4 Scope sign-off  (checkpoint)

## s3 - Research plan
- [x] s3.1 Hypotheses / approach families
- [x] s3.2 Experiment matrix + baselines + ablations
- [x] s3.3 Compute & time budget allocation
- [x] s3.4 User approves plan + compute cap  (input checkpoint)

## s4 - Data preparation
- [x] s4.1 Acquire / validate dataset access
- [x] s4.2 EDA: distributions, missingness, leakage risks
- [x] s4.3 Build splits (group-aware / time-based)
- [x] s4.4 Preprocessing pipeline
- [x] s4.5 Data card + sign-off  (checkpoint)  — taken autonomously; decision recorded in the stage report

## s5 - Experimentation
- [x] s5.1 Smoke tests / toy runs  — both paths clean; L40S throughput measured at 3,994 tok/s and the C sweep resized to fit
- [~] s5.2 Baselines  — harness verified; reference row B1 is complete at full counts on all four components (tool calling 0.6700 native / 0.5355 ours, structured output 0.1355, instruction following 0.8170, and a guardrail flag rate of 0.0074 with zero false alarms on the clean control arm), reads behind the 1B incumbent on tool calling, and sits at the floor on the safety axis; the 4-bit serving path is BUILT AND VERIFIED on attempt 5 (llama.cpp b10622 for sm_89, in shared storage, 196 generated tok/s over 8 slots, determinism held, a full Q4 pass priced at 1.71 h, 51% under its estimate), B2 is running on it as the first of the three 4-bit rows, the competitor row B4 that carries the operative threshold is still running, and B3, B5, the bf16 runtime reference, B6 and the B1 re-score go in that order as slots free; not ticked until B2, B3 and B5 exist
- [ ] s5.3 Broad sweep  — the data component is generating (a42a6c21, CPU only): defective tool returns built by the vendored probe transforms and paired 1:1 with their own intact counterparts, two defect kinds held out of training, taught phrasings split against the frozen detector, plus an enlarged clean_corpus control arm from the held-out split; the stop/go gate into D has been amended in plan.md to test the pre-registered criterion (flag rate >= 0.35, false flags <= 0.15) instead of a regression a 0.0074 rate makes impossible; still to do before the arms freeze: wire clean_control_object into s5-eval once s5.2 closes, give B1 a supplementary pass over the new arm, and carry the 64.0M-token per-arm budget in as an explicit parameter
- [ ] s5.4 Direction decision with researcher  (checkpoint)
- [ ] s5.5 Detailed runs
- [ ] s5.6 Final tuning + ensembling

## s6 - Evaluation & analysis
- [ ] s6.1 Held-out eval vs. success criteria
- [ ] s6.2 Subgroup / fairness analysis
- [ ] s6.3 Calibration analysis
- [ ] s6.4 Error analysis
- [ ] s6.5 Robustness / ablations
- [ ] s6.6 Story checkpoint: agree the paper's story  (input checkpoint)

## s7 - Reporting & delivery
- [ ] s7.1 Academic paper (LaTeX → rendered PDF)
- [ ] s7.2 Model card
- [ ] s7.3 Reproducibility packages (core kit + chosen artifacts)  (input checkpoint)
- [ ] s7.4 Final sign-off  (checkpoint)
