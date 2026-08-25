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
- [~] s5.2 Baselines  — harness verified; reference row B1 is complete at full counts on all four components (tool calling 0.6700 native / 0.5355 ours, structured output 0.1355, instruction following 0.8170, and a guardrail flag rate of 0.0074 with zero false alarms on the clean control arm), reads behind the 1B incumbent on tool calling, and sits at the floor on the safety axis; the 4-bit serving path is BUILT AND VERIFIED on attempt 5 (llama.cpp b10622 for sm_89, in shared storage, 196 generated tok/s over 8 slots, determinism held, a full Q4 pass priced at 1.71 h, 51% under its estimate), B2's first launch provisioned a gcp machine and ran nothing (COMPLETE, progress 0, 15 empty provider polls, no artifacts) and is relaunched as c4091ac9 pinned to aws; reading that job record also caught that the recorded queue commands for all three 4-bit rows would have run 40 items per component while labelling themselves full passes, since profile is only a label in s5-eval/main.py:327-338, and the commands are corrected; the competitor row B4 that carries the operative threshold is still running, and B3, B5, the bf16 runtime reference, B6 and the B1 re-score go in that order as slots free, each pinned to aws and each passing all five item-count overrides; not ticked until B2, B3 and B5 exist
- [ ] s5.3 Broad sweep  — the data component's first run (a42a6c21) failed six of its nine gates and wrote nothing, which is the gates working: its decontamination rule dropped 2,166 of 2,166 rows on a 13-gram that is the probe system preamble written at s4 to match the corpus's own rendering convention, and its no-fabrication check tested single-digit payload leaves as substrings of prose; both are fixed and measured against the held-out split (0 of 156 rows flagged on question overlap, 13 by two tighter nets), mode assignment now falls through the rotation for the 494 sources whose payloads echo no call argument, a 23-check fixture over the real probe bank covers every fix, attempt 2 (f11afe02) then wrote 5,202 rows and 2.87M tokens and cleared seven of nine gates including detector coverage at 0.704, failing only on a no-fabrication check that discarded 34.9% of sources because corpus tool returns echo the called function's name back in the payload while the target names the tool (forbidden list now scoped to the whole prompt: 19 of 73 held-out leaks down to 0) and on a clean-control floor of 200 the test split cannot reach at 68 items (arm now drawn from both held-out splits, floor 100 with the false-flag-ceiling reason stated); attempt 3 (7248507b) passed all nine gates in 4 m 53 s on CPU and wrote 7,988 rows and 4.35M tokens to tidepool/s5.3/tooldata/ with zero target leaks, zero question-gram drops, detector coverage 0.6838 inside the [0.60, 0.90] band and a 138-item clean control arm drawn 70/68 from the validation and test splits, so the guardrail supervision is built and the data half of s5.3 is done; the corpus ceiling is about 4,000 pairs rather than the 24,000 the parameter allows, since only 4,031 of 494,341 training rows carry a usable tool return; the stop/go gate into D has been amended in plan.md to test the pre-registered criterion (flag rate >= 0.35, false flags <= 0.15) instead of a regression a 0.0074 rate makes impossible; one imbalance recorded for the sweep: silently_truncated carries 1,042 rows against roughly 580 per common mode because it absorbs the 494 wrong_entity skips through the fixed rotation order, and a per-source random offset is the one-line fix if the model over-fits that phrasing; the sweep task is now authored and registered as s5-sft-sweep (37186b22): one code path for all eight arms selected by -p arm=<id>, the tool_guardrail role wired into the sampler as a second source in the same id space, the 64.0M-token per-arm budget carried as an explicit parameter that the run asserts against itself, the guardrail block dosed at 2 passes (8.70M tokens, 13.6% of budget) and taken out of the budget rather than added to it so C2' is cost-matched, entropy-weighted loss for C4 centred on the batch mean so beta cannot act as a hidden learning-rate multiplier, a full-parameter mode for C3 at LR 1e-5 with micro-batch 1 and accumulation 16 to hold the effective batch at 16, replay wired for C5 and failing loudly when no replay set is configured, and all arms scored on plain cross-entropy rather than their own objective; a CPU fixture test (test_sampler.py, 25 checks, all passing) covers equal cost per arm, guardrail presence and absence, and identical base-row order across arms; the design record is in runs/s5.3-sweep.md; no sweeps: block was used and the reasoning is recorded there; the 138-item clean control arm is now WIRED into s5-eval through a new clean_control_object setting, reported as its own false-flag rate beside the frozen 30-item arm and pooled in a third key, with a nine-check test (test_probes_additive.py) proving none of the eight pre-existing summary keys move and the new keys read empty when the arm is unconfigured, which caught a real additivity break where the envelope-depth table aggregated over every tool_return row and would have changed what depth meant for the four published baseline rows (the corpus arm now has its own depth_clean_corpus bucket); the edit is AUTHORED AND TESTED BUT NOT APPLIED to the server, since two baseline rows are running against the current harness, and it goes up once s5.2's rows are down; still to do before the arms run: apply that edit, give B1 a supplementary pass over the new arm, smoke the sweep task on arm C4, and generate the self-distillation replay set at size (about 8,000 prompts) since C5a and C5b cannot run without it
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
