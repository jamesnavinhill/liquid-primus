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
- [x] s5.2 Baselines  — harness verified; reference row B1 is complete at full counts on all four components (tool calling 0.6700 native / 0.5355 ours, structured output 0.1355, instruction following 0.8170, and a guardrail flag rate of 0.0074 with zero false alarms on the clean control arm), reads behind the 1B incumbent on tool calling, and sits at the floor on the safety axis; the 4-bit serving path is BUILT AND VERIFIED on attempt 5 (llama.cpp b10622 for sm_89, in shared storage, 196 generated tok/s over 8 slots, determinism held, a full Q4 pass priced at 1.71 h, 51% under its estimate), B2's first launch provisioned a gcp machine and ran nothing (COMPLETE, progress 0, 15 empty provider polls, no artifacts) and is relaunched as c4091ac9 pinned to aws; reading that job record also caught that the recorded queue commands for all three 4-bit rows would have run 40 items per component while labelling themselves full passes, since profile is only a label in s5-eval/main.py:327-338, and the commands are corrected; B3 (unsloth UD-Q4_K_XL) is now complete too at 0.5837 native tool calling, 0.4339 on the s4 text convention, 0.1195 structured output, 0.8022 instruction following and the same 0.0074 flag rate with zero false alarms, so the vendor's quantization-aware build beats the best cheap community quant by 5.6 points on tool calling and the two agree to within 2.4 points on the text convention, which is the ambiguity the runtime control resolves; the clean-control harness edit was therefore applied at 17:19Z once B3 was down (arm verified in storage first: 138 items, all tagged clean_corpus) rather than waiting for B4, which takes B1r, B5 and B6 off the supplementary re-score list and leaves only B1 through B4 on it; B1r (c3886996), the runtime-vs-quantization control, came down at 0.6252 native tool calling, 0.4669 on the text convention, 0.1345 structured output, 0.8115 instruction following and a 0.0074 flag rate with zero false alarms on both the frozen and the new 138-item clean arm, and it splits by axis: on the text convention it beats both 4-bit rows, so most of their shortfall there is the llama.cpp runtime rather than quantization, but on native tool calling it trails B2 (0.6252 vs 0.6392), so quantization-aware training more than recovers the runtime's cost on that axis; B4 (c319ebb1, Granite 4.0-1b, full precision) and B5 (64fa9373, Nova Function-Calling Q4_K_M) then landed together at 19:06:47Z and 19:33:21Z, both success, both verified against full artifact lists, at 4.141 and 0.642 GPU-h, giving Granite 0.7641 native tool calling / 0.1817 text convention / 0.3205 structured output / 0.7394 instruction following / 0.0000 flag rate and Nova 0.6821 / 0.2681 / 0.0710 / 0.5379 / 0.0185 with zero false alarms on both arms; the finding that closes the substage is that BOTH competitors win native tool calling and lose every other axis, and the two calling surfaces order in opposite directions (the two rows that beat us natively are the two worst on the text convention, Granite by 35.4 points), so the pre-registered tool-calling bar is a question about which surface we intend to serve rather than one number to beat and s6 should carry the two surfaces separately rather than pooled; Granite's 0.0000 flag rate is uninformative rather than clean, since a model that never flags cannot false-flag, and no baseline comes near the pre-registered 0.35 target; B6 (LFM2.5-1.2B-Base at screening) was DROPPED as redundant, since the sweep's own C1 reference arm measures the base checkpoint directly and at full budget, and both freed slots went to the s5.3 prerequisites instead; B1, B2, B3 and B4 remain on the supplementary clean-arm re-score list as zero-dependency filler work for any otherwise-idle slot; ledger cleaned in the same pass (stale total_spent 3.871 corrected to a real 10.860 of 145 across 30 entries, none left open, and an orphaned RUNNING duplicate of c4091ac9 removed without changing any recorded spend)

- [~] s5.3 Broad sweep  — ALL EIGHT ARMS ARE TRAINED AND ALL EIGHT ARE NOW SCORED. Pass 3 (ade4c590, C4+C6) landed at 23:20:04Z, success, 2/2 arms rc=0, 2.58 GPU-h, 30 artifacts + pack_summary verified, full item counts asserted on all four components; its promoted score.json was cross-checked field by field against the job record's own results block with exact agreement, so nothing is withdrawn from this pass. C4 bfcl 0.7128 / ifstruct 0.1445 / ifeval 0.7468 / flag 0.6519 / false-flag 0.0; C6 bfcl 0.7237 / ifstruct 0.1210 / ifeval 0.7024 / flag 0.6222 / false-flag 0.0. C6's rank-64 adapter cost 1.7% more wall clock than C4's rank-16, so rank does not survive the merge as a serving cost. CORRECTION to the pass-4 line previously carried here: it reported C5a as 0.7017 / 0.1360 / 0.7420 and C5b as 0.7025 / 0.1370 / 0.7355 with 'both at zero false alarms'. Those figures match no file on disk. The promoted score.json says C5a bfcl 0.7013 / ifstruct 0.1515 / ifeval 0.7837 / flag 0.5741 / FALSE-FLAG 0.10 and C5b bfcl 0.7227 / ifstruct 0.1095 / ifeval 0.7708 / flag 0.6333 / false-flag 0.0, which is what runs/s5.3-sweep.md recorded at 23:12Z; C5a is the one arm in the sweep that raises a false alarm at all. FINAL GATE-1 TALLY (detection >= 0.70, false-flag <= 0.15 on the 30-item synthetic clean arm): C7 0.7222 PASS; C4 0.6519, C5b 0.6333, C6 0.6222, C1 0.6111, C3 0.6037, C5a 0.5741, C2p 0.0074 all fail. One arm clears it, so the pre-registered rescope trigger (EVERY arm failing gate 1) DOES NOT FIRE and s5.5 has a direction. Two readings worth carrying into s5.4: the two arms that score HIGHEST on tool calling (C6 0.7237, C5b 0.7227) are both gate-1 failures, so ranking on the ranking metric alone would have selected against the project's first priority; and validation loss points the wrong way at both ends, its best arm (C2p 0.1511) posting the sweep's lowest detection rate and its worst (C7 0.1565) the highest, with BFCL spanning only 0.0224 across all eight arms against 0.7148 on the flag rate. Remaining before s5.4: the probes pass over all eight arms with the 138-item corpus clean arm is QUEUED on both cards (9ba23f0c C1,C2p,C3,C4 and 2c91ab0d C5a,C5b,C6,C7, four arms a card at 5.5 GB, byte-identical settings) and is the arm the s5.4 false-flag reading is written against; then the comparison job over all eight arms in one call, since Holm is applied across the whole family and splitting it would report p-values too small in both halves. Spend 52.070 of 145.
- [ ] s5.4 Direction decision with researcher  (checkpoint)  — THE DECISION RULE IS PRE-REGISTERED, written 2026-08-26 08:20Z while 385e210a was at 62% and no arm had a task score; it is in the s5 report under "The s5.4 decision rule, written before the numbers" and any departure will be recorded as one. Ranking metric: BFCLv3 overall, category-macro, better of the two calling conventions per arm, exactly as B1-B6 were scored, because that is the metric the pre-registered claim is written against. EVIDENCE GATE IS THE PAIRED TEST, NOT THE RANKING METRIC: an arm beats C1 only if its item-weighted paired delta survives Holm correction across the whole s5-compare family, and where the macro composite and the paired delta DISAGREE IN SIGN both are reported and no winner is claimed on that axis. Three gates first: (1) detection >= 0.70 with false-flag <= 0.15 on the clean items -- AMENDED 08:34Z on a configuration fact, before any arm had a score: clean_control_object is empty in pack.yaml and no arm queue script sets it, so NO ARM IS SCORED AGAINST THE 138-ITEM CORPUS CLEAN ARM. Not fixed by setting it for later passes, because C1 would then have no corpus cell for any arm to pair against; all eight arms stay identical and gate 1 is evaluated on the 30-item synthetic arm, which bounds the false-flag RATE only to about one item in thirty but does support the paired comparison s5.4 needs (an arm flagging a fifth of clean returns puts six discordant pairs against C1's zero). A probes-only pass with clean_control_object set is queued for every arm after the four scoring passes and before s5.5, 138 extra items an arm and a few card-minutes for all eight, and THE RESCOPE TRIGGER IS READ AGAINST THAT, not against the thirty; (2) no BFCLv3 category more than 3.0 below the matched base and IFEval no more than 2.0 below; (3) IFStruct at or above C1's. IF NOTHING SEPARATES (the likely outcome, since three of four arms tied within 0.0006 on val_loss) s5.4 does NOT pick the numerically highest arm: it picks the cheapest recipe passing all three gates, ranked r16 > r64 > full-parameter, no replay > 1% > 5%, reference mixture > raw, and records that the choice was made on cost because the arms were indistinguishable. A recorded non-separation is a finding about the sweep and goes in the paper as one. RESCOPE TRIGGER: if EVERY arm fails gate 1, the guardrail axis is not reachable from this training set, s5.5 has no direction worth detailing, and that stops for the operator rather than being decided under autonomous mode.
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
