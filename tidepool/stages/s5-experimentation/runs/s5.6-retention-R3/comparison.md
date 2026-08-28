# s5.6 arms against R3-F16, paired by item

Paired per-item comparison against the reference arm. The interval is a percentile bootstrap over the three paired-difference counts, which is exact for a statistic that is a function of those counts alone. The test is McNemar's exact two-sided test on the discordant pairs. The family correction is Holm-Bonferroni across every arm-by-component cell in this run. Deltas are item-weighted; the macro column is descriptive and carries no test.

Bootstrap resamples 10000, seed 20260826, 20 comparison(s) in the corrected family.

| arm | component | n | ref | arm | delta | 95% CI | macro | McNemar p | Holm p | reads as |
|---|---|--:|--:|--:|--:|---|--:|--:|--:|---|
| R3-Q4_0 | bfcl_native | 3490 | 0.6762 | 0.6430 | -0.0332 | [-0.0453, -0.0212] | -0.0270 | 8.063e-08 | 1.532e-06 | separates |
| R3-Q4_0 | bfcl_text | 3490 | 0.6756 | 0.6352 | -0.0404 | [-0.0510, -0.0298] | -0.0443 | 8.368e-14 | 1.674e-12 | separates |
| R3-Q4_0 | ifstruct | 2000 | 0.0860 | 0.0735 | -0.0125 | [-0.0235, -0.0015] |  | 0.0287 | 0.488 | separates before correction only |
| R3-Q4_0 | ifeval | 541 | 0.7468 | 0.7320 | -0.0148 | [-0.0370, +0.0074] |  | 0.2682 | 1 | indistinguishable |
| R3-Q4_0 | probes_detect | 270 | 0.9148 | 0.9296 | +0.0148 | [-0.0037, +0.0370] |  | 0.2891 | 1 | indistinguishable |
| R3-Q4_0 | probes_clean | 30 | 0.9000 | 0.8333 | -0.0667 | [-0.1667, +0.0000] |  | 0.5 | 1 | indistinguishable |
| R3-Q4_0 | probes_clean_corpus | 138 | 0.6957 | 0.6449 | -0.0507 | [-0.1014, -0.0072] |  | 0.06543 | 0.9814 | separates before correction only |
| R3-Q4_0 | probes_stack_idiom | 144 | 0.2361 | 0.2292 | -0.0069 | [-0.0625, +0.0556] |  | 1 | 1 | indistinguishable |
| R3-Q4_0 | probes_flag_detect | 270 | 0.7407 | 0.7037 | -0.0370 | [-0.0889, +0.0148] |  | 0.2026 | 1 | indistinguishable |
| R3-Q4_0 | probes_flag_clean_corpus | 138 | 0.9855 | 0.9783 | -0.0072 | [-0.0362, +0.0145] |  | 1 | 1 | indistinguishable |
| R3-Q4_K_M | bfcl_native | 3490 | 0.6762 | 0.6668 | -0.0095 | [-0.0203, +0.0011] | +0.0056 | 0.1019 | 1 | indistinguishable |
| R3-Q4_K_M | bfcl_text | 3490 | 0.6756 | 0.6696 | -0.0060 | [-0.0152, +0.0029] | -0.0285 | 0.2085 | 1 | indistinguishable |
| R3-Q4_K_M | ifstruct | 2000 | 0.0860 | 0.0735 | -0.0125 | [-0.0220, -0.0025] |  | 0.01439 | 0.259 | separates before correction only |
| R3-Q4_K_M | ifeval | 541 | 0.7468 | 0.7486 | +0.0018 | [-0.0222, +0.0259] |  | 1 | 1 | indistinguishable |
| R3-Q4_K_M | probes_detect | 270 | 0.9148 | 0.9111 | -0.0037 | [-0.0185, +0.0111] |  | 1 | 1 | indistinguishable |
| R3-Q4_K_M | probes_clean | 30 | 0.9000 | 0.9000 | +0.0000 | [+0.0000, +0.0000] |  | 1 | 1 | indistinguishable |
| R3-Q4_K_M | probes_clean_corpus | 138 | 0.6957 | 0.6957 | +0.0000 | [-0.0435, +0.0435] |  | 1 | 1 | indistinguishable |
| R3-Q4_K_M | probes_stack_idiom | 144 | 0.2361 | 0.2014 | -0.0347 | [-0.0903, +0.0278] |  | 0.3593 | 1 | indistinguishable |
| R3-Q4_K_M | probes_flag_detect | 270 | 0.7407 | 0.7815 | +0.0407 | [+0.0037, +0.0778] |  | 0.05224 | 0.8358 | separates before correction only |
| R3-Q4_K_M | probes_flag_clean_corpus | 138 | 0.9855 | 0.9638 | -0.0217 | [-0.0507, +0.0000] |  | 0.25 | 1 | indistinguishable |

## Retention against R3-F16

Each arm's headline rate over the reference's on the same axis. The bar is >= 93% on every axis and >= 97% on the mean.

| arm | axis | ref rate | arm rate | absolute | retention | >= floor | ratio well conditioned |
|---|---|--:|--:|--:|--:|:-:|:-:|
| R3-Q4_0 | bfcl_composite | 0.7238 | 0.6795 | -0.0443 | 93.9% | yes | yes |
| R3-Q4_0 | ifeval_prompt_strict | 0.7468 | 0.7320 | -0.0148 | 98.0% | yes | yes |
| R3-Q4_0 | ifstruct_validity | 0.0860 | 0.0735 | -0.0125 | 85.5% | NO | no (ref rate < 0.30) |
| R3-Q4_0 | probe_stack_idiom | 0.2361 | 0.2292 | -0.0069 | 97.1% | yes | no (ref rate < 0.30) |
| R3-Q4_K_M | bfcl_composite | 0.7238 | 0.6953 | -0.0285 | 96.1% | yes | yes |
| R3-Q4_K_M | ifeval_prompt_strict | 0.7468 | 0.7486 | +0.0018 | 100.2% | yes | yes |
| R3-Q4_K_M | ifstruct_validity | 0.0860 | 0.0735 | -0.0125 | 85.5% | NO | no (ref rate < 0.30) |
| R3-Q4_K_M | probe_stack_idiom | 0.2361 | 0.2014 | -0.0347 | 85.3% | NO | no (ref rate < 0.30) |

| arm | axes | mean | min | on axis | verdict |
|---|--:|--:|--:|---|---|
| R3-Q4_0 | 4 | 93.6% | 85.5% | ifstruct_validity | FAILS on ifstruct_validity |
| R3-Q4_K_M | 4 | 91.8% | 85.3% | probe_stack_idiom | FAILS on ifstruct_validity, probe_stack_idiom |

A ratio is only as well conditioned as its denominator: where the reference rate is under 0.30, a few absolute points move the ratio by more than ten of them, so read those rows with the absolute column and the paired interval above and not on their own. The bar is applied as pre-registered either way.

`delta` is the arm's rate minus the reference's over the items both scored. `macro` averages the per-group deltas with equal weight per group and has no interval, so it is a description of the same items and not a second test.

## Notes

- R3-F16/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- R3-F16/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- R3-Q4_0/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- R3-Q4_0/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- R3-Q4_K_M/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- R3-Q4_K_M/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
