# s5.6 arms against C7-F16, paired by item

Paired per-item comparison against the reference arm. The interval is a percentile bootstrap over the three paired-difference counts, which is exact for a statistic that is a function of those counts alone. The test is McNemar's exact two-sided test on the discordant pairs. The family correction is Holm-Bonferroni across every arm-by-component cell in this run. Deltas are item-weighted; the macro column is descriptive and carries no test.

Bootstrap resamples 10000, seed 20260826, 20 comparison(s) in the corrected family.

| arm | component | n | ref | arm | delta | 95% CI | macro | McNemar p | Holm p | reads as |
|---|---|--:|--:|--:|--:|---|--:|--:|--:|---|
| C7-Q4_0 | bfcl_native | 3490 | 0.6874 | 0.6679 | -0.0195 | [-0.0309, -0.0080] | -0.0497 | 0.001025 | 0.01743 | separates |
| C7-Q4_0 | bfcl_text | 3490 | 0.6481 | 0.6212 | -0.0269 | [-0.0364, -0.0172] | -0.0336 | 3.626e-08 | 6.89e-07 | separates |
| C7-Q4_0 | ifstruct | 2000 | 0.1380 | 0.1150 | -0.0230 | [-0.0375, -0.0085] |  | 0.002563 | 0.04101 | separates |
| C7-Q4_0 | ifeval | 541 | 0.7394 | 0.7301 | -0.0092 | [-0.0370, +0.0185] |  | 0.6089 | 1 | indistinguishable |
| C7-Q4_0 | probes_detect | 270 | 0.9370 | 0.9333 | -0.0037 | [-0.0185, +0.0111] |  | 1 | 1 | indistinguishable |
| C7-Q4_0 | probes_clean | 30 | 0.9000 | 0.9000 | +0.0000 | [+0.0000, +0.0000] |  | 1 | 1 | indistinguishable |
| C7-Q4_0 | probes_clean_corpus | 138 | 0.7101 | 0.6667 | -0.0435 | [-0.0942, +0.0072] |  | 0.146 | 1 | indistinguishable |
| C7-Q4_0 | probes_stack_idiom | 144 | 0.1944 | 0.1875 | -0.0069 | [-0.0625, +0.0417] |  | 1 | 1 | indistinguishable |
| C7-Q4_0 | probes_flag_detect | 270 | 0.7444 | 0.7519 | +0.0074 | [-0.0333, +0.0481] |  | 0.8555 | 1 | indistinguishable |
| C7-Q4_0 | probes_flag_clean_corpus | 138 | 0.9855 | 0.9348 | -0.0507 | [-0.0870, -0.0145] |  | 0.01563 | 0.2344 | separates before correction only |
| C7-Q4_K_M | bfcl_native | 3490 | 0.6874 | 0.6622 | -0.0252 | [-0.0355, -0.0149] | -0.0032 | 3.16e-06 | 5.688e-05 | separates |
| C7-Q4_K_M | bfcl_text | 3490 | 0.6481 | 0.6215 | -0.0266 | [-0.0352, -0.0183] | -0.0318 | 7.086e-10 | 1.417e-08 | separates |
| C7-Q4_K_M | ifstruct | 2000 | 0.1380 | 0.1355 | -0.0025 | [-0.0160, +0.0110] |  | 0.7651 | 1 | indistinguishable |
| C7-Q4_K_M | ifeval | 541 | 0.7394 | 0.7320 | -0.0074 | [-0.0351, +0.0203] |  | 0.6989 | 1 | indistinguishable |
| C7-Q4_K_M | probes_detect | 270 | 0.9370 | 0.9370 | +0.0000 | [+0.0000, +0.0000] |  | 1 | 1 | indistinguishable |
| C7-Q4_K_M | probes_clean | 30 | 0.9000 | 0.9000 | +0.0000 | [+0.0000, +0.0000] |  | 1 | 1 | indistinguishable |
| C7-Q4_K_M | probes_clean_corpus | 138 | 0.7101 | 0.6884 | -0.0217 | [-0.0725, +0.0290] |  | 0.5811 | 1 | indistinguishable |
| C7-Q4_K_M | probes_stack_idiom | 144 | 0.1944 | 0.1667 | -0.0278 | [-0.0903, +0.0347] |  | 0.5235 | 1 | indistinguishable |
| C7-Q4_K_M | probes_flag_detect | 270 | 0.7444 | 0.7370 | -0.0074 | [-0.0370, +0.0222] |  | 0.8036 | 1 | indistinguishable |
| C7-Q4_K_M | probes_flag_clean_corpus | 138 | 0.9855 | 0.9493 | -0.0362 | [-0.0725, -0.0072] |  | 0.0625 | 0.875 | separates before correction only |

## Retention against C7-F16

Each arm's headline rate over the reference's on the same axis. The bar is >= 93% on every axis and >= 97% on the mean.

| arm | axis | ref rate | arm rate | absolute | retention | >= floor | ratio well conditioned |
|---|---|--:|--:|--:|--:|:-:|:-:|
| C7-Q4_0 | bfcl_composite | 0.7057 | 0.6721 | -0.0336 | 95.2% | yes | yes |
| C7-Q4_0 | ifeval_prompt_strict | 0.7394 | 0.7301 | -0.0093 | 98.7% | yes | yes |
| C7-Q4_0 | ifstruct_validity | 0.1380 | 0.1150 | -0.0230 | 83.3% | NO | no (ref rate < 0.30) |
| C7-Q4_0 | probe_stack_idiom | 0.1944 | 0.1875 | -0.0069 | 96.4% | yes | no (ref rate < 0.30) |
| C7-Q4_K_M | bfcl_composite | 0.7057 | 0.6739 | -0.0318 | 95.5% | yes | yes |
| C7-Q4_K_M | ifeval_prompt_strict | 0.7394 | 0.7320 | -0.0074 | 99.0% | yes | yes |
| C7-Q4_K_M | ifstruct_validity | 0.1380 | 0.1355 | -0.0025 | 98.2% | yes | no (ref rate < 0.30) |
| C7-Q4_K_M | probe_stack_idiom | 0.1944 | 0.1667 | -0.0278 | 85.7% | NO | no (ref rate < 0.30) |

| arm | axes | mean | min | on axis | verdict |
|---|--:|--:|--:|---|---|
| C7-Q4_0 | 4 | 93.4% | 83.3% | ifstruct_validity | FAILS on ifstruct_validity |
| C7-Q4_K_M | 4 | 94.6% | 85.7% | probe_stack_idiom | FAILS on probe_stack_idiom |

A ratio is only as well conditioned as its denominator: where the reference rate is under 0.30, a few absolute points move the ratio by more than ten of them, so read those rows with the absolute column and the paired interval above and not on their own. The bar is applied as pre-registered either way.

`delta` is the arm's rate minus the reference's over the items both scored. `macro` averages the per-group deltas with equal weight per group and has no interval, so it is a description of the same items and not a second test.

## Notes

- C7-F16/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C7-F16/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C7-Q4_0/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C7-Q4_0/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C7-Q4_K_M/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C7-Q4_K_M/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
