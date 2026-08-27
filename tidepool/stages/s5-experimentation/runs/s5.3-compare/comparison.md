# s5.3 arms against C1, paired by item

Paired per-item comparison against the reference arm. The interval is a percentile bootstrap over the three paired-difference counts, which is exact for a statistic that is a function of those counts alone. The test is McNemar's exact two-sided test on the discordant pairs. The family correction is Holm-Bonferroni across every arm-by-component cell in this run. Deltas are item-weighted; the macro column is descriptive and carries no test.

Bootstrap resamples 10000, seed 20260826, 70 comparison(s) in the corrected family.

| arm | component | n | ref | arm | delta | 95% CI | macro | McNemar p | Holm p | reads as |
|---|---|--:|--:|--:|--:|---|--:|--:|--:|---|
| C2p | bfcl_native | 3490 | 0.7138 | 0.6785 | -0.0352 | [-0.0476, -0.0229] | +0.0001 | 2.399e-08 | 1.584e-06 | separates |
| C2p | bfcl_text | 3490 | 0.6742 | 0.6378 | -0.0364 | [-0.0458, -0.0272] | -0.0126 | 7.824e-15 | 5.398e-13 | separates |
| C2p | ifstruct | 2000 | 0.1375 | 0.1390 | +0.0015 | [-0.0125, +0.0155] |  | 0.89 | 1 | indistinguishable |
| C2p | ifeval | 541 | 0.7523 | 0.7283 | -0.0240 | [-0.0536, +0.0037] |  | 0.1299 | 1 | indistinguishable |
| C2p | probes_detect | 270 | 0.9259 | 0.7815 | -0.1444 | [-0.1852, -0.1037] |  | 3.638e-12 | 2.474e-10 | separates |
| C2p | probes_clean | 30 | 0.9000 | 0.8000 | -0.1000 | [-0.2000, +0.0000] |  | 0.25 | 1 | indistinguishable |
| C2p | probes_clean_corpus | 138 | 0.7319 | 0.7464 | +0.0145 | [-0.0290, +0.0580] |  | 0.7539 | 1 | indistinguishable |
| C2p | probes_stack_idiom | 144 | 0.1806 | 0.2500 | +0.0694 | [+0.0139, +0.1181] |  | 0.02127 | 1 | separates before correction only |
| C2p | probes_flag_detect | 270 | 0.6111 | 0.0074 | -0.6037 | [-0.6630, -0.5444] |  | 1.711e-49 | 1.197e-47 | separates |
| C2p | probes_flag_clean_corpus | 138 | 0.9855 | 1.0000 | +0.0145 | [+0.0000, +0.0362] |  | 0.5 | 1 | indistinguishable |
| C3 | bfcl_native | 3490 | 0.7138 | 0.7178 | +0.0040 | [-0.0063, +0.0140] | +0.0254 | 0.4756 | 1 | indistinguishable |
| C3 | bfcl_text | 3490 | 0.6742 | 0.6745 | +0.0003 | [-0.0092, +0.0095] | -0.0086 | 1 | 1 | indistinguishable |
| C3 | ifstruct | 2000 | 0.1375 | 0.1450 | +0.0075 | [-0.0065, +0.0220] |  | 0.3258 | 1 | indistinguishable |
| C3 | ifeval | 541 | 0.7523 | 0.7837 | +0.0314 | [+0.0018, +0.0610] |  | 0.04296 | 1 | separates before correction only |
| C3 | probes_detect | 270 | 0.9259 | 0.8926 | -0.0333 | [-0.0630, -0.0037] |  | 0.04904 | 1 | separates before correction only |
| C3 | probes_clean | 30 | 0.9000 | 0.7333 | -0.1667 | [-0.3000, -0.0333] |  | 0.0625 | 1 | separates before correction only |
| C3 | probes_clean_corpus | 138 | 0.7319 | 0.7174 | -0.0145 | [-0.0652, +0.0362] |  | 0.7905 | 1 | indistinguishable |
| C3 | probes_stack_idiom | 144 | 0.1806 | 0.2292 | +0.0486 | [-0.0139, +0.1111] |  | 0.1892 | 1 | indistinguishable |
| C3 | probes_flag_detect | 270 | 0.6111 | 0.6037 | -0.0074 | [-0.0704, +0.0593] |  | 0.9099 | 1 | indistinguishable |
| C3 | probes_flag_clean_corpus | 138 | 0.9855 | 0.9783 | -0.0072 | [-0.0435, +0.0217] |  | 1 | 1 | indistinguishable |
| C4 | bfcl_native | 3490 | 0.7138 | 0.6991 | -0.0146 | [-0.0229, -0.0063] | -0.0092 | 0.0006539 | 0.04054 | separates |
| C4 | bfcl_text | 3490 | 0.6742 | 0.6719 | -0.0023 | [-0.0080, +0.0034] | -0.0068 | 0.4968 | 1 | indistinguishable |
| C4 | ifstruct | 2000 | 0.1375 | 0.1445 | +0.0070 | [-0.0040, +0.0180] |  | 0.2467 | 1 | indistinguishable |
| C4 | ifeval | 541 | 0.7523 | 0.7468 | -0.0055 | [-0.0277, +0.0166] |  | 0.7428 | 1 | indistinguishable |
| C4 | probes_detect | 270 | 0.9259 | 0.9333 | +0.0074 | [+0.0000, +0.0185] |  | 0.5 | 1 | indistinguishable |
| C4 | probes_clean | 30 | 0.9000 | 0.8667 | -0.0333 | [-0.1000, +0.0000] |  | 1 | 1 | indistinguishable |
| C4 | probes_clean_corpus | 138 | 0.7319 | 0.7246 | -0.0072 | [-0.0290, +0.0145] |  | 1 | 1 | indistinguishable |
| C4 | probes_stack_idiom | 144 | 0.1806 | 0.2431 | +0.0625 | [+0.0139, +0.1111] |  | 0.02246 | 1 | separates before correction only |
| C4 | probes_flag_detect | 270 | 0.6111 | 0.6519 | +0.0407 | [-0.0037, +0.0852] |  | 0.1173 | 1 | indistinguishable |
| C4 | probes_flag_clean_corpus | 138 | 0.9855 | 0.9710 | -0.0145 | [-0.0362, +0.0000] |  | 0.5 | 1 | indistinguishable |
| C5a | bfcl_native | 3490 | 0.7138 | 0.7086 | -0.0052 | [-0.0138, +0.0034] | -0.0114 | 0.2704 | 1 | indistinguishable |
| C5a | bfcl_text | 3490 | 0.6742 | 0.6636 | -0.0106 | [-0.0183, -0.0029] | -0.0183 | 0.007954 | 0.4772 | separates before correction only |
| C5a | ifstruct | 2000 | 0.1375 | 0.1515 | +0.0140 | [-0.0005, +0.0285] |  | 0.06973 | 1 | indistinguishable |
| C5a | ifeval | 541 | 0.7523 | 0.7837 | +0.0314 | [+0.0018, +0.0610] |  | 0.05329 | 1 | separates before correction only |
| C5a | probes_detect | 270 | 0.9259 | 0.9519 | +0.0259 | [+0.0037, +0.0519] |  | 0.06543 | 1 | separates before correction only |
| C5a | probes_clean | 30 | 0.9000 | 0.8000 | -0.1000 | [-0.2333, +0.0000] |  | 0.25 | 1 | indistinguishable |
| C5a | probes_clean_corpus | 138 | 0.7319 | 0.7029 | -0.0290 | [-0.0652, +0.0000] |  | 0.2188 | 1 | indistinguishable |
| C5a | probes_stack_idiom | 144 | 0.1806 | 0.1944 | +0.0139 | [-0.0417, +0.0694] |  | 0.8145 | 1 | indistinguishable |
| C5a | probes_flag_detect | 270 | 0.6111 | 0.5741 | -0.0370 | [-0.0926, +0.0185] |  | 0.237 | 1 | indistinguishable |
| C5a | probes_flag_clean_corpus | 138 | 0.9855 | 0.9783 | -0.0072 | [-0.0217, +0.0000] |  | 1 | 1 | indistinguishable |
| C5b | bfcl_native | 3490 | 0.7138 | 0.7020 | -0.0117 | [-0.0212, -0.0023] | +0.0098 | 0.01766 | 1 | separates before correction only |
| C5b | bfcl_text | 3490 | 0.6742 | 0.6791 | +0.0049 | [-0.0034, +0.0135] | +0.0031 | 0.2882 | 1 | indistinguishable |
| C5b | ifstruct | 2000 | 0.1375 | 0.1095 | -0.0280 | [-0.0435, -0.0120] |  | 0.000561 | 0.03534 | separates |
| C5b | ifeval | 541 | 0.7523 | 0.7708 | +0.0185 | [-0.0129, +0.0499] |  | 0.3143 | 1 | indistinguishable |
| C5b | probes_detect | 270 | 0.9259 | 0.9407 | +0.0148 | [-0.0037, +0.0333] |  | 0.2188 | 1 | indistinguishable |
| C5b | probes_clean | 30 | 0.9000 | 0.8000 | -0.1000 | [-0.2000, +0.0000] |  | 0.25 | 1 | indistinguishable |
| C5b | probes_clean_corpus | 138 | 0.7319 | 0.7246 | -0.0072 | [-0.0435, +0.0290] |  | 1 | 1 | indistinguishable |
| C5b | probes_stack_idiom | 144 | 0.1806 | 0.2083 | +0.0278 | [-0.0417, +0.0972] |  | 0.5413 | 1 | indistinguishable |
| C5b | probes_flag_detect | 270 | 0.6111 | 0.6333 | +0.0222 | [-0.0222, +0.0704] |  | 0.4296 | 1 | indistinguishable |
| C5b | probes_flag_clean_corpus | 138 | 0.9855 | 1.0000 | +0.0145 | [+0.0000, +0.0362] |  | 0.5 | 1 | indistinguishable |
| C6 | bfcl_native | 3490 | 0.7138 | 0.7301 | +0.0163 | [+0.0034, +0.0289] | +0.0469 | 0.01193 | 0.7039 | separates before correction only |
| C6 | bfcl_text | 3490 | 0.6742 | 0.6817 | +0.0075 | [-0.0020, +0.0172] | -0.0019 | 0.1392 | 1 | indistinguishable |
| C6 | ifstruct | 2000 | 0.1375 | 0.1210 | -0.0165 | [-0.0320, -0.0010] |  | 0.04152 | 1 | separates before correction only |
| C6 | ifeval | 541 | 0.7523 | 0.7024 | -0.0499 | [-0.0813, -0.0185] |  | 0.002799 | 0.1707 | separates before correction only |
| C6 | probes_detect | 270 | 0.9259 | 0.9407 | +0.0148 | [-0.0037, +0.0333] |  | 0.2188 | 1 | indistinguishable |
| C6 | probes_clean | 30 | 0.9000 | 0.7667 | -0.1333 | [-0.2667, -0.0333] |  | 0.125 | 1 | separates before correction only |
| C6 | probes_clean_corpus | 138 | 0.7319 | 0.7101 | -0.0217 | [-0.0580, +0.0145] |  | 0.4531 | 1 | indistinguishable |
| C6 | probes_stack_idiom | 144 | 0.1806 | 0.2153 | +0.0347 | [-0.0278, +0.0972] |  | 0.4049 | 1 | indistinguishable |
| C6 | probes_flag_detect | 270 | 0.6111 | 0.6222 | +0.0111 | [-0.0370, +0.0593] |  | 0.7608 | 1 | indistinguishable |
| C6 | probes_flag_clean_corpus | 138 | 0.9855 | 1.0000 | +0.0145 | [+0.0000, +0.0362] |  | 0.5 | 1 | indistinguishable |
| C7 | bfcl_native | 3490 | 0.7138 | 0.6883 | -0.0255 | [-0.0358, -0.0152] | -0.0229 | 1.908e-06 | 0.000124 | separates |
| C7 | bfcl_text | 3490 | 0.6742 | 0.6456 | -0.0287 | [-0.0381, -0.0189] | -0.0147 | 1.012e-08 | 6.783e-07 | separates |
| C7 | ifstruct | 2000 | 0.1375 | 0.1400 | +0.0025 | [-0.0120, +0.0165] |  | 0.7811 | 1 | indistinguishable |
| C7 | ifeval | 541 | 0.7523 | 0.7338 | -0.0185 | [-0.0462, +0.0092] |  | 0.237 | 1 | indistinguishable |
| C7 | probes_detect | 270 | 0.9259 | 0.9370 | +0.0111 | [+0.0000, +0.0259] |  | 0.25 | 1 | indistinguishable |
| C7 | probes_clean | 30 | 0.9000 | 0.9000 | +0.0000 | [+0.0000, +0.0000] |  | 1 | 1 | indistinguishable |
| C7 | probes_clean_corpus | 138 | 0.7319 | 0.6957 | -0.0362 | [-0.0725, +0.0000] |  | 0.125 | 1 | indistinguishable |
| C7 | probes_stack_idiom | 144 | 0.1806 | 0.1944 | +0.0139 | [-0.0417, +0.0694] |  | 0.8036 | 1 | indistinguishable |
| C7 | probes_flag_detect | 270 | 0.6111 | 0.7222 | +0.1111 | [+0.0556, +0.1667] |  | 0.0001763 | 0.01128 | separates |
| C7 | probes_flag_clean_corpus | 138 | 0.9855 | 0.9783 | -0.0072 | [-0.0217, +0.0000] |  | 1 | 1 | indistinguishable |

`delta` is the arm's rate minus the reference's over the items both scored. `macro` averages the per-group deltas with equal weight per group and has no interval, so it is a description of the same items and not a second test.

## Notes

- C1/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C1/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C2p/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C2p/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C3/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C3/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C4/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C4/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C5a/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C5a/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C5b/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C5b/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C6/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C6/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C7/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C7/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
