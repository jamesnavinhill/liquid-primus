# s5.3 arms against R3, paired by item

Paired per-item comparison against the reference arm. The interval is a percentile bootstrap over the three paired-difference counts, which is exact for a statistic that is a function of those counts alone. The test is McNemar's exact two-sided test on the discordant pairs. The family correction is Holm-Bonferroni across every arm-by-component cell in this run. Deltas are item-weighted; the macro column is descriptive and carries no test.

Bootstrap resamples 10000, seed 20260826, 40 comparison(s) in the corrected family.

| arm | component | n | ref | arm | delta | 95% CI | macro | McNemar p | Holm p | reads as |
|---|---|--:|--:|--:|--:|---|--:|--:|--:|---|
| R1 | bfcl_native | 3490 | 0.6854 | 0.6954 | +0.0100 | [-0.0003, +0.0203] | +0.0130 | 0.06623 | 1 | indistinguishable |
| R1 | bfcl_text | 3490 | 0.6739 | 0.6622 | -0.0117 | [-0.0206, -0.0029] | -0.0102 | 0.01046 | 0.3032 | separates before correction only |
| R1 | ifstruct | 2000 | 0.0885 | 0.0820 | -0.0065 | [-0.0160, +0.0030] |  | 0.2229 | 1 | indistinguishable |
| R1 | ifeval | 541 | 0.7560 | 0.7689 | +0.0129 | [-0.0111, +0.0370] |  | 0.3604 | 1 | indistinguishable |
| R1 | probes_detect | 270 | 0.9185 | 0.9185 | +0.0000 | [-0.0222, +0.0222] |  | 1 | 1 | indistinguishable |
| R1 | probes_clean | 30 | 0.9000 | 0.8333 | -0.0667 | [-0.1667, +0.0000] |  | 0.5 | 1 | indistinguishable |
| R1 | probes_clean_corpus | 138 | 0.6884 | 0.7391 | +0.0507 | [+0.0145, +0.0942] |  | 0.03906 | 1 | separates before correction only |
| R1 | probes_stack_idiom | 144 | 0.2222 | 0.2083 | -0.0139 | [-0.0556, +0.0278] |  | 0.7539 | 1 | indistinguishable |
| R1 | probes_flag_detect | 270 | 0.7148 | 0.6148 | -0.1000 | [-0.1556, -0.0444] |  | 0.0005844 | 0.0187 | separates |
| R1 | probes_flag_clean_corpus | 138 | 0.9855 | 0.9783 | -0.0072 | [-0.0362, +0.0145] |  | 1 | 1 | indistinguishable |
| R2 | bfcl_native | 3490 | 0.6854 | 0.6943 | +0.0089 | [-0.0009, +0.0183] | +0.0263 | 0.0867 | 1 | indistinguishable |
| R2 | bfcl_text | 3490 | 0.6739 | 0.6567 | -0.0172 | [-0.0261, -0.0083] | -0.0125 | 0.0001387 | 0.004576 | separates |
| R2 | ifstruct | 2000 | 0.0885 | 0.0850 | -0.0035 | [-0.0130, +0.0060] |  | 0.5203 | 1 | indistinguishable |
| R2 | ifeval | 541 | 0.7560 | 0.7689 | +0.0129 | [-0.0074, +0.0333] |  | 0.281 | 1 | indistinguishable |
| R2 | probes_detect | 270 | 0.9185 | 0.9222 | +0.0037 | [-0.0222, +0.0296] |  | 1 | 1 | indistinguishable |
| R2 | probes_clean | 30 | 0.9000 | 0.9000 | +0.0000 | [+0.0000, +0.0000] |  | 1 | 1 | indistinguishable |
| R2 | probes_clean_corpus | 138 | 0.6884 | 0.7174 | +0.0290 | [+0.0000, +0.0652] |  | 0.2188 | 1 | indistinguishable |
| R2 | probes_stack_idiom | 144 | 0.2222 | 0.2361 | +0.0139 | [-0.0208, +0.0556] |  | 0.7266 | 1 | indistinguishable |
| R2 | probes_flag_detect | 270 | 0.7148 | 0.7111 | -0.0037 | [-0.0481, +0.0407] |  | 1 | 1 | indistinguishable |
| R2 | probes_flag_clean_corpus | 138 | 0.9855 | 0.9928 | +0.0072 | [+0.0000, +0.0217] |  | 1 | 1 | indistinguishable |
| C7 | bfcl_native | 3490 | 0.6854 | 0.6883 | +0.0029 | [-0.0086, +0.0143] | +0.0384 | 0.6643 | 1 | indistinguishable |
| C7 | bfcl_text | 3490 | 0.6739 | 0.6456 | -0.0284 | [-0.0381, -0.0186] | -0.0086 | 2.341e-08 | 8.428e-07 | separates |
| C7 | ifstruct | 2000 | 0.0885 | 0.1400 | +0.0515 | [+0.0360, +0.0670] |  | 9.431e-11 | 3.49e-09 | separates |
| C7 | ifeval | 541 | 0.7560 | 0.7338 | -0.0222 | [-0.0573, +0.0129] |  | 0.2513 | 1 | indistinguishable |
| C7 | probes_detect | 270 | 0.9185 | 0.9370 | +0.0185 | [+0.0037, +0.0370] |  | 0.0625 | 1 | separates before correction only |
| C7 | probes_clean | 30 | 0.9000 | 0.9000 | +0.0000 | [+0.0000, +0.0000] |  | 1 | 1 | indistinguishable |
| C7 | probes_clean_corpus | 138 | 0.6884 | 0.6957 | +0.0072 | [-0.0362, +0.0507] |  | 1 | 1 | indistinguishable |
| C7 | probes_stack_idiom | 144 | 0.2222 | 0.1944 | -0.0278 | [-0.0903, +0.0417] |  | 0.5235 | 1 | indistinguishable |
| C7 | probes_flag_detect | 270 | 0.7148 | 0.7222 | +0.0074 | [-0.0333, +0.0481] |  | 0.8555 | 1 | indistinguishable |
| C7 | probes_flag_clean_corpus | 138 | 0.9855 | 0.9783 | -0.0072 | [-0.0362, +0.0145] |  | 1 | 1 | indistinguishable |
| B1 | bfcl_native | 3490 | 0.6854 | 0.6822 | -0.0032 | [-0.0175, +0.0112] | +0.0494 | 0.6929 | 1 | indistinguishable |
| B1 | bfcl_text | 3490 | 0.6739 | 0.5504 | -0.1235 | [-0.1410, -0.1060] | -0.1730 | 2.095e-43 | 8.171e-42 | separates |
| B1 | ifstruct | 2000 | 0.0885 | 0.1355 | +0.0470 | [+0.0335, +0.0605] |  | 1.599e-11 | 6.076e-10 | separates |
| B1 | ifeval | 541 | 0.7560 | 0.8189 | +0.0628 | [+0.0351, +0.0906] |  | 1.743e-05 | 0.0005926 | separates |
| B1 | probes_detect | 270 | 0.9185 | 0.8074 | -0.1111 | [-0.1519, -0.0667] |  | 6.039e-07 | 2.113e-05 | separates |
| B1 | probes_clean | 30 | 0.9000 | 0.6333 | -0.2667 | [-0.4333, -0.1333] |  | 0.007813 | 0.2422 | separates before correction only |
| B1 | probes_clean_corpus | 138 | 0.6884 | 0.5870 | -0.1014 | [-0.1739, -0.0290] |  | 0.009355 | 0.2807 | separates before correction only |
| B1 | probes_stack_idiom | 144 | 0.2222 | 0.2361 | +0.0139 | [-0.0347, +0.0625] |  | 0.7744 | 1 | indistinguishable |
| B1 | probes_flag_detect | 270 | 0.7148 | 0.0074 | -0.7074 | [-0.7630, -0.6519] |  | 6.372e-58 | 2.549e-56 | separates |
| B1 | probes_flag_clean_corpus | 138 | 0.9855 | 1.0000 | +0.0145 | [+0.0000, +0.0362] |  | 0.5 | 1 | indistinguishable |

`delta` is the arm's rate minus the reference's over the items both scored. `macro` averages the per-group deltas with equal weight per group and has no interval, so it is a description of the same items and not a second test.

## Notes

- R1/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- R1/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- R2/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- R2/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- R3/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- R3/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C7/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- C7/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- B1/bfcl_native: 1 repeated id(s), kept as <id>#k and paired by occurrence
- B1/bfcl_text: 1 repeated id(s), kept as <id>#k and paired by occurrence
- B1 has no score.json in storage, so any rate it carries is not cross-checked
