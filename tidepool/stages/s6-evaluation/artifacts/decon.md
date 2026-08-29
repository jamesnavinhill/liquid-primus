# s6.4 — decontamination re-check

13-gram overlap between the SHIPPED splits and the exact eval items the reported numbers came from, counted separately for the model's input and its training target.

## What was indexed

| eval set | items | 13-grams | note |
|---|--:|--:|---|
| bfcl_scored | 3490 | 596640 | gorilla-llm/Berkeley-Function-Calling-Leaderboard, 3489 of 3489 scored ids found |
| ifbench_mt_ie | 1774 | 536324 | allenai/IFBench_multi-turn |
| ifbench_mt_if | 1387 | 407076 | allenai/IFBench_multi-turn |
| ifbench_test | 300 | 13169 | allenai/IFBench_test |
| ifeval | 541 | 13627 | google/IFEval |
| ifstruct | 2000 | 599717 | https://raw.githubusercontent.com/Liquid4All/ifstruct/main/data/test.jsonl |
| probes | 434 | 33565 | tidepool/s4.4/probes/probes.jsonl |

Unique 13-grams across every set: **908734**.

## What the shipped splits contain

| split | rows | rows hit | rate | in input | in TARGET | both sides |
|---|--:|--:|--:|--:|--:|--:|
| train | 494341 | 56707 | 11.47120% | 56689 | 56 | 38 |
| val | 12439 | 2669 | 21.45670% | 2669 | 4 | 4 |
| test | 11550 | 2692 | 23.30740% | 2691 | 2 | 1 |

The **TARGET** column is the one `s4.3` could not report. It checked benchmark n-grams against training prompts; a benchmark answer sitting in a training target is the overlap that would actually teach the benchmark, and it needed the shipped conversations to see it.

## Where the training overlap sits

| corpus | rows | hit | rate | in input | in target |
|---|--:|--:|--:|--:|--:|
| apigen | 42346 | 42346 | 100.00000% | 42346 | 38 |
| toolace | 8181 | 8181 | 100.00000% | 8181 | 0 |
| hermes_glaive | 3748 | 3748 | 100.00000% | 3748 | 0 |
| hermes_fc | 1504 | 1504 | 100.00000% | 1504 | 0 |
| hermes_fc_st | 910 | 910 | 100.00000% | 910 | 0 |
| codefeedback | 128736 | 18 | 0.01400% | 0 | 18 |

| eval set | windows matched | of those, in a target |
|---|--:|--:|
| probes | 1417225 | 0 |
| bfcl_scored | 219 | 150 |

### The worst rows, with the overlapping text

A hit is not automatically a leak. A shared instruction preamble and a shared JSON schema both produce long exact overlaps and teach nothing about any answer, which is why the matched span is printed rather than counted.

- `apigen` row `49f715104758d93a185c7afb627bca4d2860ef3a3c5b2cab70c0666274c4ef6c` (train), 25 window(s) in the input and 8 in the target, against `probes` item `tool_return/corrupted/truncated_json/d1/000`:
  > tool_call block per call containing a json object with name and arguments when
- `apigen` row `7bb2db474aad83ff38370ce214ff21a5e4dedd1488bde70ca33ee8f46244c5b6` (train), 25 window(s) in the input and 6 in the target, against `probes` item `tool_return/corrupted/truncated_json/d1/000`:
  > tool_call block per call containing a json object with name and arguments when
- `apigen` row `adb61e12a9824bfd28fcdd0f72a9354462d7a6b427f46ca78776e5f8b2d55c8a` (train), 25 window(s) in the input and 4 in the target, against `probes` item `tool_return/corrupted/truncated_json/d1/000`:
  > tool_call block per call containing a json object with name and arguments when
- `apigen` row `a4f1529cd77a3b4776fb3b91fa8803e7dea5626b797b01ec68c36d2d8780d4ae` (train), 25 window(s) in the input and 4 in the target, against `probes` item `tool_return/corrupted/truncated_json/d1/000`:
  > tool_call block per call containing a json object with name and arguments when
- `apigen` row `242e7da5b7655ba19b7d67d45c77f5ca90a5dbfd314b459708b44b54ae47d49b` (train), 25 window(s) in the input and 4 in the target, against `probes` item `tool_return/corrupted/truncated_json/d1/000`:
  > tool_call block per call containing a json object with name and arguments when
- `apigen` row `af613b1247ccd57e7c0b8586b8a4397c4a05967ecca2bd8c54094defcbfcc6bf` (train), 25 window(s) in the input and 4 in the target, against `probes` item `tool_return/corrupted/truncated_json/d1/000`:
  > tool_call block per call containing a json object with name and arguments when
- `apigen` row `21a0f4f9b2913d27747d7c563f72f8eaf987dac33619deaa89b50659601d83cf` (train), 25 window(s) in the input and 4 in the target, against `probes` item `tool_return/corrupted/truncated_json/d1/000`:
  > tool_call block per call containing a json object with name and arguments when
- `apigen` row `41390b630b66bb6fbcbf9b3c47461d58097a5e7478d13616259c20e75d7bcb94` (train), 25 window(s) in the input and 4 in the target, against `probes` item `tool_return/corrupted/truncated_json/d1/000`:
  > tool_call block per call containing a json object with name and arguments when
- `apigen` row `a69fd606d1f34de1776a10bb6342b6e0f093d62a3467d0d27906296dee647853` (train), 25 window(s) in the input and 4 in the target, against `probes` item `tool_return/corrupted/truncated_json/d1/000`:
  > tool_call block per call containing a json object with name and arguments when
- `apigen` row `6ae0e060edea35426541007ff8b6b500832112074545fc543664cead2a508add` (train), 25 window(s) in the input and 4 in the target, against `probes` item `tool_return/corrupted/truncated_json/d1/000`:
  > tool_call block per call containing a json object with name and arguments when
- `apigen` row `06bd6c7c3fe0cf666083f866f3d558bced8832a1502bd7777469a2a8a3e6de53` (train), 25 window(s) in the input and 4 in the target, against `probes` item `tool_return/corrupted/truncated_json/d1/000`:
  > tool_call block per call containing a json object with name and arguments when
- `apigen` row `5d47d58be3dc03c5c5936fd74c3057b632c20921dd53dfb91a4ae5ad37a73e22` (train), 25 window(s) in the input and 4 in the target, against `probes` item `tool_return/corrupted/truncated_json/d1/000`:
  > tool_call block per call containing a json object with name and arguments when
- `apigen` row `66c159f6d812eb70b2ddf0e655780434da62e239509fcd82c9cc2edabd854a16` (train), 25 window(s) in the input and 4 in the target, against `probes` item `tool_return/corrupted/truncated_json/d1/000`:
  > tool_call block per call containing a json object with name and arguments when
- `apigen` row `e1d2adc0df8016f3dfc7c73e1f6cb015854b601a22963f40d636bcc8b06a50e2` (train), 25 window(s) in the input and 4 in the target, against `probes` item `tool_return/corrupted/truncated_json/d1/000`:
  > tool_call block per call containing a json object with name and arguments when
- `apigen` row `db3c21c85e66bab8f8770006fba5d14c9257eb21729f1b123ee2c3e4b28c6bc9` (train), 25 window(s) in the input and 4 in the target, against `probes` item `tool_return/corrupted/truncated_json/d1/000`:
  > tool_call block per call containing a json object with name and arguments when

### Every eval item whose text appears in a training TARGET

The count of these items is the contamination finding, and a count cannot be read on its own: a benchmark answer reproduced verbatim and a shared boilerplate phrase both land here. The matched span is printed for each.

- `bfcl_scored` item `parallel_129`, first seen in `apigen` row `9fccbabb5134bb3a22c93eb1b78c7412ef7b40104d7406c00d039b60c65e41d9`:
  > 34 35 36 37 38 39 40 41 42 43 44 45 46
- `bfcl_scored` item `parallel_multiple_84`, first seen in `codefeedback` row `e924a8813b6e881c`:
  > finding the least common multiple lcm and the greatest common divisor gcd of
- `bfcl_scored` item `parallel_158`, first seen in `codefeedback` row `283d31e620a48c00`:
  > normal distribution with a mean of 5 and a standard deviation of 2

## Per-set passes

The index above keeps one owner per 13-gram, so a span shared by two eval sets counts for whichever was indexed first. Each set below was re-scanned against an index containing only itself, which removes that competition; these are the exact figures for the sets that carry reported results.

| eval set | grams | split | rows | hit | rate | in target |
|---|--:|---|--:|--:|--:|--:|
| bfcl_scored | 198023 | train | 494341 | 99 | 0.02000% | 56 |
| bfcl_scored | 198023 | val | 12439 | 10 | 0.08040% | 4 |
| bfcl_scored | 198023 | test | 11550 | 4 | 0.03460% | 2 |
| probes | 5545 | train | 494341 | 56689 | 11.46760% | 0 |
| probes | 5545 | val | 12439 | 2669 | 21.45670% | 0 |
| probes | 5545 | test | 11550 | 2691 | 23.29870% | 0 |
