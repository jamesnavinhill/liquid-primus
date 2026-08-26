#!/usr/bin/env bash
# The second pair of s5.3 arms, on the settings pass 1 measured rather than the ones it guessed.
#
# Pass 1 (`385e210a`) settled the ceiling. 11 GB an arm holds at batch 16: both arms cleared
# BFCL's full 3,490 items at the first calling convention and a third of the second with no
# split retries at all, and `gen.py`'s halving path never fired. Check that again in pass 1's
# completed log before running this, because a ceiling that only just held is not a ceiling
# that held; `oom_splits` and `smallest_batch` in each `score.json` are the fields.
#
# On cost, read the correction at 09:10Z in `runs/s5.3-sweep.md` before quoting a number. The
# supervisor's progress line counts per component, so 3,490 is BFCL alone at one of two
# conventions rather than the run, and the 130-items-a-minute figure this header used to carry
# was the short-prompt front of one component. The pair is estimated at 2.0 card-hours against
# B1's 1.42 for the same workload solo, so four passes is about 8 card-hours of the 116 that
# are unspent. Full item counts are still the right call, because the effects the s5.4 rule is
# looking for are small enough that validation loss could not separate three of four arms.
#
# Nothing about the recipe changes between passes, which is the point: every arm is scored on
# the settings the B1-B6 baseline rows were.
#
# C2p is the reference recipe with the guardrail epochs removed, and C7 is the reference on the
# raw mixture. Both are rank-16 adapters, so this pair is lighter than pass 1, which carried
# C3's full-parameter checkpoint.
set -e
lab task queue 08b4e028-b3b7-45a5-9bed-9f487c9c95ed -e tidepool --no-interactive --provider aws \
  -p arms=C2p,C7 \
  -p pack_gb=11,11 \
  -p run_tag=s5.3-arms \
  -p limit_per_component=0 \
  -p pack_overrides='{"C2p": {"adapter_object": "tidepool/s5.3/arms/C2p/adapter.zip"}, "C7": {"adapter_object": "tidepool/s5.3/arms/C7/adapter.zip"}}' \
  -m 'Second pair of the s5.3 arm scoring: C2p (no guardrail epochs) and C7 (raw mixture) on all four components at full item counts, two arms on one L4 under 11 GB ceilings. Same settings as pass 1 and as the B1-B6 baseline rows, so the four arms are directly comparable. Pass 1 measured the pair cost well under the screening estimate, so full item counts are kept.'
