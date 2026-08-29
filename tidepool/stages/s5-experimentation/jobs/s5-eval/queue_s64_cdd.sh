#!/usr/bin/env bash
# s6.4 -- the black-box memorization check, pre-registered in plan.md as part of the
# decontamination re-check. Two arms on one L4, transformers on both sides.
#
# WHAT THE RAW NUMBER IS NOT. CDD (`2402.15938v3`) samples a model n times at temperature t,
# measures each sample's edit distance to the greedy completion, and calls an item leaked when
# an unusually large fraction sit within alpha of it. It was validated on free-form text, where
# a peaked output distribution is surprising. Tool calling is not free-form: the answer is a
# function name from a supplied list and a small object of typed arguments, and a model that
# has learned only the format will emit near-identical strings at any temperature. The fixture
# suite asserts that degenerate case directly. So the raw figure is uninformative here and is
# never the reported result.
#
# WHAT IT IS. Every item is sampled on B1, the published base that never saw the training mix,
# and on R3, whose training rows the decontamination scan searched. Schema rigidity belongs to
# the task, shows up on both arms and cancels in the difference. The reported quantity is the
# paired difference, tested with a sign test over item-level differences.
#
# AND WHY THE STRATA. Fine-tuning collapses output entropy on its own, so a uniform lift over
# the base is the ORDINARY result of supervised fine-tuning and says nothing about leakage.
# The design is an interaction: eval items whose text appears in a shipped training TARGET
# against matched siblings with no overlap, matched on category then nearest prompt length.
# Contamination predicts a larger lift on the overlap stratum. Entropy collapse predicts the
# same lift on both.
#
# WHAT THE THREE FLAGGED ITEMS TURNED OUT TO BE. Pass 4 of the scan (job `705a9d7d`) quotes the
# matched span for every target-side item, and read that way none of the three is a leaked
# answer. `parallel_129` matches on a run of consecutive integers, `34 35 36 ... 46`, inside a
# `valid_values` list in an APIGen tool call for a different function. `parallel_multiple_84`
# matches the English phrase `finding the least common multiple lcm and the greatest common
# divisor gcd of` inside a CodeFeedback prose answer about quaternions. `parallel_158` matches
# `normal distribution with a mean of 5 and a standard deviation of 2` inside a C++ answer.
# BFCL's ground truth is a tool call with a named function and typed arguments, and none of the
# three training targets contains one. So the overlap stratum is three n-gram artifacts and the
# interaction is expected to be null; it is queued anyway because it was pre-registered and
# because sampling behaviour is the one check that does not depend on the scan's own
# assumptions -- a 13-gram index cannot see a paraphrased answer, and this can.
#
# THE STRATA ARE SMALL, AND THAT IS THE FINDING RATHER THAN A FLAW IN THIS RUN. The re-check
# named three scored BFCL items inside a training target out of 3,489, so the interaction runs
# on six items and is reported as underpowered. It is kept because it was pre-registered and
# because a null on six matched items is still worth more than no matched items at all. The
# `background` stratum is what carries the other reading: 100 further items spread round-robin
# across the eleven categories, outside the matched design, pooled for the base-versus-tuned
# difference and excluded from either arm of the interaction.
#
# Cost: 106 items x 52 generations x 2 arms, both packed on one L4 at 11 GB each. Estimated
# 3-5 GPU-hours; `s6.4` has spent none so far, every other component of it being CPU work.
set -e
# The pack registered with the `cdd` component and its parameters DECLARED: `task upload`
# refuses task.yaml, so a pack whose new parameters are only in the local manifest would take
# every `cdd_*` flag on the queue line and silently run with the code defaults.
TASK=${TASK:-4a00a123-9f2d-4176-b462-acc956edbc32}
# The three scored BFCL items pass 4 found inside a shipped training target, inline: a list this
# short does not need a storage round trip, and having the ids on the queue line means the run's
# own parameters record which items were flagged rather than pointing at an object that could be
# rewritten later. Override IDS with a storage object if the list ever grows.
IDS=${IDS:-'["parallel_129","parallel_158","parallel_multiple_84"]'}
OV=$(python3 <<'JSON'
import json
arms = {
    # arm: adapter object ("" = the published base checkpoint, untrained by us)
    "B1": "",
    "R3": "tidepool/s5.3/arms/R3/adapter.zip",
}
print(json.dumps({a: {"adapter_object": o, "run_tag": "s6.4-cdd-%s" % a}
                  for a, o in arms.items()}, sort_keys=True))
JSON
)
python3 -c "import json,sys; d=json.loads(sys.argv[1]); assert sorted(d)==['B1','R3']; assert d['B1']['adapter_object']==''; print('overrides parse:', json.dumps(d, indent=1))" "$OV"

lab task queue "$TASK" -e tidepool --no-interactive --provider aws \
  -p arms=B1,R3 \
  -p pack_gb=11,11 \
  -p pack_headroom=0.92 \
  -p pack_overrides="$OV" \
  -p stagger_seconds=45 \
  -p components=cdd \
  -p backend=hf \
  -p base_model=LiquidAI/LFM2.5-1.2B-Instruct \
  -p batch_size=8 \
  -p limit_per_component=0 \
  -p limit_cdd=0 \
  -p cdd_style=native_tools \
  -p cdd_samples=51 \
  -p cdd_temperature=0.8 \
  -p cdd_alpha=0.05 \
  -p cdd_xi=0.01 \
  -p cdd_l_cap=100 \
  -p cdd_seed=0 \
  -p cdd_sample_chunk=8 \
  -p cdd_max_pairs=60 \
  -p cdd_background=100 \
  -p cdd_contaminated_ids="$IDS" \
  -p profile=full \
  -p run_tag=s6.4-cdd \
  -m 'Black-box memorization check for s6.4, on the base checkpoint and the tuned finalist,
two arms on one L4 and transformers on both so the two are served identically. 51 samples per
item at temperature 0.8 against one greedy completion, edit distance, and the peakedness
statistic from 2402.15938v3 at the paper defaults. The raw figure is not the result: a rigid
tool-call schema drives peakedness toward its ceiling with no contamination present, which the
fixtures assert directly. Every item is measured on both arms and the reported quantity is the
paired difference, split into eval items whose text appears in a shipped training target,
matched clean siblings, and a background sample across all eleven categories. The interaction
between the first two is the contamination test and is underpowered by design, because the
re-check found only three contaminated items; the background stratum carries the separate
reading, on how far this fine-tune narrowed the output distribution.'
