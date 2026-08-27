#!/bin/sh
# Queue the s5.5 ladder: one card, five arms, two replay buffers and three trainers.
#
# s5.4 left the sweep with no arm through all three gates and one binding constraint: every
# specialized arm loses 3.5 to 11.7 IFEval points against a 2.0 allowance. C5a and C5b varied
# the replay FRACTION over a buffer whose composition was left proportional to the pool, moved
# IFEval two to three points, and stopped. Reading the pool afterwards showed why the axis
# stalled: the constraint-bearing slice of `antidoom-mix-v1.0` is 10.5% of it, so a 5% replay
# dose delivers verifiable-constraint instruction following at half a percent of the training
# tokens. Dose and composition were confounded because only dose ever moved.
#
# This pack separates them, at C7's recipe and the same fixed 64M-token budget every C arm got:
#
#   RBP  proportional buffer, 32,000 prompts, strata {}          (what C5a/C5b replayed)
#   RBC  reweighted buffer,   32,000 prompts, half constraint    (the new lever)
#   R1   5% replay of RBC     -> against C7, does a small reweighted dose move IFEval at all
#   R2   20% replay of RBP    -> against R3, isolates composition at a fixed dose
#   R3   20% replay of RBC    -> against R1, isolates dose over one and the same buffer
#
# Both buffers are 32,000 prompts rather than the 8,000 of s5.3 so that a 20% dose (12.8M
# tokens of a 64M budget) makes at most one pass; c5b_dose_tokens is raised to 12.8M so the
# generator's own repeat-count gate checks the dose these arms will actually take.
#
# The reweighted shares are 0.47 autoif and 0.03 ifstruct, which is half the buffer. The split
# between the two is set by supply: the decontaminated pool holds 46,734 `open_perfectblend_autoif`
# rows but only 1,279 `ifstruct_train_generated` ones, because IFStruct is an evaluation set here
# and the s4 contamination filter dropped 18,721 of its 20,000 training rows. A 0.05 ifstruct
# share would ask for 1,600 rows the pool cannot supply and replay.py would fail the run.
#
# Ceilings: 18 GB for each generator, 9.5 for each LoRA trainer, both measured at s5.3. Peak
# concurrent demand is 37.0 GB of a 43.9 GB limit, set by RBP still generating while RBC has
# already released R1 and R3. The generators write to two DIFFERENT config keys, which is what
# lets one pack carry two of them: `pack_provides` resolves the object path by key, so a shared
# key would have the second registration overwrite the first and two trainers would silently
# read the wrong buffer.
set -e
lab task queue cfc6dec7-cffb-40f2-ab9b-47dc12e0720c -e tidepool --no-interactive --provider aws \
  -p arms=RBP,RBC,R1,R2,R3 \
  -p pack_scripts=RBP=replay.py,RBC=replay.py \
  -p pack_after=R1=RBC,R2=RBP,R3=RBC \
  -p pack_provides=RBP=replay_object_proportional:replay.jsonl.gz,RBC=replay_object_constraint:replay.jsonl.gz \
  -p pack_gb=18,18,9.5,9.5,9.5 \
  -p pack_headroom=0.92 \
  -p pack_overrides='{"RBP": {"strata": "{}", "dest_prefix": "tidepool/s5.5/replay_proportional"}, "RBC": {"strata": "{\"open_perfectblend_autoif\": 0.47, \"ifstruct_train_generated\": 0.03}", "dest_prefix": "tidepool/s5.5/replay_constraint"}}' \
  -p n_prompts=32000 \
  -p c5b_dose_tokens=12800000 \
  -p stagger_seconds=60 \
  -p run_tag=s5.5-packR \
  -m 'The s5.5 replay ladder on one card: two 32,000-prompt self-distillation buffers generated side by side, then three C7-recipe trainers that read them. RBP samples the prompt pool proportionally, the way the s5.3 buffer was built; RBC reweights it to half verifiable-constraint rows (0.47 open_perfectblend_autoif, 0.03 ifstruct_train_generated, the second capped by a pool that holds only 1,279 of them after decontamination). R1 takes 5% replay of RBC, R2 20% of RBP, R3 20% of RBC, all at the same fixed 64M-token budget as the C sweep. R3-R2 isolates composition at a fixed dose, R3-R1 isolates dose over one buffer, R1-C7 asks whether a small reweighted dose moves IFEval at all. Ceilings 18/18/9.5/9.5/9.5 GB, peak 37.0 of a 43.9 GB limit, set by RBP generating while RBC has already released R1 and R3.'
