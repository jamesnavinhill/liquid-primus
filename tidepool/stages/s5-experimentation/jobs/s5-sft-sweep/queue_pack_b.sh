#!/bin/sh
# Queue Pack B the moment a GPU slot frees.
#
# The platform refuses a submission that would exceed the account's two-GPU limit rather than
# holding it in a queue, so Pack B cannot be lodged in advance -- it has to be submitted after
# C1 (job e4bd367a) finishes. This script is that submission, written down so it goes out
# unchanged whenever the slot opens, rather than being reconstructed from memory.
#
# Pack B is the rest of the sweep on one card: RB generates the self-distillation replay
# buffer, C4 and C6 train beside it from the start, and C5a and C5b are held until RB exits
# clean and then read its file off local disk. Ceilings are the measured footprints plus the
# CUDA context: 18 GB for the generator, 13 for the entropy-weighted arm, 9.5 for each LoRA
# arm. Peak concurrent demand is 41.5 GB of a 42.9 GB limit, and it is set AFTER the generator
# has handed over -- C4 + C6 + C5a + C5b -- which is only affordable because a dependency
# means the generator's 18 GB is not charged at the same time.
set -e
lab task queue cfc6dec7-cffb-40f2-ab9b-47dc12e0720c -e tidepool --no-interactive --provider aws \
  -p arms=RB,C4,C6,C5a,C5b \
  -p pack_scripts=RB=replay.py \
  -p pack_after=C5a=RB,C5b=RB \
  -p pack_provides=RB=replay_object:replay.jsonl.gz \
  -p replay_object=tidepool/s5.3/replay/replay.jsonl.gz \
  -p pack_gb=18,13,9.5,9.5,9.5 \
  -p pack_headroom=0.9 \
  -p stagger_seconds=60 \
  -p run_tag=s5.3-packB \
  -m 'Pack B: the last four sweep arms plus the buffer two of them read, on one card. RB generates the self-distillation replay buffer with replay.py; C4 and C6 train alongside it from the start; C5a and C5b are held until RB exits clean, then read its file off local disk while the supervisor uploads it to shared storage once. Ceilings 18/13/9.5/9.5/9.5 GB, peak 41.5 of a 42.9 GB limit, set by C4+C6+C5a+C5b after RB has handed over.'
