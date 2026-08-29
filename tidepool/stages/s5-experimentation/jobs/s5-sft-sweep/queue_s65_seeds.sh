#!/bin/sh
# s6.5 seed replication of the named final configuration: two more draws of R3, one card.
#
# Everything s7 will say about the final checkpoint rests on a single training draw. Two
# questions need an error bar before it can be said. Does R3's tool-calling gain over the
# matched base survive resampling, and is C7's structured-output advantage over R3 -- the only
# reason C7 is carried as a second finalist -- a property of the recipe or of the draw?
#
# What varies and what does not. `seed` sets the LoRA initialization and the dropout masks.
# It does NOT reshuffle the training rows: `sample.choose` is deterministic and the loader runs
# with shuffle off, by design, because identical row order across arms is what makes the sweep's
# arms paired. So these are three draws of the initialization, not three draws of the data
# order, and the report says so rather than calling them "three seeds" and leaving it.
#
# The two buffers already exist in shared storage from the s5.5 ladder, so this pack carries
# trainers only and no generators. Ceilings are the 9.5 GB measured for a rank-16 LoRA arm at
# s5.3; peak concurrent demand is 19.0 GB of a 43.9 GB limit, so the card is under-filled and
# a third arm would fit -- there is no third draw to run, and padding the pack with an
# unrelated arm would put an unreviewed recipe on the critical path of a replication.
#
# WHY THE FIRST ATTEMPT (job 1d7ee70d) DIED, and what changed. Both arms exited rc=1 inside a
# minute with "pack child was given no local path for tidepool/s5.5/replay_constraint/...".
# The supervisor stages shared inputs by scanning config keys that END IN `_object`, and the
# buffer path lived only in the ARMS recipe inside main.py, which the supervisor never reads.
# At s5.5 that was invisible because RBC GENERATED the buffer in the same pack, so its local
# path was registered by `pack_provides`; and the pack.yaml default that names it is called
# `replay_object_constraint`, which does not end in `_object` and so is never staged either.
# A pack of trainers with no generator has to be told the object explicitly, so it now is.
#
# WHY THE SECOND ATTEMPT (job f1d0b3df) DIED. The buffer staged correctly this time and both
# arms reached training, then both exited rc=1 at about a minute inside `multiprocessing`'s
# forkserver handshake: the DataLoader worker reset the connection during the authkey exchange.
# Each arm's console carries TWO wandb runs, which is the tell that the worker re-executed
# module scope. `loader_workers=0` removes the worker machinery from this run. It moves no
# number: workers only tokenize and collate, the loader runs with shuffle off, and the sampler
# is deterministic, so a replication with 0 workers is the same arithmetic as R3 with 2.
set -e
lab task queue cfc6dec7-cffb-40f2-ab9b-47dc12e0720c -e tidepool --no-interactive --provider aws \
  -p arms=R3b,R3c \
  -p replay_object=tidepool/s5.5/replay_constraint/replay.jsonl.gz \
  -p loader_workers=0 \
  -p pack_gb=9.5,9.5 \
  -p pack_headroom=0.92 \
  -p stagger_seconds=60 \
  -p run_tag=s6.5-seeds \
  -m 'Seed replication of the named final configuration R3, two further draws on one L40S. R3b (seed 41) and R3c (seed 97) carry byte-identical recipes to R3 -- raw mix, 20% replay of the reweighted constraint buffer, 64M-token budget, rank-16 LoRA -- and differ only in the seed that sets LoRA initialization and dropout masks. Row order is deterministic and unchanged by construction, so these are three draws of the initialization rather than of the data order. The pair answers two questions s7 needs: whether R3s tool-calling gain over the matched base survives resampling, and whether C7s structured-output advantage over R3, the sole reason C7 is carried as a second finalist, is a recipe difference or a draw difference. Ceilings 9.5 GB each, peak 19.0 of a 43.9 GB limit.'
