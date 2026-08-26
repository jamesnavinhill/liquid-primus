#!/usr/bin/env bash
# The first half of the sweep's arms, scored on the metrics that will actually decide s5.4.
#
# Two arms an L4, not four. Job `97939c69` tried four under 5.2 GB ceilings and every one of
# them died on its first BFCL batch with 4.41 GB allocated and 772 MB more wanted, one of them
# while 17.33 GB of the card sat free -- the ceiling killed it, not the card. Scoring needs a
# little over 5.2 GB an arm at batch 16, so 11 GB is the sized figure and two is what an L4
# holds. Packing four would also have bought little: generation saturates the card, unlike
# training, which leaves it idle between steps.
#
# The card stays an L4 because every full-precision number in this project was taken on one
# and these arms have to be comparable to the B1-B6 baseline rows. Batch size, item counts and
# component list are the baseline settings unchanged, for the same reason: three of these arms
# sit within 0.0006 of each other on validation loss, so anything that could move a number by
# that much has to be held still.
#
# C1 is the reference recipe and C3 tunes all 1.2B weights. C3's checkpoint is a full model
# rather than an adapter, and this is the first real scoring run down that path.
set -e
lab task queue 08b4e028-b3b7-45a5-9bed-9f487c9c95ed -e tidepool --no-interactive --provider aws \
  -p arms=C1,C3 \
  -p pack_gb=11,11 \
  -p run_tag=s5.3-arms \
  -p limit_per_component=0 \
  -p pack_overrides='{"C1": {"adapter_object": "tidepool/s5.3/arms/C1/adapter.zip"}, "C3": {"adapter_object": "tidepool/s5.3/arms/C3/adapter.zip"}}' \
  -m 'First half of the s5.3 arm scoring: C1 (reference LoRA r16) and C3 (full-parameter) on all four components at full item counts, two arms on one L4 under 11 GB ceilings. The sweep chose its arms on validation loss and three of the four tied within 0.0006, so s5.4 has to be decided on these task metrics instead. Settings are the B1-B6 baseline settings unchanged so the arms are comparable to the baseline rows.'
