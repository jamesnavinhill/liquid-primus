#!/usr/bin/env bash
# Any pair of s5.3 arms, on the settings passes 1 and 2 were run with. Passes 3 and 4 use this.
#
# `queue_arms_pass1.sh` and `queue_arms_pass2.sh` are kept as records of exactly what was
# queued for C1+C3 and C2p+C7, reasoning included. This one exists so the remaining pairs
# cannot drift from them by retyping: three of the arms sit within 0.0006 of each other on
# validation loss, so a settings difference between pairs would be larger than the effect the
# pairs exist to measure. Card, batch size, item counts and component list are the B1-B6
# baseline settings, unchanged, in every pass.
#
# Pack B's arms (C4, C5a, C5b, C6) need their adapters moved into shared storage first, the
# same hand move Pack A's needed, because the patched supervisor is deliberately not on the
# sweep task while Pack B is live on it:
#
#   python3 ../s5-compare/promote_scores.py 1df3bf2b --arms C4,C6 --include adapter.zip --dry-run
#
# Pair them by cost, not by name. Pack A's ordering lesson was that step count predicts the
# tail rather than tuning method, and here the equivalent is that two arms on a card finish
# together only if they are the same size of job; a pair is one card-hour of wall clock at the
# slower arm's rate whatever the faster one does.
set -e
ARMS=${1:?usage: queue_arms_pair.sh C4,C6}
N=$(python3 -c 'import sys; print(len([a for a in sys.argv[1].split(",") if a]))' "$ARMS")
[ "$N" = "2" ] || { echo "two arms a card at 11 GB: $ARMS is $N" >&2; exit 2; }
OV=$(python3 -c '
import json, sys
print(json.dumps({a: {"adapter_object": "tidepool/s5.3/arms/%s/adapter.zip" % a}
                  for a in sys.argv[1].split(",") if a}))' "$ARMS")
lab task queue 08b4e028-b3b7-45a5-9bed-9f487c9c95ed -e tidepool --no-interactive --provider aws \
  -p arms="$ARMS" \
  -p pack_gb=11,11 \
  -p run_tag=s5.3-arms \
  -p limit_per_component=0 \
  -p pack_overrides="$OV" \
  -m "s5.3 arm scoring, arms $ARMS, all four components at full item counts, two arms on one L4
under 11 GB ceilings. Identical settings to passes 1 and 2 and to the B1-B6 baseline rows, so
every arm in the sweep is comparable to every other and to the baselines. The sweep chose its
arms on validation loss and three of the four finished arms tied within 0.0006, so s5.4 is
decided on these task metrics."
