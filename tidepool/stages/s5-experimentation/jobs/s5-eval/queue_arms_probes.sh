#!/usr/bin/env bash
# The corpus clean arm, added to every sweep arm. Probes only, four arms a card.
#
# WHY THIS EXISTS. The arm scoring passes inherited `clean_control_object: ""` from
# `pack.yaml`, where it is empty on purpose so a re-run of an already-published baseline row
# stays byte-comparable to the row it replaces. The consequence, caught at 08:34Z on 26 Aug
# by reading the queued configuration rather than assuming it, is that no arm is scored
# against the 138-item corpus clean arm. Thirty synthetic clean items bound a false-flag rate
# only to about one item in thirty, and the s5.4 reliability gate is written against 0.15.
#
# It was not fixed by setting the parameter for the later passes: C1 would then have no
# corpus cell for any arm to be paired against, which is worse than not having the cell. All
# eight arms were scored identically, and this pass adds the wider arm to all eight at once.
#
# SIZING. Probes-only, so the 3.1k-token BFCL tool schemas that set the 11 GB ceiling are not
# in play and four arms fit where two did. 5.5 GB an arm is deliberately near the measured
# 5.2 GB floor: `gen.py` now halves a group it cannot fit and retries, so a ceiling set too
# low costs throughput and no longer costs the arm. `oom_splits` and `smallest_batch` in each
# `score.json` say whether it was too low, which is the measurement for next time.
#
# PROMOTION. The output collides with the full pass's `scored_probes.jsonl`, and it should:
# the same graded and synthetic-clean items plus the corpus arm, generated on the same model
# under greedy decoding, so the wider file supersedes the narrower one. Promote ONLY that
# file, with --force, and leave the full pass's `score.json` alone, since a probes-only
# score.json carries no tool-calling, structured-output or instruction-following rates and
# overwriting it would destroy what the comparison cross-checks against:
#
#   python3 ../s5-compare/promote_scores.py <job> --arms C1,C2p,C3,C4 \
#       --include scored_probes.jsonl --force
#
# The shared items are also a free consistency check, and it is now possible to run it:
# --force keeps the bytes it replaces as `scored_probes.jsonl.superseded-by-<job>` in the same
# directory. Greedy argmax with left padding does not depend on how prompts were grouped, so a
# verdict that moved between the two passes on a shared probe item is a real finding about
# determinism and not a rounding difference. Join the two files on `id` after promoting.
set -e
ARMS=${1:?usage: queue_arms_probes.sh C1,C2p,C3,C4}
OV=$(python3 -c '
import json, sys
arms = [a for a in sys.argv[1].split(",") if a]
print(json.dumps({a: {"adapter_object": "tidepool/s5.3/arms/%s/adapter.zip" % a}
                  for a in arms}))' "$ARMS")
GB=$(python3 -c 'import sys; print(",".join(["5.5"] * len([a for a in sys.argv[1].split(",") if a])))' "$ARMS")
lab task queue 08b4e028-b3b7-45a5-9bed-9f487c9c95ed -e tidepool --no-interactive --provider aws \
  -p arms="$ARMS" \
  -p pack_gb="$GB" \
  -p run_tag=s5.3-arms-probes \
  -p components=probes \
  -p limit_per_component=0 \
  -p clean_control_object=tidepool/s5.3/tooldata/clean_control.jsonl \
  -p pack_overrides="$OV" \
  -m "Probes only, with the 138-item corpus clean control arm, over sweep arms $ARMS. The arm
scoring passes inherited an empty clean_control_object, so none of the eight arms had the wider
clean arm and thirty synthetic items cannot price a 0.15 false-flag ceiling. All eight arms were
scored identically, so the arm is added to all eight here rather than to the later passes only.
Four arms a card at 5.5 GB, near the measured 5.2 GB floor, because probes carry none of BFCL's
3k-token tool schemas and gen.py now halves a group it cannot fit."
