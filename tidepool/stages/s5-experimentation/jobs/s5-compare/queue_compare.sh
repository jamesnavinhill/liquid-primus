#!/usr/bin/env bash
# The paired comparison behind the s5.4 decision. CPU only; no GPU-hours.
#
# Run it once every arm being compared has its `scored_*.jsonl` and `score.json` under
# `tidepool/s5.3/arms/<arm>/`. The scoring jobs write those as job artifacts under an
# `<arm>__` prefix, so `promote_scores.py <job> --arms ...` moves them across first.
#
# `arms` must list every arm in one call, and one call only. The Holm correction is applied
# across the cells in a single run, so splitting the arms over two jobs would correct each
# half against a family half the size and report p-values that are too small in both.
set -e
ARMS=${1:?usage: queue_compare.sh C1,C2p,C3,C7[,...]}
lab task queue ecd12a36-da98-4a18-bc38-2889905746e9 -e tidepool --no-interactive \
  --provider "${PROVIDER:-aws}" \
  -p arms="$ARMS" \
  -p reference=C1 \
  -m "Paired per-item comparison of the s5.3 sweep arms ($ARMS) against the C1 reference, on
CPU. The sweep chose arms on validation loss and three of the four finished arms tied within
0.0006, so s5.4 is decided on the task metrics instead. Each cell carries a 95% bootstrap
interval and an exact McNemar p, Holm-corrected across every arm-by-component cell in this
run, so a direction is only claimed where the difference survives the whole family."
