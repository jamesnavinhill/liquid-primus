#!/usr/bin/env bash
# s6.4 -- the paired reading of the memorization check. CPU only; no GPU-hours.
#
# The generation pass (job `4bab8d12`) samples every selected BFCL item 51 times at temperature
# 0.8 plus one greedy completion, on the untrained base and on the tuned finalist, and writes
# one row per item per arm carrying the peakedness statistic from `2402.15938v3`. This job is
# what turns two files of peakedness numbers into a result, and it exists as its own CPU pass
# for the same reason every other s6 comparison does: the statistics were wrong twice in this
# project before they were fixtured, and re-deriving them costs nothing when the generation is
# already on disk.
#
# It asserts, rather than assumes, that the two arms were measured on the SAME THING: the same
# item keys, the same strata, and byte-identical prompts. A peakedness difference between two
# arms that saw different prompt bytes is not a difference in memorization.
#
# What it reports:
#   * the pooled paired difference, tested with an exact two-sided sign test over item-level
#     differences (ties excluded, as the test requires);
#   * the interaction, overlap stratum minus control stratum, by permutation. Background items
#     are pooled into the first reading and excluded from this one -- folding unmatched items
#     into either arm would compare matched against unmatched and call the difference
#     contamination;
#   * a saturation guard, because a base arm already sitting at the ceiling leaves no room for
#     a lift to be visible and the pooled number would then be silence, not evidence.
#
# BEFORE RUNNING THIS, promote the generation pass's per-arm files into shared storage. The
# packed supervisor attaches each child's output as a job artifact under an `<arm>__` prefix
# rather than uploading it, so the two files arrive as `B1__cdd_items.jsonl` and
# `R3__cdd_items.jsonl` inside the job's artifact zip:
#
#   lab job download 4bab8d12-fb07-4bb6-882a-2b259228c05a -e tidepool -o /tmp/cddgen
#   cd /tmp/cddgen && unzip -o artifacts_*.zip
#   for a in B1 R3; do
#     mkdir -p /tmp/cddgen/$a && cp ${a}__cdd_items.jsonl /tmp/cddgen/$a/cdd_items.jsonl
#     lab storage upload /tmp/cddgen/$a/cdd_items.jsonl --dest tidepool/s6.4/cdd/$a --no-interactive
#   done
set -e
TASK=${TASK:-0ee71e63-759f-4e28-9b0b-f690f583ab78}
lab task queue "$TASK" -e tidepool --no-interactive --provider "${PROVIDER:-aws}" \
  -p base_arm=B1 \
  -p tuned_arm=R3 \
  -p items_prefix=tidepool/s6.4/cdd \
  -p xi=0.01 \
  -p resamples=20000 \
  -p seed=0 \
  -p examples=15 \
  -m 'Paired reading of the s6.4 memorization check. Same items on the untrained base and the
tuned finalist, asserted identical down to the prompt bytes. The pooled base-versus-tuned
difference in peakedness is tested with an exact sign test and reports how far this fine-tune
narrowed the output distribution; the overlap-minus-control interaction is the contamination
test and is underpowered by design, the re-check having found only three flagged items, none of
which its quoted spans support as a leaked answer.'
