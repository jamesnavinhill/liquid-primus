#!/usr/bin/env bash
# The paired retention comparison behind H4's stop/go. CPU only; no GPU-hours.
#
# One job per CHECKPOINT, not one job for the study. Retention is a ratio against the arm's
# own full-precision self, so R3's three arms are compared against R3-F16 and C7's against
# C7-F16, and the two families never mix. A single run cannot do both: the task takes one
# `reference`, and that limit is the right shape here rather than something to work around.
#
# Splitting also splits the Holm family, 8 cells per run instead of 16. For this claim that
# is the STRICTER direction, not the looser one: a smaller family makes each corrected
# p-value smaller, so a degradation is easier to flag, and H4 is a claim that quality was
# HELD. The verdict itself rests on the pre-registered per-axis floor and the mean, with the
# intervals beside them, and not on failing to reject.
#
#   ./queue_s56_retention.sh r3
#   ./queue_s56_retention.sh c7
#   ./queue_s56_retention.sh r3 Q4_K_L,Q4_K_XL   # the s5.6 G4a recovery arms, on their own
#
# The second argument names which 4-bit forms to compare, defaulting to the two G1 forms. The
# G4a recovery arms are compared in their OWN run rather than appended to the G1 run, for the
# same reason the checkpoints do not share one: a run of four arms is a Holm family of 16 cells
# instead of 8, and a larger family makes every corrected p LARGER, which is the looser
# direction for a claim that quality was held. Keeping the family shape identical across the
# two rungs also keeps their numbers comparable to each other, which is the whole point of a
# recovery rung. The reference is the same already-scored F16 export either way.
set -e
CK=$(echo "${1:?usage: queue_s56_retention.sh r3|c7}" | tr '[:lower:]' '[:upper:]')
case "$CK" in
  R3|C7) ;;
  *) echo "checkpoint must be r3 or c7, got $1" >&2; exit 2 ;;
esac

REF="$CK-F16"
FORMATS=${2:-Q4_0,Q4_K_M}
ARMS="$REF"
for F in ${FORMATS//,/ }; do ARMS="$ARMS,$CK-$F"; done
PREFIX=tidepool/s5.6/arms

# Refuse to spend a job on a family that is not all present. A comparison over two of three
# arms is not a smaller version of this comparison; it is a different one, and it would be
# recorded under the same name.
for A in $(echo "$ARMS" | tr ',' ' '); do
  for F in score.json eval_summary.json scored_bfcl_native_tools.jsonl \
           scored_bfcl_tools_text.jsonl scored_ifeval.jsonl scored_ifstruct.jsonl \
           scored_probes.jsonl; do
    lab --format json storage ls 2>/dev/null | grep -q "$PREFIX/$A/$F" \
      || { echo "missing: $PREFIX/$A/$F -- promote that arm first" >&2; exit 3; }
  done
done
echo "all $CK arms in this family ($ARMS) are present under $PREFIX"

lab task queue ecd12a36-da98-4a18-bc38-2889905746e9 -e tidepool --no-interactive \
  --provider "${PROVIDER:-aws}" \
  -p arms="$ARMS" \
  -p reference="$REF" \
  -p completions_prefix="$PREFIX" \
  -p retention=true \
  -m "H4 retention for the $CK checkpoint, forms $FORMATS: its 4-bit GGUF forms against its own
full-precision export, paired item by item, on CPU. The reference is $REF and never the other
checkpoint or the base model, so the ratio prices the quantization and not the fine-tune.
Each cell carries a 95% bootstrap interval and an exact McNemar p, Holm-corrected across the
cells in this run; the retention table applies the pre-registered stop/go from s3 unchanged,
>= 93% on every axis and >= 97% on the mean. Where an axis's full-precision rate is near the
floor a few absolute points move the ratio by ten, so those rows carry the absolute delta and
are flagged rather than exempted."
