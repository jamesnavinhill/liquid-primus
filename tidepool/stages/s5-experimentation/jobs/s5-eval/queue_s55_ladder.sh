#!/usr/bin/env bash
# The s5.5 replay ladder, scored on the settings every other arm in this project was scored on.
#
# Two cards, because three full-pass arms do not fit on one: an arm needs 11 GB of allocator
# space at BFCL's batch and the pack's limit is 22.08 (24.0 GB card at headroom 0.92).
#
#   ./queue_s55_ladder.sh a    # R1 + R2            -- 11 + 11        = 22.0
#   ./queue_s55_ladder.sh b    # R3 + B1p + B4p     -- 11 + 5.5 + 5.5 = 22.0
#
# WHY CARD B CARRIES PASSENGERS. R3 alone would leave half a card idle for two and a half
# hours. B1 and B4 probes-only are the two rows `queue_baselines_probes.sh hf` was written to
# queue, they are five minutes of generation each, and they release their slices long before
# R3's tail. They are also not optional for this substage: the R arms are being scored WITH
# the 138-item corpus clean arm, and a false-flag rate on 168 clean items can only be read
# against a base model measured on the same 168. Running them here rather than in their own
# pass costs nothing and removes a card-hour from the s6 list.
#
# ISOLATION. Each arm is a child process under its own `set_per_process_memory_fraction`
# ceiling, so a passenger that dies takes nothing else with it -- R3's number survives B4
# failing, and B4 is the only arm here on a vendor whose loader this project has run once.
#
# PROMOTION AFTERWARDS, and the prefixes differ on purpose:
#
#   python3 ../s5-compare/promote_scores.py <job-a> --arms R1,R2
#   python3 ../s5-compare/promote_scores.py <job-b> --arms R3
#   python3 ../s5-compare/promote_scores.py <job-b> --arms B1,B4 \
#       --prefix tidepool/s5.2 --include scored_probes.jsonl
#
# The baselines go to `tidepool/s5.2`. Promoting them to the sweep prefix would overwrite
# `tidepool/s5.3/arms/B1/scored_probes.jsonl`, which is the replay-derived file the recorded
# `c1add251` comparison was computed from.
set -e
MODE=${1:?usage: queue_s55_ladder.sh a|b}
TASK=08b4e028-b3b7-45a5-9bed-9f487c9c95ed
CLEAN=tidepool/s5.3/tooldata/clean_control.jsonl

case "$MODE" in
a)
  ARMS=R1,R2
  GB=11,11
  OV=$(cat <<'JSON'
{"R1": {"adapter_object": "tidepool/s5.3/arms/R1/adapter.zip"},
 "R2": {"adapter_object": "tidepool/s5.3/arms/R2/adapter.zip"}}
JSON
)
  NOTE="R1 (0.05 dose, reweighted buffer) and R2 (0.20 dose, proportional buffer)."
  ;;
b)
  ARMS=R3,B1,B4
  GB=11,5.5,5.5
  OV=$(cat <<'JSON'
{"R3": {"adapter_object": "tidepool/s5.3/arms/R3/adapter.zip"},
 "B1": {"adapter_object": "", "components": "probes",
        "base_model": "LiquidAI/LFM2.5-1.2B-Instruct"},
 "B4": {"adapter_object": "", "components": "probes",
        "base_model": "ibm-granite/granite-4.0-1b"}}
JSON
)
  NOTE="R3 (0.20 nominal dose, reweighted buffer) full pass, plus the B1 and B4 probes rows
riding the spare 11 GB so the corpus clean arm is measured on the base model and on the
same-size competitor."
  ;;
*) echo "mode must be a or b, got $MODE" >&2; exit 2 ;;
esac

# Check the pack before spending a card on it. A malformed override does not fail loudly: the
# supervisor merges what it can and the arm scores the wrong weights, which looks like a result.
python3 - "$OV" "$ARMS" "$GB" <<'CHECK'
import json, sys
ov = json.loads(sys.argv[1])
arms = [a for a in sys.argv[2].split(",") if a]
gb = [float(x) for x in sys.argv[3].split(",") if x.strip()]
assert sorted(ov) == sorted(arms), "pack_overrides names %s, arms are %s" % (sorted(ov), arms)
assert len(gb) == len(arms), "pack_gb has %d entries for %d arms" % (len(gb), len(arms))
limit = 24.0 * 0.92
assert sum(gb) <= limit + 1e-6, "pack peaks at %.2f GB against a %.2f GB limit" % (sum(gb), limit)
for a, v in sorted(ov.items()):
    ad = v.get("adapter_object", "")
    if ad:
        assert ad == "tidepool/s5.3/arms/%s/adapter.zip" % a, "%s: adapter path %r" % (a, ad)
    else:
        assert v.get("components") == "probes", (
            "%s has no adapter and is not a probes row -- it would score the base model "
            "on all four components under an arm name" % a)
        assert v["base_model"].count("/") == 1, "%s: base_model %r" % (a, v["base_model"])
print("pack ok: %s at %s GB, %.2f of %.2f" % (", ".join(arms), sys.argv[3], sum(gb), limit))
CHECK

lab task queue "$TASK" -e tidepool --no-interactive --provider aws \
  -p arms="$ARMS" \
  -p pack_gb="$GB" \
  -p run_tag=s5.5-ladder \
  -p limit_per_component=0 \
  -p limit_probes=0 \
  -p clean_control_object="$CLEAN" \
  -p pack_overrides="$OV" \
  -m "s5.5 replay-ladder scoring, card $MODE: $ARMS. $NOTE
All four components at full item counts on one L4, the same card, batch size, item counts and
component list every s5.2 baseline and s5.3 sweep arm was measured on. The one difference from
the C arms is clean_control_object: the R arms carry the 138-item corpus clean arm as well as
the frozen 30, which is additive (test_probes_additive.py) and leaves every pre-existing
summary key meaning what it meant when B1 through B6 were measured."
