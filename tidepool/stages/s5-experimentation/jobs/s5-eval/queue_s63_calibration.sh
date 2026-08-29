#!/bin/sh
# s6.3 -- is the guardrail's blind spot a threshold or an absent signal? One card, three arms.
#
# THE QUESTION s6.2 LEFT OPEN. The deployed guardrail is a regex family over free text: one
# operating point and no score behind it. Split by corruption, it turned out to be a
# structural-integrity detector -- at or near ceiling on the seven modes that leave the
# payload malformed, and near zero on the two that do not. R3-F16 flags 1.000 of silently
# truncated bodies and 0.067 of stale as-of dates. From the flag alone those two numbers are
# unreadable: either the model cannot tell a stale date from a fresh one, or it can and the
# single threshold the regex encodes sits above wherever it puts them. The first is a data
# problem for s7 and beyond; the second is a knob, and a deployment could turn it today.
#
# WHAT THIS MEASURES. Each tool-return probe's own messages are replayed with one frozen
# auditor turn appended, and the first token's probability mass on "yes" against "no" becomes
# a suspicion score, p_no / (p_yes + p_no). One forward pass per item, nothing sampled,
# nothing decoded, so the number cannot depend on a stopping rule. From it: AUROC over the
# 270 defective items against the 138-item clean corpus, the same AUROC per corruption mode
# against that shared clean pool, detection rate at four false-alarm budgets, ECE with
# reliability bins, and the scalar's AUROC against each arm's OWN recorded regex flags --
# which is what separates "cannot see it" from "sees it and does not say it".
#
# IT IS A SECOND INSTRUMENT, NOT THE DEPLOYED ONE. A teacher-forced verdict on a direct
# question is an easier task than volunteering a warning mid-answer, so a high AUROC here
# does NOT license claiming the shipped guardrail catches those modes. What it licenses is
# the opposite direction: an AUROC at chance would close the threshold hypothesis outright.
#
# THREE ARMS AND WHY THESE THREE. B1 is the untrained base, which flags essentially nothing;
# without it a high AUROC on R3 could not be attributed to the training rather than to the
# base model already carrying the signal. R3 is the named configuration and C7 the runner-up.
# All three are full precision on transformers, which is the only backend that exposes token
# probabilities and is by plan the one every FP number in this project comes from. A 4-bit
# arm raises rather than quietly reporting a curve taken on different numerics.
#
# THE FREE-FORM FLAGS ARE READ, NOT REGENERATED. `free_form_scored_object` points each arm at
# the scored probe file the sweep already wrote, so the scalar is ranked against the flags
# s6.1 and s6.2 actually published. Regenerating 602 completions per arm to recover them
# would cost the whole run and answer a slightly different question, because the free-form
# pass samples.
#
# PACKING THREE IS SAFE. The s5.6 collapse was compute contention between llama.cpp servers
# generating tokens; this generates none. Each arm holds a 1.2B model in bf16 and runs one
# prefill per batch, and the head is restricted to the last position so the logits tensor is
# batch x 1 x vocab rather than batch x sequence x vocab. 7 GB ceilings on a 24 GB L4 with
# 0.92 headroom leaves each arm roughly three times what it needs.
set -e
TASK=08b4e028-b3b7-45a5-9bed-9f487c9c95ed
OV=$(python3 <<'JSON'
import json
arms = {
    # arm: adapter object ("" = the untrained base)
    "B1": "",
    "R3": "tidepool/s5.3/arms/R3/adapter.zip",
    "C7": "tidepool/s5.3/arms/C7/adapter.zip",
}
out = {}
for arm, adapter in arms.items():
    out[arm] = {
        "adapter_object": adapter,
        "free_form_scored_object": "tidepool/s5.3/arms/%s/scored_probes.jsonl" % arm,
        "run_tag": "s6.3-calibration-%s" % arm,
    }
print(json.dumps(out, sort_keys=True))
JSON
)
python3 -c "import json,sys; d=json.loads(sys.argv[1]); assert sorted(d)==['B1','C7','R3']; assert d['B1']['adapter_object']=='' ; print('overrides parse:', json.dumps(d, indent=1))" "$OV"

lab task queue "$TASK" -e tidepool --no-interactive --provider aws \
  -p arms=B1,R3,C7 \
  -p pack_gb=7,7,7 \
  -p pack_headroom=0.92 \
  -p pack_overrides="$OV" \
  -p stagger_seconds=20 \
  -p components=calibration \
  -p clean_control_object=tidepool/s5.3/tooldata/clean_control.jsonl \
  -p probes_object=tidepool/s4.4/probes/probes.jsonl \
  -p backend=hf \
  -p base_model=LiquidAI/LFM2.5-1.2B-Instruct \
  -p batch_size=8 \
  -p limit_per_component=0 \
  -p limit_calibration=0 \
  -p profile=full \
  -p run_tag=s6.3-calibration \
  -m 'Calibration of the tool-return guardrail on the base model and the two finalists, three arms on one L4, one forward pass per probe and no tokens generated. s6.2 split the flag by corruption and found a structural-integrity detector: at or near ceiling on the seven modes that leave the payload malformed, near zero on wrong_entity and stale_as_of, the two that do not. The flag is a regex family over free text with one operating point and no score behind it, so those two numbers cannot be read as either a high threshold or an absent signal. This appends one frozen auditor turn to each probes own messages and reads the first tokens probability mass on yes against no, giving AUROC over 270 defective items against the 138-item clean corpus, the same per corruption mode, detection rate at four false-alarm budgets, ECE with reliability bins, and the scalars AUROC against each arms own recorded regex flags, which separates cannot see it from sees it and does not say it. B1 is in so a high AUROC on R3 can be attributed to the training rather than to the base already carrying the signal. Each arm reads its recorded scored probe file rather than regenerating 602 completions to recover the flags. A teacher-forced verdict is an easier task than volunteering a warning mid-answer, so this cannot license a claim about the shipped guardrail; chance-level AUROC would close the threshold hypothesis outright.'
