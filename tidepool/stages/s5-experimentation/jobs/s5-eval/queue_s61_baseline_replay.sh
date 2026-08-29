#!/bin/sh
# s6.1 -- put every baseline row back on the current graders, one card, four arms, no GPU work.
#
# WHY THIS IS NOT OPTIONAL. B1's score-only replay at s5.3 moved its own tool-calling composite
# from 0.6700 to 0.6649 and its instruction following from 0.8170 to 0.8189: the graders were
# edited after s5.2 was measured, and every arm from s5.3 onward was scored by the edited ones.
# B2, B3, B4 and B5 were all measured in the same s5.2 window as B1 and have never been
# replayed, so the final table would otherwise difference the finalists against four rows taken
# with a different instrument -- and the pre-registered claim names two of them directly, the
# matched base and the Nova community finetune.
#
# It buys detail as well as comparability. The replay writes per-item scored files, per-category
# tool-calling tables, the structured-output partial-credit score and the instruction-level
# IFEval pair, none of which these rows printed the first time and all of which s6.2 and s6.4
# read.
#
# PACKING FOUR ARMS IS SAFE HERE, and it is worth saying why, because packing four SCORING arms
# on one card is exactly what corrupted two of three arms at s5.6. That failure was compute
# contention between three llama.cpp servers generating tokens. A replay generates nothing and
# loads no weights: it reads saved text, re-renders prompts on CPU, and runs the graders. The
# card is idle throughout and the work is CPU-bound across 16 cores. The ceilings below are
# nominal for that reason.
#
# BASE MODEL PER ARM, AND WHY IT DIFFERS. The replay re-renders every prompt and checks it
# against the hash the original run recorded, so it has to render with the same tokenizer that
# generated the text. Which repo that was depends on the original backend: a llama.cpp row
# rendered with `tokenizer_repo`, a transformers row with `base_model`. B5 is the one that
# diverges -- Nova ships its own chat template, a third the size of the base one -- so its
# replay names the Nova repo. A wrong choice here is loud rather than silent: the hash check
# fires on every item and the run says so.
set -e
TASK=08b4e028-b3b7-45a5-9bed-9f487c9c95ed
OV=$(python3 <<'JSON'
import json
rows = {
    # arm: (completions dir, tokenizer repo that rendered them, extra config)
    "B2": ("tidepool/s5.2/B2", "LiquidAI/LFM2.5-1.2B-Instruct", {}),
    "B3": ("tidepool/s5.2/B3", "LiquidAI/LFM2.5-1.2B-Instruct", {}),
    "B4": ("tidepool/s5.2/B4", "ibm-granite/granite-4.0-1b", {}),
    # B5 alone carries the 138-item clean-corpus arm in its saved completions, because it ran
    # after that edit went up. Unset, the harness would rebuild a probe set without those items
    # and leave 138 completions unscored.
    "B5": ("tidepool/s5.2/B5", "NovachronoAI/LFM2.5-1.2B-Nova-Function-Calling",
           {"clean_control_object": "tidepool/s5.3/tooldata/clean_control.jsonl"}),
}
out = {}
for arm, (obj, tokrepo, extra) in rows.items():
    cfg = {"rescore_object": obj, "base_model": tokrepo,
           "adapter_object": "", "run_tag": "s6.1-replay-%s" % arm}
    cfg.update(extra)
    out[arm] = cfg
print(json.dumps(out, sort_keys=True))
JSON
)
python3 -c "import json,sys; d=json.loads(sys.argv[1]); assert sorted(d)==['B2','B3','B4','B5']; print('overrides parse:', json.dumps(d, indent=1))" "$OV"

lab task queue "$TASK" -e tidepool --no-interactive --provider aws \
  -p arms=B2,B3,B4,B5 \
  -p pack_gb=4,4,4,4 \
  -p pack_headroom=0.92 \
  -p pack_overrides="$OV" \
  -p stagger_seconds=20 \
  -p components=bfcl,ifstruct,ifeval,probes \
  -p bfcl_styles=tools_text,native_tools \
  -p limit_per_component=0 \
  -p limit_bfcl=0 -p limit_ifstruct=0 -p limit_ifeval=0 -p limit_probes=0 \
  -p profile=full \
  -p run_tag=s6.1-baseline-replay \
  -m 'Score-only replay of the four s5.2 baseline rows through the current graders, four arms on one L4, no weights loaded and no tokens generated. B1s replay at s5.3 moved its own composite by half a point, which is direct evidence the graders were edited after s5.2 was measured; B2, B3, B4 and B5 have never been replayed, so the s6 table would otherwise difference the finalists against rows taken on a different instrument, including the two the pre-registered claim names by hand. Each arm re-renders its prompts with the tokenizer that generated them, which is the base repo for the transformers row (B4, Granite) and the serving tokenizer for the llama.cpp rows (B5 alone diverges, on Novas own chat template). Packing four is safe here in a way it was not at s5.6: a replay generates nothing, so there is no compute contention on the card. Also recovers per-category tables, structured-output partial credit and the instruction-level IFEval pair, which these rows never printed.'
