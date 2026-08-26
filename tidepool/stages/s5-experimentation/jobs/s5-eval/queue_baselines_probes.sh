#!/usr/bin/env bash
# The corpus clean arm, added to the s5.2 baseline rows. Probes only, packed by backend.
#
# WHY. The 138-item corpus clean arm was built at s5.3, after B1 through B6 were measured, and
# wiring it in mid-flight would have put the baseline rows on two versions of the harness. So
# every published baseline carries only the 30 synthetic clean items, which bound a false-flag
# rate to about one item in thirty. The sweep arms get the wider arm from
# `queue_arms_probes.sh`; this is the same pass over the baselines, so an arm's false-flag rate
# can be read against the base model's on the same items rather than across two clean sets.
#
# It is supplementary. The s5.4 decision rule is written against sweep arms paired with C1 and
# does not read a baseline row, so nothing here blocks the direction decision. It matters for
# s6, where the reported guardrail numbers are compared to the published model's.
#
# COST. 434 probe items plus 138 corpus items is 572 an arm, and the measured full-count rate
# is about 130 items a minute an arm, so an arm is under five minutes of generation. The six
# rows together are well inside one card-hour including model loads. Packing is for tidiness
# here, not for savings.
#
# TWO PASSES, BY BACKEND, AND THE REASON IS THE CEILING. `pack_gb` is enforced with
# `torch.cuda.set_per_process_memory_fraction`, which binds an arm that allocates through torch
# and does nothing at all to a llama.cpp server in its own process. The transformers rows can
# therefore be packed under real ceilings, and the 4-bit rows are sized by their context and
# slot count instead: `gguf_parallel` times `gguf_ctx_per_slot` is the KV cache each server
# reserves, and 4 slots of 4,096 is ample for a probe item whose prompt is a tool return and
# whose answer is capped at 320 new tokens. Ports do not collide: each child adds its position
# in the pack to `gguf_port`.
#
#   ./queue_baselines_probes.sh hf     # B1, B4      -- transformers, real ceilings
#   ./queue_baselines_probes.sh gguf   # B1r, B2, B3, B5 -- llama.cpp, sized by context
#
# PROMOTION. Baselines do not live with the sweep arms, so the destination has to be said:
#
#   python3 ../s5-compare/promote_scores.py <job> --arms B1,B4 \
#       --prefix tidepool/s5.2 --include scored_probes.jsonl
#
# There is nothing at that path yet for these rows, so no --force and no superseded copy: the
# baselines' original probe verdicts are inside their own jobs' artifacts and stay there.
set -e
MODE=${1:?usage: queue_baselines_probes.sh hf|gguf}
TASK=08b4e028-b3b7-45a5-9bed-9f487c9c95ed
LLAMA=tidepool/llama-b10622-sm89.tar.gz
LFM=LiquidAI/LFM2.5-1.2B-Instruct

case "$MODE" in
hf)
  # B1 is the reference row: the published full-precision instruct model, transformers path.
  # B4 is Granite 4.0-1b, a different vendor's model of the same size, on the same source B1
  # ran on. Both at 5.5 GB, the measured probes-only figure.
  ARMS=B1,B4
  GB=5.5,5.5
  OV=$(cat <<'JSON'
{"B1": {"base_model": "LiquidAI/LFM2.5-1.2B-Instruct"},
 "B4": {"base_model": "ibm-granite/granite-4.0-1b"}}
JSON
)
  EXTRA=(-p backend=hf)
  ;;
gguf)
  # Four llama.cpp rows. B1r is the runtime reference at bf16 -- same weights as B1, same
  # server, no quantization -- so a B2-minus-B1r gap is quantization alone and a B2-minus-B1
  # gap also contains the runtime. B2 is the vendor's quantization-aware build, B3 the best
  # cheap community quant of the same weights, B5 a competitor's function-calling fine-tune
  # on its own template, which is why B5 alone overrides the tokenizer.
  ARMS=B1r,B2,B3,B5
  GB=""
  OV=$(cat <<'JSON'
{"B1r": {"gguf_repo": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
         "gguf_file": "LFM2.5-1.2B-Instruct-BF16.gguf",
         "tokenizer_repo": "LiquidAI/LFM2.5-1.2B-Instruct"},
 "B2":  {"gguf_repo": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
         "gguf_file": "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf",
         "tokenizer_repo": "LiquidAI/LFM2.5-1.2B-Instruct"},
 "B3":  {"gguf_repo": "unsloth/LFM2.5-1.2B-Instruct-GGUF",
         "gguf_file": "LFM2.5-1.2B-Instruct-UD-Q4_K_XL.gguf",
         "tokenizer_repo": "LiquidAI/LFM2.5-1.2B-Instruct"},
 "B5":  {"gguf_repo": "NovachronoAI/LFM2.5-1.2B-Nova-Function-Calling-GGUF",
         "gguf_file": "LFM2.5-1.2B-Nova-Function-Calling.Q4_K_M.gguf",
         "tokenizer_repo": "NovachronoAI/LFM2.5-1.2B-Nova-Function-Calling"}}
JSON
)
  EXTRA=(-p backend=gguf -p llama_object="$LLAMA" -p gguf_parallel=4 -p gguf_ctx_per_slot=4096)
  ;;
*) echo "mode must be hf or gguf, got $MODE" >&2; exit 2 ;;
esac

# Validate the overrides before spending a card on them. A malformed value here does not fail
# loudly: the supervisor merges what it can and the arm runs against the wrong weights, which
# looks exactly like a result. `gguf_file` names a file inside the repo, so a slash in it sends
# llama.cpp looking for a subdirectory that is not there.
python3 - "$OV" "$ARMS" <<'CHECK'
import json, sys
ov, arms = json.loads(sys.argv[1]), [a for a in sys.argv[2].split(",") if a]
assert sorted(ov) == sorted(arms), "pack_overrides names %s, arms are %s" % (sorted(ov), arms)
for a, v in sorted(ov.items()):
    if "gguf_file" in v:
        assert "/" not in v["gguf_file"], (
            "%s: gguf_file must be a bare filename, got %r" % (a, v["gguf_file"]))
        for k in ("gguf_repo", "tokenizer_repo"):
            assert v[k].count("/") == 1, "%s: %s must be org/name, got %r" % (a, k, v[k])
    else:
        assert v["base_model"].count("/") == 1, (
            "%s: base_model must be org/name, got %r" % (a, v["base_model"]))
print("overrides check ok for %s" % ", ".join(sorted(ov)))
CHECK
set -- "$TASK" -e tidepool --no-interactive --provider aws \
  -p arms="$ARMS" \
  -p run_tag="s5.2-clean-probes-$MODE" \
  -p components=probes \
  -p limit_per_component=0 \
  -p limit_probes=0 \
  -p probes_object=tidepool/s4.4/probes/probes.jsonl \
  -p clean_control_object=tidepool/s5.3/tooldata/clean_control.jsonl \
  -p base_model="$LFM" \
  -p pack_overrides="$OV" \
  "${EXTRA[@]}"
[ -n "$GB" ] && set -- "$@" -p pack_gb="$GB"
lab task queue "$@" -m "Probes plus the 138-item corpus clean control over the s5.2 baseline
rows ($ARMS), $MODE backend. The control arm was built after these rows were measured, so every
published baseline carries only the 30 synthetic clean items and cannot price a false-flag rate
below about one in thirty. Supplementary to s5.4, which reads sweep arms only; it matters at s6,
where an arm's guardrail numbers are compared to the published model's. Split by backend because
pack_gb is a torch memory fraction and does not bind a llama.cpp server: the transformers rows
run under real ceilings, the 4-bit rows are sized by 4 slots of 4,096 context."
