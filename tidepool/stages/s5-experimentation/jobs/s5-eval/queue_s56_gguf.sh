#!/usr/bin/env bash
# s5.6 G1, quality half — the six exported GGUFs scored on the same harness, the same items
# and the same card every full-precision row in this project was measured on.
#
#   ./queue_s56_gguf.sh r3            # R3-F16 + R3-Q4_0 + R3-Q4_K_M, one job each
#   ./queue_s56_gguf.sh c7            # C7-F16 + C7-Q4_0 + C7-Q4_K_M, one job each
#   ./queue_s56_gguf.sh r3 Q4_0,Q4_K_M  # just these two -- e.g. resuming after a 2-GPU
#                                        # account cap left one checkpoint's arms partly
#                                        # queued, or relaunching one arm that failed alone
#
# WHY THREE POINTS PER CHECKPOINT AND NOT TWO. A Q4-against-FP delta taken across
# `transformers` and `llama.cpp` is a delta between two deployed artifacts -- weights and
# runtime together -- and H4 is a claim about quantization. The F16 GGUF row is the same
# weights at full precision on the 4-bit runtime, so (FP -> F16) is the runtime's cost and
# (F16 -> Q4) is quantization's. The project already paid for this distinction once, at s5.2,
# where it split B2's shortfall into a runtime half and a quantization half.
#
# WHY ONE ARM PER CARD NOW, NOT THREE. This job used to pack all three arms onto one L4
# concurrently: three llama-servers, each `--parallel 8`, sharing one card. `pack_gb`
# ceilings the arms' *VRAM* (8, 4, 4 GB of 24), but a 4-bit arm's serving cost is compute, not
# memory -- `set_per_process_memory_fraction` cannot ceiling a llama.cpp child at all, and
# nothing in the pack supervisor throttles concurrent GPU compute. Attempt 2 (job 46bd54cb,
# 2026-08-28) proved that gap real: three 8-slot servers contending for one L4's SMs collapsed
# per-token throughput from ~18 tok/s to 0.28 tok/s over the run, individual HTTP requests to
# the two arms that shared the busiest window timed out, and `gen_gguf.py`'s request handler
# turned each timeout into a silent empty-string completion (logged only for the first 5
# failures) rather than a retry or an abort -- so R3-F16 and R3-Q4_0 finished reporting
# `completion_status: success` with 0/3490, 0/2000, 0/602 non-empty completions on three of
# four components, while R3-Q4_K_M, which ran alone for most of the job, was clean. A job that
# looks like a result and is not, is worse than a slower job.
#
# So the fix moves the isolation the pack supervisor already promises for memory and
# filesystem (see pack.py's docstring) up one level, to compute: one arm gets the whole card,
# every time. Each format is now its own `lab task queue` call using the same task, same
# weights, same item counts, same 8 slots x 8192 context serving shape B2, B3 and B5 were
# measured on, same clean-control arm -- only the packing arrangement changes. The prior
# rationale for packing (packing this checkpoint's three arms took ~6 card-hours against ~14
# unpacked) bought GPU-hours at the cost of exactly the failure mode above; three unpacked
# per-checkpoint jobs cost more wall clock but no more genuine ambiguity than a job dying
# outright, and the checkpoint's retention set is still complete once all three land.
#
# PROMOTION AFTERWARDS (one job per arm now):
#   python3 ../s5-compare/promote_scores.py <job-F16> --arms R3-F16 --prefix tidepool/s5.6/arms
#   python3 ../s5-compare/promote_scores.py <job-Q4_0> --arms R3-Q4_0 --prefix tidepool/s5.6/arms
#   python3 ../s5-compare/promote_scores.py <job-Q4_K_M> --arms R3-Q4_K_M --prefix tidepool/s5.6/arms
set -e
MODE=${1:?usage: queue_s56_gguf.sh r3|c7}
TASK=08b4e028-b3b7-45a5-9bed-9f487c9c95ed
CLEAN=tidepool/s5.3/tooldata/clean_control.jsonl
LLAMA=tidepool/llama-b10622-sm89.tar.gz

case "$MODE" in
r3) CK=R3 ;;
c7) CK=C7 ;;
*) echo "mode must be r3 or c7, got $MODE" >&2; exit 2 ;;
esac

# Q4_K_L and Q4_K_XL are the s5.6 G4a recovery arms: a Q4_K_M body with the token-embedding and
# output tensors held at q6_K and q8_0 respectively. They are a few tens of MB larger than
# Q4_K_M on disk, so 5 GB is the same generous ceiling with the same headroom.
declare -A GB_FOR=( [F16]=8 [Q4_0]=4 [Q4_K_M]=4 [Q4_K_L]=5 [Q4_K_XL]=5 )
FORMATS=${2:-F16,Q4_0,Q4_K_M}

for FMT in ${FORMATS//,/ }; do
  if [ -z "${GB_FOR[$FMT]:-}" ]; then
    echo "unknown format $FMT, must be one of ${!GB_FOR[*]}" >&2
    exit 2
  fi
  ARM="$CK-$FMT"
  GB="${GB_FOR[$FMT]}"
  OV=$(python3 - "$CK" "$FMT" <<'JSON'
import json, sys
ck, fmt = sys.argv[1], sys.argv[2]
print(json.dumps({
    "%s-%s" % (ck, fmt): {
        "adapter_object": "",
        "gguf_object": "tidepool/s5.6/%s/%s-%s.gguf" % (ck, ck, fmt),
        "run_tag": "s5.6-%s-%s" % (ck, fmt),
    }
}, sort_keys=True))
JSON
  )

  # Check the pack before spending a card on it. A malformed override does not fail loudly:
  # the supervisor merges what it can and the arm scores the wrong file, which looks like a
  # result -- and here the label IS the retention number's attribution, so a mislabelled arm
  # is worse than a failed one.
  python3 - "$OV" "$ARM" "$GB" "$CK" <<'CHECK'
import json, sys
ov = json.loads(sys.argv[1])
arm = sys.argv[2]
gb = float(sys.argv[3])
ck = sys.argv[4]
assert sorted(ov) == [arm], "pack_overrides names %s, arm is %s" % (sorted(ov), arm)
limit = 24.0 * 0.92
assert gb <= limit + 1e-6, "arm alone peaks at %.2f GB against a %.2f GB limit" % (gb, limit)
fmt = arm.split("-", 1)[1]
v = ov[arm]
assert v["gguf_object"] == "tidepool/s5.6/%s/%s-%s.gguf" % (ck, ck, fmt), \
    "%s serves %r, which is not the file its name claims" % (arm, v["gguf_object"])
assert v["adapter_object"] == "", "%s sets an adapter; the gguf backend refuses one" % arm
assert "__" not in arm, "%s would not survive the artifact prefix strip" % arm
assert "gguf_repo" not in v and "gguf_file" not in v, \
    "%s sets a Hub source as well as a stored one; resolve_gguf refuses both" % arm
print("pack ok: %s alone at %.1f of %.2f GB" % (arm, gb, limit))
CHECK

  lab task queue "$TASK" -e tidepool --no-interactive --provider aws \
    -p arms="$ARM" \
    -p pack_gb="$GB" \
    -p backend=gguf \
    -p llama_object="$LLAMA" \
    -p tokenizer_repo=LiquidAI/LFM2.5-1.2B-Instruct \
    -p run_tag=s5.6-export-quality \
    -p limit_per_component=0 \
    -p limit_probes=0 \
    -p clean_control_object="$CLEAN" \
    -p pack_overrides="$OV" \
    -m "s5.6 G1 quality half, $ARM: one exported serving point for one checkpoint, alone on
one L4. Queued as a solo arm (attempt 3) after job 46bd54cb showed that packing three
concurrent llama-servers on one L4 collapses generation throughput under compute
contention and turns request timeouts into silent empty completions rather than a loud
failure -- see the queue script's header for the full account. Same weights, same item
counts, same component list, same clean-control arm, same 8 slots x 8192 context serving
shape B2, B3 and B5 were measured on; only the packing arrangement changed.

Retention is read against $CK's own full-precision row, never across checkpoints and never
against the base. Pre-registered before the numbers: if $CK-F16 scores identically to the
other checkpoint's F16 row across all axes, the two merges produced the same weights and the
export is wrong, not the retention."
done
