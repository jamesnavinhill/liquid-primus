#!/usr/bin/env bash
# Stage the packed evaluation task: this directory's harness plus the sweep's supervisor.
#
# `pack.py` is not copied into the source tree here on purpose. There is one supervisor in
# this project and it lives with the sweep; a second copy would drift, and the drift would
# show up as two packs that schedule differently on the same config. It is fetched at build
# time instead, and the contract tests that guard it are run from where they live.
set -e
SRC=$(cd "$(dirname "$0")" && pwd)
SWEEP="$SRC/../s5-sft-sweep"
DST=${1:-/tmp/s5-eval-pack}
rm -rf "$DST"
mkdir -p "$DST"
for f in "$SRC"/*.py; do
  case "$(basename "$f")" in
    test_*) continue ;;
  esac
  cp "$f" "$DST/"
done
cp "$SWEEP/pack.py" "$DST/pack.py"
cp "$SRC/pack.yaml" "$DST/task.yaml"
python3 -m py_compile "$DST"/*.py
# The contract tests read main.py, replay.py and ../s5-eval/main.py out of the source tree,
# so run them from the sweep: a staged directory that compiles but breaks the packing
# contract is exactly the failure this arrangement exists to prevent.
python3 "$SWEEP/test_pack_isolation.py" >/dev/null
python3 "$SWEEP/test_pack_schedule.py" >/dev/null
python3 "$SRC/test_resolve_adapter.py" >/dev/null
python3 "$SRC/test_resolve_gguf.py" >/dev/null
# The serving loop, against a fake model that refuses any batch above a set size: batch
# halving on out-of-memory, and the calibration forward pass that reads one position and
# generates nothing. Both are fixture-only.
(cd "$SRC" && python3 test_gen_oom_split.py) >/dev/null
# The guardrail calibration scalar: rank statistic, operating-point search, reliability
# bins and the yes/no token resolution. All fixture-only, no model, so it is a build check.
(cd "$SRC" && python3 test_probes_calib.py) >/dev/null
# The memorization check: the bounded edit distance against an unbounded reference, Peak and
# its length cap, and the stratum matching that pairs a contaminated eval item with a clean
# sibling. One of these fixtures asserts that a rigid schema alone drives peakedness to 1.0
# with no contamination anywhere, which is the reason the reported figure is a paired
# difference and not the raw number.
(cd "$SRC" && python3 test_cdd.py) >/dev/null
# The serving path has to LOAD before an arm may serve: the first s5.6 quality pass lost
# six arms to a missing CUDA shared object at startup, so the repair and the whole-function
# unbound-name scan of load_gguf are build-time checks like the rest.
(cd "$SRC" && python3 -m unittest -q -b test_loader_repair) >/dev/null 2>&1
rm -rf "$DST/__pycache__"
echo "contract and scheduling checks pass"
echo "built $DST from $SRC + $SWEEP/pack.py:"
ls "$DST"
