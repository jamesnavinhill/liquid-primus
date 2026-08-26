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
rm -rf "$DST/__pycache__"
echo "contract and scheduling checks pass"
echo "built $DST from $SRC + $SWEEP/pack.py:"
ls "$DST"
