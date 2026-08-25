#!/bin/sh
# Build the s5-sft-pack task directory from this one.
#
# The packed task and the solo task run THE SAME main.py -- that is the point, and it is what
# lets a packed arm be compared against C1, which ran solo. Two registered tasks need two
# directories, so the staging dir is generated here rather than maintained by hand; a drifted
# copy would silently make the sweep un-paired.
#
# replay.py is the same arrangement for the self-distillation generator: the pack runs it as
# an arm so the buffer is produced on the card that is about to consume it, and the contract
# test checks that it and main.py still share one packing preamble.
#
#   ./sync_pack_task.sh            build /tmp/s5-sft-pack
#   ./sync_pack_task.sh <dir>      build somewhere else
#
# Then: lab task add <dir> -e tidepool --no-interactive
# Or, to update an existing registration: lab task edit <id> -e tidepool --from-dir <dir>
set -e
SRC=$(cd "$(dirname "$0")" && pwd)
DST=${1:-/tmp/s5-sft-pack}
rm -rf "$DST"
mkdir -p "$DST"
for f in main.py sample.py textify.py pack.py replay.py; do
  cp "$SRC/$f" "$DST/$f"
done
cp "$SRC/pack.yaml" "$DST/task.yaml"
python3 -m py_compile "$DST"/*.py
# The contract tests read main.py, replay.py and pack.py out of the source directory, so run
# them here: a staged directory that compiles but breaks the packing contract is the failure
# this whole arrangement exists to prevent.
python3 "$SRC/test_pack_isolation.py" >/dev/null
python3 "$SRC/test_pack_progress.py" >/dev/null
python3 "$SRC/test_pack_schedule.py" >/dev/null
rm -rf "$DST/__pycache__"
echo "contract, progress and scheduling checks pass"
echo "built $DST from $SRC:"
ls -l "$DST"
