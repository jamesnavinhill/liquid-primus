#!/usr/bin/env bash
# Stage the s6.4 error-analysis task: the driver and nothing else.
#
# The tests stay behind. They are the reason the macro arithmetic can be trusted -- the whole
# grader-edit finding is that one item explains a composite move to four decimal places, and a
# formula dividing by the wrong count would reconcile with itself and with nothing else -- and
# they run here, before the staging directory exists, so a broken tabulation never reaches a job.
set -e
SRC=$(cd "$(dirname "$0")" && pwd)
DST=${1:-/tmp/s6-errors}
SUMMARY=$(cd "$SRC" && python3 test_main.py | tail -1)
rm -rf "$DST"
mkdir -p "$DST"
cp "$SRC/main.py" "$DST/"
cp "$SRC/task.yaml" "$DST/task.yaml"
python3 -m py_compile "$DST"/*.py
rm -rf "$DST/__pycache__"
echo "$SUMMARY"
echo "built $DST from $SRC:"
ls "$DST"
