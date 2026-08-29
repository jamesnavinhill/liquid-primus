#!/usr/bin/env bash
# Stage the s6.4 decontamination re-check: the driver and nothing else.
#
# The fixtures stay behind and run first. Every number this job can report is a count of
# overlaps, and the failure mode that matters is a clean-looking zero produced by a rule that
# was never going to match anything -- an off-by-one on the window length, a normalizer that
# leaves case in, an index that failed to build. Each of those is a fixture here, chosen so
# that making the mistake changes a number the fixture checks.
set -e
SRC=$(cd "$(dirname "$0")" && pwd)
DST=${1:-/tmp/s6-decon}
PY=${PYTHON:-$HOME/.venvs/s6dec/bin/python}
SUMMARY=$(cd "$SRC" && "$PY" test_main.py | tail -1)
rm -rf "$DST"
mkdir -p "$DST"
cp "$SRC/main.py" "$DST/"
cp "$SRC/task.yaml" "$DST/task.yaml"
"$PY" -m py_compile "$DST"/*.py
rm -rf "$DST/__pycache__"
echo "$SUMMARY"
echo "built $DST from $SRC:"
ls "$DST"
