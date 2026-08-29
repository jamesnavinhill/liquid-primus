#!/usr/bin/env bash
# Stage the s6.4 peakedness comparison: the driver and nothing else.
#
# The fixtures stay behind and run first. The failure this job is exposed to is a statistic
# that looks fine and answers a different question: a pairing that silently drops the
# duplicated BFCL id, a sign test that counts ties as evidence, a permutation test whose
# p-value can reach zero. Each of those is a fixture, chosen so that making the mistake moves
# a number the fixture checks.
set -e
SRC=$(cd "$(dirname "$0")" && pwd)
DST=${1:-/tmp/s6-cdd}
PY=${PYTHON:-python3}
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
