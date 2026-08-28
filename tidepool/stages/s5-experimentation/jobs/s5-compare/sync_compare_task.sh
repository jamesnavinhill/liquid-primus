#!/usr/bin/env bash
# Stage the comparison task: the driver, the statistics, and nothing else.
#
# The tests and `promote_scores.py` stay behind on purpose. The tests are the reason this
# task can be trusted and they are run here, before the staging directory is built, so a
# broken statistic never reaches a job; shipping them would only make the job import a test
# harness it has no use for. `promote_scores.py` is orchestration that runs in the sandbox
# and talks to the CLI, which is exactly what a job cannot do.
set -e
SRC=$(cd "$(dirname "$0")" && pwd)
DST=${1:-/tmp/s5-compare}
python3 "$SRC/test_stats.py" >/dev/null
python3 "$SRC/test_main.py" >/dev/null
python3 "$SRC/test_retention.py" >/dev/null
rm -rf "$DST"
mkdir -p "$DST"
cp "$SRC/main.py" "$SRC/stats.py" "$DST/"
cp "$SRC/task.yaml" "$DST/task.yaml"
python3 -m py_compile "$DST"/*.py
rm -rf "$DST/__pycache__"
echo "statistics, driver and retention checks pass"
echo "built $DST from $SRC:"
ls "$DST"
