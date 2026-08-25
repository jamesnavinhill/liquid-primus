"""Fixture test for the pack supervisor's progress reading.

A pack runs for hours, so its progress bar is the only thing anyone watching can see. The
reading is done by tailing each child's console for its last `step i/n` line, and the ways
that goes wrong are all silent: a console that does not exist yet, a child that has printed
nothing, a crashed child holding the mean down forever, or a regex that matches the wrong
number and pins the bar at 100% an hour in.

pack.py talks to the job API at import, so the function is lifted out of the source and run
against temporary consoles instead.
"""

import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "pack.py")).read()

start = SRC.index("STEP_RE = re.compile")
end = SRC.index("done = {}\nlast_report")
ns = {"os": os, "re": re}
exec(compile(SRC[start:end], "<pack-progress>", "exec"), ns)

tmp = tempfile.mkdtemp()
ns["OUT"] = tmp
fails = []


def console(arm, text):
    d = os.path.join(tmp, arm)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "console.log"), "w") as fh:
        fh.write(text)


def check(label, arms, done, want_frac, want_live=None):
    ns["ARMS"], ns["done"] = arms, done
    frac, live = ns["progress_fraction"]()
    if abs(frac - want_frac) > 1e-6:
        fails.append("%s: fraction %.4f, expected %.4f" % (label, frac, want_frac))
    if want_live is not None and live != want_live:
        fails.append("%s: live %r, expected %r" % (label, live, want_live))


# 1. nothing has started: no consoles at all
check("no consoles yet", ["C1", "C4"], {}, 0.0, [])

# 2. one arm mid-training, one still loading the model
console("C1", "loading model\nstep 210/8416  loss 0.4  1 tokens  5000 tok/s\n")
console("C4", "loading LiquidAI/LFM2.5-1.2B-Instruct\n")
check("one training, one loading", ["C1", "C4"], {},
      (210 / 8416 + 0.0) / 2, ["C1 210/8416"])

# 3. the LAST step line wins, not the first -- an earlier line would pin the bar low
console("C1", "step 1/8416\nstep 210/8416\nstep 4208/8416\n")
check("last line wins", ["C1"], {}, 0.5, ["C1 4208/8416"])

# 4. a crashed arm counts as complete whatever its console said, so one failure cannot hold
#    the whole pack's bar down for the remaining hours
console("C3", "step 12/8416\nTraceback (most recent call last):\nRuntimeError: CUDA OOM\n")
check("crashed arm does not hold the bar", ["C1", "C3"], {"C3": 1}, (0.5 + 1.0) / 2,
      ["C1 4208/8416"])

# 5. all done
check("all done", ["C1", "C3"], {"C1": 0, "C3": 1}, 1.0, [])

# 6. noise that looks like a step line but is not, and a genuine one after it
console("C6", "resuming from step 900/900 checkpoint\nstep 7/8416  loss 1.1\n")
check("real line after lookalike", ["C6"], {}, 7 / 8416, ["C6 7/8416"])

# 7. an unreadable console must not take the supervisor down
d = os.path.join(tmp, "C7")
os.makedirs(d, exist_ok=True)
os.mkdir(os.path.join(d, "console.log"))          # a directory where a file should be
check("unreadable console is survivable", ["C7"], {}, 0.0, [])

shutil.rmtree(tmp, ignore_errors=True)
if fails:
    print("FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("pack progress reading holds across 7 cases")
