"""The H4 retention table, over hand-built summaries with the answer known in advance.

Retention is a ratio, and a ratio has two ways to lie that neither the paired test above it
nor the assertion machinery around it will catch. It can be taken against the wrong
denominator -- another checkpoint, or the base model -- which prices the fine-tune and calls
it quantization damage. And it can be read off an axis whose reference rate is near the
floor, where a movement of one absolute point is a movement of ten ratio points and the
verdict is really a statement about sampling noise.

So the fixtures below fix a reference whose rates are known, and check: that every ratio is
taken against the arm's own reference row, that the pre-registered bar is applied exactly as
written rather than softened for the near-floor axis, that the near-floor axis is FLAGGED
rather than dropped, and that a missing or zero rate is skipped with a note instead of
producing a ratio out of nothing.
"""

import json
import os
import sys
import tempfile
import types

ROOT = tempfile.mkdtemp(prefix="s5ret-")
LOGS, ARTIFACTS, FINISHED = [], [], {}
CONFIG = {}


class _Lab:
    def init(self):
        pass

    def get_config(self):
        return dict(CONFIG)

    def log(self, msg):
        LOGS.append(str(msg))

    def storage_download(self, obj):
        raise RuntimeError("no such object: %s" % obj)

    def save_artifact(self, path):
        ARTIFACTS.append(os.path.basename(path))

    def finish(self, message=None, score=None):
        FINISHED["message"] = message


mod = types.ModuleType("lab")
mod.lab = _Lab()
sys.modules["lab"] = mod

os.chdir(ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main                                                        # noqa: E402

FAILED = []
N = 0


def ok(name, cond, detail=""):
    global N
    N += 1
    if not cond:
        FAILED.append("%s %s" % (name, detail))
        print("FAIL %s %s" % (name, detail))
    else:
        print("ok   %s" % name)


def arm(bfcl=None, ifeval=None, ifstruct=None, idiom=None):
    """An arm's merged summary, carrying only the four axes the table names."""
    d = {}
    for key, val in (("bfcl_ast_composite", bfcl), ("ifeval_prompt_strict", ifeval),
                     ("ifstruct_validity", ifstruct), ("probe_stack_idiom", idiom)):
        if val is not None:
            d[key] = val
    return ({}, d)


AXES = dict(main.DEFAULT_RETENTION_AXES)


# --- the real R3 shape: three arms, the reference at full precision -----------------------
# Rates as measured by ba294aa7 / 6d91edf4 / b928b8fc, so the fixture and the run agree.
loaded = {
    "R3-F16": arm(0.7238, 0.7468, 0.086, 0.2361111111111111),
    "R3-Q4_0": arm(0.6795, 0.7320, 0.0735, 0.22916666666666666),
    "R3-Q4_K_M": arm(0.6953, 0.7486, 0.0735, 0.2013888888888889),
}
main.NOTES[:] = []
rows, verdict = main.retention_table(loaded, "R3-F16", AXES, 0.93, 0.97)

ok("every non-reference arm gets every axis", len(rows) == 8, "got %d" % len(rows))
ok("the reference arm is not compared against itself",
   all(r["arm"] != "R3-F16" for r in rows))

by = {(r["arm"], r["axis"]): r for r in rows}

ok("the ratio is the arm's rate over the reference's",
   by[("R3-Q4_0", "bfcl_composite")]["retention"] == round(0.6795 / 0.7238, 6),
   str(by[("R3-Q4_0", "bfcl_composite")]["retention"]))
ok("every row carries the denominator it used",
   all(abs(r["reference_rate"] - loaded["R3-F16"][1][r["path"]]) < 1e-6 for r in rows))
ok("the absolute delta travels with the ratio",
   abs(by[("R3-Q4_0", "ifstruct_validity")]["absolute_delta"] - (0.0735 - 0.086)) < 1e-9)

# A ratio taken against the OTHER checkpoint would price the fine-tune, not the quantization.
# There is no such row: the table only ever divides by the reference passed in.
ok("no row divides by a rate outside the reference arm",
   all(r["reference_rate"] in
       {round(v, 6) for v in loaded["R3-F16"][1].values()} for r in rows))

# --- the bar is applied as pre-registered, including where that is inconvenient ------------
ok("a comfortable axis meets the floor",
   by[("R3-Q4_0", "ifeval_prompt_strict")]["meets_axis_floor"] is True)
ok("the near-floor axis FAILS the floor rather than being exempted",
   by[("R3-Q4_0", "ifstruct_validity")]["meets_axis_floor"] is False,
   "%.4f" % by[("R3-Q4_0", "ifstruct_validity")]["retention"])
ok("the near-floor axis is still IN the table",
   ("R3-Q4_K_M", "ifstruct_validity") in by)
ok("a near-floor axis is flagged as an ill-conditioned ratio",
   by[("R3-Q4_0", "ifstruct_validity")]["ratio_is_well_conditioned"] is False)
ok("a well-based axis is not flagged",
   by[("R3-Q4_0", "bfcl_composite")]["ratio_is_well_conditioned"] is True)
ok("the flag is keyed on the REFERENCE rate, not on who won",
   all(r["ratio_is_well_conditioned"] ==
       (r["reference_rate"] >= main.RETENTION_WELL_CONDITIONED_FLOOR) for r in rows))

# --- the arm verdict ----------------------------------------------------------------------
ok("an arm failing one axis fails overall",
   verdict["R3-Q4_0"]["passes"] is False)
ok("the failing axis is named", verdict["R3-Q4_0"]["failing_axes"] == ["ifstruct_validity"])
ok("the worst axis is reported with the verdict",
   verdict["R3-Q4_0"]["min_axis"] == "ifstruct_validity")
ok("Q4_K_M fails on both of its sub-93% axes",
   verdict["R3-Q4_K_M"]["failing_axes"] == ["ifstruct_validity", "probe_stack_idiom"],
   str(verdict["R3-Q4_K_M"]["failing_axes"]))
ok("the mean is over the axes actually present",
   verdict["R3-Q4_0"]["n_axes"] == 4)

# An arm that clears every axis and the mean passes, so the criterion is not simply always
# false: a gate that can never open is not a gate.
clean = {"F": arm(0.70, 0.70, 0.70, 0.70), "Q": arm(0.69, 0.69, 0.69, 0.69)}
_r2, v2 = main.retention_table(clean, "F", AXES, 0.93, 0.97)
ok("an arm inside the bar passes", v2["Q"]["passes"] is True,
   json.dumps(v2["Q"]))
# ... and the mean alone does not rescue an arm that fails one axis.
# 0.63/0.70 is 90.0%, and three untouched axes carry the mean to 97.5% -- so the mean
# clears its bar while one axis is under its own, which is exactly the case the per-axis
# floor exists for and the case an average-only reading would wave through.
mixed = {"F": arm(0.70, 0.70, 0.70, 0.70), "Q": arm(0.70, 0.70, 0.70, 0.63)}
_r3, v3 = main.retention_table(mixed, "F", AXES, 0.93, 0.97)
ok("a healthy mean does not rescue a failed axis",
   v3["Q"]["meets_mean"] is True and v3["Q"]["passes"] is False, json.dumps(v3["Q"]))

# --- absences are skipped with a note, never invented --------------------------------------
main.NOTES[:] = []
missing = {"F": arm(0.70, 0.70, 0.70, 0.70), "Q": arm(0.69, 0.69, None, 0.69)}
r4, v4 = main.retention_table(missing, "F", AXES, 0.93, 0.97)
ok("a missing axis produces no row", len(r4) == 3, "got %d" % len(r4))
ok("a missing axis is noted", any("ifstruct" in n for n in main.NOTES), str(main.NOTES))
ok("the mean is taken over the axes that exist", v4["Q"]["n_axes"] == 3)

main.NOTES[:] = []
zero = {"F": arm(0.70, 0.70, 0.0, 0.70), "Q": arm(0.69, 0.69, 0.02, 0.69)}
r5, _v5 = main.retention_table(zero, "F", AXES, 0.93, 0.97)
ok("a zero reference rate produces no ratio",
   all(r["axis"] != "ifstruct_validity" for r in r5))
ok("a zero reference rate is noted", any("undefined" in n for n in main.NOTES),
   str(main.NOTES))

# --- the rendered table carries the numbers a reader judges on ----------------------------
summary = {
    "reference": "R3-F16", "arms": list(loaded), "components": {},
    "completions_prefix": "tidepool/s5.6/arms", "resamples": 10, "seed": 1,
    "cells": [], "n_comparisons": 0, "n_separating_by_interval": 0,
    "n_significant_family_corrected": 0, "notes": [], "assertion_failures": [],
    "method": "m",
    "retention": {"axes": AXES, "min_axis": 0.93, "min_mean": 0.97,
                  "well_conditioned_floor": main.RETENTION_WELL_CONDITIONED_FLOOR,
                  "rows": rows, "by_arm": verdict},
}
text = main.render(summary)
ok("the rendered table names the arm family it read", "s5.6 arms against R3-F16" in text,
   text.splitlines()[0])
ok("the rendered table shows the reference rate beside the ratio", "0.0860" in text)
ok("the rendered table marks the failing axis", "| NO |" in text)
ok("the rendered verdict says what it failed on", "FAILS on ifstruct_validity" in text)
ok("the rendered table warns about the near-floor denominator",
   "well conditioned" in text and "0.30" in text)

# --- retention is opt-in, because the wrong reference makes it a lie ----------------------
# main() must not build a retention table for a run that did not ask for one. The s5.3
# comparison's reference is C1, a sibling checkpoint; a ratio against it prices the
# fine-tune. Guarding on the flag rather than on the reference name is what keeps a future
# comparison from inheriting the table by accident.
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"),
           encoding="utf-8").read()
ok("main() gates the table on an explicit flag", "if want_retention:" in src)
ok("the flag defaults to off", 'C("retention", "")' in src)
ok("the summary records whether it was asked for", '"requested": want_retention' in src)


print("\n%d checks, %d failed" % (N, len(FAILED)))
sys.exit(1 if FAILED else 0)
