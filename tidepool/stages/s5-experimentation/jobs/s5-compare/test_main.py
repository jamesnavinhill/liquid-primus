"""The comparison driver over hand-built fixtures, with the lab SDK stubbed out.

Three things can go wrong in a pass like this and none of them raise. The join can match
the wrong items, in which case every number is a fiction about a set that does not exist.
The row filters can select the wrong population, which is how a false-flag rate ends up
averaged into a detection rate. And a missing file can quietly shrink the family the
correction is applied over, which makes every surviving p-value too small.

So the fixtures are built with the answer known in advance: arms that differ on exactly the
items named below, groups sized so the macro figure and the item-weighted figure disagree by
a known amount, and one arm whose file is deliberately absent.
"""

import json
import os
import shutil
import sys
import tempfile
import types

ROOT = tempfile.mkdtemp(prefix="s5cmp-")
STORE = os.path.join(ROOT, "store")
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
        path = os.path.join(STORE, obj)
        if not os.path.exists(path):
            raise RuntimeError("no such object: %s" % obj)
        return path

    def save_artifact(self, path):
        ARTIFACTS.append(os.path.basename(path))

    def finish(self, message=None, score=None):
        FINISHED["message"] = message
        FINISHED["score"] = score


mod = types.ModuleType("lab")
mod.lab = _Lab()
sys.modules["lab"] = mod

os.chdir(ROOT)
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


def put(arm, fname, rows):
    d = os.path.join(STORE, "tidepool/s5.3/arms", arm)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, fname), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def bfcl_rows(flags, categories):
    return [{"id": "b%d" % i, "category": categories[i], "style": "native_tools",
             "correct": bool(f)} for i, f in enumerate(flags)]


# ---------------------------------------------------------------- fixture: two BFCL arms
#
# Ten items in two categories. The reference is right on the first five and wrong on the
# rest; the candidate flips items 5 and 6 to correct and item 0 to wrong. Item-weighted that
# is +1/10. Category `simple` holds items 0-1 and `live` holds 2-9, so per category the
# deltas are -1/2 and +2/8, and the macro figure is -0.125, the opposite sign. The point of
# the fixture is that the two columns disagree and both are reported.
CATS = ["simple", "simple"] + ["live"] * 8
put("REF", "scored_bfcl_native_tools.jsonl",
    bfcl_rows([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], CATS))
put("CAND", "scored_bfcl_native_tools.jsonl",
    bfcl_rows([0, 1, 1, 1, 1, 1, 1, 0, 0, 0], CATS))
for arm, rate in (("REF", 0.5), ("CAND", 0.6)):
    d = os.path.join(STORE, "tidepool/s5.3/arms", arm)
    with open(os.path.join(d, "score.json"), "w", encoding="utf-8") as fh:
        json.dump({"bfcl": {"styles": {"native_tools": {"item_weighted": rate}}}}, fh)

# ---------------------------------------------------------------- fixture: one probe file
#
# One file, four populations. The candidate flags harder: it gains two of the malformed
# items and loses one clean one. A component table that pooled these would report +1/13 and
# hide the trade entirely.
def probes(malformed, clean, corpus, stack):
    rows = []
    for i, f in enumerate(malformed):
        rows.append({"id": "m%d" % i, "probe": "tool_return",
                     "arm": "corrupted" if i % 2 else "contradicted",
                     "correct": bool(f)})
    for i, f in enumerate(clean):
        rows.append({"id": "c%d" % i, "probe": "tool_return", "arm": "clean",
                     "correct": bool(f)})
    for i, f in enumerate(corpus):
        rows.append({"id": "k%d" % i, "probe": "tool_return", "arm": "clean_corpus",
                     "correct": bool(f)})
    for i, f in enumerate(stack):
        rows.append({"id": "s%d" % i, "probe": "stack_idiom", "arm": "stack",
                     "correct": bool(f)})
    return rows


put("REF", "scored_probes.jsonl", probes([1, 0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 0]))
put("CAND", "scored_probes.jsonl", probes([1, 1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0]))

# CAND has no ifeval file at all: the cell must be dropped, named, and left out of the family.
put("REF", "scored_ifeval.jsonl", [{"id": "e%d" % i, "correct": True} for i in range(4)])

CONFIG.update({
    "arms": "REF,CAND",
    "reference": "REF",
    "resamples": 2000,
    "seed": 7,
    "components": json.dumps({
        "bfcl_native": {"file": "scored_bfcl_native_tools.jsonl",
                        "check": "bfcl.styles.native_tools.item_weighted",
                        "group_field": "category"},
        "ifeval": {"file": "scored_ifeval.jsonl"},
        "probes_detect": {"file": "scored_probes.jsonl",
                          "where": {"probe": ["tool_return"],
                                    "arm": ["corrupted", "contradicted"]}},
        "probes_clean": {"file": "scored_probes.jsonl",
                         "where": {"probe": ["tool_return"], "arm": ["clean"]}},
        "probes_stack_idiom": {"file": "scored_probes.jsonl",
                               "where": {"probe": ["stack_idiom"]}},
    }),
})

S = main.main()
CELLS = {c["component"]: c for c in S["cells"]}

# ---- the join and the arithmetic
ok("every cell is CAND against REF", all(c["arm"] == "CAND" for c in S["cells"]))
b = CELLS["bfcl_native"]
ok("bfcl matched all ten items", b["n_matched"] == 10, b["n_matched"])
ok("bfcl reference rate is 0.5", abs(b["reference_rate"] - 0.5) < 1e-9, b["reference_rate"])
ok("bfcl arm rate is 0.6", abs(b["arm_rate"] - 0.6) < 1e-9, b["arm_rate"])
ok("bfcl delta is +0.1", abs(b["delta"] - 0.1) < 1e-9, b["delta"])
ok("bfcl discordant counts are 1 against 2",
   (b["only_reference_correct"], b["only_arm_correct"]) == (1, 2),
   (b["only_reference_correct"], b["only_arm_correct"]))
ok("bfcl agreeing pairs are the remaining 7", b["agree"] == 7, b["agree"])
# 2 categories, deltas -0.5 and +0.25, mean -0.125. The item-weighted figure is positive,
# so a run that reported only one of the two columns would report the opposite direction.
ok("bfcl macro delta is -0.125 and opposes the item-weighted delta",
   abs(b["macro_delta_descriptive"] + 0.125) < 1e-9 and b["delta"] > 0,
   b["macro_delta_descriptive"])
ok("bfcl macro saw both categories", b["macro_groups"] == 2, b["macro_groups"])

# ---- the row filters
d = CELLS["probes_detect"]
ok("detection selected only the four malformed probes", d["n_matched"] == 4, d["n_matched"])
ok("detection delta is +0.5", abs(d["delta"] - 0.5) < 1e-9, d["delta"])
c = CELLS["probes_clean"]
ok("the clean arm is its own cell of three", c["n_matched"] == 3, c["n_matched"])
ok("the clean arm delta is negative and shows the trade", c["delta"] < 0, c["delta"])
s = CELLS["probes_stack_idiom"]
ok("stack idiom selected two items", s["n_matched"] == 2, s["n_matched"])
ok("stack idiom is unchanged", s["delta"] == 0.0 and s["p_mcnemar_exact"] == 1.0,
   (s["delta"], s["p_mcnemar_exact"]))
ok("no probe cell is the pooled thirteen", all(
    c2["n_matched"] != 13 for c2 in S["cells"]))
ok("the four probe populations are not double-counted",
   sum(CELLS[k]["n_matched"] for k in ("probes_detect", "probes_clean",
                                       "probes_stack_idiom")) == 9)

# ---- a missing side
ok("ifeval was dropped, not compared", "ifeval" not in CELLS)
ok("the missing ifeval file is named in the notes",
   any("ifeval" in n and "missing" in n.lower() for n in S["notes"]), S["notes"])
ok("the family is the four cells that had both sides", S["n_comparisons"] == 4,
   S["n_comparisons"])

# ---- correction and reporting
ok("Holm p is at least the raw p everywhere",
   all(c2["p_holm"] >= c2["p_mcnemar_exact"] - 1e-12 for c2 in S["cells"])) 
ok("nothing this small survives family correction",
   S["n_significant_family_corrected"] == 0, S["n_significant_family_corrected"])
ok("the recomputed rates reconciled with both arms' score.json",
   not any("recomputed rate" in f for f in S["assertion_failures"]),
   S["assertion_failures"])
ok("both artifacts were saved",
   set(ARTIFACTS) == {"comparison.json", "comparison.md"}, ARTIFACTS)
ok("the completion message counts the family and the failures",
   "4 comparison(s)" in FINISHED["message"] and "0 assertion failure(s)"
   in FINISHED["message"], FINISHED.get("message"))
md = open(os.path.join(ROOT, "out", "comparison.md"), encoding="utf-8").read()
ok("the table names every compared cell", all(k in md for k in CELLS))
ok("the table says the macro column carries no test", "no interval" in md)
ok("indistinguishable cells are labelled as such", "indistinguishable" in md)

# ---- a wrong reference is refused rather than guessed at
CONFIG["reference"] = "NOPE"
try:
    main.main()
    ok("a reference outside the arm list is refused", False, "no SystemExit")
except SystemExit as exc:
    ok("a reference outside the arm list is refused", "reference" in str(exc), str(exc))
CONFIG["reference"] = "REF"
CONFIG["arms"] = "REF"
try:
    main.main()
    ok("a single arm is refused", False, "no SystemExit")
except SystemExit as exc:
    ok("a single arm is refused", "at least two" in str(exc), str(exc))

# ---- a repeated id is reported rather than silently shrinking the denominator
CONFIG["arms"] = "REF,DUP"
put("DUP", "scored_ifeval.jsonl",
    [{"id": "e0", "correct": True}, {"id": "e0", "correct": False},
     {"id": "e1", "correct": True}, {"id": "e2", "correct": True},
     {"id": "e3", "correct": True}])
CONFIG["components"] = json.dumps({"ifeval": {"file": "scored_ifeval.jsonl"}})
S2 = main.main()
ok("a repeated item id is an assertion failure",
   any("repeated item ids" in f for f in S2["assertion_failures"]),
   S2["assertion_failures"])
ok("the later row wins, so the duplicate scores as its second value",
   S2["cells"][0]["only_reference_correct"] == 1, S2["cells"][0])

# ---- a filter that selects nothing is the expected real case, not an error
#
# None of the eight sweep variants was scored against the corpus clean arm, so
# `probes_clean_corpus` will select zero rows out of a file that exists and is otherwise
# fine. The cell has to drop out with a note, and the corrected family has to shrink with
# it: a cell silently compared over zero items, or counted in the family and never
# reported, would make every surviving p-value in the run too small.
# Rewritten without the corpus arm, which is exactly how the real files look: the harness
# only emits those rows when `clean_control_object` is set, and no arm queue script set it.
put("REF", "scored_probes.jsonl", probes([1, 0, 0, 0], [1, 1, 1], [], [1, 0]))
put("CAND", "scored_probes.jsonl", probes([1, 1, 1, 0], [1, 1, 0], [], [1, 0]))
CONFIG["arms"] = "REF,CAND"
CONFIG["components"] = json.dumps({
    "probes_detect": {"file": "scored_probes.jsonl",
                      "where": {"probe": ["tool_return"],
                                "arm": ["corrupted", "contradicted"]}},
    "probes_clean_corpus": {"file": "scored_probes.jsonl",
                            "where": {"probe": ["tool_return"], "arm": ["clean_corpus"]}},
})
S3 = main.main()
ok("a filter selecting no rows drops its cell",
   [c["component"] for c in S3["cells"]] == ["probes_detect"],
   [c["component"] for c in S3["cells"]])
ok("the empty population is named in the notes",
   any("clean_corpus" in n and "selected no rows" in n for n in S3["notes"]), S3["notes"])
ok("the corrected family shrank to the one real cell", S3["n_comparisons"] == 1,
   S3["n_comparisons"])
ok("an empty population is not an assertion failure", not S3["assertion_failures"],
   S3["assertion_failures"])

print("\n%d checks, %d failed" % (N, len(FAILED)))
shutil.rmtree(ROOT, ignore_errors=True)
sys.exit(1 if FAILED else 0)
