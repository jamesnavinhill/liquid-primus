"""The s6.4 driver over fixtures whose answer is known before the code runs.

Three things this pass could get wrong without raising. The macro arithmetic could be off
by a factor -- it is the load-bearing claim, since the whole grader-edit finding is that one
item explains a composite move to four decimal places, and a formula that divides by the
wrong count would reconcile with itself and with nothing else. The envelope/schema split
could put a mixed failure on the wrong side, which would turn "lost the wrapper" into "lost
the schema" or the reverse. And the loss tabulation could count items in the abstain half,
which is exactly the pooling `s6.2` had to undo.

So the fixtures are built with the flip set, the category sizes, and the error lists chosen
so each of those errors changes a number this file checks.
"""

import json
import os
import sys
import tempfile
import types

ROOT = tempfile.mkdtemp(prefix="s6err-")
STORE = os.path.join(ROOT, "store")
LOGS, ARTIFACTS = [], []
CONFIG = {}


FINISHED = []


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

    # `finish` is the real SDK call and `job_complete` is not in the SDK at all -- the decon
    # job at this same substage wrote both its artifacts and was then marked FAILED for calling
    # the name that does not exist. The stub carries only the real one, so a wrong name here
    # shows up as an empty FINISHED rather than as a silently swallowed exception.
    def finish(self, message=None, score=None):
        FINISHED.append((message, score))


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
        FAILED.append("%s %s" % (name, repr(detail)))
        print("FAIL %s %s" % (name, repr(detail)))
    else:
        print("ok   %s" % name)


def put(obj, rows):
    path = os.path.join(STORE, obj)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------- 1
# Two categories, 4 items and 20 items. One item flips wrong in the 20-item category, so a
# two-category macro moves by exactly 1/(2*20) = 0.025 and by nothing else. An arm that
# divided by the item count instead would report 1/24 and still look plausible.

def native(cat, n, wrong=()):
    return [{"id": "%s_%d" % (cat, i), "category": cat, "style": "native_tools",
             "correct": i not in wrong,
             "reason": "" if i not in wrong else "made no call when one applies",
             "n_calls": 0 if i in wrong else 1,
             "calls": []} for i in range(n)]


OLD = native("simple", 4) + native("live_relevance", 20)
NEW = native("simple", 4) + native("live_relevance", 20, wrong=(7,))
put("pre/A/scored_bfcl_native_tools.jsonl", OLD)
put("post/A/scored_bfcl_native_tools.jsonl", NEW)
# B shares the flipped item's current verdict, so the flip transfers; it also has its own
# unrelated wrong item, which must NOT be counted as a flip.
put("post/B/scored_bfcl_native_tools.jsonl",
    native("simple", 4, wrong=(0,)) + native("live_relevance", 20, wrong=(7,)))
# C is right where A went wrong, so nothing transfers to it.
put("post/C/scored_bfcl_native_tools.jsonl", native("simple", 4) + native("live_relevance", 20))

g = main.grader_delta({"A": "pre/A/scored_bfcl_native_tools.jsonl"},
                      lambda a: "post", "scored_bfcl_native_tools.jsonl",
                      {"B": (3.0 / 4 + 20.0 / 20) / 2, "C": 1.0})
oa = g["observed"]["A"]
ok("1.1 one flip found", oa["n_flips"] == 1, oa)
ok("1.2 the flip is named", g["flips"][0]["id"] == "live_relevance_7", g["flips"])
ok("1.3 macro moves by 1/(k*n)", abs(oa["observed_delta"] + 0.025) < 1e-9, oa)
ok("1.4 the flips explain all of it", oa["arithmetic_reconciles"], oa)
ok("1.5 the implied delta is the observed one",
   abs(oa["delta_implied_by_flips"] - oa["observed_delta"]) < 1e-12, oa)

pb = g["predicted"]["B"]
ok("1.6 a shared flip transfers", [i["id"] for i in pb["items"]] == ["live_relevance_7"], pb)
ok("1.7 B's own unrelated failure is not a flip", len(pb["items"]) == 1, pb)
ok("1.8 the prediction lands on B", abs(pb["residual"]) < 1e-9, pb)
pc = main.grader_delta({}, lambda a: "post", "scored_bfcl_native_tools.jsonl",
                       {"C": 1.0})["predicted"]["C"]
ok("1.9 nothing transfers to an arm that is right there", pc["items"] == [], pc)

# --------------------------------------------------------------------------- 1b
# The repeated id, which is the defect the first pass of this job shipped with. BFCL v3
# gives two different `live_relevance` questions the same id. Keying on the bare id and
# keeping the first occurrence drops the second from every join, so a flip that lands on the
# second occurrence is invisible: the job reports zero flips while the composite has plainly
# moved, and only the arithmetic check notices. Occurrence keying is what makes it visible.

def dup(rows, i, j):
    out = [dict(r) for r in rows]
    out[j] = dict(out[j], id=out[i]["id"])
    return out


DOLD = native("simple", 4) + dup(native("live_relevance", 20), 7, 8)
DNEW = native("simple", 4) + dup(native("live_relevance", 20, wrong=(8,)), 7, 8)
put("pre/D/scored_bfcl_native_tools.jsonl", DOLD)
put("post/D/scored_bfcl_native_tools.jsonl", DNEW)
gd = main.grader_delta({"D": "pre/D/scored_bfcl_native_tools.jsonl"},
                       lambda a: "post", "scored_bfcl_native_tools.jsonl", {})
od = gd["observed"]["D"]
ok("1b.1 the repeat is counted", od["repeated_ids_pre"] == 1
   and od["repeated_ids_post"] == 1, od)
ok("1b.2 both occurrences survive the join", od["n_shared"] == 24, od)
ok("1b.3 the flip on the SECOND occurrence is found", od["n_flips"] == 1, od)
ok("1b.4 it is keyed as <id>#2", gd["flips"][0]["id"] == "live_relevance_7#2", gd["flips"])
ok("1b.5 the arithmetic still reconciles", od["arithmetic_reconciles"], od)
ok("1b.6 the composite moved by 1/(k*n)", abs(od["observed_delta"] + 0.025) < 1e-9, od)
ok("1b.7 the id order matches", od["same_id_order"], od)
ok("1b.8 a lost parse is flagged as one", gd["flips"][0]["parse_changed"]
   and gd["flips"][0]["old_n_calls"] == 1 and gd["flips"][0]["new_n_calls"] == 0,
   gd["flips"][0])

# A post-edit file that walks the corpus in a different order makes occurrence pairing
# meaningless, so it has to be an assertion failure rather than a silently wrong join.
before = len(main.FAILURES)
put("post/E/scored_bfcl_native_tools.jsonl", list(reversed(DNEW)))
put("pre/E/scored_bfcl_native_tools.jsonl", DOLD)
ge = main.grader_delta({"E": "pre/E/scored_bfcl_native_tools.jsonl"},
                       lambda a: "post", "scored_bfcl_native_tools.jsonl", {})
ok("1b.9 a reordered file is caught", not ge["observed"]["E"]["same_id_order"]
   and len(main.FAILURES) > before, main.FAILURES[before:])
del main.FAILURES[before:]

# ---------------------------------------------------------------------------- 2
# The envelope reader, on the four shapes a completion can have.
ok("2.1 opened and closed", main.envelope("```json\n{}\n```")["closed"], "")
ok("2.2 opened only", main.envelope("```yaml\na: 1\n") == {
    "fences": 1, "opened": True, "closed": False, "lang": "yaml", "chars": 13,
    "empty": False}, main.envelope("```yaml\na: 1\n"))
ok("2.3 no fence at all", not main.envelope("just prose")["opened"], "")
ok("2.4 empty completion", main.envelope("   ")["empty"], "")
ok("2.5 the language tag is read", main.envelope("```JSON\n{}\n```")["lang"] == "json", "")

# ---------------------------------------------------------------------------- 3
# Envelope failures against schema failures. Six items: two valid, two failing only on the
# wrapper, one failing on the schema, one failing on both. The mixed item must count as a
# schema failure -- a model that produced a fence AND a wrong document has not merely lost
# the wrapper, and putting it on the envelope side is how this pass would overstate its own
# finding.
SC = [
    {"id": "a", "correct": True, "output_format": "json", "detail": {"score": 1.0,
                                                                     "errors": []}},
    {"id": "b", "correct": True, "output_format": "yaml", "detail": {"score": 1.0,
                                                                     "errors": []}},
    {"id": "c", "correct": False, "output_format": "json",
     "detail": {"score": 0.0, "errors": ["Unclosed code block"]}},
    {"id": "d", "correct": False, "output_format": "json",
     "detail": {"score": 0.0, "errors": ["Response must use a code block but none was found"]}},
    {"id": "e", "correct": False, "output_format": "json",
     "detail": {"score": 0.5, "errors": ["required field missing: title"]}},
    {"id": "f", "correct": False, "output_format": "json",
     "detail": {"score": 0.0, "errors": ["Unclosed code block", "required field missing: x"]}},
]
put("post/A/scored_ifstruct.jsonl", SC)
put("comps/A/completions_ifstruct.jsonl", [
    {"id": "a", "completion": "```json\n{}\n```"},
    {"id": "b", "completion": "```yaml\na: 1\n```"},
    {"id": "c", "completion": "```json\n{\n"},
    {"id": "d", "completion": "no fence here"},
    {"id": "e", "completion": "```json\n{}\n```"},
    {"id": "f", "completion": "```json\n{\n"},
])
i = main.ifstruct_errors(["A"], lambda a: "post", "comps", "scored_ifstruct.jsonl",
                         "completions_ifstruct.jsonl", 20)["A"]
j = i["by_format"]["json"]
ok("3.1 formats are kept apart", sorted(i["by_format"]) == ["json", "yaml"], i["by_format"])
ok("3.2 envelope-only counts the two pure wrapper failures", j["envelope_only"] == 2, j)
ok("3.3 a mixed failure is a schema failure", j["schema_any"] == 2, j)
ok("3.4 envelope share is over failures, not items",
   abs(j["envelope_share_of_failures"] - 2.0 / 4.0) < 1e-9, j)
ok("3.5 an unclosed fence is not a missing one",
   (j["fence_missing"], j["fence_unclosed"]) == (1, 2), j)
ok("3.6 partial credit is averaged over items",
   abs(j["mean_partial_score"] - 1.5 / 5.0) < 1e-9, j)
ok("3.7 completions joined", i["joined_completions"] == 6, i)

# Raw validity mixes a schema failure with a fence failure. Conditioning on the items
# whose envelope was fine is the column that separates them: two of the six json items
# are envelope-only, leaving three clean-envelope items of which one validated.
ok("3.7b conditional validity drops the envelope-only items",
   (j["clean_envelope_items"], j["validity_given_clean_envelope"]) == (3, round(1 / 3, 6)),
   j)

# A missing completions file must leave the verdict columns intact and only empty the
# envelope ones: the error composition is the finding, and the wrapper is the corroboration.
del main.NOTES[:]
i2 = main.ifstruct_errors(["A"], lambda a: "post", "nowhere", "scored_ifstruct.jsonl",
                          "completions_ifstruct.jsonl", 20)["A"]
j2 = i2["by_format"]["json"]
ok("3.8 verdicts survive a missing completions file",
   (j2["envelope_only"], j2["schema_any"], j2["n"]) == (2, 2, 5), j2)
ok("3.9 the envelope columns are empty and said so",
   j2["fence_missing"] == 0 and any("has no" in n for n in main.NOTES), main.NOTES)

# ---------------------------------------------------------------------------- 4
# Losses are counted on the call-warranted half only. The fixture loses one item in
# `simple` and one in `live_relevance`; only the first may appear.
put("post/R/scored_bfcl_native_tools.jsonl",
    native("simple", 4, wrong=(1,)) + native("live_relevance", 20, wrong=(3,)))
put("comps/R/completions_bfcl_native_tools.jsonl",
    [{"id": "simple_1", "completion": "I cannot answer that with these tools."},
     {"id": "live_relevance_3", "completion": "x"}])
del main.NOTES[:]
n = main.native_losses(["A", "R"], "A", lambda a: "post",
                       "comps", "scored_bfcl_native_tools.jsonl",
                       "completions_bfcl_native_tools.jsonl", 5)
# A is the reference and its own file has live_relevance_7 wrong, which is in the abstain
# half and must not create a gain either.
r = n["arms"]["R"]
ok("4.1 only the call-warranted half is counted", r["lost"] == 1, r)
ok("4.2 the abstain half creates no gain", r["gained"] == 0, r)
ok("4.3 the denominator is the call-warranted half", r["n_call_warranted"] == 4, r)
ok("4.4 the reason is collapsed to a family",
   r["by_reason"] == {"did not call": 1}, r["by_reason"])
ok("4.5 prose with no call is recorded as prose",
   r["what_it_emitted"]["emitted prose only"] == 1, r["what_it_emitted"])
ok("4.6 the lost item is named", r["examples"][0]["id"] == "simple_1", r["examples"])

ok("4.7 reason families", (main.reason_family("expected 1 call(s), parsed 0"),
                           main.reason_family(""),
                           main.reason_family("wrong function name: foo")) ==
   ("did not call", "correct", "wrong function"), "")

# The grader writes one clause per ground-truth call, prefixed `gt<i>/call<j>:`. Left
# whole, every distinct function name becomes its own family and the table degenerates
# into a list of singletons that says nothing. The prefix is stripped and the first
# clause decides, so two items that failed the same way land in the same row.
ok("4.8 per-call clauses collapse onto the first",
   (main.reason_family("gt0/call0: wrong function: total_revenue"),
    main.reason_family("gt0/call1: wrong function: q; gt1/call0: wrong function: r"),
    main.reason_family("gt0/call0: wrong argument value for 'limit'")) ==
   ("wrong function", "wrong function", "wrong arguments"), "")

# ---------------------------------------------------------------------------- 5
# The whole pass end to end, and the rendered table.
CONFIG.update({"arms": "A,R", "reference": "A", "scored_prefix": "post",
               "completions_prefix": "comps",
               "pre_edit_objects": json.dumps({"A": "pre/A/scored_bfcl_native_tools.jsonl"}),
               "recorded_pre_edit_composite": json.dumps({"R": 0.9}),
               "components": ""})
main.CFG = dict(CONFIG)
del main.NOTES[:]
del main.FAILURES[:]
main.main()
doc = json.load(open(os.path.join(ROOT, "out", "errors.json"), encoding="utf-8"))
md = open(os.path.join(ROOT, "out", "errors.md"), encoding="utf-8").read()
ok("5.1 all three components ran",
   all(k in doc for k in ("grader_delta", "ifstruct_errors", "native_losses")), sorted(doc))
ok("5.2 both artifacts saved", sorted(ARTIFACTS)[-2:] == ["errors.json", "errors.md"],
   ARTIFACTS)
ok("5.3 the flipped item is named in the table", "live_relevance_7" in md, "")
ok("5.4 the envelope column is rendered", "envelope share" in md, "")
ok("5.5 no assertion failed on a clean fixture", doc["assertion_failures"] == [],
   doc["assertion_failures"])
ok("5.6 the two halves are recorded in the summary",
   len(doc["call_warranted"]) == 8 and len(doc["abstain_warranted"]) == 3, "")

# A component list restricts the pass.
CONFIG["components"] = "ifstruct_errors"
main.CFG = dict(CONFIG)
main.main()
doc2 = json.load(open(os.path.join(ROOT, "out", "errors.json"), encoding="utf-8"))
ok("5.7 components= runs only what it names",
   "ifstruct_errors" in doc2 and "native_losses" not in doc2, sorted(doc2))

# Both passes above must have closed out on the real SDK call, and the score they closed
# with is the assertion-failure count -- zero on a fixture that is meant to reconcile.
ok("5.8 each pass closed out, scored by assertion failures",
   len(FINISHED) == 2 and all(f[1] == 0 for f in FINISHED), FINISHED)

print("\n%d checks, %d failed" % (N, len(FAILED)))
sys.exit(1 if FAILED else 0)
