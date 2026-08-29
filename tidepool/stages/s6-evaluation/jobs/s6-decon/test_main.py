"""The decontamination re-check over fixtures whose answer is known before the code runs.

Four things this pass could get wrong while still producing a plausible-looking zero. The
n-gram rule could be off by one, so a 12-word overlap counts or a 13-word one does not. The
normalization could fail to collapse case and punctuation, which would make the whole check
report clean because real overlaps rarely survive verbatim. The input/target split could put
an assistant-turn leak on the input side, which is precisely the distinction this job exists
to draw. And the BFCL restriction could index the whole benchmark rather than the ids that
were scored, which would report contamination of items no number came from.

Each fixture below is built so that one of those errors changes a number checked here.
"""

import gzip
import json
import os
import sys
import tempfile
import types

ROOT = tempfile.mkdtemp(prefix="s6dec-")
STORE = os.path.join(ROOT, "store")
ARTIFACTS = []
CONFIG = {}


FINISHED = []


class _Lab:
    def init(self):
        pass

    def get_config(self):
        return dict(CONFIG)

    def log(self, msg):
        pass

    def update_progress(self, pct):
        pass

    def storage_download(self, obj):
        path = os.path.join(STORE, obj)
        if not os.path.exists(path):
            raise RuntimeError("no such object: %s" % obj)
        return path

    def save_artifact(self, path):
        ARTIFACTS.append(os.path.basename(path))

    def finish(self, message=None, score=None):
        FINISHED.append((message, score))

    # DELIBERATELY ABSENT: `job_complete`. The worker that ran 154c72a4 had an SDK without it,
    # and the pass was marked FAILED after both artifacts were already written -- a full set of
    # correct numbers under a status that says not to trust them. The stub reproduces that SDK
    # so the fixtures fail if the closing call is ever left unguarded again.


mod = types.ModuleType("lab")
mod.lab = _Lab()
sys.modules["lab"] = mod

# The eval index reaches the network three ways and none of them belongs in a fixture run.
IFSTRUCT_TEXT = [""]


class _Resp(object):
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


req = types.ModuleType("requests")
req.get = lambda url, timeout=None: _Resp(IFSTRUCT_TEXT[0])
sys.modules["requests"] = req

BFCL_FILES = {}


def _list_repo_files(repo_id, repo_type=None, token=None):
    return sorted(BFCL_FILES)


def _hf_hub_download(repo_id, filename, repo_type=None, token=None):
    path = os.path.join(ROOT, "hf", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for e in BFCL_FILES[filename]:
            fh.write(json.dumps(e) + "\n")
    return path


hub = types.ModuleType("huggingface_hub")
hub.list_repo_files = _list_repo_files
hub.hf_hub_download = _hf_hub_download
sys.modules["huggingface_hub"] = hub

os.chdir(ROOT)
CONFIG.update({"ngram_n": 13, "max_chars_for_grams": 20000, "eval_sets": "",
               "bfcl_ids_from": "scored.jsonl", "ifstruct_url": "http://fixture/test.jsonl",
               "probes_object": "probes.jsonl", "worst_rows": 25, "examples": 15})
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


def put(obj, rows, gz=False):
    path = os.path.join(STORE, obj)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    opener = gzip.open if gz else open
    with opener(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


# ---------------------------------------------------------------------------- 1
# The gram rule itself. THIRTEEN words is the boundary, so a twelve-word sentence
# contributes nothing and a thirteen-word one contributes exactly one window.

W = "alpha bravo charlie delta echo foxtrot golf hotel india juliett kilo lima mike"
ok("1.1 twelve words yield nothing",
   main.grams(main.words_of(" ".join(W.split()[:12]))).size == 0, "")
ok("1.2 thirteen words yield one window",
   main.grams(main.words_of(W)).size == 1, "")
ok("1.3 fourteen words yield two", main.grams(main.words_of(W + " november")).size == 2, "")
ok("1.4 case and punctuation are normalized away",
   list(main.grams(main.words_of(W)))
   == list(main.grams(main.words_of("Alpha, BRAVO; charlie -- delta echo foxtrot golf "
                                    "hotel india juliett kilo lima mike!"))), "")
ok("1.5 a different word is a different window",
   list(main.grams(main.words_of(W)))
   != list(main.grams(main.words_of(W.replace("mike", "november")))), "")

# ---------------------------------------------------------------------------- 2
# The index owns its grams, so a hit names the item it came from and can quote it.

idx = main.Index()
idx.add_set("benchA", [("a1", W), ("a2", "one two three four five six seven eight nine "
                                          "ten eleven twelve thirteen")], {})
idx.add_set("benchB", [("b1", "wholly unrelated text that shares no window at all with "
                              "anything above it here")], {})
idx.freeze()
n, by, items = idx.hits(main.grams(main.words_of("preamble words " + W + " trailing")))
ok("2.1 the planted window is found", n == 1, (n, by))
ok("2.2 it is attributed to the right set", by == {"benchA": 1}, by)
label, item_id, text = idx.quote(items[0], main.grams(main.words_of(W)))
ok("2.3 the overlapping span is quoted", (label, item_id) == ("benchA", "a1")
   and text == W, (label, item_id, text))
ok("2.4 an unrelated text hits nothing",
   idx.hits(main.grams(main.words_of("nothing here matches any indexed window whatsoever "
                                     "not one single time")))[0] == 0, "")

# ---------------------------------------------------------------------------- 3
# Input versus target. The SAME leaked sentence is placed in a user turn in one row and in
# an assistant turn in another, and the two must land in different columns.

put("probes.jsonl", [])
# 2,000 short items, so the count assertion sees the real corpus size and none of them is
# long enough to contribute a window. The overlap fixtures are BFCL's job.
IFSTRUCT_TEXT[0] = "\n".join(json.dumps({"id": "if%d" % k, "prompt": "short prompt %d" % k})
                             for k in range(2000))
put("scored.jsonl", [{"id": "kept_1"}])
BFCL_FILES.clear()
BFCL_FILES["BFCL_v3_live_simple.json"] = [
    {"id": "kept_1", "question": "leaked " + W + " tail",
     "function": [{"name": "f", "description": "x"}]},
    {"id": "never_scored", "question": "secret " + " ".join(
        "zulu%d" % i for i in range(14)) + " end"},
]
rows = [
    {"c": "toolace", "i": "r1", "s": "train",
     "messages": [{"role": "user", "content": "please " + W + " now"},
                  {"role": "assistant", "content": "sure thing"}]},
    {"c": "toolace", "i": "r2", "s": "train",
     "messages": [{"role": "user", "content": "unrelated question entirely"},
                  {"role": "assistant", "content": "here it is " + W + " done"}]},
    {"c": "apigen", "i": "r3", "s": "train",
     "messages": [{"role": "user", "content": "nothing to see"},
                  {"role": "assistant", "content": "nor here"}]},
    {"c": "apigen", "i": "r4", "s": "train",
     "messages": [{"role": "user", "content": "secret " + " ".join(
         "zulu%d" % i for i in range(14)) + " end"},
                  {"role": "assistant", "content": "ok"}]},
]
put("train.jsonl.gz", rows, gz=True)
CONFIG.update({"train_object": "train.jsonl.gz", "val_object": "", "test_object": ""})
del main.NOTES[:]
del main.FAILURES[:]
idx2 = main.build_index()
ok("3.0 a second freeze does not empty the index",
   (idx2.freeze() or idx2.all.size) > 0, idx2.all.size)
ok("3.1 only the scored BFCL id is indexed",
   main.NOTES == [] and idx2.report["bfcl_scored"]["rows"] == 1, idx2.report)
ok("3.2 the ids-found check passes",
   idx2.report["bfcl_scored"]["ids_found"] == 1, idx2.report)
s = main.scan(idx2, "train.jsonl.gz", 25, 15)
ok("3.3 two rows are hit", s["rows_hit"] == 2, s)
ok("3.4 one hit is in the input", s["rows_hit_in_input"] == 1, s)
ok("3.5 one hit is in the TARGET", s["rows_hit_in_target"] == 1, s)
ok("3.6 neither row hit on both sides", s["rows_hit_on_both_sides"] == 0, s)
ok("3.7 the unscored BFCL item is invisible",
   s["by_corpus"]["apigen"]["hit"] == 0, s["by_corpus"])
ok("3.8 the hits are attributed to the corpus that carries them",
   s["by_corpus"]["toolace"]["hit"] == 2, s["by_corpus"])
ok("3.9 the target windows are counted separately",
   s["target_windows_by_eval_set"].get("bfcl_scored") == 1
   and s["windows_by_eval_set"].get("bfcl_scored") == 2,
   (s["windows_by_eval_set"], s["target_windows_by_eval_set"]))
ok("3.10 the worst rows quote the overlap",
   any(W in (w["overlapping_text"] or "") for w in s["worst"]), s["worst"])
ok("3.11 the eval item is named", s["worst"][0]["eval_item"] == "kept_1", s["worst"][0])
# The target-side finding has to be readable, not just counted. Pass 3 named three BFCL items
# and recorded nothing about what any of them shared, which is the difference between a leaked
# answer and a shared boilerplate phrase and therefore the whole content of the finding.
_ex = s["eval_target_overlap_examples"]
ok("3.12 the target-side item carries the span it shares",
   len(_ex) == 1 and _ex[0]["eval_item"] == "kept_1" and W.split()[0] in _ex[0]["overlapping_text"],
   _ex)
ok("3.13 and the training row it was found in",
   (_ex[0]["corpus"], _ex[0]["row_id"]) == ("toolace", "r2"), _ex[0])
ok("3.14 an input-side-only hit contributes no target example",
   all(e["row_id"] != "r1" for e in _ex), _ex)

# The one-owner-per-gram limitation, and the pass that removes it. `freeze` keeps a single
# owner for each 13-gram, so a span two sets share is credited to whichever was indexed first
# and the other set's count is a remainder. On the real data that put 1.42 million of 1.42
# million matching windows under `probes`, on the strength of one shared instruction template.
_shared = main.Index()
_shared.add_set("first", [("f1", W)], {})
_shared.add_set("second", [("s1", W)], {})
_shared.freeze()
_n, _by, _items = _shared.hits(main.grams(main.words_of("please " + W + " now")))
ok("3.15 a span two sets share is credited to only one of them",
   _n > 0 and list(_by) == ["first"], (_n, _by))
main.ONLY = {"second"}
_solo = main.Index()
_solo.add_set("first", [("f1", W)], {})
_solo.add_set("second", [("s1", W)], {})
_solo.freeze()
main.ONLY = set()
ok("3.16 a single-set pass indexes only that set",
   _solo.sets == ["second"] and list(_solo.report) == ["second"], _solo.sets)
_n2, _by2, _ = _solo.hits(main.grams(main.words_of("please " + W + " now")))
ok("3.17 and the set that was losing the span now sees it",
   _n2 == _n and list(_by2) == ["second"], (_n2, _by2))

# ---------------------------------------------------------------------------- 4
# A clean split reports clean, and it does so with a non-empty index. A zero produced by an
# index that failed to build is the one failure mode this job could not otherwise notice,
# so it is an assertion rather than a number.

put("clean.jsonl.gz", [{"c": "sql_ctx", "i": "c%d" % k, "s": "train",
                        "messages": [{"role": "user", "content": "select %d from t" % k},
                                     {"role": "assistant", "content": "SELECT %d;" % k}]}
                       for k in range(20)], gz=True)
sc = main.scan(idx2, "clean.jsonl.gz", 25, 15)
ok("4.1 a clean split hits nothing", sc["rows_hit"] == 0 and sc["rows"] == 20, sc)
ok("4.2 and the index it was checked against is not empty", idx2.all.size > 0, "")
empty = main.Index()
empty.freeze()
del main.FAILURES[:]
main.check("index_not_empty", empty.all.size > 0, "empty")
ok("4.3 an empty index is an assertion failure", len(main.FAILURES) == 1, main.FAILURES)
del main.FAILURES[:]

# ---------------------------------------------------------------------------- 5
# The whole pass end to end, including the artifacts and the rendered table.

del ARTIFACTS[:]
del main.NOTES[:]
CONFIG.update({"train_object": "train.jsonl.gz", "val_object": "clean.jsonl.gz",
               "test_object": "", "per_set_scan": "bfcl_scored,not_indexed"})
main.CFG = dict(CONFIG)          # the driver reads its config once, at import
main.main()
ok("5.1 both artifacts saved", sorted(ARTIFACTS) == ["decon.json", "decon.md"], ARTIFACTS)
doc = open(os.path.join(main.OUT, "decon.md"), encoding="utf-8").read()
out = json.load(open(os.path.join(main.OUT, "decon.json"), encoding="utf-8"))
ok("5.2 no assertion failed on the fixture", out["assertion_failures"] == [],
   out["assertion_failures"])
ok("5.3 both splits are reported",
   set(out["splits"]) == {"train", "val"}, list(out["splits"]))
ok("5.4 the target column is rendered", "in TARGET" in doc, doc[:400])
ok("5.5 the overlapping text reaches the document", W in doc, "")
ok("5.6 the clean split shows zero", out["splits"]["val"]["rows_hit"] == 0,
   out["splits"]["val"])

# The close-out is not a formality. A run that computes everything and then raises on the
# way out is a correct answer under a FAILED status, which nobody may quote.
ok("5.7 the run closed out on the real SDK call",
   len(FINISHED) == 1 and "decontamination" in (FINISHED[0][0] or ""), FINISHED)

# The named-item output. A count of contaminated rows cannot be stratified on; the peakedness
# check downstream picks its overlap arm out of this map, so an empty or aggregate-only field
# here would silently turn that test into an unstratified one.
tr = out["splits"]["train"]
ok("5.8 the hit eval items are named on the target side",
   any(v for v in tr["eval_items_hit_in_target"].values()), tr["eval_items_hit_in_target"])
ok("5.9 a capped list says what it dropped",
   set(tr["eval_items_truncated"]) == {"target", "input"}, tr["eval_items_truncated"])

# The per-set pass, end to end through the driver: an exact figure for the set that carries
# the reported numbers, and a note rather than a silent skip for a label that is not indexed.
ok("5.10 the named set was re-scanned on its own",
   "bfcl_scored" in (out.get("per_set") or {}), list(out.get("per_set") or {}))
ok("5.11 its train figures match the shared pass where no other set competes for the span",
   out["per_set"]["bfcl_scored"]["splits"]["train"]["rows_hit"] ==
   out["splits"]["train"]["rows_hit"], out["per_set"]["bfcl_scored"]["splits"]["train"])
ok("5.12 an unknown label is reported rather than skipped in silence",
   any("not_indexed" in n for n in out["notes"]), out["notes"])
ok("5.13 the rendered report carries the per-set table and the target-side spans",
   "## Per-set passes" in doc and "appears in a training TARGET" in doc, "")

print("\n%d checks, %d failed" % (N, len(FAILED)))
if FAILED:
    raise SystemExit(1)
