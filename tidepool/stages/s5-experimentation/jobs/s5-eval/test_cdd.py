"""Fixtures for the peakedness module. No model, no card: strings and item dicts only."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdd                                                          # noqa: E402

N, FAILED = 0, []


def ok(name, cond, detail=""):
    global N
    N += 1
    if cond:
        print("ok   %s" % name)
    else:
        FAILED.append(name)
        print("FAIL %s   %s" % (name, detail))


# ---------------------------------------------------------------------------- 1
# The bounded edit distance is exact at and below the cutoff and only refuses above it.
# Everything the statistic reports -- an exact repeat, a near repeat -- is inside the band,
# so a wrong answer there would be a wrong Peak rather than a slow one.
ok("1.1 identical strings are distance zero",
   cdd.bounded_edit_distance("abc", "abc", 5) == 0)
ok("1.2 one substitution", cdd.bounded_edit_distance("abc", "abd", 5) == 1)
ok("1.3 one insertion", cdd.bounded_edit_distance("abc", "abxc", 5) == 1)
ok("1.4 one deletion", cdd.bounded_edit_distance("abxc", "abc", 5) == 1)
ok("1.5 exact below the cutoff", cdd.bounded_edit_distance("kitten", "sitting", 5) == 3)
ok("1.6 refused above the cutoff, not wrong",
   cdd.bounded_edit_distance("kitten", "sitting", 2) == 3, "")
ok("1.7 a length gap past the cutoff short-circuits",
   cdd.bounded_edit_distance("a", "a" * 40, 5) == 6)
ok("1.8 the empty string against a long one",
   cdd.bounded_edit_distance("", "abcdefgh", 3) == 4)
ok("1.9 symmetric", cdd.bounded_edit_distance("abcdef", "abdcef", 4) ==
   cdd.bounded_edit_distance("abdcef", "abcdef", 4))

# Against a full unbounded reference, on strings short enough to compute both ways.
def slow(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


bad = []
words = ["", "a", "ab", "abc", "tool_call", "tool_calls", "get_weather", "get_weathe",
         '{"name": "x"}', '{"name": "y"}', '{"nme": "x"}']
for x in words:
    for y in words:
        want = slow(x, y)
        got = cdd.bounded_edit_distance(x, y, 4)
        if want <= 4 and got != want:
            bad.append((x, y, want, got))
        if want > 4 and got != 5:
            bad.append((x, y, want, got))
ok("1.10 matches an unbounded reference on every pair", not bad, bad[:3])

# ---------------------------------------------------------------------------- 2
# Peak itself, and the l cap. Twenty samples: ten are the greedy text exactly, five are one
# character off, five are unrelated. With l capped at 100 and alpha 0.05 the threshold is 5,
# so the first fifteen are inside it.
G = "x" * 80
S = [G] * 10 + [G[:-1] + "y"] * 5 + ["z" * 80] * 5
p = cdd.peakedness(G, S, alpha=0.05, l_cap=100)
ok("2.1 threshold is floor(alpha * l)", (p["l"], p["threshold"]) == (80, 4), p)
ok("2.2 peak counts everything within the threshold", p["n_within"] == 15, p)
ok("2.3 exact repeats are counted separately", p["n_exact"] == 10, p)
ok("2.4 peak is a fraction of the samples", abs(p["peak"] - 15 / 20) < 1e-12, p)
ok("2.5 distinct samples is the diversity floor", p["distinct_samples"] == 3, p)

# The cap is what stops a long answer from buying itself a wide threshold. Without it, a
# 2,000-character completion would call anything within 100 edits "the same answer".
long_g = "q" * 2000
pc = cdd.peakedness(long_g, [long_g, "q" * 1900 + "r" * 100], alpha=0.05, l_cap=100)
ok("2.6 l is capped, so a long answer does not widen its own threshold",
   (pc["l"], pc["threshold"], pc["n_within"]) == (100, 5, 1), pc)

ok("2.7 no samples is not a crash", cdd.peakedness("a", [])["peak"] == 0.0)

# A rigid schema with no memorization at all still looks peaked. This is the reason the job
# runs a base model on the same items: the number below is not evidence of anything on its own.
rigid = cdd.peakedness('{"name": "get_weather", "args": {"city": "Paris"}}',
                       ['{"name": "get_weather", "args": {"city": "Paris"}}'] * 51,
                       alpha=0.05, l_cap=100)
ok("2.8 a fixed schema alone produces peak 1.0", rigid["peak"] == 1.0, rigid)

# ---------------------------------------------------------------------------- 3
# Control matching. Same category, nearest prompt length, consumed without replacement, and
# deterministic in the order the flagged items are given.
ITEMS = [{"id": "simple_1", "category": "simple"},
         {"id": "simple_2", "category": "simple"},
         {"id": "simple_3", "category": "simple"},
         {"id": "simple_4", "category": "simple"},
         {"id": "live_simple_1", "category": "live_simple"},
         {"id": "live_simple_2", "category": "live_simple"}]
LEN = {"simple_1": 100, "simple_2": 900, "simple_3": 105, "simple_4": 110,
       "live_simple_1": 50, "live_simple_2": 55}
pairs, missing, unmatched = cdd.match_controls(
    ITEMS, ["simple_1", "live_simple_1"], lambda it: LEN[it["id"]], 10)
got = {f["id"]: c["id"] for f, c in pairs}
ok("3.1 the control is the nearest length in the same category",
   got.get("simple_1") == "simple_3", got)
ok("3.2 categories never cross", got.get("live_simple_1") == "live_simple_2", got)
ok("3.3 a flagged item is never its own control",
   all(f["id"] != c["id"] for f, c in pairs), got)

# Two flagged items in one category must not be handed the same control.
pairs2, _, _ = cdd.match_controls(ITEMS, ["simple_1", "simple_4"],
                                 lambda it: LEN[it["id"]], 10)
ctrl = [c["id"] for _f, c in pairs2]
ok("3.4 a control is consumed, not reused", len(ctrl) == len(set(ctrl)), ctrl)

ok("3.5 the order is deterministic",
   [f["id"] for f, _ in cdd.match_controls(ITEMS, ["simple_4", "simple_1"],
                                           lambda it: LEN[it["id"]], 10)[0]]
   == [f["id"] for f, _ in pairs2], "")

pairs3, missing3, _ = cdd.match_controls(ITEMS, ["simple_1", "not_an_item"],
                                         lambda it: LEN[it["id"]], 10)
ok("3.6 an id that is not in the scored set is reported, not silently dropped",
   missing3 == ["not_an_item"] and len(pairs3) == 1, (missing3, len(pairs3)))

_, _, un4 = cdd.match_controls([{"id": "solo", "category": "only"}], ["solo"],
                               lambda it: 1, 10)
ok("3.7 a category with no unflagged sibling is reported", un4 == ["solo"], un4)

ok("3.8 max_pairs caps the design",
   len(cdd.match_controls(ITEMS, ["simple_1", "simple_4"],
                          lambda it: LEN[it["id"]], 1)[0]) == 1, "")

# ---------------------------------------------------------------------------- 4
# The id list, inline and from storage, including the shape the decon job actually writes.
d = tempfile.mkdtemp()
with open(os.path.join(d, "ids.json"), "w", encoding="utf-8") as fh:
    json.dump({"contaminated_eval_items": ["a", "b"]}, fh)
ok("4.1 inline JSON", cdd.load_id_list('["x","y"]', lambda o: o) == ["x", "y"])
ok("4.2 empty is empty", cdd.load_id_list("", lambda o: o) == [])
ok("4.3 a storage object holding the decon field",
   cdd.load_id_list("ids.json", lambda o: os.path.join(d, o)) == ["a", "b"])

# 3.8 -- the duplicate id. BFCL v3 ships two different `live_relevance` questions under
# `live_relevance_3-3-0`. Keying on the bare id drops one; keying on the occurrence keeps both
# and, when the id is flagged, keeps either from being handed back as a clean control.
_dup = [{"id": "live_relevance_3-3-0", "category": "live_relevance"},
        {"id": "live_relevance_3-3-0", "category": "live_relevance"},
        {"id": "live_relevance_9-9-0", "category": "live_relevance"},
        {"id": "live_relevance_9-9-1", "category": "live_relevance"}]
for _i, _it in enumerate(_dup):
    _it["k"] = "%s#%d" % (_it["id"], 1 if _i != 1 else 2)
_pairs, _missing, _unmatched = cdd.match_controls(
    _dup, ["live_relevance_3-3-0"], lambda it: 100, 0, key=lambda it: it["k"])
ok("3.8 a duplicated id flags both occurrences and neither becomes a control",
   ([p[0]["k"] for p in _pairs], [p[1]["id"] for p in _pairs], _missing, _unmatched) ==
   (["live_relevance_3-3-0#1", "live_relevance_3-3-0#2"],
    ["live_relevance_9-9-0", "live_relevance_9-9-1"], [], []),
   (_pairs, _missing, _unmatched))

print("\n%d checks, %d failed" % (N, len(FAILED)))
sys.exit(1 if FAILED else 0)
