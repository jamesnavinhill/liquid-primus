"""The paired peakedness comparison over fixtures whose answer is known before the code runs.

Five things this job could get wrong while still printing a plausible table. The pairing could
key on the bare BFCL id, which pairs one arm's `live_relevance_3-3-0` against the other arm's
second question of the same id. The sign test could count ties as evidence, which manufactures
significance out of items that did not move. It could use the normal approximation, which is
wrong in the tail that decides the result at these sample sizes. The permutation p-value could
reach exactly zero, which no permutation test may report. And the interaction could quietly
fall back to a pooled main effect when the strata are missing, which is the one result this
substage must never present as an answer to the contamination question.

Each fixture below is built so that one of those errors changes a number checked here.
"""

import json
import math
import os
import sys
import tempfile
import types

ROOT = tempfile.mkdtemp(prefix="s6cdd-")
STORE = os.path.join(ROOT, "store")
os.makedirs(STORE, exist_ok=True)
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

    def storage_download(self, obj):
        path = os.path.join(STORE, obj)
        if not os.path.exists(path):
            raise RuntimeError("no such object: %s" % obj)
        return path

    def save_artifact(self, path):
        ARTIFACTS.append(os.path.basename(path))

    def finish(self, message=None, score=None):
        FINISHED.append((message, score))

    # DELIBERATELY ABSENT: `job_complete`. The SDK on the worker does not have it, and job
    # 154c72a4 was marked FAILED on that call after writing correct artifacts. The stub
    # reproduces that SDK so the fixtures fail if the closing call is ever changed back.


_mod = types.ModuleType("lab")
_mod.lab = _Lab()
sys.modules["lab"] = _mod

os.chdir(ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main                                                        # noqa: E402

PASS = FAIL = 0


def ok(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok   %s" % name)
    else:
        FAIL += 1
        print("FAIL %s   %s" % (name, detail))


# ---------------------------------------------------------------- 1. the sign test

ok("1.1 no evidence from nothing", main.sign_test_exact(0, 0) == 1.0)
ok("1.2 a perfect split is p=1", main.sign_test_exact(5, 5) == 1.0)
# 2 * (C(6,0) + C(6,1)) / 64 = 2 * 7 / 64.
ok("1.3 six pairs, one against, matches the closed form",
   abs(main.sign_test_exact(1, 5) - (2.0 * 7 / 64)) < 1e-12, main.sign_test_exact(1, 5))
ok("1.4 the test is symmetric in its arguments",
   main.sign_test_exact(1, 5) == main.sign_test_exact(5, 1))
# All twenty on one side: 2 / 2^20. A normal approximation lands near 1e-5 and would pass a
# loose tolerance, so the fixture checks the exact value.
ok("1.5 twenty of twenty is exactly 2^-19",
   abs(main.sign_test_exact(0, 20) - 2.0 ** -19) < 1e-15, main.sign_test_exact(0, 20))
ok("1.6 p never exceeds 1", all(main.sign_test_exact(a, b) <= 1.0
                                for a in range(6) for b in range(6)))
ok("1.7 the tail is monotone in how lopsided the split is",
   main.sign_test_exact(0, 10) < main.sign_test_exact(2, 8) < main.sign_test_exact(4, 6))

# ---------------------------------------------------------------- 2. the permutation test

_sep = main.permutation_diff([1.0] * 12, [0.0] * 12, resamples=2000, seed=1)
ok("2.1 a clean separation reports the true difference", _sep["diff"] == 1.0, _sep)
ok("2.2 a permutation p-value is never zero", _sep["p"] > 0.0, _sep)
ok("2.3 the floor is 1/(resamples+1), as reported at six decimals",
   _sep["p"] == round(1.0 / 2001, 6), _sep)
_same = main.permutation_diff([0.2] * 10, [0.2] * 10, resamples=500, seed=1)
ok("2.4 identical strata give no difference and no significance",
   _same["diff"] == 0.0 and _same["p"] == 1.0, _same)
ok("2.5 the same seed gives the same p twice",
   main.permutation_diff([0.1, 0.4, 0.2], [0.3, 0.0, 0.5], resamples=300, seed=7) ==
   main.permutation_diff([0.1, 0.4, 0.2], [0.3, 0.0, 0.5], resamples=300, seed=7))
ok("2.6 an empty stratum is reported rather than divided by",
   main.permutation_diff([], [0.1], resamples=10, seed=0)["resamples"] == 0)
_neg = main.permutation_diff([0.0] * 8, [1.0] * 8, resamples=1000, seed=2)
ok("2.7 the sign of the difference is preserved", _neg["diff"] == -1.0, _neg)

# ---------------------------------------------------------------- 3. loading and keying

_dup = os.path.join(ROOT, "dup.jsonl")
with open(_dup, "w", encoding="utf-8") as fh:
    for r in [{"id": "live_relevance_3-3-0", "peak": 0.1},
              {"id": "live_relevance_3-3-0", "peak": 0.9},
              {"id": "simple_0", "peak": 0.5}]:
        fh.write(json.dumps(r) + "\n")
_loaded = main.load_items(_dup)
ok("3.1 a file with a duplicated id keeps both questions", len(_loaded) == 3, sorted(_loaded))
ok("3.2 the occurrence convention is <id>#k",
   sorted(_loaded) == ["live_relevance_3-3-0#1", "live_relevance_3-3-0#2", "simple_0#1"],
   sorted(_loaded))
ok("3.3 the second occurrence keeps its own peak",
   _loaded["live_relevance_3-3-0#2"]["peak"] == 0.9)
_keyed = os.path.join(ROOT, "keyed.jsonl")
with open(_keyed, "w", encoding="utf-8") as fh:
    fh.write(json.dumps({"id": "a", "key": "a#7", "peak": 0.3}) + "\n")
    fh.write("\n")
ok("3.4 an explicit key is honoured and blank lines are skipped",
   list(main.load_items(_keyed)) == ["a#7"], list(main.load_items(_keyed)))


# ---------------------------------------------------------------- 4. the comparison

def arm(peaks, stratum_of, sha_of=None, exact=0, distinct=1):
    rows = {}
    for k, p in peaks.items():
        rows[k] = {"id": k.split("#")[0], "key": k, "category": "live_simple",
                   "stratum": stratum_of[k], "peak": p, "n_exact": exact,
                   "distinct_samples": distinct,
                   "prompt_sha": (sha_of or {}).get(k, "sha" + k)}
    return rows


_keys = ["o%d#1" % i for i in range(6)] + ["c%d#1" % i for i in range(6)]
_strata = {k: ("overlap" if k.startswith("o") else "control") for k in _keys}
# The contamination shape: the tuned arm gains 0.30 on the overlap items and 0.05 on the
# controls, so the pooled lift is positive on both and the interaction is what distinguishes
# them. A job that reported only the pooled number would call both of these the same result.
_base = arm({k: 0.40 for k in _keys}, _strata)
_tuned = arm({k: 0.70 if k.startswith("o") else 0.45 for k in _keys}, _strata)
main.FAILURES[:] = []
main.NOTES[:] = []
_res, _rows = main.compare("B1", "R3", _base, _tuned, 0.01, 2000, 0, 5)
ok("4.1 no assertion failures on well-formed input", main.FAILURES == [], main.FAILURES)
ok("4.2 every shared item is paired", _res["n_items"] == 12, _res["n_items"])
ok("4.3 the pooled lift is the mean of both strata",
   abs(_res["overall"]["mean_diff"] - 0.175) < 1e-9, _res["overall"]["mean_diff"])
ok("4.4 the pooled sign test sees twelve items move up",
   (_res["overall"]["n_up"], _res["overall"]["n_down"], _res["overall"]["n_tied"]) ==
   (12, 0, 0), _res["overall"])
ok("4.5 each stratum carries its own lift",
   (_res["by_stratum"]["overlap"]["mean_diff"],
    _res["by_stratum"]["control"]["mean_diff"]) == (0.3, 0.05), _res["by_stratum"])
ok("4.6 the interaction is the difference of differences",
   abs(_res["interaction"]["diff"] - 0.25) < 1e-9, _res["interaction"])
ok("4.7 the interaction is significant when only the overlap stratum moves",
   _res["interaction"]["stratified"] and _res["interaction"]["p"] < 0.01,
   _res["interaction"])

# Entropy collapse: the SAME lift on both strata. The pooled number is identical in size to a
# real contamination signal, and only the interaction tells them apart. This is the fixture
# that justifies the whole stratified design.
main.FAILURES[:] = []
main.NOTES[:] = []
_flat = arm({k: 0.70 for k in _keys}, _strata)
_res2, _ = main.compare("B1", "R3", _base, _flat, 0.01, 2000, 0, 5)
ok("4.8 a uniform lift still moves every item and passes the pooled sign test",
   _res2["overall"]["n_up"] == 12 and _res2["overall"]["sign_test_p"] < 0.001,
   _res2["overall"])
ok("4.9 but the interaction is flat, which is what entropy collapse looks like",
   _res2["interaction"]["diff"] == 0.0 and _res2["interaction"]["p"] == 1.0,
   _res2["interaction"])

# A prompt-hash mismatch means the two arms answered different questions under one key.
main.FAILURES[:] = []
main.NOTES[:] = []
_odd = arm({k: 0.5 for k in _keys}, _strata, sha_of={"o0#1": "different"})
main.compare("B1", "R3", _base, _odd, 0.01, 200, 0, 5)
ok("4.10 differing prompt bytes are an assertion failure",
   any(f.startswith("cdd_same_prompt_bytes") for f in main.FAILURES), main.FAILURES)

# No strata at all. The pooled difference survives; the claim must not.
main.FAILURES[:] = []
main.NOTES[:] = []
_un = {k: "unstratified" for k in _keys}
_res3, _ = main.compare("B1", "R3", arm({k: 0.4 for k in _keys}, _un),
                        arm({k: 0.9 for k in _keys}, _un), 0.01, 200, 0, 5)
ok("4.11 an unstratified run reports no interaction",
   _res3["interaction"]["stratified"] is False, _res3["interaction"])
ok("4.12 and says in a note that the pooled number does not answer the question",
   any("entropy collapse" in n or "does not" in n or "not answered" in n
       for n in main.NOTES), main.NOTES)

# Saturation. A base arm already at the ceiling cannot be beaten, and a null there is a
# property of the statistic rather than a finding about the model.
main.FAILURES[:] = []
main.NOTES[:] = []
_sat_base = arm({k: 1.0 for k in _keys}, _strata, exact=51, distinct=1)
_sat_tuned = arm({k: 1.0 for k in _keys}, _strata, exact=51, distinct=1)
_res4, _ = main.compare("B1", "R3", _sat_base, _sat_tuned, 0.01, 200, 0, 5)
ok("4.13 a saturated base arm is called out rather than reported as a null",
   any("bounded near zero" in n for n in main.NOTES), main.NOTES)
ok("4.14 the exact-repeat column is carried so a saturated run still has a reading",
   _res4["overall"]["tuned_mean_exact"] == 51.0, _res4["overall"])

# Items present on one side only are dropped from the pairing and reported, never zero-filled.
main.FAILURES[:] = []
main.NOTES[:] = []
_short = {k: v for k, v in _tuned.items() if k != "o0#1"}
_res5, _ = main.compare("B1", "R3", _base, _short, 0.01, 200, 0, 5)
ok("4.15 an unpaired item is dropped, not treated as a zero difference",
   _res5["n_items"] == 11 and _res5["only_in_base"] == ["o0#1"], _res5["n_items"])

# Background items are outside the matched design. They belong to the pooled reading and must
# not enter either arm of the interaction, where they would put unmatched items against matched
# ones and let the difference be read as contamination.
main.FAILURES[:] = []
main.NOTES[:] = []
_bkeys = _keys + ["b%d#1" % i for i in range(8)]
_bstrata = dict(_strata)
_bstrata.update({k: "background" for k in _bkeys if k.startswith("b")})
_bbase = arm({k: 0.40 for k in _bkeys}, _bstrata)
_btuned = arm({k: 0.70 if k.startswith("o") else 0.45 for k in _bkeys}, _bstrata)
_res6, _ = main.compare("B1", "R3", _bbase, _btuned, 0.01, 2000, 0, 5)
ok("4.16 background items are pooled with the rest",
   _res6["n_items"] == 20 and _res6["by_stratum"]["background"]["n"] == 8,
   _res6["by_stratum"])
ok("4.17 but the interaction still compares only overlap against control",
   (_res6["interaction"]["n_a"], _res6["interaction"]["n_b"]) == (6, 6) and
   abs(_res6["interaction"]["diff"] - 0.25) < 1e-9, _res6["interaction"])

# ---------------------------------------------------------------- 5. the report and close-out

_md = main.render(_res)
ok("5.1 the report names both arms in its title",
   _md.splitlines()[0] == "# `s6.4` — peakedness, R3 against B1", _md.splitlines()[0])
ok("5.2 it warns that the raw figure is not the result",
   "not the result" in _md and "no contamination present" in _md)
ok("5.3 it prints the interaction with its p-value", "Difference of differences" in _md)
ok("5.4 an unstratified report refuses the interaction instead of printing one",
   "does not separate leakage" in main.render(_res3))

CONFIG.clear()
CONFIG.update({"base_arm": "B1", "tuned_arm": "R3", "items_prefix": "t/cdd",
               "resamples": 200, "examples": 3})
main.CFG = dict(CONFIG)
for _a, _rows_in in (("B1", _base), ("R3", _tuned)):
    _d = os.path.join(STORE, "t", "cdd", _a)
    os.makedirs(_d, exist_ok=True)
    with open(os.path.join(_d, "cdd_items.jsonl"), "w", encoding="utf-8") as fh:
        for _k in sorted(_rows_in):
            fh.write(json.dumps(_rows_in[_k], sort_keys=True) + "\n")
main.FAILURES[:] = []
main.NOTES[:] = []
ARTIFACTS[:] = []
FINISHED[:] = []
main.main()
ok("5.5 the driver saves all three artifacts",
   sorted(ARTIFACTS) == ["cdd_compare.json", "cdd_compare.md", "cdd_paired.jsonl"], ARTIFACTS)
ok("5.6 the pass is closed out on the real SDK call, scored by assertion failures",
   len(FINISHED) == 1 and FINISHED[0][1] == 0, FINISHED)
ok("5.7 the close-out message says whether the run was stratified",
   "stratified" in FINISHED[0][0] and "UNSTRATIFIED" not in FINISHED[0][0], FINISHED)
_paired = os.path.join(ROOT, "out", "cdd_paired.jsonl")
ok("5.8 the per-item file has one row per pair",
   sum(1 for _ in open(_paired, encoding="utf-8")) == 12)

print("")
print("%d checks, %d failed" % (PASS + FAIL, FAIL))
sys.exit(1 if FAIL else 0)
