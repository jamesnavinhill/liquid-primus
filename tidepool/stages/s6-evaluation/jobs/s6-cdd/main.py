"""s6.4 — the paired peakedness comparison, base against tuned.

The generation half of this test runs inside the evaluation pack, where each arm is its own
process on one card and no process can see another's items. Every statistic worth reporting
needs both arms at once, so they live here, in a CPU job, exactly as `s5-compare` carries the
paired statistics for the pack that `s5-eval` generates.

WHAT THE RAW NUMBER IS NOT. Contamination Detection via output Distribution (`2402.15938v3`)
samples a model n times at temperature t, measures the edit distance from each sample to the
greedy completion, and calls an item leaked when an unusually large fraction of samples sit
within alpha of it. The method was validated on free-form text, where a peaked output
distribution is genuinely surprising. Tool calling is not free-form: the answer is a function
name from a supplied list and a small object of typed arguments, and a model that has merely
learned the format will emit near-identical strings at any temperature. Our own fixtures
assert the degenerate case directly, that a fixed schema alone drives Peak to 1.0 with no
contamination anywhere. So a high peakedness figure on this task is uninformative on its own,
and the reported result is never the raw number.

WHAT IT IS. Every item is sampled on BOTH arms: `B1`, the published base checkpoint that
never saw the training mix, and `R3`, the tuned checkpoint whose training rows the
decontamination scan searched. Schema rigidity is a property of the task, so it is present in
both arms and cancels in the difference. The reported figure is the paired difference, and the
sign test over item-level differences is the honest test of it because the differences are
neither normal nor independent of item difficulty.

WHY THAT IS STILL NOT ENOUGH, AND WHAT THE STRATA FIX. Fine-tuning collapses output entropy on
its own. An arm trained on any tool-calling data at all will be more peaked than its base on
EVERY tool-calling item, contaminated or not, so a uniform positive difference is the expected
result of ordinary supervised fine-tuning and says nothing about leakage. The design is
therefore an interaction, not a main effect: half the items are eval items whose text appears
in a shipped training TARGET, the other half are matched siblings with no overlap at all,
matched on category and then on nearest prompt length. Contamination predicts a LARGER lift on
the overlap stratum than on the control stratum. Entropy collapse predicts the same lift on
both. The permutation test over the paired differences is what separates them, and it is the
number this substage reports.

Both arms must have seen identical prompt bytes for the pairing to mean anything, so each
item's prompt hash is compared across arms and a mismatch is an assertion failure rather than
a footnote.
"""

import json
import math
import os
import random

from lab import lab

lab.init()
CFG = lab.get_config()
OUT = os.path.join(os.getcwd(), "out")
os.makedirs(OUT, exist_ok=True)
FAILURES = []
NOTES = []


def log(msg):
    print(msg, flush=True)
    try:
        lab.log(msg)
    except Exception:                                              # noqa: BLE001
        pass


def C(key, default=None):
    v = CFG.get(key, default)
    if isinstance(v, str) and isinstance(default, int):
        return int(v)
    if isinstance(v, str) and isinstance(default, float):
        return float(v)
    if isinstance(v, str) and isinstance(default, bool):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return v


def check(name, cond, detail=""):
    if not cond:
        FAILURES.append("%s: %s" % (name, detail))
        log("ASSERTION FAILED %s: %s" % (name, detail))
    return bool(cond)


def save(obj, name):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as fh:
        if name.endswith(".json"):
            json.dump(obj, fh, indent=1, sort_keys=True, ensure_ascii=False)
        else:
            fh.write(obj)
    try:
        lab.save_artifact(path)
        log("saved artifact %s (%d bytes)" % (name, os.path.getsize(path)))
    except Exception as exc:                                       # noqa: BLE001
        log("save_artifact failed for %s: %s" % (name, exc))
    return path


# ---------------------------------------------------------------- statistics

def sign_test_exact(minus, plus):
    """Two-sided exact sign test. Ties are excluded by the caller, as the test requires.

    Exact rather than normal-approximate because the paired sets here are small by design --
    sixty pairs is the configured ceiling -- and the normal approximation is poor in the tail
    that decides the result.
    """
    n = minus + plus
    if n == 0:
        return 1.0
    k = min(minus, plus)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def permutation_diff(a, b, resamples=20000, seed=0):
    """Difference of means with a two-sided label-permutation p-value.

    Used for the interaction: `a` is the per-item paired difference on the overlap stratum and
    `b` the same on the control stratum, so the statistic is a difference of differences and
    the null being shuffled is that an item's stratum does not predict how much the tuned arm
    gained over the base. Permutation rather than a t-test because peakedness is bounded on
    [0, 1] and saturates against the upper bound on exactly the items of interest.
    """
    if not a or not b:
        return {"diff": 0.0, "p": 1.0, "resamples": 0, "n_a": len(a), "n_b": len(b)}
    obs = sum(a) / len(a) - sum(b) / len(b)
    pool = list(a) + list(b)
    na = len(a)
    rng = random.Random(seed)
    hits = 0
    for _ in range(resamples):
        rng.shuffle(pool)
        d = sum(pool[:na]) / na - sum(pool[na:]) / (len(pool) - na)
        if abs(d) >= abs(obs) - 1e-12:
            hits += 1
    return {"diff": round(obs, 6), "p": round((hits + 1) / (resamples + 1), 6),
            "resamples": resamples, "n_a": na, "n_b": len(b)}


def mean(xs):
    return (sum(xs) / len(xs)) if xs else 0.0


# ---------------------------------------------------------------- loading

def load_items(path):
    """One arm's `cdd_items.jsonl`, keyed by occurrence.

    Keyed on `key` and not on `id`: BFCL v3 ships two different `live_relevance` questions
    under the id `live_relevance_3-3-0`, and a dict keyed on the bare id would pair one arm's
    first question against the other arm's second. Older rows without a `key` field fall back
    to their id with an occurrence counter, which reproduces the same convention.
    """
    rows, seen = {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            k = r.get("key")
            if not k:
                seen[r["id"]] = seen.get(r["id"], 0) + 1
                k = "%s#%d" % (r["id"], seen[r["id"]])
            rows[k] = r
    return rows


def fetch(obj, dest):
    """Pull one storage object into its own fresh directory.

    Its own directory because `lab storage download` into a populated one has been observed to
    APPEND to an existing same-named file rather than truncate it, which would turn two arms'
    item files into one file of doubled length and pair every item against itself.
    """
    os.makedirs(dest, exist_ok=True)
    local = lab.storage_download(obj)
    return local


# ---------------------------------------------------------------- comparison

def compare(base_arm, tuned_arm, base, tuned, xi, resamples, seed, examples):
    shared = sorted(set(base) & set(tuned))
    check("cdd_shared_items", len(shared) > 0,
          "%d base, %d tuned, %d shared" % (len(base), len(tuned), len(shared)))
    only_base = sorted(set(base) - set(tuned))
    only_tuned = sorted(set(tuned) - set(base))
    if only_base or only_tuned:
        NOTES.append("%d item(s) only in %s and %d only in %s; the comparison uses the %d "
                     "they share" % (len(only_base), base_arm, len(only_tuned), tuned_arm,
                                     len(shared)))

    # Identical prompt bytes on both sides, or the pairing compares two different questions.
    sha_mismatch = [k for k in shared
                    if base[k].get("prompt_sha") != tuned[k].get("prompt_sha")]
    check("cdd_same_prompt_bytes", not sha_mismatch,
          "%d item(s) differ, first: %s" % (len(sha_mismatch), sha_mismatch[:3]))
    strat_mismatch = [k for k in shared if base[k].get("stratum") != tuned[k].get("stratum")]
    check("cdd_same_strata", not strat_mismatch,
          "%d item(s) differ, first: %s" % (len(strat_mismatch), strat_mismatch[:3]))

    per_item, by_stratum = [], {}
    for k in shared:
        b, t = base[k], tuned[k]
        d = t["peak"] - b["peak"]
        row = {"key": k, "id": b.get("id", k.split("#")[0]),
               "category": b.get("category", ""), "stratum": b.get("stratum", "unstratified"),
               "base_peak": round(b["peak"], 6), "tuned_peak": round(t["peak"], 6),
               "diff": round(d, 6),
               "base_exact": b.get("n_exact", 0), "tuned_exact": t.get("n_exact", 0),
               "base_distinct": b.get("distinct_samples", 0),
               "tuned_distinct": t.get("distinct_samples", 0),
               "base_leaked": bool(b["peak"] > xi), "tuned_leaked": bool(t["peak"] > xi)}
        per_item.append(row)
        by_stratum.setdefault(row["stratum"], []).append(row)

    def block(rows):
        diffs = [r["diff"] for r in rows]
        plus = sum(1 for d in diffs if d > 0)
        minus = sum(1 for d in diffs if d < 0)
        ties = sum(1 for d in diffs if d == 0)
        return {"n": len(rows),
                "base_avg_peak": round(mean([r["base_peak"] for r in rows]), 6),
                "tuned_avg_peak": round(mean([r["tuned_peak"] for r in rows]), 6),
                "mean_diff": round(mean(diffs), 6),
                "median_diff": round(sorted(diffs)[len(diffs) // 2], 6) if diffs else 0.0,
                "n_up": plus, "n_down": minus, "n_tied": ties,
                "sign_test_p": round(sign_test_exact(minus, plus), 6),
                "base_leak_ratio": round(mean([1.0 if r["base_leaked"] else 0.0
                                               for r in rows]), 6),
                "tuned_leak_ratio": round(mean([1.0 if r["tuned_leaked"] else 0.0
                                                for r in rows]), 6),
                # Exact repeats and distinct-sample counts are reported alongside Peak because
                # Peak saturates: once every sample is within alpha of the greedy text it can
                # go no higher, and these two keep moving after it stops.
                "base_mean_exact": round(mean([r["base_exact"] for r in rows]), 4),
                "tuned_mean_exact": round(mean([r["tuned_exact"] for r in rows]), 4),
                "base_mean_distinct": round(mean([r["base_distinct"] for r in rows]), 4),
                "tuned_mean_distinct": round(mean([r["tuned_distinct"] for r in rows]), 4)}

    overall = block(per_item)
    strata = {name: block(rows) for name, rows in sorted(by_stratum.items())}

    # The interaction. Absent strata means the generation pass had no contaminated id list,
    # which leaves a main effect that cannot separate leakage from ordinary entropy collapse.
    # `background` items are outside the matched design and belong to the pooled reading only;
    # folding them into either arm of the interaction would compare matched items against
    # unmatched ones and call the difference contamination.
    over = [r["diff"] for r in by_stratum.get("overlap", [])]
    ctrl = [r["diff"] for r in by_stratum.get("control", [])]
    if over and ctrl:
        interaction = permutation_diff(over, ctrl, resamples=resamples, seed=seed)
        interaction["stratified"] = True
    else:
        interaction = {"stratified": False, "diff": 0.0, "p": 1.0, "resamples": 0,
                       "n_a": len(over), "n_b": len(ctrl)}
        NOTES.append("no overlap/control strata in the item files, so only the pooled paired "
                     "difference is available and the contamination question is not answered "
                     "by it: an equal lift on contaminated and clean items is what ordinary "
                     "fine-tuning produces")

    # Saturation guard. If the base arm is already at the ceiling on most items, the paired
    # difference has nowhere to go and a null result is a property of the statistic.
    sat = sum(1 for r in per_item if r["base_peak"] >= 0.999)
    if per_item and sat / len(per_item) > 0.5:
        NOTES.append("the base arm is already at Peak >= 0.999 on %d of %d items (%.1f%%), so "
                     "the paired difference is bounded near zero by the statistic itself; read "
                     "the exact-repeat and distinct-sample columns instead"
                     % (sat, len(per_item), 100.0 * sat / len(per_item)))

    movers = sorted(per_item, key=lambda r: -abs(r["diff"]))[:examples]
    return {"base_arm": base_arm, "tuned_arm": tuned_arm, "xi": xi,
            "n_items": len(per_item), "overall": overall, "by_stratum": strata,
            "interaction": interaction, "largest_movers": movers,
            "only_in_base": only_base[:20], "only_in_tuned": only_tuned[:20]}, per_item


# ---------------------------------------------------------------- report

def render(res):
    o, L = res["overall"], []
    L.append("# `s6.4` — peakedness, %s against %s" % (res["tuned_arm"], res["base_arm"]))
    L.append("")
    L.append("%d paired items, xi = %g. The raw peakedness figure is not the result: a fixed "
             "tool-call schema drives it toward 1.0 with no contamination present, which is "
             "why every item is measured on both arms and the reported quantity is the "
             "difference." % (res["n_items"], res["xi"]))
    L.append("")
    L.append("## Pooled")
    L.append("")
    L.append("| | base %s | tuned %s | diff |" % (res["base_arm"], res["tuned_arm"]))
    L.append("| --- | ---: | ---: | ---: |")
    L.append("| avg Peak | %.4f | %.4f | %+.4f |"
             % (o["base_avg_peak"], o["tuned_avg_peak"], o["mean_diff"]))
    L.append("| leak ratio at xi | %.4f | %.4f | %+.4f |"
             % (o["base_leak_ratio"], o["tuned_leak_ratio"],
                o["tuned_leak_ratio"] - o["base_leak_ratio"]))
    L.append("| mean exact repeats | %.2f | %.2f | %+.2f |"
             % (o["base_mean_exact"], o["tuned_mean_exact"],
                o["tuned_mean_exact"] - o["base_mean_exact"]))
    L.append("| mean distinct samples | %.2f | %.2f | %+.2f |"
             % (o["base_mean_distinct"], o["tuned_mean_distinct"],
                o["tuned_mean_distinct"] - o["base_mean_distinct"]))
    L.append("")
    L.append("Sign test over item differences: %d up, %d down, %d tied, p = %.4f."
             % (o["n_up"], o["n_down"], o["n_tied"], o["sign_test_p"]))
    L.append("")
    L.append("## By stratum")
    L.append("")
    L.append("| stratum | n | base | tuned | mean diff | up/down/tied | sign p |")
    L.append("| --- | ---: | ---: | ---: | ---: | :---: | ---: |")
    for name, b in res["by_stratum"].items():
        L.append("| %s | %d | %.4f | %.4f | %+.4f | %d/%d/%d | %.4f |"
                 % (name, b["n"], b["base_avg_peak"], b["tuned_avg_peak"], b["mean_diff"],
                    b["n_up"], b["n_down"], b["n_tied"], b["sign_test_p"]))
    L.append("")
    it = res["interaction"]
    L.append("## Interaction")
    L.append("")
    if it["stratified"]:
        L.append("Difference of differences, overlap minus control: %+.4f, permutation p = "
                 "%.4f over %d resamples (%d overlap items, %d control items). Contamination "
                 "predicts a positive value here. A lift of the same size on both strata is "
                 "what ordinary post-training entropy collapse looks like."
                 % (it["diff"], it["p"], it["resamples"], it["n_a"], it["n_b"]))
    else:
        L.append("Not available: the generation pass ran without strata, so the pooled "
                 "difference above is a main effect and does not separate leakage from "
                 "entropy collapse.")
    L.append("")
    L.append("## Largest movers")
    L.append("")
    L.append("| item | category | stratum | base | tuned | diff |")
    L.append("| --- | --- | --- | ---: | ---: | ---: |")
    for r in res["largest_movers"]:
        L.append("| `%s` | %s | %s | %.4f | %.4f | %+.4f |"
                 % (r["key"], r["category"], r["stratum"], r["base_peak"], r["tuned_peak"],
                    r["diff"]))
    L.append("")
    if NOTES:
        L.append("## Notes")
        L.append("")
        for n in NOTES:
            L.append("- %s" % n)
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- driver

def main():
    base_arm = str(C("base_arm", "B1"))
    tuned_arm = str(C("tuned_arm", "R3"))
    prefix = str(C("items_prefix", "")).rstrip("/")
    objects = C("arm_objects", "") or ""
    if isinstance(objects, str):
        objects = json.loads(objects) if objects.strip() else {}
    xi = float(C("xi", 0.01))
    resamples = int(C("resamples", 20000))
    seed = int(C("seed", 0))
    examples = int(C("examples", 15))

    loaded = {}
    for arm in (base_arm, tuned_arm):
        obj = objects.get(arm) or "%s/%s/cdd_items.jsonl" % (prefix, arm)
        log("fetching %s for %s" % (obj, arm))
        local = fetch(obj, os.path.join(os.getcwd(), "in", arm))
        loaded[arm] = load_items(local)
        log("  %s: %d item(s)" % (arm, len(loaded[arm])))

    res, per_item = compare(base_arm, tuned_arm, loaded[base_arm], loaded[tuned_arm],
                            xi, resamples, seed, examples)
    res["notes"] = NOTES
    res["failures"] = FAILURES
    save(res, "cdd_compare.json")
    save("\n".join(json.dumps(r, sort_keys=True) for r in per_item), "cdd_paired.jsonl")
    save(render(res), "cdd_compare.md")

    it = res["interaction"]
    msg = ("s6.4 peakedness: %d paired items, pooled mean difference %+.4f (sign test p=%.4f); "
           "overlap-minus-control %+.4f (permutation p=%.4f, %s). %d assertion failure(s), "
           "%d note(s)."
           % (res["n_items"], res["overall"]["mean_diff"], res["overall"]["sign_test_p"],
              it["diff"], it["p"], "stratified" if it["stratified"] else "UNSTRATIFIED",
              len(FAILURES), len(NOTES)))
    log(msg)
    # `finish`, not `job_complete`: the latter is absent from the SDK on the worker and is how
    # job 154c72a4 came to die after writing correct artifacts. The score is the assertion
    # failure count, so the dashboard row says at a glance whether the pass checked out.
    try:
        lab.finish(message=msg, score=len(FAILURES))
    except Exception:                                              # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
