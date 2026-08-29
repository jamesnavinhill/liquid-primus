"""Black-box memorization check: does the tuned model collapse onto one answer?

`plan.md` pre-registers a decontamination re-check "including the black-box peakedness test
(`2402.15938v3`)". The n-gram half of that answers whether benchmark text is IN the training
data. It cannot answer the question the reader actually has, which is whether any of it was
MEMORIZED -- a shared instruction preamble and a shared JSON schema both produce long exact
overlaps and teach nothing about any answer. CDD answers that from sampled text alone: a model
that has memorized an item keeps returning the same answer when you ask it to be random.

The method, exactly as the paper defines it. Take the greedy completion s_0, then n samples at
temperature t. Let l be the longest sample, capped at 100. Peak(x) is the fraction of samples
whose edit distance to s_0 is at most alpha*l, and an item is called leaked when Peak(x) > xi.
The paper's defaults are alpha=0.05, xi=0.01, n=51, t=0.8, and they are the defaults here.

WHAT THIS JOB ADDS, AND WHY IT IS NOT OPTIONAL. Our own reading notes on that paper raised an
objection to using it here at all: the "answer" to a BFCL item is a tool call against a fixed
schema, so a model with no contamination whatsoever should still produce near-identical text
every time, and a raw Peak number would read as memorization when it is only the schema being
rigid. The paper never tested a rigid-schema task, and it hand-sets xi per domain.

So the raw number is not the result. Every item is run through BOTH the untrained base and the
tuned checkpoint, and the statistic is the PAIRED DIFFERENCE. The base has never seen a token of
our training mix, so whatever peakedness it shows on an item is that item's schema rigidity, and
the tuned model's excess over it is what training added. A design where the negative control is
a different model on the same items needs no threshold to be interpretable.

The second layer is the stratification. `cdd_contaminated_ids` names the eval items the n-gram
pass found inside a training TARGET; the controls are items with no overlap at all, matched to
them one for one on category and prompt length. Contamination predicts an interaction, not a
main effect: the tuned model should pull away from the base further on the overlapping items
than on the matched ones. A uniform lift across both strata is not contamination, it is what
supervised fine-tuning does to output entropy, and reporting it as contamination would be the
error this design exists to prevent.

Nothing here is compared against a number measured elsewhere, but it is still taken on the same
card and the same serving path as everything else full-precision in this project, because the
two arms must be comparable to each other and the cheapest way to guarantee that is to change
nothing.
"""

import json
import math
import os
import random


# ---------------------------------------------------------------------------- edit distance

def bounded_edit_distance(a, b, cutoff):
    """Levenshtein distance between two strings, or cutoff+1 once it is certain to exceed it.

    Banded: only the diagonal band of width 2*cutoff+1 can ever hold a value at or below the
    cutoff, so the rest of the matrix is not computed. The whole point of Peak is an indicator
    at a small threshold, and the exact value of a distance of 900 is not used anywhere, so
    paying O(len^2) to learn it would be several minutes of card time spent on nothing.

    Returns the true distance when it is <= cutoff, and cutoff+1 otherwise. That makes
    `d == 0` (an exact repeat) and `d <= threshold` both exact, which are the two things
    reported.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > cutoff:
        return cutoff + 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    if la == 0:
        # An empty string against a non-empty one is lb deletions, and the band below has no
        # row to walk. Left to the loop it returned cutoff+1 for every such pair, which is
        # correct only when lb genuinely exceeds the cutoff and wrong whenever it does not.
        return lb if lb <= cutoff else cutoff + 1
    inf = cutoff + 1
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        lo = max(1, j - cutoff)
        hi = min(la, j + cutoff)
        cur = [inf] * (la + 1)
        cur[lo - 1] = inf if lo - 1 > 0 else j
        bj = b[j - 1]
        best = inf
        for i in range(lo, hi + 1):
            cost = 0 if a[i - 1] == bj else 1
            v = min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + cost)
            cur[i] = v
            if v < best:
                best = v
        if best > cutoff:
            return cutoff + 1
        prev = cur
    return prev[la] if prev[la] <= cutoff else cutoff + 1


def peakedness(greedy, samples, alpha=0.05, l_cap=100):
    """Peak(M;x) and the pieces it is built from, for one item on one arm."""
    if not samples:
        return {"n_samples": 0, "peak": 0.0, "l": 0, "threshold": 0,
                "n_within": 0, "n_exact": 0, "distinct_samples": 0}
    l = min(max(len(s) for s in samples), l_cap)
    thr = int(math.floor(alpha * l))
    within = exact = 0
    for s in samples:
        d = bounded_edit_distance(s, greedy, thr)
        if d <= thr:
            within += 1
        if d == 0:
            exact += 1
    return {"n_samples": len(samples), "peak": within / len(samples), "l": l,
            "threshold": thr, "n_within": within, "n_exact": exact,
            "distinct_samples": len(set(samples))}


# The cross-arm statistics do NOT live here. This module runs inside the evaluation pack,
# where each arm is its own process and no process can see another's items; the paired sign
# test and the stratum permutation test need both arms at once and live in the `s6-cdd`
# comparison job, alongside every other paired statistic in this project.

# ---------------------------------------------------------------------------- item selection

def load_id_list(raw, storage):
    """A JSON list of ids, given inline or as a storage object holding one."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        return [str(x) for x in json.loads(raw)]
    path = storage(raw)
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if isinstance(doc, dict):
        doc = doc.get("contaminated_eval_items") or doc.get("ids") or []
    return [str(x) for x in doc]


def match_controls(items, flagged_ids, prompt_len, max_pairs, key=None):
    """One control per flagged item: same category, nearest prompt length, no overlap.

    Matching on category matters more than it looks. The eleven BFCL categories differ in how
    much freedom the answer has -- `irrelevance` invites prose, `parallel_multiple` forces a
    fixed shape -- so an unmatched control set would compare peakedness across task types and
    call the difference contamination. Length is the tiebreak because a longer prompt pins the
    answer down harder.

    `key` names each item for ordering and reporting, defaulting to its bare BFCL id. The
    caller passes an occurrence key because BFCL v3 ships two different `live_relevance`
    questions under the id `live_relevance_3-3-0`, and keying a dict on the bare id drops one
    of them silently. Flag membership is still tested on the bare id: the decontamination scan
    reports the id it saw, and where an id covers two questions, flagging both is the
    conservative reading and keeps either from landing in the control pool.

    Deterministic: flagged items are taken in sorted key order and each control is consumed, so
    the same inputs always give the same pairs.
    """
    key = key or (lambda it: it["id"])
    flag = set(flagged_ids)
    flagged = sorted([it for it in items if it["id"] in flag], key=key)
    missing = sorted(flag - {it["id"] for it in items})
    pool = {}
    for it in items:
        if it["id"] in flag:
            continue
        pool.setdefault(it["category"], []).append(it)
    pairs, unmatched = [], []
    for f in (flagged[:max_pairs] if max_pairs else flagged):
        cands = pool.get(f["category"]) or []
        if not cands:
            unmatched.append(key(f))
            continue
        target = prompt_len(f)
        best = min(cands, key=lambda c: (abs(prompt_len(c) - target), key(c)))
        cands.remove(best)
        pairs.append((f, best))
    return pairs, missing, unmatched
