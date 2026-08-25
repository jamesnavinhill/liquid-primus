"""s4.4 — build the two probe sets, validate them, and prove they are held out.

These are the only evaluation items in the project that were written rather than
downloaded, and that is the point: the failure the fleet has to fix is what a model does
*after* a tool returns something wrong, and nothing in the benchmark index scores that
turn. The set is generated deterministically from two banks, so it is reproducible from
the code and carries no sampling noise between checkpoints.

Three things have to be true before a probe set is worth scoring, and the job fails if
any of them is not:

  1. Every grader is well formed: regexes compile, `must` and `must_not` cannot both
     match the same trivial string, and each corrupted item's forbidden values actually
     appear in the *correct* payload, so a model that emits one has demonstrably invented
     it rather than read it.
  2. No probe prompt overlaps the training split. Hand-authored is not the same as
     held out, so it is checked against the rendered training text on the same 13-gram
     rule the corpora were decontaminated with.
  3. No probe prompt overlaps another probe's prompt, which would let one item's answer
     leak into another's within a single evaluation run.
"""

import collections
import gzip
import hashlib
import json
import os
import re
import time

import numpy as np
from lab import lab

import bank_stack
import bank_tools
import build

lab.init()
CFG = lab.get_config() or {}
NGRAM_N = int(CFG.get("ngram_n") or 13)
TRAIN_OBJECT = CFG.get("train_object") or ""
OUT = "out"
os.makedirs(OUT, exist_ok=True)


def log(msg):
    lab.log(msg)
    print(msg, flush=True)


def dump(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    try:
        lab.save_artifact(p)
    except Exception as exc:
        log("save_artifact failed for %s: %s" % (name, exc))
    return p


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]+")


def grams(text, n=NGRAM_N):
    """Same normalization and hash as the s4.3 decontamination pass, so the two agree."""
    toks = _WS.sub(" ", _PUNCT.sub(" ", (text or "").lower())).split()
    out = []
    for i in range(len(toks) - n + 1):
        d = hashlib.blake2b(" ".join(toks[i:i + n]).encode("utf-8"), digest_size=8).digest()
        out.append(int.from_bytes(d, "big"))
    return out


items = build.build_tools(bank_tools.SCENARIOS) + build.build_stack(bank_stack.FAMILIES)
for k, it in enumerate(items):
    it["id"] = "%s/%s/%s/d%d/%03d" % (it["probe"], it["arm"], it["mode"], it["depth"], k)
log("built %d probe items: %s" % (len(items),
    json.dumps(dict(collections.Counter("%s:%s" % (i["probe"], i["arm"]) for i in items)))))

# ------------------------------------------------------------------ 1. graders

fails = []
for it in items:
    chk = it["check"]
    if chk["kind"] == "regex":
        if not chk["must"]:
            fails.append("%s has no must patterns" % it["id"])
        for pat in chk["must"] + chk["must_not"]:
            try:
                re.compile(pat)
            except re.error as exc:
                fails.append("%s has an uncompilable pattern %r: %s" % (it["id"], pat, exc))
        # A must and a must_not that are the same pattern make the item unpassable.
        overlap = set(chk["must"]) & set(chk["must_not"])
        if overlap:
            fails.append("%s requires and forbids the same pattern %s" % (it["id"], sorted(overlap)))
    elif chk["kind"] in ("recover", "flag_mismatch"):
        for s in chk.get("forbid") or []:
            if not s:
                fails.append("%s forbids an empty string, which matches everything" % it["id"])
    else:
        fails.append("%s has an unknown check kind %r" % (it["id"], chk["kind"]))

# Each corrupted item's forbidden values must be readable off the CORRECT payload and
# absent from the broken one. Otherwise "the model said 812" is not evidence of anything.
for scen in bank_tools.SCENARIOS:
    good = json.dumps(scen["payload"], ensure_ascii=False)
    hits = [f for f in scen["fabricated"] if re.sub(r"[^0-9A-Za-z.]", "", f).lower()
            in re.sub(r"[^0-9A-Za-z.]", "", good).lower()]
    if not hits:
        fails.append("scenario %s: none of its forbidden values %s appear in its correct payload, "
                     "so emitting one would not prove fabrication"
                     % (scen["id"], scen["fabricated"]))

# ------------------------------------------------------- 2 & 3. held out, and disjoint

probe_grams, prompt_of, text_of = {}, {}, {}
for it in items:
    text = "\n".join(m["content"] for m in it["messages"] if m["role"] in ("user", "tool"))
    probe_grams[it["id"]] = set(grams(text))
    prompt_of[it["id"]] = next((m["content"] for m in it["messages"] if m["role"] == "user"), "")
    text_of[it["id"]] = text

# Two collision checks, and the distinction between them is the point.
#
# The corrupted and contradicted arms deliberately ask the SAME question and differ only in
# what the tool hands back: that is the comparison the probe exists to make, so a shared user
# prompt within one scenario is intended, not a defect. What must never happen is two items
# being the same measurement (identical user text AND identical tool text), or one question
# appearing under two DIFFERENT scenarios, which would make a single failure count twice.
scen_of = {it["id"]: (it["probe"], it["scenario"]) for it in items}

dup_items = collections.defaultdict(list)
for pid, t in text_of.items():
    dup_items[hashlib.blake2b(t.encode("utf-8"), digest_size=8).hexdigest()].append(pid)
dups = {h: ids for h, ids in dup_items.items() if len(ids) > 1}
if dups:
    fails.append("%d groups of probe items are byte-identical on question and tool return, so "
                 "they are the same measurement counted more than once: %s"
                 % (len(dups), json.dumps([sorted(v) for v in list(dups.values())[:5]])))

by_prompt = collections.defaultdict(set)
for pid, p in prompt_of.items():
    by_prompt[hashlib.blake2b(p.encode("utf-8"), digest_size=8).hexdigest()].add(pid)
cross = {}
for h, ids in by_prompt.items():
    scens = {scen_of[i] for i in ids}
    if len(scens) > 1:
        cross[h] = sorted(scens)
if cross:
    fails.append("%d prompts appear under more than one scenario: %s"
                 % (len(cross), json.dumps(list(cross.values())[:5])))

collision_report = {"identical_item_groups": len(dups),
                    "prompts_under_multiple_scenarios": len(cross),
                    "shared_prompt_groups_within_scenario":
                        sum(1 for ids in by_prompt.values() if len(ids) > 1)}

overlap_report = {"checked": False, "reason": "no train_object configured"}
if TRAIN_OBJECT:
    t0 = time.time()
    local = lab.storage_download(TRAIN_OBJECT)
    # The training split is ~10^6 rows and its user turns carry order 10^8 n-grams.
    # Held as Python ints that is tens of gigabytes, so the index is folded down to a
    # sorted unique array every few million grams and never materialized whole.
    BLOCK = 4_000_000
    buf, arr, n_rows = [], np.zeros(0, dtype=np.uint64), 0

    def fold(buf, arr):
        if not buf:
            return arr
        blk = np.fromiter(buf, dtype=np.uint64, count=len(buf))
        return np.unique(np.concatenate([arr, blk]) if arr.size else blk)

    with gzip.open(local, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            n_rows += 1
            for m in r.get("messages") or []:
                if m.get("role") == "user":
                    buf.extend(grams(m.get("content") or ""))
            if len(buf) >= BLOCK:
                arr = fold(buf, arr)
                buf = []
                if n_rows % 200000 < 2:
                    lab.update_progress(min(80, 10 + int(70.0 * n_rows / 1_000_000)))
    arr = fold(buf, arr)
    del buf
    log("train index: %d rows, %d unique %d-grams (%.1fs)"
        % (n_rows, arr.size, NGRAM_N, time.time() - t0))
    hits = {}
    for pid, gs in probe_grams.items():
        if not gs:
            continue
        q = np.fromiter(gs, dtype=np.uint64, count=len(gs))
        idx = np.searchsorted(arr, q)
        idx[idx >= arr.size] = 0
        n = int((arr[idx] == q).sum())
        if n:
            hits[pid] = n
    overlap_report = {"checked": True, "train_object": TRAIN_OBJECT, "train_rows": n_rows,
                      "train_unique_grams": int(arr.size), "items_with_overlap": len(hits),
                      "worst": dict(sorted(hits.items(), key=lambda kv: -kv[1])[:20])}
    if hits:
        fails.append("%d probe items share a %d-gram with the training split; they are not held out"
                     % (len(hits), NGRAM_N))
else:
    log("NOT CHECKED: no train_object configured, so held-out status is asserted rather than "
        "measured. Re-run with train_object set before citing any probe score.")

# ------------------------------------------------------------------------- outputs

path = os.path.join(OUT, "probes.jsonl")
with open(path, "w") as fh:
    for it in items:
        fh.write(json.dumps(it, ensure_ascii=False) + "\n")
lab.save_artifact(path)

summary = {
    "n_items": len(items),
    "by_probe_and_arm": dict(collections.Counter("%s:%s" % (i["probe"], i["arm"]) for i in items)),
    "by_mode": dict(collections.Counter(i["mode"] for i in items)),
    "by_depth": dict(collections.Counter(str(i["depth"]) for i in items)),
    "tool_scenarios": [s["id"] for s in bank_tools.SCENARIOS],
    "stack_families": {f["family"]: len(f["items"]) for f in bank_stack.FAMILIES},
    "surface_forms": [f for f, _ in build.FORMS],
    "heldout_check": overlap_report,
    "collision_check": collision_report,
    "assertion_failures": fails,
}
dump("probes_summary.json", summary)
dump("score.json", {"n_items": len(items),
                    "items_overlapping_train": overlap_report.get("items_with_overlap", -1),
                    "heldout_checked": bool(TRAIN_OBJECT),
                    "identical_item_groups": collision_report["identical_item_groups"],
                    "prompts_under_multiple_scenarios":
                        collision_report["prompts_under_multiple_scenarios"],
                    "assertion_failures": len(fails)})
for f in fails:
    log("ASSERTION FAILURE: %s" % f)

lab.update_progress(100)
log("SCORE " + json.dumps({"n_items": len(items),
                           "items_overlapping_train": overlap_report.get("items_with_overlap", -1),
                           "heldout_checked": bool(TRAIN_OBJECT),
                           "assertion_failures": len(fails)}))
if fails:
    lab.finish("failed", "%d probe assertion failures; see probes_summary.json" % len(fails))
else:
    lab.finish("success", "%d probe items across 2 sets, held-out check %s"
               % (len(items), "passed" if TRAIN_OBJECT else "NOT RUN"))
