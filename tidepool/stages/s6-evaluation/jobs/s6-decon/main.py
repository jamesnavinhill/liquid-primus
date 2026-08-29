"""s6.4 — decontamination re-check, against the files the models were actually trained on.

`s4.3` built a 13-gram index over eleven benchmark corpora, measured overlap against every
row of the raw mix, flagged the contaminated rows, and dropped them from every split. It
recorded 799 rows of 1,047,820 as contaminated and closed its own coverage gap by adding both
`allenai/IFBench_multi-turn` configs to the index. So the drop happened. What has never been
measured is the thing a paper actually claims, and the two are not the same statement:

  * `s4.3` checked BENCHMARK n-grams against TRAINING PROMPTS. The shipped rows are full
    conversations, and the assistant turns were never checked. A benchmark answer appearing
    in a training target is the leak that actually teaches the benchmark, and a prompt-only
    rule cannot see it.
  * `s4.3` checked against the benchmark corpora WHOLE. The reported numbers come from a
    specific 3,489 BFCL ids, a specific 2,000 IFStruct items, and a specific probe bank. An
    overlap against an item nobody scored is not contamination of any result.
  * `s4.3` checked the PRE-DROP mix. Whether the drop worked is a property of the shipped
    file, and nothing has read the shipped file for it.

This job answers the narrow version. It indexes the exact eval items behind the reported
numbers, streams `train.jsonl.gz` (and `val`/`test`) row by row, and reports overlap split by
whether it lands in the model's input or in its target. A row is reported per corpus, per
eval set, and by side, and the worst rows are named with the overlapping text spelled out, so
a reader can judge whether a hit is a leak or a boilerplate collision.

The rule is `s4.3`'s: lowercase, strip punctuation to spaces, collapse whitespace, split on
spaces, hash every window of `ngram_n` consecutive words. The HASH is not `s4.3`'s. That job
took a blake2b digest per window, which costs one hash call per word position and is fine
over prompts; the shipped conversations are several times longer and the eval side is indexed
in the same process, so this uses a polynomial hash over per-word blake2b ids, evaluated with
numpy. Both sides use the same function, the word ids are a deterministic digest rather than
an enumeration, and the whole comparison is inside one process, so the change is a speed
choice with one consequence worth stating: the hash is 64-bit, and over the roughly 2e8
windows this pass computes the chance of any accidental 64-bit collision is on the order of
1e-3. A collision would create a FALSE overlap, never hide a real one, so the direction of
the error is toward reporting contamination that is not there.
"""

import collections
import gzip
import hashlib
import json
import os
import time

import numpy as np

from lab import lab

lab.init()
CFG = lab.get_config()
OUT = os.path.join(os.getcwd(), "out")
os.makedirs(OUT, exist_ok=True)
FAILURES = []
NOTES = []
# When non-empty, `Index.add_set` indexes only these labels. See `per_set_scan` in main().
ONLY = set()
HF_TOKEN = os.environ.get("HF_TOKEN") or None


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


def fetch(obj):
    """A storage object as a local path. A directory landing is unwrapped."""
    p = lab.storage_download(obj)
    if os.path.isdir(p):
        base = os.path.basename(obj.rstrip("/"))
        cand = os.path.join(p, base)
        if os.path.exists(cand):
            return cand
        files = [os.path.join(dp, f) for dp, _d, fs in os.walk(p) for f in fs]
        if len(files) == 1:
            return files[0]
        raise RuntimeError("%s landed as a directory holding %d files" % (obj, len(files)))
    return p


# ------------------------------------------------------------------ text + grams

import re                                                          # noqa: E402

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")
NGRAM_N = C("ngram_n", 13)
MAX_CHARS = C("max_chars_for_grams", 20000)
# An odd multiplier, so the polynomial hash is a bijection on each coordinate mod 2**64.
BASE = np.uint64(0x9E3779B97F4A7C15)
POW = np.array([pow(int(BASE), j, 1 << 64) for j in range(NGRAM_N)], dtype=np.uint64)
_WORD_ID = {}


def normalize(text):
    return _WS.sub(" ", _PUNCT.sub(" ", (text or "").lower())).strip()


def words_of(text):
    return normalize(text[:MAX_CHARS] if text else "").split()


def wid(word):
    v = _WORD_ID.get(word)
    if v is None:
        v = int.from_bytes(hashlib.blake2b(word.encode("utf-8", "replace"),
                                           digest_size=8).digest(), "big")
        _WORD_ID[word] = v
    return v


def grams(words):
    """Every window of NGRAM_N words, as a uint64 polynomial hash. Wraps mod 2**64."""
    if len(words) < NGRAM_N:
        return np.zeros(0, dtype=np.uint64)
    a = np.fromiter((wid(w) for w in words), dtype=np.uint64, count=len(words))
    win = np.lib.stride_tricks.sliding_window_view(a, NGRAM_N)
    with np.errstate(over="ignore"):
        return (win * POW).sum(axis=1, dtype=np.uint64)


def harvest(row, prefer):
    """The text of one eval item. `prefer` names the fields to read when they exist."""
    if prefer:
        vals = [row.get(k) for k in prefer
                if isinstance(row.get(k), str) and row.get(k).strip()]
        if vals:
            return "\n".join(vals)
    out = []

    def walk(o, depth=0):
        if depth > 6 or len(out) > 64:
            return
        if isinstance(o, str):
            if len(o) >= 30:
                out.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, depth + 1)

    walk(row)
    return "\n".join(out)


# --------------------------------------------------------------------- the index

class Index(object):
    """Every eval item's 13-grams, sorted, each one owned by the item it came from.

    Owning matters. A count of contaminated rows says how bad it is; naming the eval item a
    training row overlaps says whether it is a leak at all, because the two common false
    positives -- a shared instruction preamble and a shared JSON schema -- are obvious the
    moment the matched text is printed beside the item it matched.
    """

    def __init__(self):
        self.sets = []
        self._g, self._s, self._i = [], [], []
        self.items = []                # (set_idx, item_id, words)
        self.report = {}
        self.all = None

    def add_set(self, label, rows, meta):
        # A single-set pass skips every other set here rather than in each caller, so the
        # `only` filter cannot drift out of step with the list of sets the index knows about.
        if ONLY and label not in ONLY:
            return
        t0 = time.time()
        si = len(self.sets)
        self.sets.append(label)
        n, total = 0, 0
        for item_id, text in rows:
            w = words_of(text)
            gg = grams(w)
            if gg.size:
                self._g.append(gg)
                self._s.append(np.full(gg.size, si, dtype=np.uint8))
                self._i.append(np.full(gg.size, len(self.items), dtype=np.int32))
            self.items.append((si, item_id, w))
            total += int(gg.size)
            n += 1
        d = {"rows": n, "grams": total, "seconds": round(time.time() - t0, 1)}
        d.update(meta or {})
        self.report[label] = d
        log("  %-14s %6d items %9d grams  %.1fs" % (label, n, total, d["seconds"]))

    def freeze(self):
        if getattr(self, "all", None) is not None:
            return                      # idempotent: a second call must not empty the index
        if not self._g:
            self.all = np.zeros(0, np.uint64)
            self.own_set = np.zeros(0, np.uint8)
            self.own_item = np.zeros(0, np.int32)
            return
        g = np.concatenate(self._g)
        s = np.concatenate(self._s)
        i = np.concatenate(self._i)
        self._g = self._s = self._i = None
        order = np.argsort(g, kind="stable")
        g, s, i = g[order], s[order], i[order]
        keep = np.empty(g.size, dtype=bool)
        keep[0] = True
        np.not_equal(g[1:], g[:-1], out=keep[1:])
        self.all, self.own_set, self.own_item = g[keep], s[keep], i[keep]

    def hits(self, gg):
        """(number of matching windows, {set label: count}, [item index]) for one text."""
        if gg.size == 0 or self.all.size == 0:
            return 0, {}, []
        j = np.searchsorted(self.all, gg)
        j[j >= self.all.size] = 0
        m = self.all[j] == gg
        if not m.any():
            return 0, {}, []
        js = j[m]
        by = collections.Counter(self.sets[k] for k in self.own_set[js].tolist())
        return int(m.sum()), dict(by), sorted(set(self.own_item[js].tolist()))

    def quote(self, item_idx, gg):
        """The first overlapping 13-gram, as words, for a row that hit this item."""
        si, item_id, w = self.items[item_idx]
        mine = grams(w)
        if mine.size == 0 or gg.size == 0:
            return self.sets[si], item_id, ""
        common = np.intersect1d(mine, gg, assume_unique=False)
        if common.size == 0:
            return self.sets[si], item_id, ""
        at = int(np.nonzero(mine == common[0])[0][0])
        return self.sets[si], item_id, " ".join(w[at:at + NGRAM_N])


# ------------------------------------------------------------------- eval corpora

HF_SETS = {
    "ifeval":       ("google/IFEval", "default", "train", ["prompt"]),
    "ifbench_test": ("allenai/IFBench_test", "default", "train", ["prompt", "instruction"]),
    "ifbench_mt_if": ("allenai/IFBench_multi-turn", "ifbench_constraints", "test",
                      ["prompt", "instruction"]),
    "ifbench_mt_ie": ("allenai/IFBench_multi-turn", "ifeval_constraints", "test",
                      ["prompt", "instruction"]),
}
BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


def build_index():
    idx = Index()

    # BFCL, restricted to the ids actually scored. An overlap against an item no reported
    # number came from is not contamination of any result, and BFCL v3 is much larger than
    # the slice this project evaluates on.
    used = set()
    try:
        with open(fetch(C("bfcl_ids_from")), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    used.add(str(json.loads(line)["id"]))
    except Exception as exc:                                       # noqa: BLE001
        NOTES.append("could not read the scored BFCL ids (%s), so BFCL is not indexed" % exc)
    if used:
        from huggingface_hub import hf_hub_download, list_repo_files
        rows, found = [], set()
        files = sorted(f for f in list_repo_files(BFCL_REPO, repo_type="dataset",
                                                  token=HF_TOKEN)
                       if f.startswith("BFCL_v3_") and f.endswith(".json"))
        for fname in files:
            path = hf_hub_download(BFCL_REPO, fname, repo_type="dataset", token=HF_TOKEN)
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:                              # noqa: BLE001
                        continue
                    rid = str(entry.get("id"))
                    if rid not in used:
                        continue
                    found.add(rid)
                    rows.append((rid, harvest(entry, [])))
        idx.add_set("bfcl_scored", rows,
                    {"hf_id": BFCL_REPO, "files": len(files), "ids_asked": len(used),
                     "ids_found": len(found)})
        check("bfcl_ids_resolve", len(found) >= 0.99 * len(used),
              "only %d of %d scored BFCL ids were found upstream, so the index is not the "
              "set the numbers came from" % (len(found), len(used)))

    # IFStruct, from the same pinned raw URL the eval pack fetches, so the items are the
    # ones that were scored rather than a mirror of them.
    try:
        import requests
        r = requests.get(C("ifstruct_url"), timeout=120)
        r.raise_for_status()
        rows = []
        for k, line in enumerate(r.text.splitlines()):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            rows.append((str(e.get("id", k)), harvest(e, ["prompt"])))
        idx.add_set("ifstruct", rows,
                    {"url": C("ifstruct_url"),
                     "sha256": hashlib.sha256(r.text.encode()).hexdigest()})
        check("ifstruct_count", len(rows) == 2000, "%d items, expected 2000" % len(rows))
    except Exception as exc:                                       # noqa: BLE001
        NOTES.append("IFStruct not indexed (%s)" % exc)

    # The probe bank. Hand-authored here, so a hit would mean our own probe text reached the
    # training mix, which is a different failure from a public benchmark leaking.
    try:
        rows = []
        with open(fetch(C("probes_object")), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                text = "\n".join(m.get("content") or "" for m in (e.get("messages") or []))
                rows.append((str(e.get("id")), text))
        idx.add_set("probes", rows, {"object": C("probes_object")})
    except Exception as exc:                                       # noqa: BLE001
        NOTES.append("probe bank not indexed (%s)" % exc)

    wanted = [s.strip() for s in str(C("eval_sets", "")).split(",") if s.strip()]
    if wanted:
        from datasets import load_dataset
        for label in wanted:
            spec = HF_SETS.get(label)
            if spec is None:
                NOTES.append("no such eval set %r, skipped" % label)
                continue
            hf_id, config, split, prefer = spec
            try:
                ds = load_dataset(hf_id, config, split=split, token=HF_TOKEN)
                rows = [(str(k), harvest(row, prefer)) for k, row in enumerate(ds)]
                idx.add_set(label, rows, {"hf_id": hf_id, "config": config, "split": split})
            except Exception as exc:                               # noqa: BLE001
                NOTES.append("%s not indexed (%s)" % (label, exc))
    idx.freeze()
    return idx


# --------------------------------------------------------------- the shipped file

INPUT_ROLES = ("system", "user", "tool", "function", "ipython", "observation")


def sides(messages):
    """(what the model reads, what the model is trained to produce)."""
    inp, tgt = [], []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if not isinstance(c, str) or not c:
            continue
        (inp if str(m.get("role", "")).lower() in INPUT_ROLES else tgt).append(c)
    return "\n".join(inp), "\n".join(tgt)


def scan(idx, obj, worst_rows, examples, item_cap=200):
    """Stream one shipped split and count overlap, by corpus and by side."""
    path = fetch(obj)
    per_corpus = collections.defaultdict(
        lambda: {"rows": 0, "hit": 0, "hit_input": 0, "hit_target": 0})
    by_set = collections.Counter()
    by_set_target = collections.Counter()
    # Which eval ITEMS were touched, counted in training rows rather than windows. The
    # window counts above say how much overlap there is; only a named item list can be acted
    # on. The peakedness check at the end of this substage is stratified on exactly this
    # output -- eval items whose text appears in a training TARGET against matched items with
    # no overlap at all -- so an aggregate here would leave that test with nothing to sample.
    item_rows_input = collections.Counter()
    item_rows_target = collections.Counter()
    # An eval item named on the TARGET side is the whole contamination finding, and a count
    # cannot be read: the two ways a benchmark answer appears verbatim in a training answer
    # are a genuine leak and a shared boilerplate phrase, and only the matched text tells them
    # apart. Pass 3 named three BFCL items without recording what any of them shared, which
    # left the finding unreadable. One example per item, the first row that hit it.
    item_target_examples = {}
    worst = []
    n_rows, n_hit, n_input, n_target, n_both = 0, 0, 0, 0, 0
    t0 = time.time()
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:                                      # noqa: BLE001
                continue
            n_rows += 1
            corpus = str(row.get("c", "?"))
            per_corpus[corpus]["rows"] += 1
            itxt, ttxt = sides(row.get("messages"))
            gi, gt = grams(words_of(itxt)), grams(words_of(ttxt))
            ni, si_, items_i = idx.hits(gi)
            nt, st_, items_t = idx.hits(gt)
            if not ni and not nt:
                continue
            n_hit += 1
            per_corpus[corpus]["hit"] += 1
            if ni:
                n_input += 1
                per_corpus[corpus]["hit_input"] += 1
            if nt:
                n_target += 1
                per_corpus[corpus]["hit_target"] += 1
            if ni and nt:
                n_both += 1
            for k, v in si_.items():
                by_set[k] += v
            for k, v in st_.items():
                by_set[k] += v
                by_set_target[k] += v
            for ix in items_i:
                item_rows_input[ix] += 1
            for ix in items_t:
                item_rows_target[ix] += 1
                if ix not in item_target_examples:
                    lbl, iid, txt = idx.quote(ix, gt)
                    item_target_examples[ix] = {
                        "eval_set": lbl, "eval_item": iid, "corpus": corpus,
                        "row_id": str(row.get("i", "")), "overlapping_text": txt,
                        "training_target_excerpt": ttxt[:600]}
            worst.append((ni + nt, corpus, str(row.get("i", "")), str(row.get("s", "")),
                          ni, nt, si_, st_, items_i or items_t,
                          gi if ni else gt))
            worst.sort(key=lambda t: -t[0])
            del worst[worst_rows:]
    named = []
    for w in worst[:examples]:
        label, item_id, text = ("", "", "")
        if w[8]:
            label, item_id, text = idx.quote(w[8][0], w[9])
        named.append({"corpus": w[1], "row_id": w[2], "split": w[3],
                      "windows_in_input": w[4], "windows_in_target": w[5],
                      "sets_in_input": w[6], "sets_in_target": w[7],
                      "eval_set": label, "eval_item": item_id, "overlapping_text": text})
    def item_map(counter, cap):
        """{eval set: {item id: training rows that overlap it}}, largest first, capped.

        The target-side map is never capped in practice because the counts are tiny; the
        input-side one can name tens of thousands of items and the cap keeps the artifact
        readable. `truncated` records what the cap dropped, so a reader never mistakes a
        capped list for a complete one.
        """
        per_set = {}
        for ix, n in counter.most_common():
            label = idx.sets[idx.items[ix][0]]
            per_set.setdefault(label, []).append((idx.items[ix][1], n))
        out_map, dropped = {}, {}
        for label, pairs in per_set.items():
            out_map[label] = {i: n for i, n in pairs[:cap]}
            dropped[label] = max(0, len(pairs) - cap)
        return out_map, dropped, {k: len(v) for k, v in per_set.items()}

    tgt_map, tgt_dropped, tgt_total = item_map(item_rows_target, 100000)
    inp_map, inp_dropped, inp_total = item_map(item_rows_input, item_cap)
    out = {
        "object": obj, "rows": n_rows, "rows_hit": n_hit,
        "rows_hit_in_input": n_input, "rows_hit_in_target": n_target,
        "rows_hit_on_both_sides": n_both,
        "rate": round(n_hit / max(1, n_rows), 6),
        "target_rate": round(n_target / max(1, n_rows), 6),
        "windows_by_eval_set": dict(by_set.most_common()),
        "target_windows_by_eval_set": dict(by_set_target.most_common()),
        "by_corpus": {k: dict(v, rate=round(v["hit"] / max(1, v["rows"]), 6))
                      for k, v in sorted(per_corpus.items())},
        "eval_items_hit_in_target": tgt_map,
        "eval_items_hit_in_target_count": tgt_total,
        "eval_items_hit_in_input": inp_map,
        "eval_items_hit_in_input_count": inp_total,
        "eval_items_truncated": {"target": tgt_dropped, "input": inp_dropped},
        "eval_target_overlap_examples": [
            item_target_examples[ix] for ix, _n in item_rows_target.most_common()],
        "worst": named, "seconds": round(time.time() - t0, 1),
    }
    log("%s: %d rows, %d hit (%.4f%%), %d of them in a training TARGET, %.0fs"
        % (obj, n_rows, n_hit, 100.0 * out["rate"], n_target, out["seconds"]))
    return out


# ------------------------------------------------------------------------ render

def render(s):
    o = ["# s6.4 — decontamination re-check", "",
         "%d-gram overlap between the SHIPPED splits and the exact eval items the reported "
         "numbers came from, counted separately for the model's input and its training "
         "target." % NGRAM_N, "",
         "## What was indexed", "",
         "| eval set | items | 13-grams | note |", "|---|--:|--:|---|"]
    for label, v in sorted((s.get("index") or {}).items()):
        note = v.get("hf_id") or v.get("url") or v.get("object") or ""
        if v.get("ids_asked"):
            note = "%s, %d of %d scored ids found" % (note, v["ids_found"], v["ids_asked"])
        o.append("| %s | %d | %d | %s |" % (label, v["rows"], v["grams"], note))
    o += ["", "Unique 13-grams across every set: **%d**." % s.get("unique_grams", 0), ""]
    o += ["## What the shipped splits contain", "",
          "| split | rows | rows hit | rate | in input | in TARGET | both sides |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for name in ("train", "val", "test"):
        v = (s.get("splits") or {}).get(name)
        if not v:
            continue
        o.append("| %s | %d | %d | %.5f%% | %d | %d | %d |"
                 % (name, v["rows"], v["rows_hit"], 100.0 * v["rate"],
                    v["rows_hit_in_input"], v["rows_hit_in_target"],
                    v["rows_hit_on_both_sides"]))
    o += ["", "The **TARGET** column is the one `s4.3` could not report. It checked benchmark "
          "n-grams against training prompts; a benchmark answer sitting in a training target "
          "is the overlap that would actually teach the benchmark, and it needed the shipped "
          "conversations to see it.", ""]
    tr = (s.get("splits") or {}).get("train")
    if tr:
        o += ["## Where the training overlap sits", "",
              "| corpus | rows | hit | rate | in input | in target |",
              "|---|--:|--:|--:|--:|--:|"]
        for c, v in sorted(tr["by_corpus"].items(), key=lambda kv: -kv[1]["hit"]):
            if not v["hit"]:
                continue
            o.append("| %s | %d | %d | %.5f%% | %d | %d |"
                     % (c, v["rows"], v["hit"], 100.0 * v["rate"], v["hit_input"],
                        v["hit_target"]))
        if not any(v["hit"] for v in tr["by_corpus"].values()):
            o.append("| _no corpus contributed a single overlapping window_ | | | | | |")
        o += ["", "| eval set | windows matched | of those, in a target |",
              "|---|--:|--:|"]
        for k, v in (tr["windows_by_eval_set"] or {"(none)": 0}).items():
            o.append("| %s | %s | %s |"
                     % (k, v, (tr["target_windows_by_eval_set"] or {}).get(k, 0)))
        if tr.get("worst"):
            o += ["", "### The worst rows, with the overlapping text", "",
                  "A hit is not automatically a leak. A shared instruction preamble and a "
                  "shared JSON schema both produce long exact overlaps and teach nothing "
                  "about any answer, which is why the matched span is printed rather than "
                  "counted.", ""]
            for w in tr["worst"]:
                o += ["- `%s` row `%s` (%s), %d window(s) in the input and %d in the target, "
                      "against `%s` item `%s`:"
                      % (w["corpus"], w["row_id"], w["split"], w["windows_in_input"],
                         w["windows_in_target"], w["eval_set"] or "?",
                         w["eval_item"] or "?"),
                      "  > %s" % (w["overlapping_text"] or "(no span recovered)")]
        if tr.get("eval_target_overlap_examples"):
            o += ["", "### Every eval item whose text appears in a training TARGET", "",
                  "The count of these items is the contamination finding, and a count cannot "
                  "be read on its own: a benchmark answer reproduced verbatim and a shared "
                  "boilerplate phrase both land here. The matched span is printed for each.",
                  ""]
            for e in tr["eval_target_overlap_examples"]:
                o += ["- `%s` item `%s`, first seen in `%s` row `%s`:"
                      % (e["eval_set"], e["eval_item"], e["corpus"], e["row_id"]),
                      "  > %s" % (e["overlapping_text"] or "(no span recovered)")]

    if s.get("per_set"):
        o += ["", "## Per-set passes", "",
              "The index above keeps one owner per 13-gram, so a span shared by two eval sets "
              "counts for whichever was indexed first. Each set below was re-scanned against "
              "an index containing only itself, which removes that competition; these are the "
              "exact figures for the sets that carry reported results.", "",
              "| eval set | grams | split | rows | hit | rate | in target |",
              "|---|--:|---|--:|--:|--:|--:|"]
        for label, blk in sorted(s["per_set"].items()):
            for name in ("train", "val", "test"):
                v = (blk.get("splits") or {}).get(name)
                if not v:
                    continue
                o.append("| %s | %d | %s | %d | %d | %.5f%% | %d |"
                         % (label, blk["unique_grams"], name, v["rows"], v["rows_hit"],
                            100.0 * v["rate"], v["rows_hit_in_target"]))

    if s.get("notes"):
        o += ["", "## Notes", ""] + ["- %s" % n for n in s["notes"]]
    if s.get("assertion_failures"):
        o += ["", "## Assertion failures", ""] + ["- %s" % f
                                                  for f in s["assertion_failures"]]
    return "\n".join(o) + "\n"


def main():
    log("building the eval index (n=%d)" % NGRAM_N)
    idx = build_index()
    check("index_not_empty", idx.all.size > 0,
          "no eval item produced a single %d-gram, so every overlap number below is zero "
          "by construction" % NGRAM_N)
    log("index: %d unique %d-grams over %d set(s)"
        % (idx.all.size, NGRAM_N, len(idx.sets)))
    try:
        lab.update_progress(25)
    except Exception:                                              # noqa: BLE001
        pass

    worst_rows, examples = C("worst_rows", 25), C("examples", 15)
    item_cap = int(C("item_report_cap", 200))
    splits = {}
    for name, key in (("train", "train_object"), ("val", "val_object"),
                      ("test", "test_object")):
        obj = C(key)
        if not obj:
            continue
        try:
            splits[name] = scan(idx, obj, worst_rows, examples, item_cap)
        except Exception as exc:                                   # noqa: BLE001
            NOTES.append("%s not scanned (%s: %s)" % (name, type(exc).__name__, exc))
            log("%s FAILED: %s" % (name, exc))
        try:
            lab.update_progress(25 + 25 * len(splits))
        except Exception:                                          # noqa: BLE001
            pass

    # PER-SET PASSES. `Index.freeze` deduplicates grams and keeps ONE owner per gram, so a
    # 13-gram shared by two eval items -- or by two eval SETS -- is attributed to whichever was
    # indexed first. Pass 3 showed how far that can distort a reading: 1.42 million of the
    # 1.42 million matching windows were attributed to `probes`, on the strength of one shared
    # instruction template, and every other set's count is a remainder after that. A set scanned
    # against an index containing only itself has no such competition, so these passes are the
    # exact numbers for the sets that carry reported results, and the shared pass above stays
    # as the cheap all-sets sweep it was built to be.
    global ONLY
    per_set = {}
    for label in [x.strip() for x in str(C("per_set_scan", "") or "").split(",") if x.strip()]:
        if label not in idx.sets:
            NOTES.append("per-set pass skipped for %s: not in the index" % label)
            continue
        ONLY = {label}
        try:
            one = build_index()
            one_splits = {}
            for name, key in (("train", "train_object"), ("val", "val_object"),
                              ("test", "test_object")):
                obj = C(key)
                if obj:
                    one_splits[name] = scan(one, obj, worst_rows, examples, item_cap)
            per_set[label] = {"unique_grams": int(one.all.size), "splits": one_splits}
            log("per-set %s: %d unique grams, train %d/%d rows hit, %d in a target"
                % (label, one.all.size, one_splits.get("train", {}).get("rows_hit", -1),
                   one_splits.get("train", {}).get("rows", -1),
                   one_splits.get("train", {}).get("rows_hit_in_target", -1)))
        except Exception as exc:                                   # noqa: BLE001
            NOTES.append("per-set pass failed for %s (%s: %s)" % (label, type(exc).__name__, exc))
        finally:
            ONLY = set()

    check("train_scanned", "train" in splits,
          "the shipped training split was not read, so the re-check answered nothing")
    summary = {
        "ngram_n": NGRAM_N, "max_chars_for_grams": MAX_CHARS,
        "index": idx.report, "unique_grams": int(idx.all.size),
        "splits": splits, "per_set": per_set,
        "notes": NOTES, "assertion_failures": FAILURES,
    }
    save(summary, "decon.json")
    save(render(summary), "decon.md")
    tr = splits.get("train") or {}
    msg = ("s6.4 decontamination re-check: %d of %d shipped training rows share a "
           "%d-gram with a scored eval item, %d of them in a training target. "
           "%d assertion failure(s), %d note(s)."
           % (tr.get("rows_hit", -1), tr.get("rows", -1), NGRAM_N,
              tr.get("rows_hit_in_target", -1), len(FAILURES), len(NOTES)))
    log(msg)
    # The first run of this job (154c72a4) did every scan, wrote both artifacts, and then died
    # here: the SDK on the worker has no `job_complete`, so the whole pass was marked FAILED
    # after its results were already on disk. The status is the citation, so a result under a
    # FAILED id is a result nobody should quote. Closing out is a courtesy to the dashboard and
    # must never be able to fail the run that earned the numbers.
    # `finish` is what every other job in this project closes with, and it is the call that
    # puts the message and the score on the dashboard row the operator opens. `job_complete`
    # is not in the SDK on the worker at all, which is how 154c72a4 came to die here.
    try:
        lab.finish(message=msg, score=int(tr.get("rows_hit_in_target", 0)))
    except Exception:                                              # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
