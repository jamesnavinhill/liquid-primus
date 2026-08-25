"""s5.3 supervision for unreliable tool returns, plus an enlarged clean control arm.

Why this job exists. The s5.2 baselines showed the guardrail gap is in the *data*: of
494,341 rendered training rows, 5,221 carry a tool-result turn, 75 use any doubt or
failure language, and none shows a genuinely broken return being caught. So the probes
were measuring a behaviour nothing in the corpus teaches, and B1's 0.0074 flag rate is a
statement about the training set rather than about the checkpoint.

Two artifacts come out of here.

  1. `tooldata_train.jsonl.gz` — defective tool returns paired one-for-one with their own
     intact counterparts. The defect is applied by the *vendored probe transform*, so
     train and test damage a payload with the same code, and the correct reply is a
     function of which transform ran. No teacher model and no sampling anywhere.

  2. `clean_control.jsonl` — an enlarged clean arm drawn from the *test* split's real
     intact tool returns. The frozen 30-item `clean` arm is too small to hold a false-flag
     ceiling once a model has been taught to flag: one flagged item moves it by 3.3 points.
     The new arm is additive and named `clean_corpus`, so the frozen arm, its recorded
     0.0000, and `false_flag_rate_clean` are all untouched.

Three anti-circularity measures, because a guardrail trained on the measurement is not a
guardrail:

  * the transforms are vendored, not reimplemented (`defects.py` imports `build`);
  * `null_leaf` and `stale_as_of` are held out, so 60 of the 290 tool-return probe items
    test defect kinds never trained on;
  * target phrasings are split into a pool the frozen 26-pattern detector fires on and a
    pool it is blind to, and the job fails if the pool collapses to one side. Without the
    split, a trained flag rate would report how well our phrasing was memorised.

Every gate below is fatal. A set that silently fails one of them is worse than no set,
because s5.3 would spend GPU hours learning from it before anyone noticed.
"""

import collections
import gzip
import hashlib
import json
import os
import random
import re
import time

from lab import lab

import build
import defects
import probes_score as ps
import targets

lab.init()
CFG = lab.get_config() or {}
NGRAM_N = int(CFG.get("ngram_n") or 13)
TRAIN_OBJECT = CFG.get("train_object") or "tidepool/s4.4/train.jsonl.gz"
TEST_OBJECT = CFG.get("test_object") or "tidepool/s4.4/test.jsonl.gz"
PROBES_OBJECT = CFG.get("probes_object") or "tidepool/s4.4/probes/probes.jsonl"
TOKENIZER_ID = CFG.get("tokenizer_id") or "LiquidAI/LFM2.5-1.2B-Instruct"
MAX_DEFECTIVE = int(CFG.get("max_defective") or 24000)
MAX_CONTROL = int(CFG.get("max_control") or 400)
MAX_CORPUS_SHARE = float(CFG.get("max_corpus_share") or 0.40)
OVERLAP_LIMIT = float(CFG.get("overlap_limit") or 0.30)
COVERAGE_MIN = float(CFG.get("coverage_min") or 0.60)
COVERAGE_MAX = float(CFG.get("coverage_max") or 0.90)
STORAGE_PREFIX = CFG.get("storage_prefix") or "tidepool/s5.3/tooldata"
SEED = int(CFG.get("seed") or 20260825)

OUT = "out"
os.makedirs(OUT, exist_ok=True)
FAILS = []
FACTS = {"ngram_n": NGRAM_N, "seed": SEED, "trained_modes": defects.TRAINED_MODES,
         "held_out_modes": sorted(defects.HELD_OUT_MODES)}


def log(msg):
    lab.log(msg)
    print(msg, flush=True)


def fail(msg):
    FAILS.append(msg)
    log("GATE FAILURE: %s" % msg)


def dump(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    try:
        lab.save_artifact(p)
    except Exception as exc:
        log("save_artifact failed for %s: %s" % (name, exc))
    return p


def place(path, dest):
    """Put a file in shared storage if this SDK can, and say so either way."""
    up = None
    for name in ("storage_upload", "upload_storage", "storage_put"):
        up = getattr(lab, name, None)
        if callable(up):
            break
        up = None
    if up is None:
        log("no storage upload in this SDK; %s stays on the job record for outside placement"
            % os.path.basename(path))
        return False
    obj = "%s/%s" % (dest.rstrip("/"), os.path.basename(path))
    for args, kwargs in (((path, obj), {}), ((path,), {"dest": dest}), ((path,), {})):
        try:
            up(*args, **kwargs)
            log("placed %s in shared storage" % obj)
            return True
        except Exception as exc:
            log("upload shape %r failed: %s" % (kwargs or args, exc))
    return False


# --------------------------------------------------------------- 13-gram decontamination
#
# Identical to s4's rule (`s4-probes/main.py`), deliberately: a set decontaminated on a
# different rule than the corpora were is not comparable to them.

_PUNCT = re.compile(r"[^\w\s]+")
_WS = re.compile(r"\s+")


def grams(text, n=NGRAM_N):
    toks = _WS.sub(" ", _PUNCT.sub(" ", (text or "").lower())).split()
    out = []
    for i in range(len(toks) - n + 1):
        d = hashlib.blake2b(" ".join(toks[i:i + n]).encode("utf-8"), digest_size=8).digest()
        out.append(int.from_bytes(d, "big"))
    return out


def request_text(msgs):
    """The surface that identifies the scenario: the system contract and the user turns."""
    return "\n".join(m.get("content") or "" for m in msgs
                     if m.get("role") in ("system", "user"))


def all_text(msgs):
    return "\n".join(m.get("content") or "" for m in msgs)


# ------------------------------------------------------------------------ probe reference

t0 = time.time()
probe_path = lab.storage_download(PROBES_OBJECT)
probe_items = [json.loads(l) for l in open(probe_path) if l.strip()]
probe_req = set()
probe_all = set()
for it in probe_items:
    probe_req.update(grams(request_text(it["messages"])))
    probe_all.update(grams(all_text(it["messages"])))
log("probes: %d items, %d request grams, %d whole-item grams"
    % (len(probe_items), len(probe_req), len(probe_all)))
FACTS["probe_items"] = len(probe_items)
lab.update_progress(5)


# ---------------------------------------------------------------------- source harvesting

_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_RESP = re.compile(r"^<tool_response>(.*)</tool_response>$", re.S)


def tool_payload(content):
    m = _RESP.match((content or "").strip())
    if not m:
        return None
    try:
        obj = json.loads(m.group(1).strip())
    except Exception:
        return None
    return obj if isinstance(obj, dict) and obj else None


def call_of(msgs, upto):
    """The (name, arguments) of the last tool call before index `upto`."""
    for m in reversed(msgs[:upto]):
        if m.get("role") != "assistant":
            continue
        blocks = _CALL.findall(m.get("content") or "")
        if not blocks:
            continue
        try:
            obj = json.loads(blocks[-1])
        except Exception:
            return None, {}
        args = obj.get("arguments")
        return obj.get("name"), args if isinstance(args, dict) else {}
    return None, {}


def harvest(local, split):
    """Rows with a tool return that has a JSON body, a headline value, and a real reply."""
    seen_groups = set()
    src = []
    counts = collections.Counter()
    with gzip.open(local, "rt") as fh:
        for line in fh:
            counts["rows"] += 1
            if counts["rows"] % 100000 == 0:
                log("  scanned %d %s rows, %d candidates" % (counts["rows"], split, len(src)))
            try:
                row = json.loads(line)
            except Exception:
                counts["unparsed"] += 1
                continue
            msgs = row.get("messages") or []
            for k, m in enumerate(msgs):
                if m.get("role") != "tool":
                    continue
                counts["tool_turns"] += 1
                payload = tool_payload(m.get("content"))
                if payload is None:
                    counts["body_not_json_object"] += 1
                    continue
                nxt = msgs[k + 1] if k + 1 < len(msgs) else None
                if not nxt or nxt.get("role") != "assistant" or not (nxt.get("content") or "").strip():
                    counts["no_reply_turn"] += 1
                    continue
                if "<tool_call>" in (nxt.get("content") or ""):
                    # The real reply is another call, not an answer. Teaching "flag it" here
                    # would put a prose target where the corpus itself emits a call.
                    counts["reply_is_another_call"] += 1
                    continue
                path, headline = defects.headline_value(payload)
                if not headline:
                    counts["no_headline_value"] += 1
                    continue
                counts["kept"] += 1
                seen_groups.add(row.get("g"))
                src.append({"c": row.get("c"), "i": row.get("i"), "g": row.get("g"),
                            "s": row.get("s"), "role": row.get("role"), "k": k,
                            "messages": msgs, "payload": payload,
                            "headline_path": ".".join(path or ()), "headline": headline,
                            "reply": nxt.get("content")})
                break   # one source per row: a second turn shares the request surface
    log("%s harvest: %s" % (split, json.dumps(dict(counts))))
    return src, counts, seen_groups


train_local = lab.storage_download(TRAIN_OBJECT)
train_src, train_counts, train_groups = harvest(train_local, "train")
FACTS["train_scan"] = dict(train_counts)
lab.update_progress(30)

if not train_src:
    fail("no usable tool-return rows in the training split")

# Stratify: no single corpus may supply more than MAX_CORPUS_SHARE of the sources, so the
# guardrail is not learned in one vendor's response dialect. The s5.2 note is explicit that
# specialisation belongs to the technology, not to one SDK's idiosyncrasies.
rng = random.Random(SEED)
by_corpus = collections.defaultdict(list)
for s in train_src:
    by_corpus[s["c"]].append(s)
for v in by_corpus.values():
    rng.shuffle(v)
cap = max(1, int(MAX_DEFECTIVE * MAX_CORPUS_SHARE))
selected = []
for c in sorted(by_corpus):
    selected.extend(by_corpus[c][:cap])
rng.shuffle(selected)
selected = selected[:MAX_DEFECTIVE]
FACTS["sources_by_corpus"] = {c: len(v) for c, v in sorted(by_corpus.items())}
FACTS["sources_selected"] = len(selected)
log("selected %d sources from %d corpora (cap %d each)" % (len(selected), len(by_corpus), cap))


# ----------------------------------------------------------------------------- generation

def row_id(s, mode, depth):
    return "%s|%s|%s|%s|d%d" % (s["c"], s["i"], s["k"], mode, depth)


def with_tool_body(msgs, k, body):
    out = [dict(m) for m in msgs[:k + 1]]
    out[k] = {"role": "tool", "content": "<tool_response>%s</tool_response>" % body}
    return out


pairs = []
gen_counts = collections.Counter()
for n, s in enumerate(selected):
    if n % 2000 == 0:
        lab.update_progress(min(70, 30 + int(35.0 * n / max(1, len(selected)))))
    h = hashlib.blake2b(("%s|%s|%s" % (s["c"], s["i"], s["k"])).encode("utf-8"),
                        digest_size=8).digest()
    hv = int.from_bytes(h, "big")
    mode = defects.TRAINED_MODES[hv % len(defects.TRAINED_MODES)]
    depth = 1 + (hv >> 17) % 3
    name, args = call_of(s["messages"], s["k"])
    try:
        body, why, forbid = defects.apply_defect(s["payload"], mode, depth, args)
    except Exception as exc:
        gen_counts["transform_error_" + mode] += 1
        continue
    if body is None:
        gen_counts["skipped_" + mode] += 1
        continue
    plain = defects.plain_why(mode, why)
    text, pool = targets.pick(row_id(s, mode, depth), mode, name or s["c"], why, plain)
    leak = [v for v in (forbid or []) if v.lower() in text.lower()]
    if leak:
        # A target that quotes a value the damaged response no longer carries is the exact
        # fabrication the probes forbid. Dropping is right; a nonzero count is a bug here.
        gen_counts["target_leaked_value"] += 1
        continue
    bad_msgs = with_tool_body(s["messages"], s["k"], body) + [
        {"role": "assistant", "content": text}]
    intact = json.dumps(build.wrap(s["payload"], depth), ensure_ascii=False)
    good_msgs = with_tool_body(s["messages"], s["k"], intact) + [
        {"role": "assistant", "content": s["reply"]}]
    gen_counts["mode_" + mode] += 1
    gen_counts["depth_%d" % depth] += 1
    gen_counts["pool_" + pool] += 1
    pairs.append({"src": s, "mode": mode, "depth": depth, "pool": pool, "why": why,
                  "tool": name, "forbid": forbid or [],
                  "bad": bad_msgs, "good": good_msgs})

log("generated %d pairs: %s" % (len(pairs), json.dumps(dict(gen_counts))))
FACTS["generation"] = dict(gen_counts)
lab.update_progress(70)

# Decontamination. The request surface is checked on the s4 rule (any shared 13-gram is a
# collision); the whole row is checked on an overlap *fraction*, because the synthetic
# envelope and the `silently_truncated` wrapper are boilerplate shared by construction with
# the probe items and a single shared gram there means nothing.
decon = collections.Counter()
kept = []
for p in pairs:
    rg = grams(request_text(p["bad"]))
    if probe_req.intersection(rg):
        decon["request_gram_collision"] += 1
        continue
    ag = grams(all_text(p["bad"]))
    frac = (len(probe_all.intersection(ag)) / len(ag)) if ag else 0.0
    if frac > OVERLAP_LIMIT:
        decon["whole_row_overlap_over_limit"] += 1
        continue
    p["overlap_fraction"] = round(frac, 4)
    kept.append(p)
log("decontamination: kept %d of %d, %s" % (len(kept), len(pairs), json.dumps(dict(decon))))
FACTS["decontamination"] = dict(decon)
FACTS["max_overlap_fraction"] = max([p["overlap_fraction"] for p in kept] or [0.0])

# ----------------------------------------------------------------------------- token count
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, token=os.environ.get("HF_TOKEN"))

    def n_tok(msgs):
        try:
            return len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False))
        except Exception:
            return len(tok("\n".join(m["content"] for m in msgs)).input_ids)
except Exception as exc:
    log("tokenizer unavailable (%s); token counts will be word estimates" % exc)

    def n_tok(msgs):
        return sum(len((m.get("content") or "").split()) for m in msgs) * 4 // 3

# ------------------------------------------------------------------------------ write out
train_path = os.path.join(OUT, "tooldata_train.jsonl.gz")
tok_total = 0
written = collections.Counter()
with gzip.open(train_path, "wt") as fh:
    for p in kept:
        s = p["src"]
        for arm, msgs in (("defective", p["bad"]), ("clean_pair", p["good"])):
            nt = n_tok(msgs)
            tok_total += nt
            written[arm] += 1
            fh.write(json.dumps({
                "c": "tooldata", "i": row_id(s, p["mode"], p["depth"]),
                "s": "train", "g": s["g"], "role": "tool_guardrail",
                "arm": arm, "mode": p["mode"], "depth": p["depth"],
                "pool": p["pool"] if arm == "defective" else "n/a",
                "source": {"c": s["c"], "i": s["i"], "turn": s["k"]},
                "n_tok": nt, "messages": msgs,
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
lab.save_artifact(train_path)
FACTS["rows_written"] = dict(written)
FACTS["train_tokens"] = tok_total
log("wrote %d rows (%d defective + %d clean), %.2fM tokens"
    % (sum(written.values()), written["defective"], written["clean_pair"], tok_total / 1e6))
lab.update_progress(85)

# ------------------------------------------------------------- enlarged clean control arm
test_local = lab.storage_download(TEST_OBJECT)
test_src, test_counts, test_groups = harvest(test_local, "test")
FACTS["test_scan"] = dict(test_counts)
rng2 = random.Random(SEED + 1)
rng2.shuffle(test_src)
control, ctrl_counts = [], collections.Counter()
for s in test_src:
    if len(control) >= MAX_CONTROL:
        break
    if s["g"] in train_groups:
        ctrl_counts["group_seen_in_train"] += 1
        continue
    depth = 1 + (len(control) % 3)
    body = json.dumps(build.wrap(s["payload"], depth), ensure_ascii=False)
    msgs = with_tool_body(s["messages"], s["k"], body)
    if probe_req.intersection(grams(request_text(msgs))):
        ctrl_counts["request_gram_collision"] += 1
        continue
    ctrl_counts["kept"] += 1
    control.append({
        "probe": "tool_return", "arm": "clean_corpus", "mode": "intact",
        "scenario": "%s_%s_%s" % (s["c"], s["i"], s["k"]), "depth": depth,
        "id": "clean_corpus_%s_%s_%s_d%d" % (s["c"], s["i"], s["k"], depth),
        "messages": msgs,
        "defect": "none: a real intact return from the held-out split",
        "check": {"kind": "clean", "expect": s["headline"]},
    })
ctrl_path = os.path.join(OUT, "clean_control.jsonl")
with open(ctrl_path, "w") as fh:
    for it in control:
        fh.write(json.dumps(it, ensure_ascii=False) + "\n")
lab.save_artifact(ctrl_path)
FACTS["control"] = dict(ctrl_counts)
FACTS["control_n"] = len(control)
FACTS["control_sha"] = ps.control_sha(control)
FACTS["control_depths"] = dict(collections.Counter(str(i["depth"]) for i in control))
log("control arm: %d items, sha %s, %s"
    % (len(control), FACTS["control_sha"], json.dumps(dict(ctrl_counts))))
lab.update_progress(93)

# ------------------------------------------------------------------------------ the gates
det = collections.Counter()
undetected_families = set()
for p in kept:
    text = p["bad"][-1]["content"]
    hit = ps.flagged(text)[0]
    det[(p["pool"], hit)] += 1
    if not hit:
        undetected_families.add(targets.FAMILY[p["mode"]])
n_det = sum(v for (pool, hit), v in det.items() if hit)
coverage = n_det / len(kept) if kept else 0.0
FACTS["detector_coverage"] = round(coverage, 4)
FACTS["detector_by_pool"] = {"%s/%s" % (k[0], k[1]): v for k, v in sorted(det.items())}
FACTS["undetected_families"] = sorted(undetected_families)

if not kept:
    fail("no rows survived generation and decontamination")
if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
    fail("detector coverage %.4f outside [%.2f, %.2f]; the target pool has collapsed to "
         "one side and the measured flag rate would be a recall test on our phrasing"
         % (coverage, COVERAGE_MIN, COVERAGE_MAX))
if len(undetected_families) < 3:
    fail("only %d defect families have a target the detector misses; at least 3 are needed "
         "for the flag rate to mean anything beyond phrase matching" % len(undetected_families))
if gen_counts["target_leaked_value"]:
    fail("%d targets quoted a value the damaged response no longer carries; the templates "
         "must never interpolate a payload value" % gen_counts["target_leaked_value"])
bad_modes = sorted({p["mode"] for p in kept} & defects.HELD_OUT_MODES)
if bad_modes:
    fail("held-out modes present in the training set: %s" % ", ".join(bad_modes))
if written["defective"] and abs(written["clean_pair"] / written["defective"] - 1.0) > 0.1:
    fail("clean-to-defective ratio %.3f is not the 1:1 the false-flag guard needs"
         % (written["clean_pair"] / max(1, written["defective"])))
off_split = sorted({p["src"]["s"] for p in kept} - {"train"})
if off_split:
    fail("training rows drawn from non-train splits: %s" % ", ".join(map(str, off_split)))
if decon["request_gram_collision"]:
    log("note: %d rows dropped for sharing a 13-gram with a probe request; the final set "
        "has none by construction" % decon["request_gram_collision"])
if len(control) < 200:
    fail("clean control arm has %d items; under 200 it cannot hold a false-flag ceiling "
         "any tighter than the frozen 30-item arm already does" % len(control))
missing_modes = sorted(set(defects.TRAINED_MODES) - {p["mode"] for p in kept})
if missing_modes:
    fail("trained modes with no surviving rows: %s" % ", ".join(missing_modes))

FACTS["mode_counts_final"] = dict(collections.Counter(p["mode"] for p in kept))
FACTS["depth_counts_final"] = dict(collections.Counter("d%d" % p["depth"] for p in kept))
FACTS["pool_counts_final"] = dict(collections.Counter(p["pool"] for p in kept))
FACTS["corpus_counts_final"] = dict(collections.Counter(p["src"]["c"] for p in kept))
FACTS["seconds"] = round(time.time() - t0, 1)
FACTS["gate_failures"] = FAILS
FACTS["storage_prefix"] = STORAGE_PREFIX

dump("tooldata_summary.json", FACTS)
dump("score.json", {
    "rows": sum(written.values()),
    "defective": written["defective"],
    "clean_pairs": written["clean_pair"],
    "train_tokens": tok_total,
    "detector_coverage": FACTS["detector_coverage"],
    "undetected_families": len(undetected_families),
    "control_n": len(control),
    "control_sha": FACTS["control_sha"],
    "request_gram_collisions_in_output": 0,
    "max_overlap_fraction": FACTS["max_overlap_fraction"],
    "gate_failures": len(FAILS),
})

placed = [place(train_path, STORAGE_PREFIX), place(ctrl_path, STORAGE_PREFIX)]
FACTS["placed_in_storage"] = placed
lab.update_progress(100)
log("SCORE " + json.dumps({"rows": sum(written.values()),
                           "train_tokens": tok_total,
                           "detector_coverage": FACTS["detector_coverage"],
                           "control_n": len(control),
                           "gate_failures": len(FAILS)}))
if FAILS:
    lab.finish("failed", "%d gate failures; see tooldata_summary.json" % len(FAILS))
else:
    lab.finish("success",
               "%d guardrail rows (%.2fM tokens), detector coverage %.2f over %d families, "
               "%d-item clean_corpus control arm"
               % (sum(written.values()), tok_total / 1e6, coverage,
                  len(set(FACTS["mode_counts_final"])), len(control)))
