"""s4.4 — turn the s4.3 split assignment into the corpus s5 actually trains on.

s4.3 decided, for every row in the mix, which side of the split it belongs on and
what is wrong with it. It deliberately decided nothing about text: its output is a
manifest of (corpus, row id, group, split, flag bits) and not a single training
token. This job is the other half. It re-reads the eleven source corpora, joins each
row to its manifest entry, drops what the flags say to drop, and renders the survivors
into one conversation format so the model sees a single convention rather than the
five its corpora happen to use.

Three things are settled here that could not be settled earlier:

  1. **Argument values.** The s4.3 adapters summarize schemas and count arguments;
     neither survives into renderable text. Everything model-visible is re-derived
     from the raw row through `render.py` and `canon.py`.
  2. **The suspect/contradiction intersection.** s4.3 is the first pass to flag
     over-refusal plausibility and same-query label contradiction independently, so
     the intersection exists for the first time and is what carries the drop.
  3. **Whether antidoom is training data.** The scouting pass called it a prompt set
     with no target column; its adapter reads assistant turns anyway. Rows that end on
     a user turn go to an on-policy prompt pool and are counted there.

Held-out purity was enforced in s4.3 and is re-checked here on the rendered rows,
because a rendering step that collapses two prompts to the same text would reopen a
leak that the manifest says is closed.
"""

import collections
import gzip
import hashlib
import json
import os
import time
import traceback

import numpy as np
from lab import lab

import adapters
import canon
import render

lab.init()
CFG = lab.get_config() or {}


def cfg(key, default):
    v = CFG.get(key, default)
    return type(default)(v) if v is not None and not isinstance(v, type(default)) else v


SPLITS_OBJECT = cfg("splits_object", "tidepool/s4.3/splits.jsonl.gz")
TOKENIZER_ID = cfg("tokenizer_id", "LiquidAI/LFM2.5-1.2B-Instruct")
MAX_SEQ_LEN = cfg("max_seq_len", 4096)
DEDUP = bool(cfg("dedup_triples", True))
DROP_FLAGS = [s.strip() for s in cfg("drop_flags", "").split(",") if s.strip()]
KEEP_FLAGS = [s.strip() for s in cfg("keep_flags", "").split(",") if s.strip()]
SUBSAMPLE = cfg("token_stats_subsample", 40000)

FLAG_BITS = {"contaminated": 1, "tool_name_mismatch": 2, "suspect_over_refusal": 4,
             "label_contradiction": 8, "parse_failure": 16, "duplicate_triple": 32}

OUT = "out"
os.makedirs(OUT, exist_ok=True)
HF_TOKEN = os.environ.get("HF_TOKEN") or None


def log(msg):
    lab.log(msg)
    print(msg, flush=True)


def dump(name, obj):
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    try:
        lab.save_artifact(path)
    except Exception as exc:
        log("save_artifact failed for %s: %s" % (name, exc))
    return path


def h64(*parts):
    d = hashlib.blake2b("\x1f".join(str(p) for p in parts).encode("utf-8", "replace"),
                        digest_size=8).digest()
    return int.from_bytes(d, "big")


# The registry is the s4.3 one, minus the group-key machinery it no longer needs.
CORPORA = [
    ("toolace",        "Team-ACE/ToolACE",                        "default",                 "train", adapters.toolace,            "tool"),
    ("apigen",         "argilla/Synth-APIGen-v0.1",               "default",                 "train", adapters.apigen,             "tool"),
    ("hermes_fc",      "NousResearch/hermes-function-calling-v1", "func_calling",            "train", adapters.hermes,             "tool"),
    ("hermes_fc_st",   "NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", adapters.hermes,             "tool"),
    ("hermes_glaive",  "NousResearch/hermes-function-calling-v1", "glaive_func_calling",     "train", adapters.hermes,             "tool"),
    ("hermes_json_ag", "NousResearch/hermes-function-calling-v1", "json_mode_agentic",       "train", adapters.hermes,             "struct"),
    ("hermes_json_st", "NousResearch/hermes-function-calling-v1", "json_mode_singleturn",    "train", adapters.hermes,             "struct"),
    ("sql_ctx",        "b-mc2/sql-create-context",                "default",                 "train", adapters.sql_create_context, "sql"),
    ("sql_clinton",    "Clinton/Text-to-sql-v1",                  "default",                 "train", adapters.clinton_sql,        "sql"),
    ("codefeedback",   "m-a-p/CodeFeedback-Filtered-Instruction", "default",                 "train", adapters.codefeedback,       "code"),
    ("antidoom",       "LiquidAI/antidoom-mix-v1.0",              "default",                 "train", adapters.antidoom,           "prompts"),
]


# ------------------------------------------------------------------- drop rules

# A rule is a set of flag names that must ALL be present; a row is dropped when it
# matches ANY rule. The over-refusal rule is a pair rather than a single flag on
# purpose: s4.3 is the first pass to measure refusal plausibility and same-query
# label contradiction independently, and either alone is weak evidence. A refusal
# naming a callable tool may still be the right answer for reasons the schema does
# not show; a query carrying both labels may be two legitimately different requests
# that happen to share a prompt string. Together they are a corpus contradicting
# itself about a call it could have made, which is the failure this project exists
# to fix and must not be trained on.
DEFAULT_RULES = [["contaminated"], ["tool_name_mismatch"], ["parse_failure"],
                 ["suspect_over_refusal", "label_contradiction"]]
try:
    RULES = json.loads(CFG.get("drop_rules") or "null") or DEFAULT_RULES
except Exception:
    RULES = DEFAULT_RULES
RULE_MASKS = [(("+".join(sorted(r))), sum(FLAG_BITS[f] for f in r)) for r in RULES]
log("drop rules: %s" % json.dumps([n for n, _ in RULE_MASKS]))


def dropped_by(flags):
    for name, mask in RULE_MASKS:
        if flags & mask == mask:
            return name
    return None


# ------------------------------------------------------------- the s4.3 manifest

t0 = time.time()
local = lab.storage_download(SPLITS_OBJECT)
log("manifest %s -> %s (%.1fs)" % (SPLITS_OBJECT, local, time.time() - t0))

manifest = {}
man_counts = collections.Counter()
flag_alone = collections.Counter()
with gzip.open(local, "rt") as fh:
    for line in fh:
        r = json.loads(line)
        manifest[(r["c"], str(r["i"]))] = (r["s"], int(r.get("f") or 0), r["g"])
        man_counts[(r["c"], r["s"])] += 1
        for name, bit in FLAG_BITS.items():
            if int(r.get("f") or 0) & bit:
                flag_alone[name] += 1
log("manifest rows: %d; flags set: %s" % (len(manifest), json.dumps(dict(flag_alone))))

from transformers import AutoTokenizer                          # noqa: E402
from datasets import load_dataset                               # noqa: E402

tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, token=HF_TOKEN)
log("tokenizer %s, vocab %d" % (TOKENIZER_ID, tok.vocab_size))


def n_tokens(msgs):
    """Length under the model's own chat template when it has one, plain text if not."""
    try:
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        return len(ids)
    except Exception:
        return len(tok("\n".join(m["content"] for m in msgs)).input_ids)


# ------------------------------------------------------------------- render pass

writers = {s: gzip.open(os.path.join(OUT, "%s.jsonl.gz" % s), "wt") for s in ("train", "val", "test")}
pool = gzip.open(os.path.join(OUT, "prompt_pool.jsonl.gz"), "wt")

per_corpus, unknown_types = {}, {}
kept_lens = collections.defaultdict(list)
seen_text, prompt_seen = {}, {}
totals = collections.Counter()
purity_violations = []
role_of = {}

for label, hf_id, config, split, adapter, role in CORPORA:
    t0 = time.time()
    role_of[label] = role
    c = collections.Counter()
    reasons = collections.Counter()
    rule_hits = collections.Counter()
    try:
        ds = load_dataset(hf_id, config, split=split, token=HF_TOKEN)
    except Exception as exc:
        log("=== %-14s LOAD FAILED %s: %s" % (label, type(exc).__name__, exc))
        per_corpus[label] = {"error": "%s: %s" % (type(exc).__name__, exc)}
        continue
    fn = render.RENDER[label]
    for row in ds:
        c["rows"] += 1
        try:
            rec = adapter(row)
        except Exception:
            c["adapter_error"] += 1
            continue
        key = (label, str(rec.get("row_id")))
        hit = manifest.get(key)
        if hit is None:
            c["not_in_manifest"] += 1
            continue
        dst, flags, gid = hit
        if dst == "drop":
            c["manifest_drop"] += 1
            continue
        rule = dropped_by(flags)
        if rule:
            c["flag_drop"] += 1
            rule_hits[rule] += 1
            continue
        try:
            msgs = fn(row, unknown_types)
        except render.PromptOnly as po:
            c["prompt_only"] += 1
            pool.write(json.dumps({"c": label, "i": key[1], "s": dst, "g": gid,
                                   "messages": po.messages}, ensure_ascii=False,
                                  separators=(",", ":")) + "\n")
            continue
        except render.Unrenderable as exc:
            c["unrenderable"] += 1
            reasons[str(exc)] += 1
            continue
        except Exception as exc:
            c["render_error"] += 1
            reasons["%s:%s" % (type(exc).__name__, str(exc)[:40])] += 1
            continue
        text = json.dumps(msgs, ensure_ascii=False, sort_keys=True)
        th = h64(text)
        if DEDUP and th in seen_text:
            c["dup_rendered"] += 1
            continue
        seen_text[th] = dst
        # The manifest guarantees group disjointness and, since s4.3's enforcement
        # pass, prompt disjointness on the source text. Rendering can still collapse
        # two source prompts onto one string, so the check is repeated on what is
        # actually written.
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        ph = h64(first_user)
        prev = prompt_seen.get(ph)
        if prev is None:
            prompt_seen[ph] = dst
        elif prev != dst:
            c["purity_violation"] += 1
            if len(purity_violations) < 200:
                purity_violations.append({"corpus": label, "row": key[1],
                                          "split": dst, "seen_as": prev})
            continue
        nt = n_tokens(msgs)
        if nt > MAX_SEQ_LEN:
            c["over_max_seq_len"] += 1
            continue
        c["kept"] += 1
        c["kept_" + dst] += 1
        if len(kept_lens[label]) < SUBSAMPLE:
            kept_lens[label].append(nt)
        writers[dst].write(json.dumps({"c": label, "i": key[1], "s": dst, "g": gid,
                                       "role": role, "n_tok": nt, "messages": msgs},
                                      ensure_ascii=False, separators=(",", ":")) + "\n")
    for k, v in c.items():
        totals[k] += v
    per_corpus[label] = {"hf_id": hf_id, "config": config, "role": role,
                         "counts": dict(c), "unrenderable_reasons": dict(reasons.most_common(20)),
                         "drop_rule_hits": dict(rule_hits),
                         "seconds": round(time.time() - t0, 1)}
    log("=== %-14s rows=%-7d kept=%-7d (tr/va/te %d/%d/%d)  flagdrop=%-6d unrender=%-6d pool=%-6d dup=%d"
        % (label, c["rows"], c["kept"], c["kept_train"], c["kept_val"], c["kept_test"],
           c["flag_drop"], c["unrenderable"], c["prompt_only"], c["dup_rendered"]))
    if reasons:
        log("    top unrenderable: %s" % json.dumps(dict(reasons.most_common(5))))

for w in writers.values():
    w.close()
pool.close()


# ----------------------------------------------------------------- measurements

split_tokens, split_rows = collections.Counter(), collections.Counter()
role_rows = collections.Counter()
for label, info in per_corpus.items():
    cts = info.get("counts") or {}
    for s in ("train", "val", "test"):
        split_rows[s] += cts.get("kept_" + s, 0)
    role_rows[info.get("role", "?")] += cts.get("kept", 0)

len_stats = {}
for label, xs in kept_lens.items():
    if not xs:
        continue
    a = np.array(xs, dtype=np.int64)
    len_stats[label] = {"n_sampled": int(a.size), "mean": round(float(a.mean()), 1),
                        "p50": int(np.percentile(a, 50)), "p90": int(np.percentile(a, 90)),
                        "p99": int(np.percentile(a, 99)), "max": int(a.max())}
    split_tokens[label] = int(a.sum())

all_lens = np.array([x for xs in kept_lens.values() for x in xs], dtype=np.int64)
overall = ({"n_sampled": int(all_lens.size), "mean": round(float(all_lens.mean()), 1),
            "p50": int(np.percentile(all_lens, 50)), "p90": int(np.percentile(all_lens, 90)),
            "p99": int(np.percentile(all_lens, 99)), "max": int(all_lens.max())}
           if all_lens.size else {})

# Estimated trainable tokens: mean sampled length times kept rows, per corpus, so a
# corpus whose lengths were subsampled is not read as if the sample were the whole.
est_tokens = {}
for label, st in len_stats.items():
    est_tokens[label] = int(round(st["mean"] * (per_corpus[label]["counts"].get("kept", 0))))

summary = {
    "manifest_object": SPLITS_OBJECT,
    "manifest_rows": len(manifest),
    "tokenizer": TOKENIZER_ID,
    "max_seq_len": MAX_SEQ_LEN,
    "drop_rules": [n for n, _ in RULE_MASKS],
    "flag_rows_in_manifest": dict(flag_alone),
    "totals": dict(totals),
    "rows_by_split": dict(split_rows),
    "rows_by_role": dict(role_rows),
    "length_stats_by_corpus": len_stats,
    "length_stats_overall": overall,
    "estimated_trainable_tokens_by_corpus": est_tokens,
    "estimated_trainable_tokens": int(sum(est_tokens.values())),
    "unknown_schema_types": dict(sorted(unknown_types.items(), key=lambda kv: -kv[1])[:60]),
    "purity_violations": len(purity_violations),
    "per_corpus": per_corpus,
}
dump("preprocess_summary.json", summary)
dump("purity_violations.json", purity_violations)

# ------------------------------------------------------------------- assertions

fails = []
if split_rows["train"] == 0 or split_rows["val"] == 0 or split_rows["test"] == 0:
    fails.append("an empty split: %s" % json.dumps(dict(split_rows)))
if purity_violations:
    fails.append("%d rendered rows collapsed a prompt across splits" % len(purity_violations))
for label, info in per_corpus.items():
    cts = info.get("counts") or {}
    if "error" in info:
        fails.append("%s failed to load: %s" % (label, info["error"]))
        continue
    # A corpus whose rows are almost all unrenderable means the renderer is reading it
    # wrong, which is a code fault and not a data finding. The prompt pool is exempt:
    # antidoom routing every row there is a legitimate outcome, and the point of looking.
    live = cts.get("rows", 0) - cts.get("not_in_manifest", 0) - cts.get("manifest_drop", 0)
    if live > 1000 and cts.get("kept", 0) + cts.get("prompt_only", 0) < 0.5 * live:
        fails.append("%s kept %d of %d manifest-live rows (renderer suspect): %s"
                     % (label, cts.get("kept", 0), live,
                        json.dumps(dict(list((info.get("unrenderable_reasons") or {}).items())[:3]))))
if totals["not_in_manifest"] > 0.01 * max(1, totals["rows"]):
    fails.append("%d rows (%.2f%%) had no manifest entry: the row-id join is drifting"
                 % (totals["not_in_manifest"], 100.0 * totals["not_in_manifest"] / max(1, totals["rows"])))
dump("assertions.json", {"failures": fails, "n": len(fails)})
for f in fails:
    log("ASSERTION FAILURE: %s" % f)

# ---------------------------------------------------------------------- figures

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [l for l, _, _, _, _, _ in CORPORA if l in len_stats]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    kept = [per_corpus[l]["counts"].get("kept", 0) for l in labels]
    drop = [per_corpus[l]["counts"].get("flag_drop", 0)
            + per_corpus[l]["counts"].get("unrenderable", 0)
            + per_corpus[l]["counts"].get("manifest_drop", 0) for l in labels]
    y = np.arange(len(labels))
    ax[0].barh(y, kept, color="#2b6cb0", label="kept")
    ax[0].barh(y, drop, left=kept, color="#cbd5e0", label="dropped")
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=8)
    ax[0].set_xlabel("rows"); ax[0].set_title("s4.4 kept vs dropped, by corpus")
    ax[0].legend(fontsize=8)
    ax[1].boxplot([kept_lens[l] for l in labels], vert=False, showfliers=False)
    ax[1].set_yticklabels(labels, fontsize=8)
    ax[1].axvline(MAX_SEQ_LEN, color="#c53030", ls="--", lw=1, label="max_seq_len")
    ax[1].set_xlabel("tokens per example"); ax[1].set_title("rendered length")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "fig-preprocess.png")
    fig.savefig(p, dpi=130)
    lab.save_artifact(p)
except Exception as exc:
    log("figure failed: %s" % exc)
    traceback.print_exc()

for s in ("train", "val", "test"):
    lab.save_artifact(os.path.join(OUT, "%s.jsonl.gz" % s))
lab.save_artifact(os.path.join(OUT, "prompt_pool.jsonl.gz"))

score = {"rows_train": split_rows["train"], "rows_val": split_rows["val"],
         "rows_test": split_rows["test"], "prompt_pool_rows": totals["prompt_only"],
         "estimated_trainable_tokens": summary["estimated_trainable_tokens"],
         "rows_dropped_by_flag": totals["flag_drop"],
         "rows_unrenderable": totals["unrenderable"],
         "rows_over_max_seq_len": totals["over_max_seq_len"],
         "duplicate_rendered_rows": totals["dup_rendered"],
         "purity_violations": len(purity_violations),
         "assertion_failures": len(fails)}
dump("score.json", score)
lab.update_progress(100)
log("SCORE " + json.dumps(score))
if fails:
    lab.finish("failed", "%d assertion failures; see assertions.json" % len(fails))
else:
    lab.finish("success", "s4.4: %d train / %d val / %d test rendered rows, ~%.1fM trainable tokens"
               % (split_rows["train"], split_rows["val"], split_rows["test"],
                  summary["estimated_trainable_tokens"] / 1e6))
