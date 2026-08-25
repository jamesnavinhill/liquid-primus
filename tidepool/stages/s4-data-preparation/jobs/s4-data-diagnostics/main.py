"""s4.2 — full-corpus diagnostics for the tidepool training and evaluation mix.

Produces every number the s4 stage report cites: field-level missingness, tool
schema and call distributions, the two Synth-APIGen refusal families and the
mislabelled-refusal count, tokenization under the LFM2.5 tokenizer, the s4.3
group-key size distributions, duplicate counts, and a 13-gram decontamination
pass of every training prompt against every benchmark prompt set.

Artifacts are saved as each corpus finishes, so a crash in corpus seven does not
lose corpora one through six.
"""

import collections
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback

import numpy as np
from lab import lab

import adapters

lab.init()
CFG = lab.get_config() or {}


def cfg(key, default):
    v = CFG.get(key, default)
    return type(default)(v) if v is not None and not isinstance(v, type(default)) else v


TOKENIZER_ID = cfg("tokenizer_id", "LiquidAI/LFM2.5-1.2B-Instruct")
NGRAM_N = cfg("ngram_n", 13)
FULL_BELOW = cfg("token_stats_full_below", 80000)
SUBSAMPLE = cfg("token_stats_subsample", 40000)
SEED = cfg("seed", 20260825)

OUT = "out"
os.makedirs(OUT, exist_ok=True)
HF_TOKEN = os.environ.get("HF_TOKEN") or None


def log(msg):
    lab.log(msg)
    print(msg, flush=True)


def dump(name, obj, kind=None):
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    try:
        lab.save_artifact(path, **({"type": kind} if kind else {}))
    except Exception as exc:                     # never lose the run over an upload
        log("save_artifact failed for %s: %s" % (name, exc))
    return path


# --------------------------------------------------------------- corpus registry

TRAIN = [
    # (label, hf id, config, split, adapter, role)
    ("toolace",        "Team-ACE/ToolACE",                        "default",               "train", adapters.toolace,            "tool"),
    ("apigen",         "argilla/Synth-APIGen-v0.1",               "default",               "train", adapters.apigen,             "tool"),
    ("hermes_fc",      "NousResearch/hermes-function-calling-v1", "func_calling",          "train", adapters.hermes,             "tool"),
    ("hermes_fc_st",   "NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", adapters.hermes,           "tool"),
    ("hermes_glaive",  "NousResearch/hermes-function-calling-v1", "glaive_func_calling",   "train", adapters.hermes,             "tool"),
    ("hermes_json_ag", "NousResearch/hermes-function-calling-v1", "json_mode_agentic",     "train", adapters.hermes,             "struct"),
    ("hermes_json_st", "NousResearch/hermes-function-calling-v1", "json_mode_singleturn",  "train", adapters.hermes,             "struct"),
    ("sql_ctx",        "b-mc2/sql-create-context",                "default",               "train", adapters.sql_create_context, "sql"),
    ("sql_clinton",    "Clinton/Text-to-sql-v1",                  "default",               "train", adapters.clinton_sql,        "sql"),
    ("codefeedback",   "m-a-p/CodeFeedback-Filtered-Instruction",  "default",              "train", adapters.codefeedback,       "code"),
    ("antidoom",       "LiquidAI/antidoom-mix-v1.0",              "default",               "train", adapters.prompt_only,        "prompts"),
]

EVAL = [
    ("ifstruct",  "LiquidAI/ifstruct-v1.0",                   "default",           "test",       ["prompt"]),
    ("ifeval",    "google/IFEval",                            "default",           "train",      ["prompt"]),
    ("ifbench",   "allenai/IFBench_test",                     "default",           "train",      ["prompt", "instruction"]),
    ("multi_if",  "facebook/Multi-IF",                        "default",           "train",      []),
    ("humaneval", "openai/openai_humaneval",                  "openai_humaneval",  "test",       ["prompt"]),
    ("mbpp_full", "google-research-datasets/mbpp",             "full",              "test",       ["text", "prompt"]),
    ("mbpp_san",  "google-research-datasets/mbpp",             "sanitized",         "test",       ["prompt", "text"]),
    ("mmlu_pro",  "TIGER-Lab/MMLU-Pro",                       "default",           "test",       ["question"]),
]

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


# ------------------------------------------------------------------ text + grams

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text):
    return _WS.sub(" ", _PUNCT.sub(" ", (text or "").lower())).strip()


def grams(text, n=NGRAM_N):
    words = normalize(text).split()
    if len(words) < n:
        return ()
    return tuple(
        int.from_bytes(hashlib.blake2b(" ".join(words[i:i + n]).encode(), digest_size=8).digest(), "big")
        for i in range(len(words) - n + 1)
    )


def harvest(row, prefer):
    """Prefer named prompt fields; otherwise take every long string in the row."""
    if prefer:
        vals = [row.get(k) for k in prefer if isinstance(row.get(k), str) and row.get(k).strip()]
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


# ------------------------------------------------------------------ eval corpora

from datasets import load_dataset                                # noqa: E402
from huggingface_hub import hf_hub_download, list_repo_files     # noqa: E402


def load(hf_id, config, split):
    return load_dataset(hf_id, config, split=split, token=HF_TOKEN)


eval_report = {}
eval_gram_chunks = []
eval_owner = {}          # gram-hash -> which eval set it came from (sampled, see below)

log("building benchmark n-gram index (n=%d)" % NGRAM_N)
for label, hf_id, config, split, prefer in EVAL:
    t0 = time.time()
    try:
        ds = load(hf_id, config, split)
        n, g, texts = 0, 0, []
        for row in ds:
            txt = harvest(row, prefer)
            texts.append(txt)
            n += 1
        chunk = []
        for txt in texts:
            gg = grams(txt)
            chunk.extend(gg)
            g += len(gg)
        arr = np.fromiter(chunk, dtype=np.uint64, count=len(chunk))
        eval_gram_chunks.append((label, arr))
        eval_report[label] = {"hf_id": hf_id, "config": config, "split": split,
                              "rows": n, "grams": g, "unique_grams": int(np.unique(arr).size),
                              "prefer_fields": prefer or "all-long-strings",
                              "seconds": round(time.time() - t0, 1)}
        log("  %-10s %6d rows  %9d grams" % (label, n, g))
    except Exception as exc:
        eval_report[label] = {"hf_id": hf_id, "error": "%s: %s" % (type(exc).__name__, exc)}
        log("  %-10s FAILED %s" % (label, exc))

# BFCL v3 lives as raw JSON in the repo tree, not as a datasets-server dataset.
try:
    files = [f for f in list_repo_files(BFCL_REPO, repo_type="dataset", token=HF_TOKEN)
             if f.startswith("BFCL_v3_") and f.endswith(".json")]
    n, g, chunk, cats = 0, 0, [], {}
    for fname in sorted(files):
        path = hf_hub_download(BFCL_REPO, fname, repo_type="dataset", token=HF_TOKEN)
        cn = 0
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                txt = harvest(entry, [])
                gg = grams(txt)
                chunk.extend(gg)
                g += len(gg)
                n += 1
                cn += 1
        cats[fname] = cn
    arr = np.fromiter(chunk, dtype=np.uint64, count=len(chunk))
    eval_gram_chunks.append(("bfcl_v3", arr))
    eval_report["bfcl_v3"] = {"hf_id": BFCL_REPO, "files": len(files), "rows": n, "grams": g,
                              "unique_grams": int(np.unique(arr).size), "per_file": cats}
    log("  %-10s %6d rows  %9d grams  (%d category files)" % ("bfcl_v3", n, g, len(files)))
except Exception as exc:
    eval_report["bfcl_v3"] = {"hf_id": BFCL_REPO, "error": "%s: %s" % (type(exc).__name__, exc)}
    log("  bfcl_v3 FAILED %s" % exc)

dump("eval_index.json", eval_report)

# Per-eval-set sorted arrays let a hit be attributed to the benchmark it came from.
EVAL_SORTED = [(label, np.unique(arr)) for label, arr in eval_gram_chunks if arr.size]
ALL_EVAL = np.unique(np.concatenate([a for _, a in EVAL_SORTED])) if EVAL_SORTED else np.zeros(0, np.uint64)
log("benchmark index: %d unique %d-grams across %d sets" % (ALL_EVAL.size, NGRAM_N, len(EVAL_SORTED)))
lab.update_progress(15)


def overlap_hits(gg):
    """Which benchmark sets does this prompt share a 13-gram with, and how many."""
    if not gg or ALL_EVAL.size == 0:
        return 0, []
    q = np.fromiter(gg, dtype=np.uint64, count=len(gg))
    idx = np.searchsorted(ALL_EVAL, q)
    idx[idx >= ALL_EVAL.size] = 0
    hit = ALL_EVAL[idx] == q
    total = int(hit.sum())
    if total == 0:
        return 0, []
    qh = q[hit]
    who = []
    for label, arr in EVAL_SORTED:
        j = np.searchsorted(arr, qh)
        j[j >= arr.size] = 0
        if bool((arr[j] == qh).any()):
            who.append(label)
    return total, who


# -------------------------------------------------------------------- tokenizer

TOK = None
try:
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(TOKENIZER_ID, token=HF_TOKEN)
    log("tokenizer %s loaded, vocab=%d" % (TOKENIZER_ID, TOK.vocab_size))
except Exception as exc:
    log("TOKENIZER UNAVAILABLE (%s) — token statistics will be omitted" % exc)


def pctiles(values):
    if not values:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "mean": round(float(a.mean()), 2),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def hist(counter, top=25):
    return dict(collections.Counter(counter).most_common(top))


# ------------------------------------------------------------- per-corpus pass

random.seed(SEED)
summary, flagged, prompt_hashes, group_index = {}, {}, {}, {}
figures = []

for ci, (label, hf_id, config, split, adapt, role) in enumerate(TRAIN):
    t0 = time.time()
    log("=== %s (%s / %s)" % (label, hf_id, config))
    try:
        ds = load(hf_id, config, split)
    except Exception as exc:
        summary[label] = {"hf_id": hf_id, "config": config, "role": role,
                          "error": "%s: %s" % (type(exc).__name__, exc)}
        log("  LOAD FAILED %s" % exc)
        continue

    n_rows = len(ds)
    field_missing = collections.Counter()
    field_empty = collections.Counter()
    tools_per_row, params_per_tool, required_per_tool, depth_per_tool = [], [], [], []
    calls_per_row, args_per_call = [], []
    type_vocab, container_vocab = collections.Counter(), collections.Counter()
    refusal_families = collections.Counter()
    n_no_tools = n_call_rows = n_refusal_rows = 0
    mismatch_rows, mismatch_calls, total_calls, badjson = 0, 0, 0, 0
    target_present_on_refusal = 0
    suspect_ids, mismatch_ids = [], []
    groups = collections.Counter()
    contaminated, contaminated_by = 0, collections.Counter()
    contam_ids = []
    dup_exact = collections.Counter()
    apigen_query_labels = collections.defaultdict(set)   # query -> {"call","refusal"}
    apigen_query_rows = collections.defaultdict(list)
    prompt_lens = []
    extras = collections.Counter()
    schema_formats, call_formats = collections.Counter(), collections.Counter()
    non_identifier_names = 0

    tok_idx = set(range(n_rows)) if n_rows <= FULL_BELOW else set(
        random.sample(range(n_rows), min(SUBSAMPLE, n_rows)))
    tok_prompt, tok_schema, tok_call = [], [], []

    for i, row in enumerate(ds):
        for k, v in row.items():
            if v is None:
                field_missing[k] += 1
            elif isinstance(v, str) and not v.strip():
                field_empty[k] += 1
        try:
            rec = adapt(row)
        except Exception:
            badjson += 1
            continue

        prompt = rec["prompt"] or ""
        prompt_lens.append(len(prompt))
        groups[rec["group_key"]] += 1
        h = hashlib.blake2b(normalize(prompt).encode(), digest_size=8).hexdigest()
        dup_exact[h] += 1
        prompt_hashes.setdefault(h, []).append(label)

        tools = rec["tools"]
        tools_per_row.append(len(tools))
        if not tools:
            n_no_tools += 1
        fm = rec.get("formats") or {}
        if fm.get("schema_format"):
            schema_formats[fm["schema_format"]] += 1
        for k, v in (fm.get("call_formats") or {}).items():
            call_formats[k] += v
        # Some corpora declare tool names without a parseable schema object.
        names = set(rec.get("declared_names") or [])
        for t in tools:
            names.add(t["name"])
            params_per_tool.append(t["n_params"])
            required_per_tool.append(len(t["required"]))
            depth_per_tool.append(t["depth"])
            for ty in t["types"]:
                type_vocab[ty] += 1
            container_vocab[t["container_type"]] += 1

        calls = rec["calls"]
        calls_per_row.append(len(calls))
        if calls:
            n_call_rows += 1
            bad = 0
            for c in calls:
                total_calls += 1
                n_args = c.get("n_args")
                args_per_call.append(n_args if isinstance(n_args, int)
                                     else (len(c["args"]) if isinstance(c["args"], dict) else 0))
                if c.get("name_is_identifier") is False:
                    non_identifier_names += 1
                if names and c["name"] not in names:
                    bad += 1
            if bad:
                mismatch_rows += 1
                mismatch_calls += bad
                if len(mismatch_ids) < 5000:
                    mismatch_ids.append({"row_id": rec["row_id"], "bad_calls": bad})
        if rec["refusal"]:
            n_refusal_rows += 1
            fam = "other"
            for key, text in adapters.APIGEN_REFUSALS.items():
                if rec["refusal"].strip() == text:
                    fam = key
            refusal_families[fam] += 1
            if rec["target"] and rec["target"] in names:
                target_present_on_refusal += 1
                tgt = next((t for t in tools if t["name"] == rec["target"]), None)
                if tgt and len(tgt["required"]) <= 2 and len(suspect_ids) < 20000:
                    suspect_ids.append({"row_id": rec["row_id"], "target": rec["target"],
                                        "n_required": len(tgt["required"]),
                                        "family": fam})
        for k, v in (rec.get("extra") or {}).items():
            if isinstance(v, bool):
                extras[k] += int(v)
            elif isinstance(v, int):
                extras[k] += v

        if label == "apigen":
            qn = normalize(prompt)
            apigen_query_labels[qn].add("refusal" if rec["refusal"] else "call")
            if len(apigen_query_rows[qn]) < 4:
                apigen_query_rows[qn].append(rec["row_id"])

        gg = grams(prompt)
        nhit, who = overlap_hits(gg)
        if nhit:
            contaminated += 1
            for w in who:
                contaminated_by[w] += 1
            if len(contam_ids) < 20000:
                contam_ids.append({"row_id": rec["row_id"], "n_grams": nhit, "benchmarks": who})

        if TOK is not None and i in tok_idx:
            tok_prompt.append(len(TOK(prompt, add_special_tokens=False)["input_ids"]))
            if rec["schema_text"]:
                tok_schema.append(len(TOK(rec["schema_text"][:200000], add_special_tokens=False)["input_ids"]))
            if calls:
                blob = json.dumps(calls, separators=(",", ":"))
                tok_call.append(len(TOK(blob, add_special_tokens=False)["input_ids"]) / max(1, len(calls)))

        if i and i % 25000 == 0:
            log("    %s: %d/%d rows" % (label, i, n_rows))

    contradictions = 0
    contradiction_ids = []
    if label == "apigen":
        for qn, labels in apigen_query_labels.items():
            if len(labels) > 1:
                contradictions += 1
                if len(contradiction_ids) < 5000:
                    contradiction_ids.append({"query_hash": hashlib.blake2b(qn.encode(), digest_size=8).hexdigest(),
                                              "row_ids": apigen_query_rows[qn]})

    gsizes = np.asarray(sorted(groups.values()), dtype=np.float64)
    summary[label] = {
        "hf_id": hf_id, "config": config, "split": split, "role": role,
        "rows": n_rows,
        "field_missing_null": dict(field_missing),
        "field_empty_string": dict(field_empty),
        "adapter_parse_failures": badjson,
        "prompt_chars": pctiles(prompt_lens),
        "tools_per_row": pctiles(tools_per_row),
        "rows_without_tools": n_no_tools,
        "params_per_tool": pctiles(params_per_tool),
        "required_per_tool": pctiles(required_per_tool),
        "schema_nesting_depth": pctiles(depth_per_tool),
        "declared_type_vocabulary": hist(type_vocab),
        "schema_container_type": hist(container_vocab),
        "rows_with_calls": n_call_rows,
        "calls_per_row": pctiles(calls_per_row),
        "args_per_call": pctiles(args_per_call),
        "total_calls": total_calls,
        "schema_serialization_formats": dict(schema_formats),
        "assistant_output_formats": dict(call_formats),
        "calls_with_non_identifier_names": non_identifier_names,
        "tool_name_mismatch_rows": mismatch_rows,
        "tool_name_mismatch_calls": mismatch_calls,
        "refusal_rows": n_refusal_rows,
        "refusal_families": dict(refusal_families),
        "refusal_with_target_declared": target_present_on_refusal,
        "suspect_over_refusals": len(suspect_ids),
        "label_contradictions_same_query": contradictions,
        "duplicate_prompt_groups": int(sum(1 for v in dup_exact.values() if v > 1)),
        "duplicate_prompt_rows": int(sum(v for v in dup_exact.values() if v > 1)),
        "distinct_prompts": len(dup_exact),
        "groups": {"n_groups": len(groups),
                   "sizes": pctiles(list(gsizes)),
                   "largest_group_share": round(float(gsizes.max() / max(1.0, gsizes.sum())), 5) if gsizes.size else 0.0,
                   "top10_share": round(float(gsizes[-10:].sum() / max(1.0, gsizes.sum())), 5) if gsizes.size else 0.0},
        "contaminated_rows": contaminated,
        "contaminated_rate": round(contaminated / max(1, n_rows), 5),
        "contaminated_by_benchmark": dict(contaminated_by),
        "tokens_per_prompt": pctiles(tok_prompt),
        "tokens_per_schema_block": pctiles(tok_schema),
        "tokens_per_tool_call": pctiles(tok_call),
        "token_stats_coverage": "all rows" if n_rows <= FULL_BELOW else "seeded subsample of %d" % len(tok_idx),
        "adapter_extras": dict(extras),
        "seconds": round(time.time() - t0, 1),
    }
    group_index[label] = {"n_groups": len(groups),
                          "size_histogram": hist(collections.Counter(groups.values()), top=40)}

    flagged[label] = {"contaminated": contam_ids, "tool_name_mismatch": mismatch_ids,
                      "suspect_over_refusal": suspect_ids,
                      "label_contradictions": contradiction_ids}
    dump("per_corpus_%s.json" % label, summary[label])
    dump("flagged_%s.json" % label, flagged[label], kind="evals")
    log("  done in %.1fs: %d rows, contaminated %d (%.3f%%), refusals %d, mismatch rows %d"
        % (time.time() - t0, n_rows, contaminated, 100 * contaminated / max(1, n_rows),
           n_refusal_rows, mismatch_rows))
    lab.update_progress(15 + int(70 * (ci + 1) / len(TRAIN)))

# ------------------------------------------------------- cross-corpus duplicates

cross = collections.Counter()
for h, labels in prompt_hashes.items():
    uniq = sorted(set(labels))
    if len(uniq) > 1:
        cross[" | ".join(uniq)] += 1
dump("cross_corpus_duplicate_prompts.json",
     {"pairs": dict(cross.most_common(200)),
      "total_prompts_seen_in_more_than_one_corpus": int(sum(cross.values()))})

# ---------------------------------------------------------------------- figures

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tool_sets = [(k, v) for k, v in summary.items() if v.get("role") in ("tool", "struct") and "rows" in v]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, v in tool_sets:
        axes[0].bar(k, v["tools_per_row"].get("mean", 0))
        axes[1].bar(k, v["calls_per_row"].get("mean", 0))
        axes[2].bar(k, v["tokens_per_tool_call"].get("mean", 0))
    for ax, title in zip(axes, ("mean tools per row", "mean calls per row",
                                "mean tokens per tool call (LFM2.5)")):
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=60, labelsize=7)
    fig.tight_layout()
    p = os.path.join(OUT, "fig-tool-shape.png")
    fig.savefig(p, dpi=140)
    lab.save_artifact(p)
    figures.append(p)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [k for k, v in summary.items() if "contaminated_rate" in v]
    ax.bar(labels, [100 * summary[k]["contaminated_rate"] for k in labels])
    ax.set_ylabel("%% of rows sharing a %d-gram with a benchmark" % NGRAM_N)
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    fig.tight_layout()
    p = os.path.join(OUT, "fig-contamination.png")
    fig.savefig(p, dpi=140)
    lab.save_artifact(p)
    figures.append(p)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for k, v in summary.items():
        gh = group_index.get(k, {}).get("size_histogram") or {}
        if gh:
            xs = sorted(int(x) for x in gh)
            ax.plot(xs, [gh[str(x)] if str(x) in gh else gh[x] for x in xs], marker=".", label=k)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("group size (rows sharing a split key)")
    ax.set_ylabel("number of groups")
    ax.legend(fontsize=6)
    fig.tight_layout()
    p = os.path.join(OUT, "fig-group-sizes.png")
    fig.savefig(p, dpi=140)
    lab.save_artifact(p)
    figures.append(p)
except Exception as exc:
    log("figures failed: %s" % exc)
    traceback.print_exc()

# ------------------------------------------------------------------------ wrap

tool_rows = sum(v.get("rows", 0) for v in summary.values() if v.get("role") in ("tool", "struct"))
all_rows = sum(v.get("rows", 0) for v in summary.values() if "rows" in v)
contam_rows = sum(v.get("contaminated_rows", 0) for v in summary.values())
apigen = summary.get("apigen", {})

diagnostics = {
    "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "tokenizer": {"id": TOKENIZER_ID, "available": TOK is not None,
                  "vocab_size": getattr(TOK, "vocab_size", None)},
    "ngram_n": NGRAM_N, "seed": SEED,
    "benchmark_index": eval_report,
    "corpora": summary,
    "group_index": group_index,
    "cross_corpus_duplicates": dict(cross.most_common(50)),
    "totals": {"rows_all_corpora": all_rows, "rows_tool_and_struct": tool_rows,
               "contaminated_rows": contam_rows,
               "contaminated_rate": round(contam_rows / max(1, all_rows), 5)},
}
dump("diagnostics.json", diagnostics, kind="evals")

score = {
    "rows_total": all_rows,
    "rows_tool_struct": tool_rows,
    "contaminated_rows": contam_rows,
    "contaminated_pct": round(100 * contam_rows / max(1, all_rows), 3),
    "apigen_refusal_rows": apigen.get("refusal_rows", 0),
    "apigen_suspect_over_refusals": apigen.get("suspect_over_refusals", 0),
    "apigen_label_contradictions": apigen.get("label_contradictions_same_query", 0),
    "apigen_mismatch_calls": apigen.get("tool_name_mismatch_calls", 0),
    "corpora_loaded": sum(1 for v in summary.values() if "rows" in v),
    "corpora_failed": sum(1 for v in summary.values() if "error" in v),
}
lab.update_progress(100)
log("SCORE " + json.dumps(score))
lab.finish(message="s4.2 diagnostics complete over %d rows across %d corpora"
                   % (all_rows, score["corpora_loaded"]), score=score)
