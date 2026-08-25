"""s4.3 — group-aware train/val/test splits for the tidepool mix, plus the four
measurements s4.2 left open.

s4.2 scanned 1,047,820 rows and produced the numbers the stage report cites. It
also left four things unresolved, and each of them changes how the splits must be
cut, so they are settled here in the same pass that materializes the splits:

  1. `LiquidAI/antidoom-mix-v1.0` was read with a prompt-only adapter that found
     no prompt field, so its contamination came back 0 and its group count came
     back 1. Both were the reader. Fixed adapter, real numbers.
  2. `CodeFeedback` was keyed on its `resource` field: four groups, largest 40%
     of the corpus. Re-keyed on the API surface each answer exercises.
  3. Duplication was measured on the prompt alone. The dedup rule needs the
     `(prompt, toolset, answer)` triple, because in APIGen the same query
     recurring against a different toolset is the corpus's useful variation.
  4. 865 Hermes `glaive_func_calling` rows failed the adapter. What kind of row
     they are decides whether they are recovered or dropped.

Every count s4.2 recorded is re-derived here and asserted against the recorded
value. A hard mismatch fails the job: the s4.2 id lists were saved under an
eval-results prefix the artifact download path does not serve, so re-deriving
them is how this stage gets a retrievable list, and the recorded counts are what
make the re-derivation trustworthy rather than merely new.

Split rule: group-disjoint, deterministic, size-aware. Groups are ordered by a
seeded hash of the group key, then walked to fill test to its row target, then
val, then train. Two corpora that share a group namespace are split jointly, so
a key cannot land on both sides through two different files. A group larger than
`collapse_frac` of its namespace cannot be split without dominating whichever
side it lands on, so it is declared collapsed and routed wholesale to train.
"""

import collections
import gzip
import hashlib
import json
import os
import re
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


NGRAM_N = cfg("ngram_n", 13)
SEED = cfg("seed", 20260825)
SALT = cfg("split_salt", "tidepool-s4.3")
TEST_FRAC = cfg("test_frac", 0.05)
VAL_FRAC = cfg("val_frac", 0.05)
COLLAPSE_FRAC = cfg("collapse_frac", 0.02)
TOKENIZER_ID = cfg("tokenizer_id", "LiquidAI/LFM2.5-1.2B-Instruct")
SUBSAMPLE = cfg("token_stats_subsample", 40000)
RARE_K = cfg("code_key_rare_symbols", 1)
MAX_PROMPT_CHARS = cfg("max_prompt_chars_for_grams", 20000)

OUT = "out"
os.makedirs(OUT, exist_ok=True)
HF_TOKEN = os.environ.get("HF_TOKEN") or None


def log(msg):
    lab.log(msg)
    print(msg, flush=True)


def dump(name, obj):
    """Plain artifact, no `type=`. s4.2 passed type="evals" for its id lists and
    they landed on a prefix `lab job download` does not serve; everything this
    job needs to be retrievable goes to the default artifacts prefix."""
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    try:
        lab.save_artifact(path)
    except Exception as exc:
        log("save_artifact failed for %s: %s" % (name, exc))
    return path


# --------------------------------------------------------------- corpus registry

TRAIN = [
    # (label, hf id, config, split, adapter, role, namespace)
    ("toolace",        "Team-ACE/ToolACE",                        "default",                 "train", adapters.toolace,            "tool",    "toolace"),
    ("apigen",         "argilla/Synth-APIGen-v0.1",               "default",                 "train", adapters.apigen,             "tool",    "apigen"),
    ("hermes_fc",      "NousResearch/hermes-function-calling-v1", "func_calling",            "train", adapters.hermes,             "tool",    "hermes_fc"),
    ("hermes_fc_st",   "NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", adapters.hermes,             "tool",    "hermes_fc"),
    ("hermes_glaive",  "NousResearch/hermes-function-calling-v1", "glaive_func_calling",     "train", adapters.hermes,             "tool",    "hermes_glaive"),
    ("hermes_json_ag", "NousResearch/hermes-function-calling-v1", "json_mode_agentic",       "train", adapters.hermes,             "struct",  "hermes_json"),
    ("hermes_json_st", "NousResearch/hermes-function-calling-v1", "json_mode_singleturn",    "train", adapters.hermes,             "struct",  "hermes_json"),
    ("sql_ctx",        "b-mc2/sql-create-context",                "default",                 "train", adapters.sql_create_context, "sql",     "sql"),
    ("sql_clinton",    "Clinton/Text-to-sql-v1",                  "default",                 "train", adapters.clinton_sql,        "sql",     "sql"),
    ("codefeedback",   "m-a-p/CodeFeedback-Filtered-Instruction", "default",                 "train", adapters.codefeedback,       "code",    "code"),
    ("antidoom",       "LiquidAI/antidoom-mix-v1.0",              "default",                 "train", adapters.antidoom,           "prompts", "antidoom"),
]

# `hermes_fc` and `hermes_fc_st` share all 1,161 of their distinct prompts and are
# the same corpus under two config names (s4.2, finding 4). They share a namespace
# so the duplicate pair cannot straddle the split while s4.4 decides which to drop.

EVAL = [
    ("ifstruct",     "LiquidAI/ifstruct-v1.0",       "default",          "test",  ["prompt"]),
    ("ifeval",       "google/IFEval",                "default",          "train", ["prompt"]),
    ("ifbench",      "allenai/IFBench_test",         "default",          "train", ["prompt", "instruction"]),
    ("ifbench_mt_if", "allenai/IFBench_multi-turn",  "ifbench_constraints", "test", ["prompt", "instruction"]),
    ("ifbench_mt_ie", "allenai/IFBench_multi-turn",  "ifeval_constraints",  "test", ["prompt", "instruction"]),
    ("multi_if",     "facebook/Multi-IF",            "default",          "train", []),
    ("humaneval",    "openai/openai_humaneval",      "openai_humaneval", "test",  ["prompt"]),
    ("mbpp_full",    "google-research-datasets/mbpp", "full",            "test",  ["text", "prompt"]),
    ("mbpp_san",     "google-research-datasets/mbpp", "sanitized",       "test",  ["prompt", "text"]),
    ("mmlu_pro",     "TIGER-Lab/MMLU-Pro",           "default",          "test",  ["question"]),
]
# The two IFBench multi-turn configs are new: s4.2 indexed nine benchmark sets and
# neither of these was among them, so their overlap with the mix was never measured.

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# What s4.2 recorded, per corpus. Re-derived below and compared.
S42 = {
 "toolace":        {"rows": 11300,  "contaminated_rows": 4,   "tool_name_mismatch_rows": 113, "refusal_rows": 1928, "suspect_over_refusals": 0,    "label_contradictions_same_query": 0,    "adapter_parse_failures": 0,   "calls_with_non_identifier_names": 8626, "n_groups": 10394,  "distinct_prompts": 11165},
 "apigen":         {"rows": 49402,  "contaminated_rows": 36,  "tool_name_mismatch_rows": 5,   "refusal_rows": 6000, "suspect_over_refusals": 1942, "label_contradictions_same_query": 5999, "adapter_parse_failures": 0,   "calls_with_non_identifier_names": 0,    "n_groups": 45617,  "distinct_prompts": 43103},
 "hermes_fc":      {"rows": 1893,   "contaminated_rows": 0,   "tool_name_mismatch_rows": 0,   "refusal_rows": 0,    "suspect_over_refusals": 0,    "label_contradictions_same_query": 0,    "adapter_parse_failures": 0,   "calls_with_non_identifier_names": 0,    "n_groups": 1108,   "distinct_prompts": 1161},
 "hermes_fc_st":   {"rows": 1893,   "contaminated_rows": 0,   "tool_name_mismatch_rows": 0,   "refusal_rows": 0,    "suspect_over_refusals": 0,    "label_contradictions_same_query": 0,    "adapter_parse_failures": 0,   "calls_with_non_identifier_names": 0,    "n_groups": 1108,   "distinct_prompts": 1161},
 "hermes_glaive":  {"rows": 5209,   "contaminated_rows": 0,   "tool_name_mismatch_rows": 3,   "refusal_rows": 0,    "suspect_over_refusals": 0,    "label_contradictions_same_query": 0,    "adapter_parse_failures": 865, "calls_with_non_identifier_names": 0,    "n_groups": 3406,   "distinct_prompts": 1192},
 "hermes_json_ag": {"rows": 1342,   "contaminated_rows": 0,   "tool_name_mismatch_rows": 0,   "refusal_rows": 0,    "suspect_over_refusals": 0,    "label_contradictions_same_query": 0,    "adapter_parse_failures": 0,   "calls_with_non_identifier_names": 0,    "n_groups": 834,    "distinct_prompts": 1300},
 "hermes_json_st": {"rows": 1241,   "contaminated_rows": 0,   "tool_name_mismatch_rows": 0,   "refusal_rows": 0,    "suspect_over_refusals": 0,    "label_contradictions_same_query": 0,    "adapter_parse_failures": 0,   "calls_with_non_identifier_names": 0,    "n_groups": 1238,   "distinct_prompts": 1241},
 "sql_ctx":        {"rows": 78577,  "contaminated_rows": 0,   "tool_name_mismatch_rows": 0,   "refusal_rows": 0,    "suspect_over_refusals": 0,    "label_contradictions_same_query": 0,    "adapter_parse_failures": 0,   "calls_with_non_identifier_names": 0,    "n_groups": 72942,  "distinct_prompts": 78251},
 "sql_clinton":    {"rows": 262208, "contaminated_rows": 0,   "tool_name_mismatch_rows": 0,   "refusal_rows": 0,    "suspect_over_refusals": 0,    "label_contradictions_same_query": 0,    "adapter_parse_failures": 0,   "calls_with_non_identifier_names": 0,    "n_groups": 154976, "distinct_prompts": 183041},
 "codefeedback":   {"rows": 156526, "contaminated_rows": 759, "tool_name_mismatch_rows": 0,   "refusal_rows": 0,    "suspect_over_refusals": 0,    "label_contradictions_same_query": 0,    "adapter_parse_failures": 0,   "calls_with_non_identifier_names": 0,    "n_groups": 4,      "distinct_prompts": 143726},
 "antidoom":       {"rows": 478229, "contaminated_rows": 0,   "tool_name_mismatch_rows": 0,   "refusal_rows": 0,    "suspect_over_refusals": 0,    "label_contradictions_same_query": 0,    "adapter_parse_failures": 0,   "calls_with_non_identifier_names": 0,    "n_groups": 1,      "distinct_prompts": 1},
}
# Keys whose group index or prompt surface changed by design in this job: their
# s42 group/prompt figures are superseded, not checked.
REKEYED = {"codefeedback", "antidoom"}

HARD = ("rows", "tool_name_mismatch_rows", "refusal_rows", "suspect_over_refusals",
        "label_contradictions_same_query", "calls_with_non_identifier_names")
# `adapter_parse_failures` for glaive is expected to change if the recovery pass
# below succeeds, so it is reported rather than asserted.


# ------------------------------------------------------------------ text + grams

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize(text):
    return _WS.sub(" ", _PUNCT.sub(" ", (text or "").lower())).strip()


def h64(*parts):
    b = hashlib.blake2b(digest_size=8)
    for p in parts:
        b.update(str(p).encode("utf-8", "replace"))
        b.update(b"\x1f")
    return int.from_bytes(b.digest(), "big")


def grams(text, n=NGRAM_N):
    words = normalize(text[:MAX_PROMPT_CHARS] if text else "").split()
    if len(words) < n:
        return ()
    return tuple(
        int.from_bytes(hashlib.blake2b(" ".join(words[i:i + n]).encode(), digest_size=8).digest(), "big")
        for i in range(len(words) - n + 1)
    )


def harvest(row, prefer):
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


def pctiles(values):
    if not values:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "mean": round(float(a.mean()), 2),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


# ------------------------------------------------------------------ eval corpora

from datasets import load_dataset                                # noqa: E402
from huggingface_hub import hf_hub_download, list_repo_files     # noqa: E402


def load(hf_id, config, split):
    return load_dataset(hf_id, config, split=split, token=HF_TOKEN)


eval_report, eval_chunks = {}, []
log("building benchmark n-gram index (n=%d, %d sets + BFCL)" % (NGRAM_N, len(EVAL)))
for label, hf_id, config, split, prefer in EVAL:
    t0 = time.time()
    try:
        ds = load(hf_id, config, split)
        chunk, n, g = [], 0, 0
        for row in ds:
            gg = grams(harvest(row, prefer))
            chunk.extend(gg)
            g += len(gg)
            n += 1
        arr = np.fromiter(chunk, dtype=np.uint64, count=len(chunk))
        eval_chunks.append((label, arr))
        eval_report[label] = {"hf_id": hf_id, "config": config, "split": split, "rows": n,
                              "grams": g, "unique_grams": int(np.unique(arr).size),
                              "prefer_fields": prefer or "all-long-strings",
                              "new_in_s4_3": label.startswith("ifbench_mt"),
                              "seconds": round(time.time() - t0, 1)}
        log("  %-12s %6d rows  %9d grams" % (label, n, g))
    except Exception as exc:
        eval_report[label] = {"hf_id": hf_id, "config": config,
                              "error": "%s: %s" % (type(exc).__name__, exc)}
        log("  %-12s FAILED %s" % (label, exc))

try:
    files = sorted(f for f in list_repo_files(BFCL_REPO, repo_type="dataset", token=HF_TOKEN)
                   if f.startswith("BFCL_v3_") and f.endswith(".json"))
    chunk, n, g = [], 0, 0
    for fname in files:
        path = hf_hub_download(BFCL_REPO, fname, repo_type="dataset", token=HF_TOKEN)
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                gg = grams(harvest(entry, []))
                chunk.extend(gg)
                g += len(gg)
                n += 1
    arr = np.fromiter(chunk, dtype=np.uint64, count=len(chunk))
    eval_chunks.append(("bfcl_v3", arr))
    eval_report["bfcl_v3"] = {"hf_id": BFCL_REPO, "files": len(files), "rows": n, "grams": g,
                              "unique_grams": int(np.unique(arr).size)}
    log("  %-12s %6d rows  %9d grams" % ("bfcl_v3", n, g))
except Exception as exc:
    eval_report["bfcl_v3"] = {"hf_id": BFCL_REPO, "error": "%s: %s" % (type(exc).__name__, exc)}
    log("  bfcl_v3 FAILED %s" % exc)

EVAL_SORTED = [(l, np.unique(a)) for l, a in eval_chunks if a.size]
ALL_EVAL = (np.unique(np.concatenate([a for _, a in EVAL_SORTED]))
            if EVAL_SORTED else np.zeros(0, np.uint64))
del eval_chunks
dump("eval_index.json", eval_report)
log("benchmark index: %d unique %d-grams across %d sets" % (ALL_EVAL.size, NGRAM_N, len(EVAL_SORTED)))
lab.update_progress(12)


def overlap_hits(gg):
    if not gg or ALL_EVAL.size == 0:
        return 0, []
    q = np.fromiter(gg, dtype=np.uint64, count=len(gg))
    idx = np.searchsorted(ALL_EVAL, q)
    idx[idx >= ALL_EVAL.size] = 0
    hit = ALL_EVAL[idx] == q
    if not hit.any():
        return 0, []
    qh = q[hit]
    who = []
    for label, arr in EVAL_SORTED:
        j = np.searchsorted(arr, qh)
        j[j >= arr.size] = 0
        if bool((arr[j] == qh).any()):
            who.append(label)
    return int(hit.sum()), who


# -------------------------------------------------------------------- tokenizer

TOK = None
try:
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(TOKENIZER_ID, token=HF_TOKEN)
    log("tokenizer %s loaded" % TOKENIZER_ID)
except Exception as exc:
    log("tokenizer unavailable (%s); token stats omitted" % exc)


# ------------------------------------------------------------- answer surfaces

def answer_surface(row, rec):
    """The label side of the (prompt, toolset, answer) triple."""
    if rec.get("calls"):
        return json.dumps([[c.get("name"), sorted((c.get("args") or {}).keys())]
                           for c in rec["calls"]], sort_keys=True)
    if rec.get("refusal"):
        return "refusal:" + normalize(rec["refusal"])[:300]
    for k in ("answer", "output", "response", "completion", "sql"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return normalize(v)[:2000]
    conv = adapters._loads(row.get("conversations"), []) or []
    for turn in conv:
        if isinstance(turn, dict):
            frm = (turn.get("from") or turn.get("role") or "").lower()
            if frm in ("gpt", "assistant", "model"):
                return normalize(turn.get("value") or turn.get("content") or "")[:2000]
    return ""


def toolset_sig(rec):
    tools = rec.get("tools") or []
    if not tools:
        return ""
    return adapters._group(tools)


# ------------------------------------------------ glaive parse-failure taxonomy

def hermes_row_shape(row):
    """Why did the adapter get nothing out of this row?"""
    conv = adapters._loads(row.get("conversations"), []) or []
    roles = [str((t.get("from") or t.get("role") or "?")).lower()
             for t in conv if isinstance(t, dict)]
    sys_txt = " ".join((t.get("value") or t.get("content") or "")
                       for t in conv if isinstance(t, dict)
                       and str(t.get("from") or t.get("role") or "").lower() == "system")
    raw_tools = row.get("tools")
    return {
        "roles": "|".join(roles) or "(no conversations)",
        "n_turns": len(conv),
        "tools_field": ("absent" if raw_tools is None else
                        "empty" if not str(raw_tools).strip() else "present"),
        "schema_in_system": bool(re.search(r'"(?:parameters|properties|name)"\s*:', sys_txt)
                                 or "<tools>" in sys_txt),
        "toolcall_tag_in_conv": any("<tool_call>" in (t.get("value") or t.get("content") or "")
                                    for t in conv if isinstance(t, dict)),
        "category": row.get("category"),
    }


# ------------------------------------------------------------- per-corpus pass

FLAG_CONTAM, FLAG_MISMATCH, FLAG_SUSPECT, FLAG_CONTRA, FLAG_PARSEFAIL, FLAG_DUPTRIPLE = 1, 2, 4, 8, 16, 32

rows_by_label = {}          # label -> list of (row_id, group_key, prompt_h, triple_h, flags)
summary, flags_out, extras = {}, {}, {}
prompt_owner = collections.defaultdict(set)
code_pending = []           # (row_id, prompt_h, triple_h, flags, mods, called, defined, lang)
code_mod_freq, code_sym_freq = collections.Counter(), collections.Counter()

for ci, (label, hf_id, config, split, adapt, role, ns) in enumerate(TRAIN):
    t0 = time.time()
    log("=== %s (%s / %s)" % (label, hf_id, config))
    try:
        ds = load(hf_id, config, split)
    except Exception as exc:
        summary[label] = {"hf_id": hf_id, "config": config, "role": role, "namespace": ns,
                          "error": "%s: %s" % (type(exc).__name__, exc)}
        log("  LOAD FAILED %s" % exc)
        continue

    recs = []
    n_rows = 0
    n_contam = n_mismatch = n_suspect = n_contra = n_refusal = n_parsefail = n_nonident = 0
    contam_by = collections.Counter()
    contam_ids, mismatch_ids, suspect_ids, contra_ids, parsefail_ids = [], [], [], [], []
    shapes = collections.Counter()
    shape_samples = []
    src_hist = collections.Counter()
    policy_hist = collections.Counter()
    license_hist = collections.Counter()
    tok_prompt = []
    query_labels = collections.defaultdict(set)
    query_rows = collections.defaultdict(list)

    for row in ds:
        n_rows += 1
        try:
            rec = adapt(row)
        except Exception:
            n_parsefail += 1
            continue
        prompt = rec.get("prompt") or ""
        tools = rec.get("tools") or []
        calls = rec.get("calls") or []
        rid = rec.get("row_id") or ""
        flags = 0

        # A row the adapter could not read: no prompt, and nothing to train on.
        if not prompt and not calls and not rec.get("refusal"):
            n_parsefail += 1
            flags |= FLAG_PARSEFAIL
            if len(parsefail_ids) < 20000:
                parsefail_ids.append(rid)
            if label.startswith("hermes"):
                sh = hermes_row_shape(row)
                shapes[json.dumps({k: sh[k] for k in
                                   ("roles", "tools_field", "schema_in_system",
                                    "toolcall_tag_in_conv")}, sort_keys=True)] += 1
                if len(shape_samples) < 5:
                    shape_samples.append(sh)

        declared = {t.get("name") for t in tools if t.get("name")}
        bad = [c.get("name") for c in calls
               if c.get("name") and declared and c.get("name") not in declared]
        if bad:
            n_mismatch += 1
            flags |= FLAG_MISMATCH
            if len(mismatch_ids) < 20000:
                mismatch_ids.append(rid)
        n_nonident += sum(1 for c in calls if c.get("name")
                          and not adapters._IDENT.match(str(c["name"])))

        if rec.get("refusal"):
            n_refusal += 1
            if rec.get("target") and rec["target"] in declared:
                n_suspect += 1
                flags |= FLAG_SUSPECT
                if len(suspect_ids) < 20000:
                    suspect_ids.append(rid)

        nq = normalize(prompt)
        ph = h64(nq)
        if nq:
            query_labels[ph].add("refusal" if rec.get("refusal") else "call" if calls else "other")
            if len(query_rows[ph]) < 4:
                query_rows[ph].append(rid)

        gg = grams(prompt)
        nhit, who = overlap_hits(gg)
        if nhit:
            n_contam += 1
            flags |= FLAG_CONTAM
            for w in who:
                contam_by[w] += 1
            if len(contam_ids) < 40000:
                contam_ids.append(rid)

        th = h64(nq, toolset_sig(rec), answer_surface(row, rec))
        prompt_owner[ph].add(label)

        if TOK is not None and prompt and len(tok_prompt) < SUBSAMPLE:
            tok_prompt.append(len(TOK(prompt, add_special_tokens=False)["input_ids"]))

        if label == "codefeedback":
            ex = rec.get("extra") or {}
            mods, called = ex.get("mods") or [], ex.get("called") or []
            for m in mods:
                code_mod_freq[m] += 1
            for c in called:
                code_sym_freq[c] += 1
            code_pending.append((rid, ph, th, flags, mods, called,
                                 ex.get("defined") or [], ex.get("lang")))
        else:
            recs.append((rid, rec.get("group_key") or "ungrouped", ph, th, flags))

        if label == "antidoom":
            ex = rec.get("extra") or {}
            src_hist["%s/%s" % (ex.get("source"), ex.get("source_config"))] += 1
            if ex.get("heldout_policy"):
                policy_hist[str(ex["heldout_policy"])[:180]] += 1
            license_hist[str(ex.get("source_license"))] += 1

    # Same-query label contradictions (the APIGen over-refusal test).
    for qh, kinds in query_labels.items():
        if "refusal" in kinds and "call" in kinds:
            n_contra += len(query_rows[qh])
            if len(contra_ids) < 20000:
                contra_ids.extend(query_rows[qh])

    if label != "codefeedback":
        rows_by_label[label] = recs

    summary[label] = {
        "hf_id": hf_id, "config": config, "split": split, "role": role, "namespace": ns,
        "rows": n_rows,
        "contaminated_rows": n_contam,
        "contaminated_rate": round(n_contam / max(1, n_rows), 5),
        "contaminated_by_benchmark": dict(contam_by.most_common()),
        "tool_name_mismatch_rows": n_mismatch,
        "calls_with_non_identifier_names": n_nonident,
        "refusal_rows": n_refusal,
        "suspect_over_refusals": n_suspect,
        "label_contradictions_same_query": n_contra,
        "adapter_parse_failures": n_parsefail,
        "distinct_prompts": len(query_labels),
        "tokens_per_prompt": pctiles(tok_prompt),
        "seconds": round(time.time() - t0, 1),
    }
    flags_out[label] = {"contaminated": contam_ids, "tool_name_mismatch": mismatch_ids,
                        "suspect_over_refusal": suspect_ids,
                        "label_contradictions": contra_ids,
                        "adapter_parse_failure": parsefail_ids}
    dump("flags_%s.json" % label, flags_out[label])
    if shapes:
        extras[label] = {"parse_failure_shapes": dict(shapes.most_common(20)),
                         "samples": shape_samples}
    if label == "antidoom":
        extras["antidoom"] = {"upstream_sources": dict(src_hist.most_common(200)),
                              "n_upstream_sources": len(src_hist),
                              "heldout_policies": dict(policy_hist.most_common(60)),
                              "licenses": dict(license_hist.most_common(40))}
    log("  %.1fs  rows=%d contam=%d mismatch=%d refusal=%d suspect=%d contra=%d parsefail=%d"
        % (time.time() - t0, n_rows, n_contam, n_mismatch, n_refusal, n_suspect,
           n_contra, n_parsefail))
    lab.update_progress(12 + int(58 * (ci + 1) / len(TRAIN)))


# ------------------------------------------------- CodeFeedback rarity re-keying

def code_key(mods, called, defined, lang):
    """Key on the rarest API symbols a row touches.

    Two rows that both call `numpy.linalg.svd` belong together; two rows that both
    call `print` do not, because `print` says nothing about what the row is. So the
    key is the least frequent symbol in the row, measured over the whole corpus:
    the group becomes "every answer whose most distinctive API symbol is X".

    K is 1 by design. Taking the K rarest symbols instead makes the key sensitive
    to symbols that are not the point: an answer that imports `scipy` alongside
    `numpy` gets a different 3-symbol key from one that imports `numpy` alone,
    even when both are calling the same `np.linalg.svd`. One symbol is stable
    under that. Modules are folded in at three times their raw count so a bare
    import loses to an actual call, and a symbol seen once in the whole corpus is
    a typo rather than a surface, so it is dropped before the minimum is taken.
    Rows with no surviving symbol fall back to the names they define, then to the
    language alone; the tier mix is reported so a collapsed tail stays visible.
    """
    pool = ([(code_sym_freq[c] * 1, c) for c in called if code_sym_freq[c] >= 2]
            + [(code_mod_freq[m] * 3, "mod:" + m) for m in mods if code_mod_freq[m] >= 2])
    if pool:
        pool.sort()
        return "api:%s|%s" % (lang, ",".join(s for _, s in pool[:RARE_K])), "api"
    if defined:
        return "def:%s|%s" % (lang, ",".join(sorted(defined)[:RARE_K])), "defined"
    return "bare:%s" % lang, "bare"


if code_pending:
    tiers = collections.Counter()
    recs = []
    for rid, ph, th, flags, mods, called, defined, lang in code_pending:
        gk, tier = code_key(mods, called, defined, lang)
        tiers[tier] += 1
        recs.append((rid, gk, ph, th, flags))
    rows_by_label["codefeedback"] = recs
    gsz = collections.Counter(r[1] for r in recs)
    extras["codefeedback"] = {
        "key_tiers": dict(tiers),
        "n_groups": len(gsz),
        "largest_group_share": round(max(gsz.values()) / max(1, len(recs)), 5),
        "largest_groups": dict(gsz.most_common(15)),
        "s42_key": "resource (4 groups, largest 40.2%)",
        "distinct_modules": len(code_mod_freq),
        "distinct_called_symbols": len(code_sym_freq),
        "top_modules": dict(code_mod_freq.most_common(30)),
    }
    summary["codefeedback"]["n_groups_rekeyed"] = len(gsz)
    log("codefeedback re-keyed: %d groups, largest %.3f%%, tiers %s"
        % (len(gsz), 100 * max(gsz.values()) / max(1, len(recs)), dict(tiers)))
    del code_pending
lab.update_progress(74)


# ------------------------------------------------------------ group namespaces

NS = {label: ns for label, _, _, _, _, _, ns in TRAIN}
ns_groups = collections.defaultdict(collections.Counter)     # ns -> group_key -> rows
for label, recs in rows_by_label.items():
    for _, gk, _, _, _ in recs:
        ns_groups[NS[label]][gk] += 1

split_of, ns_report = {}, {}
for ns, groups in sorted(ns_groups.items()):
    total = sum(groups.values())
    collapsed = sorted(g for g, c in groups.items() if c > COLLAPSE_FRAC * total)
    collapsed_rows = sum(groups[g] for g in collapsed)
    free = [g for g in groups if g not in set(collapsed)]
    free_rows = total - collapsed_rows
    order = sorted(free, key=lambda g: h64(ns, g, SALT))
    # Targets are fractions of ALL rows, so a large collapsed group shrinks train
    # rather than the held-out sets. Capped at 40% of the free pool each, so a
    # namespace that is mostly collapsed does not hand its entire splittable
    # remainder to eval.
    want_test = min(TEST_FRAC * total, 0.4 * free_rows)
    want_val = min(VAL_FRAC * total, 0.4 * free_rows)
    got = {"test": 0, "val": 0, "train": 0}
    for g in order:
        if got["test"] < want_test:
            dst = "test"
        elif got["val"] < want_val:
            dst = "val"
        else:
            dst = "train"
        split_of[(ns, g)] = dst
        got[dst] += groups[g]
    for g in collapsed:
        split_of[(ns, g)] = "train"
        got["train"] += groups[g]
    ns_report[ns] = {
        "corpora": sorted(l for l, n in NS.items() if n == ns and l in rows_by_label),
        "rows": total, "n_groups": len(groups),
        "collapsed_groups": len(collapsed), "collapsed_rows": collapsed_rows,
        "collapsed_share": round(collapsed_rows / max(1, total), 5),
        "collapsed_keys": collapsed[:10],
        "target_test_rows": int(want_test), "target_val_rows": int(want_val),
        "targets_capped_by_free_pool": bool(want_test < TEST_FRAC * total
                                            or want_val < VAL_FRAC * total),
        "rows_by_split": got,
        "share_by_split": {k: round(v / max(1, total), 5) for k, v in got.items()},
        "largest_group_rows": max(groups.values()),
    }
    log("ns %-12s %8d rows %7d groups  train/val/test %.3f/%.3f/%.3f  collapsed %d rows"
        % (ns, total, len(groups), got["train"] / max(1, total), got["val"] / max(1, total),
           got["test"] / max(1, total), collapsed_rows))


# ------------------------------------------------ assignment, leakage, dup audit

prompt_splits = collections.defaultdict(set)
triple_splits = collections.defaultdict(set)
triple_count = collections.Counter()
per_label_split = {}
gz_path = os.path.join(OUT, "splits.jsonl.gz")
with gzip.open(gz_path, "wt") as fh:
    for label, recs in rows_by_label.items():
        ns = NS[label]
        c = collections.Counter()
        for rid, gk, ph, th, flags in recs:
            dst = split_of[(ns, gk)]
            c[dst] += 1
            prompt_splits[ph].add(dst)
            triple_splits[th].add(dst)
            triple_count[th] += 1
            fh.write(json.dumps({"c": label, "ns": ns, "i": rid,
                                 "g": "%016x" % h64(ns, gk), "s": dst, "f": flags},
                                separators=(",", ":")) + "\n")
        per_label_split[label] = dict(c)
try:
    lab.save_artifact(gz_path)
except Exception as exc:
    log("save_artifact failed for splits.jsonl.gz: %s" % exc)
log("splits written: %.1f MB" % (os.path.getsize(gz_path) / 1e6))
lab.update_progress(86)

dup_triples = {h: n for h, n in triple_count.items() if n > 1}
straddle_prompts = sum(1 for v in prompt_splits.values() if len(v) > 1)
straddle_triples = sum(1 for v in triple_splits.values() if len(v) > 1)
cross_corpus = collections.Counter()
for ph, labs in prompt_owner.items():
    if len(labs) > 1:
        cross_corpus[" | ".join(sorted(labs))] += 1

total_rows = sum(len(r) for r in rows_by_label.values())
duplication = {
    "definition": "triple = (normalized prompt, toolset signature, normalized answer surface)",
    "rows": total_rows,
    "distinct_triples": len(triple_count),
    "duplicate_triple_groups": len(dup_triples),
    "duplicate_triple_rows": int(sum(dup_triples.values())),
    "rows_removed_by_exact_triple_dedup": int(sum(dup_triples.values()) - len(dup_triples)),
    "distinct_prompts": len(prompt_splits),
    "duplicate_prompt_rows": total_rows - len(prompt_splits),
    "cross_corpus_prompt_pairs": dict(cross_corpus.most_common(50)),
    "per_corpus": {},
}
for label, recs in rows_by_label.items():
    tc = collections.Counter(r[3] for r in recs)
    dups = {h: n for h, n in tc.items() if n > 1}
    duplication["per_corpus"][label] = {
        "rows": len(recs), "distinct_triples": len(tc),
        "duplicate_triple_rows": int(sum(dups.values())),
        "removed_by_exact_triple_dedup": int(sum(dups.values()) - len(dups)),
        "distinct_prompts": len(set(r[2] for r in recs)),
    }
dump("duplication.json", duplication)

leakage = {
    "prompts_appearing_in_more_than_one_split": straddle_prompts,
    "triples_appearing_in_more_than_one_split": straddle_triples,
    "prompt_leak_rate": round(straddle_prompts / max(1, len(prompt_splits)), 6),
    "note": ("A group-disjoint split can still straddle on prompts when the same prompt "
             "carries two different group keys. Counted here rather than assumed zero."),
}
dump("leakage_audit.json", leakage)
dump("split_summary.json", {"namespaces": ns_report, "per_corpus": per_label_split,
                            "params": {"test_frac": TEST_FRAC, "val_frac": VAL_FRAC,
                                       "collapse_frac": COLLAPSE_FRAC, "salt": SALT,
                                       "seed": SEED}})
dump("extras.json", extras)


# ---------------------------------------------------------------- assertions

checks, fails = [], []
for label, want in S42.items():
    got = summary.get(label) or {}
    if "rows" not in got:
        checks.append({"corpus": label, "field": "*", "status": "SKIP",
                       "reason": got.get("error", "corpus not loaded")})
        continue
    for field, w in want.items():
        g = got.get(field)
        hard = field in HARD
        if label in REKEYED and field in ("n_groups", "distinct_prompts", "contaminated_rows"):
            checks.append({"corpus": label, "field": field, "s42": w, "s43": g,
                           "status": "SUPERSEDED",
                           "reason": "group key or prompt surface changed by design in s4.3"})
            continue
        if field == "n_groups":
            g = len(set(r[1] for r in rows_by_label.get(label, [])))
        if field == "adapter_parse_failures" and label == "hermes_glaive":
            checks.append({"corpus": label, "field": field, "s42": w, "s43": g,
                           "status": "REPORTED", "reason": "recovery pass may change this"})
            continue
        ok = (g == w)
        checks.append({"corpus": label, "field": field, "s42": w, "s43": g,
                       "status": "OK" if ok else ("FAIL" if hard else "DRIFT")})
        if not ok and hard:
            fails.append("%s.%s: s4.2=%s s4.3=%s" % (label, field, w, g))

dump("assertions.json", {"hard_fields": list(HARD), "failures": fails, "checks": checks})
log("assertions: %d checks, %d hard failures" % (len(checks), len(fails)))


# ------------------------------------------------------------------- figures

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    nss = sorted(ns_report)
    bottom = np.zeros(len(nss))
    for dst, colour in (("train", "0.75"), ("val", "0.45"), ("test", "0.15")):
        vals = np.array([ns_report[n]["share_by_split"].get(dst, 0) for n in nss])
        axes[0].bar(nss, vals, bottom=bottom, label=dst, color=colour)
        bottom += vals
    axes[0].set_title("realized row share by split, per group namespace")
    axes[0].legend(fontsize=7)
    axes[0].tick_params(axis="x", rotation=55, labelsize=7)

    labels = sorted(rows_by_label)
    axes[1].bar(labels, [100 * summary[l]["contaminated_rate"] for l in labels], color="0.4")
    axes[1].set_ylabel("%% rows sharing a %d-gram with a benchmark" % NGRAM_N)
    axes[1].set_title("contamination, with antidoom read correctly")
    axes[1].tick_params(axis="x", rotation=55, labelsize=7)
    fig.tight_layout()
    p = os.path.join(OUT, "fig-splits.png")
    fig.savefig(p, dpi=140)
    lab.save_artifact(p)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    for label in labels:
        gsz = collections.Counter(r[1] for r in rows_by_label[label])
        hh = collections.Counter(gsz.values())
        xs = sorted(hh)
        ax.plot(xs, [hh[x] for x in xs], marker=".", label=label, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("group size (rows sharing a split key)")
    ax.set_ylabel("number of groups")
    ax.legend(fontsize=6)
    ax.set_title("group-size distribution after re-keying")
    fig.tight_layout()
    p = os.path.join(OUT, "fig-group-sizes-rekeyed.png")
    fig.savefig(p, dpi=140)
    lab.save_artifact(p)
except Exception as exc:
    log("figures failed: %s" % exc)
    traceback.print_exc()


# ---------------------------------------------------------------------- wrap

report = {
    "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "params": {"ngram_n": NGRAM_N, "seed": SEED, "salt": SALT, "test_frac": TEST_FRAC,
               "val_frac": VAL_FRAC, "collapse_frac": COLLAPSE_FRAC,
               "code_key_rare_symbols": RARE_K},
    "benchmark_index": eval_report,
    "corpora": summary,
    "namespaces": ns_report,
    "per_corpus_split": per_label_split,
    "duplication": duplication,
    "leakage": leakage,
    "extras": extras,
    "assertions": {"failures": fails, "n_checks": len(checks)},
}
dump("splits_report.json", report)

score = {
    "rows_total": total_rows,
    "namespaces": len(ns_report),
    "rows_train": sum(v.get("train", 0) for v in per_label_split.values()),
    "rows_val": sum(v.get("val", 0) for v in per_label_split.values()),
    "rows_test": sum(v.get("test", 0) for v in per_label_split.values()),
    "prompts_straddling_splits": straddle_prompts,
    "triples_straddling_splits": straddle_triples,
    "duplicate_triple_rows": duplication["duplicate_triple_rows"],
    "rows_removed_by_triple_dedup": duplication["rows_removed_by_exact_triple_dedup"],
    "antidoom_contaminated_rows": summary.get("antidoom", {}).get("contaminated_rows"),
    "codefeedback_groups": extras.get("codefeedback", {}).get("n_groups"),
    "glaive_parse_failures": summary.get("hermes_glaive", {}).get("adapter_parse_failures"),
    "assertion_failures": len(fails),
}
lab.update_progress(100)
log("SCORE " + json.dumps(score))

if fails:
    # Every artifact is already saved, so the failure is diagnosable from
    # assertions.json. The job is left FAILED rather than finished, so nothing
    # downstream can cite splits built on counts that did not reproduce.
    log("HARD ASSERTION FAILURE: " + "; ".join(fails[:10]))
    raise RuntimeError("s4.2 counts did not reproduce: " + "; ".join(fails[:10]))

lab.finish(message="s4.3 splits over %d rows in %d namespaces, %d prompts straddling a split"
                   % (total_rows, len(ns_report), straddle_prompts), score=score)
