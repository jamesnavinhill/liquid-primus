"""s5.2 — the evaluation harness. One job, one model, four components, raw text kept.

This is the most reused artifact in the project: the baseline row, all eight sweep arms,
the tuning runs and the final checkpoints are all scored by this file, so the numbers only
mean something as a set if the harness never moves underneath them. Three properties are
therefore built in rather than left to discipline.

**Everything is regenerable.** Raw completions are saved per component before any scoring
runs. A scoring bug found after eight arms have been evaluated costs a CPU rerun over the
saved text, not eight GPU passes.

**Nothing is quoted.** Every model that appears in a comparison is rerun here, including
the stock checkpoint and the competitor finetunes. Published numbers are used once, as a
check that this harness lands near them on the model they were published for, and never
as a baseline.

**The baseline is given its best shot.** Tool contracts are presented two ways, the
convention `s4` trains on and LFM2.5's own native `tools=` format, and a baseline's score
is the better of the two. A delta measured against a model prompted in a format it was
never trained for is not a delta worth reporting.

Components: BFCLv3-AST composite (`bfcl.py`), IFStruct v1.0 by Liquid's own validator
(`ifstruct_score.py`), IFEval by Google's own registry (`ifeval_score.py`), and the two
in-house probe families plus a clean control arm (`probes_score.py`).
"""

import collections
import hashlib
import json
import os
import time
import urllib.request


import adapters
import bfcl
import cdd
import gen
import gen_gguf
import ifeval_score
import ifstruct_score
import probes_calib
import probes_score
import prompting
import replay

# ------------------------------------------------------------------ packing
#
# `s6` scores eight checkpoints on four components. Run one per card that is eight card-hours
# of a job whose GPU sits near idle between generation batches; `pack.py` from the sweep runs
# several of these at once on a single card instead, one isolated child process per arm, each
# reading its own checkpoint and writing its own results directory.
#
# The contract is the sweep's, unchanged: config and output directory arrive on the
# environment, the job API belongs to the supervisor alone, shared-storage objects are
# resolved to paths the supervisor already fetched, and an assertion failure exits non-zero
# rather than calling lab.error. What is specific to an evaluation is that its arms differ in
# their *inputs* -- each one reads a different adapter -- so the objects a pack must stage
# come from the per-arm overrides as well as the shared config, and any host resource with a
# fixed default (the 4-bit server's port, its scratch directories) has to be made per-arm or
# two children will fight over it.
PACK_CHILD = os.environ.get("TIDEPOOL_PACK_CHILD") == "1"
PACK_MEMFRAC = float(os.environ.get("TIDEPOOL_PACK_MEMFRAC") or 0.0)
PACK_LOCAL = json.loads(os.environ.get("TIDEPOOL_PACK_LOCAL") or "{}")

if PACK_CHILD:
    class _SupervisedLab(object):
        """Stands in for the job API inside a packed child.

        Every reporting call in this file is already guarded by `if not PACK_CHILD`; this
        catches anything that is ever added without one, turning it into a visible no-op
        rather than a second writer on the job's stream. It says so once per method so a
        missed guard shows up in the console without flooding it.

        Storage calls are the exception and must raise. The file probes for them with
        `getattr(lab, "storage_upload", None)`, and a stub that answered would hand back a
        no-op the caller then reports as a successful upload: the weights would never reach
        shared storage and the log would say they had. Raising makes the probe fail, which is
        exactly what sends the caller down its CLI fallback -- a separate process writing to
        an arm-specific prefix, which is safe to run concurrently."""

        _warned = set()
        _must_raise = ("storage_upload", "upload_storage", "storage_put",
                       "storage_download", "get_config", "init")

        def __getattr__(self, name):
            if name in _SupervisedLab._must_raise:
                raise AttributeError(
                    "lab.%s is deliberately absent in a packed child; storage goes through "
                    "the CLI and config arrives on the environment" % name)

            def _noop(*a, **k):
                if name not in _SupervisedLab._warned:
                    _SupervisedLab._warned.add(name)
                    print("[pack] suppressed lab.%s() in child mode" % name, flush=True)
            return _noop

    lab = _SupervisedLab()
    CFG = json.loads(os.environ.get("TIDEPOOL_PACK_CFG") or "{}")
else:
    from lab import lab
    lab.init()
    CFG = lab.get_config() or {}


def storage(obj):
    """Resolve a shared-storage object to a local path.

    A packed child downloads nothing. Every arm on the card reads the same corpus, so N
    children fetching it N times would multiply the transfer, the disk and the failure
    surface for no benefit; the supervisor fetches each object once and hands the resolved
    paths down. A missing path is fatal rather than silently re-downloaded, because a child
    that quietly reaches for the network has stopped being isolated."""
    if not PACK_CHILD:
        return lab.storage_download(obj)
    p = PACK_LOCAL.get(obj)
    if not p or not os.path.exists(p):
        raise SystemExit("pack child was given no local path for %r (supervisor should have "
                         "downloaded it before spawning children)" % obj)
    return p


def C(k, default):
    v = CFG.get(k)
    return default if v is None or v == "" else v


IFSTRUCT_URL = "https://raw.githubusercontent.com/Liquid4All/ifstruct/main/data/test.jsonl"
BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
IFEVAL_REPO = "google/IFEval"
IFEVAL_FILE = "ifeval_input_data.jsonl"

# A packed child writes into `out/<arm>/`, which the supervisor sweeps at the end and uploads
# under an `<arm>__` prefix, so eight arms' completions land in one job's artifact list without
# a collision and each file still says which arm produced it.
OUT = os.environ.get("TIDEPOOL_PACK_OUT") or "out"
ARM = str(os.environ.get("TIDEPOOL_PACK_ARM") or C("arm", "") or "")
# Anything a child puts on the host -- a listening port, a download cache -- has to be
# per-arm. Two llama.cpp servers both defaulting to 8080 is not an OOM; the second one dies
# on bind and the first one silently answers both arms' requests, which would produce two
# plausible score sets from one model.
PACK_INDEX = int(os.environ.get("TIDEPOOL_PACK_INDEX") or 0)
# The benchmark caches go under the arm's directory for the same reason. Two children both
# copying BFCL into `data/bfcl` is not a crash and not an OOM: it is one of them reading a
# file the other is halfway through writing, which surfaces as a JSON parse error in a
# component that has nothing to do with downloads, or worse, does not surface at all.
DATA = os.path.join(OUT, "data")
NOTES = []

import asserts  # noqa: E402
from asserts import ASSERTS, check, check_completions  # noqa: E402


def log(msg):
    print(("[%s] %s" % (ARM, msg)) if ARM else msg, flush=True)
    if PACK_CHILD:
        return
    try:
        lab.log(str(msg))
    except Exception:                                              # noqa: BLE001
        pass


asserts.set_logger(log)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


_SAVED = set()


def save(obj, name, kind=None):
    # Two facts about the artifact store, both learned the expensive way.
    #
    # It keys on the *basename*, so `completions/x.jsonl` and `scored/x.jsonl` register as
    # the same artifact and the second silently replaces the first. The first smoke lost its
    # raw structured-output text that way. Names are flat now, and a repeat is recorded as a
    # note rather than passing unnoticed.
    #
    # And `save_artifact(path, type="evals")` files a copy under eval-results metadata
    # rather than in the generic artifact list, which takes it out of `job artifacts` and
    # out of `job download` with it. The Granite smoke wrote eleven files; the six saved
    # with no type came back and the five saved with `type="evals"` did not, though the log
    # reported every one of them as saved. Nothing here passes `type=` any more. `kind` is
    # kept as a label on the log line, and the numbers the eval pane wants reach it through
    # finish(score=...), which does show up.
    base = os.path.basename(name)
    if base in _SAVED:
        NOTES.append("artifact name reused, the earlier one is gone: %s" % base)
    _SAVED.add(base)
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(obj, (dict, list)) and name.endswith(".json"):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True, ensure_ascii=False)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            for row in obj:
                # `default=repr` on purpose: a row that will not serialize must not
                # destroy a pass whose generations are already paid for. The parser
                # normalizes values (prompting.jsonable), so this is a last resort that
                # leaves the odd value visible in the artifact rather than raising.
                fh.write(json.dumps(row, ensure_ascii=False, default=repr) + "\n")
    if PACK_CHILD:
        # The supervisor walks this directory when the arm exits and uploads what it finds,
        # so a child that also uploaded would either duplicate the artifact or race the
        # collector for the name.
        log("wrote %s (%d bytes)%s"
            % (name, os.path.getsize(path), (" [%s]" % kind) if kind else ""))
        return path
    try:
        lab.save_artifact(path)
        log("saved artifact %s (%d bytes)%s"
            % (name, os.path.getsize(path), (" [%s]" % kind) if kind else ""))
    except Exception as exc:                                       # noqa: BLE001
        log("save_artifact failed for %s: %s" % (name, exc))
    return path


# ------------------------------------------------------------------ inputs

def fetch_bfcl(categories):
    from huggingface_hub import hf_hub_download
    root = os.path.join(DATA, "bfcl")
    os.makedirs(os.path.join(root, "possible_answer"), exist_ok=True)
    shas = {}
    for cat in categories:
        for sub in ("", "possible_answer/"):
            if sub and cat in bfcl.NO_CALL_EXPECTED | bfcl.CALL_EXPECTED:
                continue
            fn = "%sBFCL_v3_%s.json" % (sub, cat)
            src = hf_hub_download(repo_id=BFCL_REPO, filename=fn, repo_type="dataset")
            dst = os.path.join(root, fn)
            with open(src, "rb") as a, open(dst, "wb") as b:
                b.write(a.read())
            shas[fn] = sha(dst)
    return root, shas


def fetch_ifstruct():
    os.makedirs(DATA, exist_ok=True)
    dst = os.path.join(DATA, "ifstruct_test.jsonl")
    with urllib.request.urlopen(IFSTRUCT_URL, timeout=120) as r, open(dst, "wb") as fh:
        fh.write(r.read())
    return dst, {"ifstruct/data/test.jsonl": sha(dst)}


def fetch_ifeval():
    from huggingface_hub import hf_hub_download
    src = hf_hub_download(repo_id=IFEVAL_REPO, filename=IFEVAL_FILE, repo_type="dataset")
    return src, {IFEVAL_FILE: sha(src)}


def fetch_probes(obj):
    local = storage(obj)
    if os.path.isdir(local):
        local = os.path.join(local, os.path.basename(obj))
    rows = []
    with open(local, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    for r in rows:
        r.setdefault("id", "%s_%s_%s_d%s" % (r["probe"], r["arm"], r["mode"], r["depth"]))
    return rows, {obj: sha(local)}


def read_flags(fh):
    """id -> whether the free-form completion tripped the guardrail regex, from a scored file."""
    flags = {}
    for line in fh:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if isinstance(r.get("detail"), dict) and "flagged" in r["detail"]:
            flags[r["id"]] = bool(r["detail"]["flagged"])
    return flags


# ------------------------------------------------------------------ components

def run_bfcl(cfg, runner, prompter, categories, styles, limit):
    root, shas = fetch_bfcl(categories)
    items = []
    for cat in categories:
        got = bfcl.load_category(root, cat, limit=limit)
        items.extend(got)
        log("  bfcl %-24s %4d items" % (cat, len(got)))
    check("bfcl_items_present", len(items) > 0, "%d items" % len(items))
    per_style, summaries = {}, {}
    for style in styles:
        if style == "native_tools" and not prompter.supports_tools_arg:
            NOTES.append("native_tools skipped: the model's chat template takes no tools= "
                         "argument, so only the s4 text convention was scored")
            continue
        prompts, kept = [], []
        for it in items:
            try:
                prompts.append(prompter.render(it["messages"], it["tools"], style=style))
                kept.append(it)
            except Exception as exc:                               # noqa: BLE001
                NOTES.append("bfcl render failed for %s (%s)" % (it["id"], exc))
        outs = runner.generate(prompts, max_new_tokens=int(cfg.get("max_new_bfcl", 320)),
                               batch_size=int(cfg.get("batch_size", 16)),
                               tag="bfcl/%s" % style,
                               ids=[it["id"] for it in kept])
        check_completions("bfcl_%s" % style, outs)
        raw, scored = [], []
        for it, p, o in zip(kept, prompts, outs):
            calls = prompting.parse_calls(o)
            ok, why = bfcl.score_item(it, calls)
            raw.append({"id": it["id"], "category": it["category"], "style": style,
                        "prompt_sha": hashlib.sha256(p.encode()).hexdigest()[:12],
                        "prompt": p, "completion": o})
            scored.append({"id": it["id"], "category": it["category"], "style": style,
                           "correct": ok, "reason": why, "n_calls": len(calls),
                           "calls": calls})
        save(raw, "completions_bfcl_%s.jsonl" % style)
        save(scored, "scored_bfcl_%s.jsonl" % style, kind="evals")
        summaries[style] = bfcl.summarize(scored)
        per_style[style] = summaries[style]["composite_category_mean"]
        log("  bfcl/%-13s composite %.4f (item-weighted %.4f)"
            % (style, summaries[style]["composite_category_mean"],
               summaries[style]["item_weighted"]))
    best = max(per_style, key=lambda s: per_style[s]) if per_style else None
    return {"styles": summaries, "best_style": best,
            "composite": per_style.get(best, 0.0),
            "input_sha": shas}


def run_ifstruct(cfg, runner, prompter, limit):
    path, shas = fetch_ifstruct()
    items = ifstruct_score.load(path, limit=limit)
    check("ifstruct_count", limit or len(items) == 2000, "%d items" % len(items))
    prompts = [prompter.render(it["messages"], None) for it in items]
    outs = runner.generate(prompts, max_new_tokens=int(cfg.get("max_new_ifstruct", 1024)),
                           batch_size=int(cfg.get("batch_size", 16)), tag="ifstruct",
                           ids=[it["id"] for it in items])
    check_completions("ifstruct", outs)
    raw, scored = [], []
    for it, p, o in zip(items, prompts, outs):
        ok, detail = ifstruct_score.score_item(it, o)
        raw.append({"id": it["id"], "prompt_sha": hashlib.sha256(p.encode()).hexdigest()[:12],
                    "completion": o})
        scored.append({"id": it["id"], "correct": ok, "detail": detail,
                       "output_format": it["spec"]["output_format"]})
    save(raw, "completions_ifstruct.jsonl")
    save(scored, "scored_ifstruct.jsonl", kind="evals")
    out = ifstruct_score.summarize(scored)
    out["input_sha"] = shas
    log("  ifstruct first-attempt validity %.4f (n=%d)"
        % (out[ifstruct_score.FIRST_ATTEMPT], out["n_items"]))
    return out


def run_ifeval(cfg, runner, prompter, limit):
    path, shas = fetch_ifeval()
    items = ifeval_score.load(path, limit=limit)
    check("ifeval_count", limit or len(items) == 541, "%d items" % len(items))
    prompts = [prompter.render(it["messages"], None) for it in items]
    outs = runner.generate(prompts, max_new_tokens=int(cfg.get("max_new_ifeval", 1024)),
                           batch_size=int(cfg.get("batch_size", 16)), tag="ifeval",
                           ids=[it["id"] for it in items])
    check_completions("ifeval", outs)
    raw, scored = [], []
    for it, p, o in zip(items, prompts, outs):
        ok, detail = ifeval_score.score_item(it, o)
        raw.append({"id": it["id"], "prompt_sha": hashlib.sha256(p.encode()).hexdigest()[:12],
                    "completion": o})
        scored.append({"id": it["id"], "correct": ok, "detail": detail})
    save(raw, "completions_ifeval.jsonl")
    save(scored, "scored_ifeval.jsonl", kind="evals")
    out = ifeval_score.summarize(scored)
    out["input_sha"] = shas
    log("  ifeval prompt-level strict %.4f loose %.4f (n=%d)"
        % (out["prompt_level_strict"], out["prompt_level_loose"], out["n_items"]))
    return out


def run_probes(cfg, runner, prompter, limit):
    graded, shas = fetch_probes(cfg.get("probes_object", "tidepool/s4.4/probes/probes.jsonl"))
    control = probes_score.build_clean()
    csha = probes_score.control_sha(control)
    save(control, "probes_control.jsonl")
    check("probes_graded_count", limit or len(graded) == 434, "%d graded" % len(graded))
    check("probes_control_count", len(control) == 30, "%d control" % len(control))
    # The enlarged clean arm, off by default so a re-run of an already-measured row is
    # byte-comparable to the row it replaces. When it is configured, its items are appended
    # under arm "clean_corpus" and every pre-existing summary key keeps its meaning.
    corpus_control = []
    cc_obj = cfg.get("clean_control_object") or ""
    if cc_obj:
        corpus_control, cc_sha = fetch_probes(cc_obj)
        shas = dict(shas or {})
        shas["clean_control"] = cc_sha if not isinstance(cc_sha, dict) else cc_sha
        check("clean_control_count", len(corpus_control) >= 100,
              "%d corpus control items" % len(corpus_control))
        bad_arm = sorted({it.get("arm") for it in corpus_control} - {"clean_corpus"})
        check("clean_control_arm_tagged", not bad_arm,
              "unexpected arm tags in the corpus control arm: %s" % bad_arm)
        log("  clean control arm: %d items from %s" % (len(corpus_control), cc_obj))
    # A replay rebuilds the clean arm from the same generator rather than reading it back,
    # so it is worth proving the two agree. If the arm has drifted, the false-flag rate
    # from the source run and the one from this pass are measured on different items.
    verify_control = getattr(runner, "verify_control", None)
    if verify_control:
        check("probes_control_matches_source", *verify_control(control))
    if limit:
        # Strided, not the first N: the graded file is ordered by arm, so a head slice
        # would smoke-test one check kind and leave the other two unproven.
        stride = max(1, len(graded) // limit)
        items = graded[::stride][:limit] + control[::max(1, len(control) // limit)][:limit]
        if corpus_control:
            cs = max(1, len(corpus_control) // limit)
            items += corpus_control[::cs][:limit]
    else:
        items = graded + control + corpus_control
    # The probes carry their own system prompt, which is the s4 tool convention with the
    # honesty clause. Nothing is added on top: `tools=None` keeps the harness out of it.
    prompts = [prompter.render(it["messages"], None) for it in items]
    outs = runner.generate(prompts, max_new_tokens=int(cfg.get("max_new_probes", 320)),
                           batch_size=int(cfg.get("batch_size", 16)), tag="probes",
                           ids=[it["id"] for it in items])
    check_completions("probes", outs)
    raw, scored = [], []
    for it, p, o in zip(items, prompts, outs):
        ok, detail = probes_score.score_item(it, o)
        raw.append({"id": it["id"], "arm": it["arm"], "mode": it["mode"],
                    "prompt_sha": hashlib.sha256(p.encode()).hexdigest()[:12],
                    "completion": o})
        scored.append({"id": it["id"], "probe": it["probe"], "arm": it["arm"],
                       "mode": it["mode"], "depth": it["depth"], "correct": ok,
                       "detail": detail})
    save(raw, "completions_probes.jsonl")
    save(scored, "scored_probes.jsonl", kind="evals")
    out = probes_score.summarize(scored)
    out["input_sha"] = shas
    out["control_sha"] = csha
    log("  probes: flag rate %s, false-flag %s (frozen n=%s), false-flag %s (corpus n=%s), "
        "stack idiom %s"
        % (out["flag_rate_malformed"], out["false_flag_rate_clean"], out["n_clean_frozen"],
           out["false_flag_rate_clean_corpus"], out["n_clean_corpus"],
           out["stack_idiom_accuracy"]))
    return out


def run_cdd(cfg, runner, prompter, categories, limit):
    """The black-box memorization check. See `cdd` for the method and the design argument.

    Runs on the same BFCL items, the same style and the same serving path as the reported
    composite, because peakedness on a differently-rendered prompt is peakedness on a different
    distribution. Sampling is the only thing that differs from the scored pass, and it is
    reached through a separate runner entry point so no benchmark path can arrive here.

    llama.cpp is refused rather than quietly substituted. The two arms are compared against each
    other, so both must serve identically; a 4-bit arm has its own sampler and its own
    numerics, and the difference between two arms served differently is not a difference
    between the arms.
    """
    sample_fn = getattr(runner, "sample", None)
    if sample_fn is None:
        raise RuntimeError("the peakedness check needs stochastic sampling and only the "
                           "transformers backend provides it; this run is on %s"
                           % type(runner).__name__)
    style = str(cfg.get("cdd_style", "native_tools"))
    n_samples = int(cfg.get("cdd_samples", 51))
    alpha = float(cfg.get("cdd_alpha", 0.05))
    xi = float(cfg.get("cdd_xi", 0.01))
    temp = float(cfg.get("cdd_temperature", 0.8))
    l_cap = int(cfg.get("cdd_l_cap", 100))
    max_pairs = int(cfg.get("cdd_max_pairs", 60))
    background = int(cfg.get("cdd_background", 0))
    chunk = int(cfg.get("cdd_sample_chunk", 8))
    max_new = int(cfg.get("max_new_bfcl", 320))

    root, shas = fetch_bfcl(categories)
    items = []
    for cat in categories:
        items.extend(bfcl.load_category(root, cat, limit=limit))
    check("cdd_items_present", len(items) > 0, "%d items" % len(items))

    # Occurrence keys, exactly as `s5-compare` and `s6-errors` pair their rows. BFCL v3 ships
    # two different `live_relevance` questions under the id `live_relevance_3-3-0`, so a dict
    # keyed on the bare id silently gives one question the other's prompt -- and sampling 51
    # completions of the wrong prompt is a defect no downstream statistic could detect.
    seen = collections.Counter()
    for it in items:
        seen[it["id"]] += 1
        it["cdd_key"] = "%s#%d" % (it["id"], seen[it["id"]])
    rendered = {}
    for it in items:
        try:
            rendered[it["cdd_key"]] = prompter.render(it["messages"], it["tools"], style=style)
        except Exception as exc:                                   # noqa: BLE001
            NOTES.append("cdd render failed for %s (%s)" % (it["cdd_key"], exc))
    items = [it for it in items if it["cdd_key"] in rendered]

    flagged = cdd.load_id_list(cfg.get("cdd_contaminated_ids", ""), storage)
    if flagged:
        pairs, missing, unmatched = cdd.match_controls(
            items, flagged, lambda it: len(rendered[it["cdd_key"]]), max_pairs,
            key=lambda it: it["cdd_key"])
        selected = [(f, "overlap") for f, _c in pairs] + [(c, "control") for _f, c in pairs]
        # The strata answer the contamination question and nothing else, and they are only as
        # large as the contamination found. The re-check named three scored BFCL items inside a
        # training target, so on this project the interaction runs on six items and is reported
        # as underpowered. The paired base-versus-tuned difference is a separate reading, on how
        # far fine-tuning narrowed the output distribution, and it wants item count rather than
        # matching -- hence a third stratum, drawn deterministically and spread across
        # categories so no single task shape dominates it.
        if background:
            used = {it["cdd_key"] for it, _s in selected}
            per_cat = {}
            for it in sorted(items, key=lambda x: x["cdd_key"]):
                if it["cdd_key"] not in used:
                    per_cat.setdefault(it["category"], []).append(it)
            extra, ring = [], sorted(per_cat)
            while ring and len(extra) < background:
                for cat in list(ring):
                    if not per_cat[cat]:
                        ring.remove(cat)
                        continue
                    extra.append(per_cat[cat].pop(0))
                    if len(extra) >= background:
                        break
            selected += [(it, "background") for it in extra]
        if missing:
            NOTES.append("%d flagged id(s) are not in the scored item set and were dropped: %s"
                         % (len(missing), ", ".join(missing[:5])))
        if unmatched:
            NOTES.append("%d flagged id(s) had no unflagged sibling in their category and were "
                         "dropped: %s" % (len(unmatched), ", ".join(unmatched[:5])))
    else:
        # No stratification available. The paired base-versus-tuned design still works and is
        # still the result; what is lost is the interaction test, which is the half that
        # distinguishes contamination from ordinary entropy collapse. Say so rather than
        # reporting a main effect as if it answered the question.
        pairs, missing, unmatched = [], [], []
        keep = sorted(items, key=lambda it: it["cdd_key"])[:2 * max_pairs]
        selected = [(it, "unstratified") for it in keep]
        NOTES.append("no contaminated id list was given, so the strata are absent and only the "
                     "paired base-versus-tuned difference is reported")

    log("  cdd %d item(s), %d sample(s) each at t=%.2f, style=%s"
        % (len(selected), n_samples, temp, style))
    rows = []
    for n, (it, stratum) in enumerate(selected):
        prompt = rendered[it["cdd_key"]]
        greedy = runner.generate([prompt], max_new_tokens=max_new, batch_size=1,
                                 tag="cdd/greedy")[0]
        samples = runner.sample(prompt, n_samples, temperature=temp,
                                max_new_tokens=max_new, chunk=chunk,
                                seed=int(cfg.get("cdd_seed", 0)), tag="cdd")
        pk = cdd.peakedness(greedy, samples, alpha=alpha, l_cap=l_cap)
        rows.append(dict(pk, id=it["id"], key=it["cdd_key"], category=it["category"],
                         stratum=stratum,
                         leaked=bool(pk["peak"] > xi),
                         prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()[:12],
                         greedy=greedy[:2000]))
        if (n + 1) % 20 == 0:
            log("    cdd %d/%d" % (n + 1, len(selected)))
    save(rows, "cdd_items.jsonl")

    by_stratum = {}
    for r in rows:
        b = by_stratum.setdefault(r["stratum"], {"n": 0, "peak_sum": 0.0, "leaked": 0,
                                                 "exact_sum": 0, "distinct_sum": 0})
        b["n"] += 1
        b["peak_sum"] += r["peak"]
        b["leaked"] += 1 if r["leaked"] else 0
        b["exact_sum"] += r["n_exact"]
        b["distinct_sum"] += r["distinct_samples"]
    for b in by_stratum.values():
        b["avg_peak"] = round(b["peak_sum"] / max(1, b["n"]), 6)
        b["leak_ratio"] = round(b["leaked"] / max(1, b["n"]), 6)
        b["mean_exact_repeats"] = round(b["exact_sum"] / max(1, b["n"]), 4)
        b["mean_distinct_samples"] = round(b["distinct_sum"] / max(1, b["n"]), 4)
        b.pop("peak_sum")
    return {"style": style, "n_samples": n_samples, "temperature": temp, "alpha": alpha,
            "xi": xi, "l_cap": l_cap, "max_new_tokens": max_new,
            "n_items": len(rows), "pairs": len(pairs), "stratified": bool(flagged),
            "by_stratum": by_stratum, "input_sha": shas,
            "avg_peak": round(sum(r["peak"] for r in rows) / max(1, len(rows)), 6),
            "leak_ratio": round(sum(1 for r in rows if r["leaked"]) / max(1, len(rows)), 6)}


def run_calibration(cfg, runner, prompter, tok, limit):
    """A scalar per tool-return probe, from one forward pass, and what it says about the flag.

    The deployed guardrail is a regex hit on free text: one operating point, no score behind
    it, so `s6.2`'s finding that the flag is at ceiling on malformed returns and near zero on
    well-formed-but-wrong ones cannot be read as either "the threshold is high" or "the signal
    is absent". Appending one frozen auditor turn and reading the first token's probability
    gives the missing scalar. See `probes_calib` for what that does and does not measure.

    Only the transformers backend can do this. llama.cpp exposes token probabilities too, but
    every full-precision number in this project comes from one serving path by design, and a
    calibration curve taken on a different one would not be comparable to the flag rate it is
    explaining. A 4-bit arm therefore raises rather than quietly reporting a second instrument.
    """
    probs_fn = getattr(runner, "first_token_probs", None)
    if probs_fn is None:
        raise RuntimeError("the calibration component reads token probabilities and only the "
                           "transformers backend provides them; this run is on %s"
                           % type(runner).__name__)
    graded, shas = fetch_probes(cfg.get("probes_object", "tidepool/s4.4/probes/probes.jsonl"))
    items = [it for it in graded if it.get("probe") == "tool_return"]
    cc_obj = cfg.get("clean_control_object") or ""
    if cc_obj:
        corpus, cc_sha = fetch_probes(cc_obj)
        items += corpus
        shas = dict(shas or {})
        shas["clean_control"] = cc_sha
    # The frozen 30-item synthetic clean arm, scored but NOT counted as a negative. The
    # false-alarm budget belongs to one clean population and the corpus arm is the one drawn
    # from real splits, so pooling the two would report a rate for neither. Carrying the
    # synthetic scores alongside gives the check that matters: if the generator's clean items
    # and the corpus's clean items land in different parts of the range, the negatives the
    # operating points were chosen on are not representative and the table says less than it
    # appears to.
    synthetic = probes_score.build_clean()
    for it in synthetic:
        it.setdefault("id", "%s_%s_%s_d%s"
                      % (it["probe"], it["arm"], it["mode"], it.get("depth")))
    items += synthetic
    shas["probes_control"] = probes_score.control_sha(synthetic)
    # The negatives ARE the measurement here, not a control: an AUROC with no clean arm is
    # undefined and an operating-point table with none is meaningless. Refuse rather than
    # emit a table of Nones that reads like a result.
    check("calibration_has_negatives",
          any(it.get("arm") in probes_calib.NEGATIVE_ARMS for it in items),
          "no %s items: pass clean_control_object" % "/".join(probes_calib.NEGATIVE_ARMS))
    if limit:
        stride = max(1, len(items) // limit)
        items = items[::stride][:limit]
    choices, resolved = probes_calib.choice_ids(tok)
    check("calibration_words_distinct", not resolved["overlapping_token_ids"],
          "yes and no share first token(s) %s, so the ratio cannot be read"
          % resolved["overlapping_token_ids"])
    prompts = [prompter.render(probes_calib.audit_messages(it), None) for it in items]
    got = probs_fn(prompts, choices, batch_size=int(cfg.get("batch_size", 16)),
                   tag="calibration")
    rows = []
    for it, pr in zip(items, got):
        rows.append({"id": it["id"], "arm": it["arm"], "mode": it["mode"],
                     "depth": it.get("depth"),
                     "p_yes": round(float((pr or {}).get("yes", 0.0)), 8),
                     "p_no": round(float((pr or {}).get("no", 0.0)), 8),
                     "score": probes_calib.suspicion(pr or {})})
    save(rows, "calibration_probes.jsonl", kind="evals")
    # The same run's free-form verdicts, if `probes` already ran. Reading them off disk keeps
    # the two components independent -- calibration can run alone -- while still answering
    # "does the scalar rank this model's own flags" whenever both are present.
    free_form = {}
    ff_path = os.path.join(OUT, "scored_probes.jsonl")
    ff_obj = (cfg.get("free_form_scored_object") or "").strip()
    if os.path.exists(ff_path):
        try:
            free_form = read_flags(open(ff_path, encoding="utf-8"))
            out_src = "this run"
        except Exception as exc:                                   # noqa: BLE001
            NOTES.append("could not read this run's free-form flags (%s)" % exc)
            out_src = None
    elif ff_obj:
        # The arm's RECORDED verdicts, from the sweep the reported flag rates came from.
        # Regenerating 602 completions to rank a scalar against them would cost the whole
        # run and answer a slightly different question, because the free-form pass samples
        # and the recorded rates are the ones s6.1 and s6.2 actually published. Joining on
        # id against the stored file ranks the scalar against the published flags.
        local = storage(ff_obj)
        if os.path.isdir(local):
            local = os.path.join(local, os.path.basename(ff_obj))
        with open(local, encoding="utf-8") as fh:
            free_form = read_flags(fh)
        shas[ff_obj] = sha(local)
        out_src = "recorded (%s)" % ff_obj
        check("calibration_free_form_overlaps",
              sum(1 for r in rows if r["id"] in free_form) >= len(rows) // 2,
              "only %d of %d scored ids appear in %s, so the ids do not join"
              % (sum(1 for r in rows if r["id"] in free_form), len(rows), ff_obj))
    else:
        out_src = None
    out = probes_calib.summarize(rows, free_form or None)
    out["input_sha"] = shas
    out["choice_tokens"] = resolved
    out["scored_with_free_form"] = bool(free_form)
    out["free_form_source"] = out_src
    log("  calibration: AUROC %s over %d defective vs %d clean, ECE %s, %s"
        % (out["auroc"], out["n_positive"], out["n_negative"], out["ece"],
           ", ".join("%s@%d%%fa=%s" % (r["detection_rate"], int(100 * r["false_alarm_budget"]),
                                       r["threshold"])
                     for r in out["operating_points"])))
    for mode, m in sorted(out["by_mode"].items()):
        log("    %-20s n=%-3d AUROC %s  mean %s" % (mode, m["n"], m["auroc"], m["mean_score"]))
    return out


# ------------------------------------------------------------------ driver

def main():
    cfg = CFG
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    if PACK_MEMFRAC > 0:
        # The ceiling is what makes a pack safe: an arm that reaches for more than its share
        # gets an OOM inside its own process and dies alone, instead of taking whichever
        # sibling happened to allocate next down with it. It binds the transformers path,
        # which allocates through torch in this process. The 4-bit path serves from a
        # separate llama.cpp process that torch cannot constrain, so a pack carrying one
        # sizes it by its context and slot count instead, and says so in the summary.
        try:
            import torch
            torch.cuda.set_per_process_memory_fraction(PACK_MEMFRAC, 0)
            log("memory ceiling %.3f of the card" % PACK_MEMFRAC)
        except Exception as exc:                                   # noqa: BLE001
            log("could not set a memory ceiling (%s)" % exc)

    base = cfg.get("base_model", "LiquidAI/LFM2.5-1.2B-Instruct")
    adapter_obj = (cfg.get("adapter_object") or "").strip()
    run_tag = cfg.get("run_tag", "s5.2-eval")
    components = [c.strip() for c in str(cfg.get("components", "bfcl,ifstruct,ifeval,probes")).split(",") if c.strip()]
    styles = [s.strip() for s in str(cfg.get("bfcl_styles", "tools_text,native_tools")).split(",") if s.strip()]
    cats = [c.strip() for c in str(cfg.get("bfcl_categories", ",".join(bfcl.DEFAULT_CATEGORIES))).split(",") if c.strip()]
    limit = int(cfg.get("limit_per_component", 0) or 0)
    # Per-component caps, because the components differ in cost by an order of magnitude and a
    # single number cannot express a screening profile. A sweep arm does not need all 2,000
    # structured-output rows to be ranked against its siblings, and it does need every probe
    # item, since the guardrail criteria are pass/fail on the whole set. Unset (-1) inherits
    # `limit_per_component`; 0 means the full set.
    def cap(name):
        v = cfg.get("limit_" + name, -1)
        v = int(v) if v not in ("", None) else -1
        return limit if v < 0 else v
    limits = {c: cap(c) for c in ("bfcl", "ifstruct", "ifeval", "probes",
                                  "calibration", "cdd")}
    profile = str(cfg.get("profile", "") or ("screening" if limit else "full"))
    log("run %s [%s]: model=%s adapter=%s components=%s limits=%s"
        % (run_tag, profile, base, adapter_obj or "-", components, limits))

    # nltk's punkt tokenizer is a runtime download and IFEval's registry needs it.
    if "ifeval" in components:
        try:
            import nltk
            nltk.download("punkt", quiet=True)
        except Exception as exc:                                   # noqa: BLE001
            NOTES.append("nltk punkt download failed (%s); IFEval sentence counts may fail" % exc)

    # Two serving paths, one per precision family, exactly as `plan.md` freezes them: every
    # full-precision row goes through `transformers`, every 4-bit row through llama.cpp. The
    # scoring, prompting and item selection below are the same code either way, so a Q4 row and
    # the FP row it is compared against differ in the backend and the weights and in nothing
    # else. Which backend produced a number is part of the summary.
    backend = str(cfg.get("backend", "hf") or "hf").lower()
    # A third mode that is not a backend: score-only. `rescore_object` points at a storage
    # directory holding a finished run's `completions_*.jsonl`, and every component below
    # reads its text from there instead of generating it. No weights are loaded and no GPU
    # is used; the tokenizer still is, because the prompts are re-rendered so the recorded
    # prompt hashes can be checked. Everything downstream of generation is the same code,
    # which is the only reason a re-scored number may be compared with the one it replaces.
    rescore_obj = (cfg.get("rescore_object") or "").strip()
    server = None
    serving = {}
    adapter_dir = None
    if rescore_obj:
        if adapter_obj:
            raise RuntimeError("a replay scores saved completions: an adapter changes the "
                               "model and cannot apply to text already generated")
        src = storage(rescore_obj)
        if os.path.isfile(src):
            src = os.path.dirname(src)
        log("re-scoring %s from %s" % (run_tag, rescore_obj))
        tok = replay.load_tokenizer(base, log=log)
        n_params = 0
        runner = replay.ReplayRunner(src, log=log)
        serving = {"backend": "replay", "rescore_object": rescore_obj,
                   "generated_here": False,
                   "note": ("completions were generated by an earlier run; only the graders "
                            "and the summary ran here")}
    elif backend == "gguf":
        if adapter_obj:
            raise RuntimeError("a LoRA adapter cannot be merged into a GGUF here: quantize the "
                               "merged checkpoint first and pass it as gguf_repo/gguf_file")
        server, tok, serving = gen_gguf.load_gguf(
            cfg, log=log, out_dir=OUT, storage=storage,
            port=int(cfg.get("gguf_port", 8080)) + PACK_INDEX)
        n_params = 0
        runner = gen_gguf.GgufRunner(server, tok, log=log)
        serving.update({"decoding": "greedy", "temperature": 0.0, "top_k": 1,
                        "cache_prompt": False})
    elif backend == "hf":
        # Two shapes of trained weights arrive here and only one of them is an adapter. A
        # full-parameter arm has no adapter to merge -- it is the model -- so it replaces the
        # base rather than sitting on top of it. Which one it is comes from looking at the
        # directory, not from a parameter, and it is recorded so a run cannot be misread later.
        weights_kind, load_from = None, base
        if adapter_obj:
            wdir = adapters.resolve(storage(adapter_obj), os.path.join(OUT, "weights"))
            weights_kind = adapters.kind(wdir)
            check("weights_loadable", weights_kind is not None,
                  "%s resolved to %s, which holds neither an adapter_config.json nor a model "
                  "config" % (adapter_obj, wdir))
            if weights_kind == adapters.FULL:
                load_from, adapter_dir = wdir, None
                log("%s is a full fine-tuned checkpoint, so it is loaded as the model itself "
                    "and nothing is merged onto it" % adapter_obj)
            else:
                adapter_dir = wdir
                log("adapter for %s resolved to %s" % (adapter_obj, adapter_dir))
        model, tok, n_params = gen.load_model(load_from, adapter_dir, log=log)
        runner = gen.Runner(model, tok, log=log)
        serving = {"backend": "transformers", "dtype": "bfloat16", "decoding": "greedy",
                   "padding_side": "left", "batch_size": int(cfg.get("batch_size", 16)),
                   "weights_kind": weights_kind or "base",
                   "loaded_from": load_from if load_from != base else base}
    else:
        raise RuntimeError("unknown backend %r: expected hf or gguf" % backend)
    if not PACK_CHILD:
        lab.update_progress(10)

    prompter = prompting.Prompter(tok, log=log)
    # Mode is picked on real rows that carry a tool turn, which is what a published
    # template is most likely to reject.
    probe_rows, _ = fetch_probes(cfg.get("probes_object", "tidepool/s4.4/probes/probes.jsonl"))
    samples = [r["messages"] for r in probe_rows if any(m["role"] == "tool" for m in r["messages"])][:6]
    prompter.pick_mode(samples or [[{"role": "user", "content": "hi"}]])
    check("template_native", prompter.mode == "native",
          "template mode is %s: %s" % (prompter.mode, prompter.note))

    results, score = {}, {}
    steps = max(len(components), 1)
    for i, comp in enumerate(components):
        log("component %s" % comp)
        if comp == "bfcl":
            results["bfcl"] = run_bfcl(cfg, runner, prompter, cats, styles, limits["bfcl"])
            score["bfcl_ast_composite"] = round(results["bfcl"]["composite"], 4)
        elif comp == "ifstruct":
            results["ifstruct"] = run_ifstruct(cfg, runner, prompter, limits["ifstruct"])
            score["ifstruct_validity"] = round(results["ifstruct"][ifstruct_score.FIRST_ATTEMPT], 4)
        elif comp == "ifeval":
            results["ifeval"] = run_ifeval(cfg, runner, prompter, limits["ifeval"])
            score["ifeval_prompt_strict"] = round(results["ifeval"]["prompt_level_strict"], 4)
        elif comp == "probes":
            results["probes"] = run_probes(cfg, runner, prompter, limits["probes"])
            p = results["probes"]
            score["probe_flag_rate"] = p["flag_rate_malformed"]
            score["probe_false_flag"] = p["false_flag_rate_clean"]
            if p.get("n_clean_corpus"):
                score["probe_false_flag_corpus"] = p["false_flag_rate_clean_corpus"]
                score["probe_false_flag_all_clean"] = p["false_flag_rate_all_clean"]
            score["probe_stack_idiom"] = p["stack_idiom_accuracy"]
        elif comp == "cdd":
            results["cdd"] = run_cdd(cfg, runner, prompter, cats, limits["cdd"])
            score["cdd_avg_peak"] = results["cdd"]["avg_peak"]
            score["cdd_leak_ratio"] = results["cdd"]["leak_ratio"]
        elif comp == "calibration":
            results["calibration"] = run_calibration(cfg, runner, prompter, tok,
                                                     limits["calibration"])
            c = results["calibration"]
            score["probe_flag_auroc"] = c["auroc"]
            score["probe_flag_ece"] = c["ece"]
        else:
            NOTES.append("unknown component ignored: %s" % comp)
        if not PACK_CHILD:
            lab.update_progress(10 + int(85 * (i + 1) / steps))

    if server is not None:
        # The server's own log carries the load line, the offload count and any slot warning,
        # which is the only place a silent CPU fallback would show up. A 4-bit number taken on
        # the CPU by accident would look fine and cost a day, so the log travels with the run.
        server.stop()
        if not PACK_CHILD:
            try:
                lab.save_artifact(server.log_path)
            except Exception as exc:                               # noqa: BLE001
                NOTES.append("could not save the server log: %r" % exc)

    # Peak GPU memory, so the next eval pack is sized on a measurement. The first packed
    # evaluation was sized by guessing 10 GB an arm on a 24 GB card, which fit two and would
    # have refused a third with no evidence either way. Reserved rather than allocated,
    # because reserved is what the arm's siblings cannot have.
    try:
        import torch as _t
        peak_gb = (round(_t.cuda.max_memory_allocated(0) / 1e9, 2)
                   if _t.cuda.is_available() else None)
        peak_reserved_gb = (round(_t.cuda.max_memory_reserved(0) / 1e9, 2)
                            if _t.cuda.is_available() else None)
        card_total_gb = (round(_t.cuda.get_device_properties(0).total_memory / 1e9, 1)
                         if _t.cuda.is_available() else None)
    except Exception:                                              # noqa: BLE001
        peak_gb = peak_reserved_gb = card_total_gb = None
    if peak_gb is not None:
        log("peak GPU memory %s GB allocated, %s GB reserved, of %s GB"
            % (peak_gb, peak_reserved_gb, card_total_gb))
        score["peak_gpu_gb"] = peak_gb
        score["peak_gpu_reserved_gb"] = peak_reserved_gb
        score["card_total_gb"] = card_total_gb

    summary = {
        "run_tag": run_tag,
        "base_model": base,
        "adapter_object": adapter_obj or None,
        "weights_kind": serving.get("weights_kind"),
        "peak_gpu_gb": peak_gb,
        "peak_gpu_reserved_gb": peak_reserved_gb,
        "card_total_gb": card_total_gb,
        "n_params": n_params,
        "serving": serving,
        "template": {"mode": prompter.mode, "note": prompter.note,
                     "supports_tools_arg": prompter.supports_tools_arg},
        "profile": profile,
        "rescore_object": rescore_obj or None,
        "rescored": bool(rescore_obj),
        "limit_per_component": limit or None,
        "limits": limits,
        "components": components,
        "results": results,
        "throughput": runner.throughput(),
        "assertions": ASSERTS,
        "assertion_failures": sum(1 for a in ASSERTS if not a["ok"]),
        "notes": NOTES,
        "wall_seconds": round(time.time() - t0, 1),
        "config": {k: cfg.get(k) for k in sorted(cfg)},
    }
    save(summary, "eval_summary.json", kind="evals")
    # Both forms of the same object, on purpose. The .json is the one a person reads; the
    # one-line .jsonl is the one that appends cleanly when several runs' summaries are
    # concatenated for a comparison table. An earlier note here blamed the extension for
    # the summary being unretrievable, which was wrong: the cause was `type="evals"`, and
    # it is fixed in save() above.
    save([summary], "eval_summary.jsonl", kind="evals")
    score["assertion_failures"] = summary["assertion_failures"]
    score["generated_tokens"] = summary["throughput"]["generated_tokens"]
    log(json.dumps({k: v for k, v in score.items()}, indent=1, sort_keys=True))
    if not PACK_CHILD:
        lab.update_progress(100)
    tput = summary["throughput"]
    # A replay generated nothing, so reporting its tokens per second would read as a
    # measurement. It reports how many saved completions it scored instead.
    done_note = (("%d completion(s) replayed" % tput.get("replayed_completions", 0))
                 if rescore_obj else ("%s tok/s" % tput["tokens_per_second"]))
    numeric = {k: v for k, v in score.items() if isinstance(v, (int, float))}
    if PACK_CHILD:
        # `score.json` is the file the supervisor reads back for every arm to build the
        # pack's own summary, so the name is part of the contract rather than a convention.
        save(dict(numeric, run_tag=run_tag, arm=ARM, components=components,
                  assertion_failures=summary["assertion_failures"],
                  wall_seconds=summary["wall_seconds"]), "score.json")
        log("%s: %d component(s), %d assertion failure(s), %s"
            % (run_tag, len(components), summary["assertion_failures"], done_note))
        # Non-zero on an assertion failure, because a packed child has no lab.error: its
        # exit code is the only channel the supervisor reads it on.
        raise SystemExit(3 if summary["assertion_failures"] else 0)
    lab.finish(message=("%s: %d component(s), %d assertion failure(s), %s"
                        % (run_tag, len(components), summary["assertion_failures"],
                           done_note)),
               score=numeric)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:                                       # noqa: BLE001
        import traceback
        traceback.print_exc()
        if not PACK_CHILD:
            try:
                lab.error(message="s5-eval failed: %s" % exc)
            except Exception:                                      # noqa: BLE001
                pass
        raise
