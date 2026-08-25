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

import hashlib
import json
import os
import time
import urllib.request

from lab import lab

import bfcl
import gen
import gen_gguf
import ifeval_score
import ifstruct_score
import probes_score
import prompting
import replay

IFSTRUCT_URL = "https://raw.githubusercontent.com/Liquid4All/ifstruct/main/data/test.jsonl"
BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
IFEVAL_REPO = "google/IFEval"
IFEVAL_FILE = "ifeval_input_data.jsonl"

OUT = "out"
NOTES = []
ASSERTS = []


def log(msg):
    print(msg, flush=True)
    try:
        lab.log(str(msg))
    except Exception:                                              # noqa: BLE001
        pass


def check(name, ok, detail=""):
    ASSERTS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:300]})
    if not ok:
        log("ASSERTION FAILED %s: %s" % (name, detail))
    return bool(ok)


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
    root = "data/bfcl"
    os.makedirs(root + "/possible_answer", exist_ok=True)
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
    os.makedirs("data", exist_ok=True)
    dst = "data/ifstruct_test.jsonl"
    with urllib.request.urlopen(IFSTRUCT_URL, timeout=120) as r, open(dst, "wb") as fh:
        fh.write(r.read())
    return dst, {"ifstruct/data/test.jsonl": sha(dst)}


def fetch_ifeval():
    from huggingface_hub import hf_hub_download
    src = hf_hub_download(repo_id=IFEVAL_REPO, filename=IFEVAL_FILE, repo_type="dataset")
    return src, {IFEVAL_FILE: sha(src)}


def fetch_probes(obj):
    local = lab.storage_download(obj)
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
    else:
        items = graded + control
    # The probes carry their own system prompt, which is the s4 tool convention with the
    # honesty clause. Nothing is added on top: `tools=None` keeps the harness out of it.
    prompts = [prompter.render(it["messages"], None) for it in items]
    outs = runner.generate(prompts, max_new_tokens=int(cfg.get("max_new_probes", 320)),
                           batch_size=int(cfg.get("batch_size", 16)), tag="probes",
                           ids=[it["id"] for it in items])
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
    log("  probes: flag rate %s, false-flag %s, stack idiom %s"
        % (out["flag_rate_malformed"], out["false_flag_rate_clean"],
           out["stack_idiom_accuracy"]))
    return out


# ------------------------------------------------------------------ driver

def main():
    lab.init()
    cfg = lab.get_config() or {}
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)

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
    limits = {c: cap(c) for c in ("bfcl", "ifstruct", "ifeval", "probes")}
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
        src = lab.storage_download(rescore_obj)
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
        server, tok, serving = gen_gguf.load_gguf(cfg, log=log)
        n_params = 0
        runner = gen_gguf.GgufRunner(server, tok, log=log)
        serving.update({"decoding": "greedy", "temperature": 0.0, "top_k": 1,
                        "cache_prompt": False})
    elif backend == "hf":
        if adapter_obj:
            adapter_dir = lab.storage_download(adapter_obj)
            log("adapter downloaded to %s" % adapter_dir)
        model, tok, n_params = gen.load_model(base, adapter_dir, log=log)
        runner = gen.Runner(model, tok, log=log)
        serving = {"backend": "transformers", "dtype": "bfloat16", "decoding": "greedy",
                   "padding_side": "left", "batch_size": int(cfg.get("batch_size", 16))}
    else:
        raise RuntimeError("unknown backend %r: expected hf or gguf" % backend)
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
            score["probe_stack_idiom"] = p["stack_idiom_accuracy"]
        else:
            NOTES.append("unknown component ignored: %s" % comp)
        lab.update_progress(10 + int(85 * (i + 1) / steps))

    if server is not None:
        # The server's own log carries the load line, the offload count and any slot warning,
        # which is the only place a silent CPU fallback would show up. A 4-bit number taken on
        # the CPU by accident would look fine and cost a day, so the log travels with the run.
        server.stop()
        try:
            lab.save_artifact(server.log_path)
        except Exception as exc:                                   # noqa: BLE001
            NOTES.append("could not save the server log: %r" % exc)

    summary = {
        "run_tag": run_tag,
        "base_model": base,
        "adapter_object": adapter_obj or None,
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
    lab.update_progress(100)
    tput = summary["throughput"]
    # A replay generated nothing, so reporting its tokens per second would read as a
    # measurement. It reports how many saved completions it scored instead.
    done_note = (("%d completion(s) replayed" % tput.get("replayed_completions", 0))
                 if rescore_obj else ("%s tok/s" % tput["tokens_per_second"]))
    lab.finish(message=("%s: %d component(s), %d assertion failure(s), %s"
                        % (run_tag, len(components), summary["assertion_failures"],
                           done_note)),
               score={k: v for k, v in score.items() if isinstance(v, (int, float))})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:                                       # noqa: BLE001
        import traceback
        traceback.print_exc()
        try:
            lab.error(message="s5-eval failed: %s" % exc)
        except Exception:                                          # noqa: BLE001
            pass
        raise
