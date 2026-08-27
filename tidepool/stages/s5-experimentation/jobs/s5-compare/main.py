"""s5.4's evidence: every sweep arm against the reference, item by item, with intervals.

The sweep was designed to choose a direction on validation loss and cannot. Three of its
four finished arms sit within 0.0006 of each other at one seed apiece, so the direction has
to come from the task scores. Reading those as a table of rates would repeat the same
mistake at a larger scale: four components, eight arms, and effects small enough that the
ranking could easily be sampling noise wearing the clothes of a finding.

So this pass does three things a table of rates cannot. It joins arms item by item, which
takes item difficulty out of the comparison and is the reason a two-point difference over a
few thousand shared items can be resolved at all. It reports an interval and an exact
paired test for every arm-by-component cell, so a cell that cannot be distinguished from
the reference says so in its own row. And it corrects across the whole family of
comparisons, because twenty-eight cells against one reference will turn up a nominally
significant result by chance about once even when every arm is identical.

It runs on CPU and draws nothing from the GPU allowance. Its input is each arm's
`scored_*.jsonl`, which is where the per-item `correct` verdict lives; the sibling
`completions_*.jsonl` holds the raw generations and carries no verdict at all. Output is an
artifact, so the s5.4 decision cites a job.

One honest limit, stated here because it is easy to miss when reading the table. The paired
difference is item-weighted. BFCL's headline in `score.json` is a macro average over
eleven categories, which weights a 40-item category like a 400-item one and therefore has
no per-item pairing to test. Where a component names a grouping field, the macro-averaged
paired difference is reported beside the item-weighted one as a descriptive figure with no
interval and no test attached, so the two are never mistaken for each other.
"""

import hashlib
import json
import os

import stats

from lab import lab

lab.init()
CFG = lab.get_config() or {}


def C(k, default):
    v = CFG.get(k)
    return default if v is None or v == "" else v


def reset():
    """Re-read the config and clear the run's accumulators.

    The config is read at import as well, which is the idiom every other task here uses.
    Reading it again at the top of the pass is what lets the pass be exercised more than
    once in one process, and clearing the note and failure lists with it is the part that
    matters: carried over, they would report a previous run's missing file as this one's.
    """
    global CFG
    CFG = lab.get_config() or {}
    del NOTES[:]
    del FAILURES[:]
    ID_SHA.clear()


OUT = "out"
NOTES = []
FAILURES = []
# (arm, component) -> sha1 of the id sequence the component selected, in file order. Only
# consulted when a component has repeated ids, where it is what makes occurrence pairing
# sound; see `read_items`.
ID_SHA = {}

# Component name -> where its per-item verdicts live, which rows count, what to cross-check
# the recomputed rate against, and (optionally) the field to macro-average over.
DEFAULT_COMPONENTS = {
    "bfcl_native": {
        "file": "scored_bfcl_native_tools.jsonl",
        "check": "eval_summary.results.bfcl.styles.native_tools.item_weighted",
        "group_field": "category",
    },
    "bfcl_text": {
        "file": "scored_bfcl_tools_text.jsonl",
        "check": "eval_summary.results.bfcl.styles.tools_text.item_weighted",
        "group_field": "category",
    },
    "ifstruct": {"file": "scored_ifstruct.jsonl"},
    "ifeval": {"file": "scored_ifeval.jsonl"},
    # The probe file carries four populations that answer different questions, and averaging
    # them would hide the trade the guardrail actually makes: an arm can raise its detection
    # rate purely by flagging more often, which shows up as a loss on the clean arms and
    # nowhere else. They are split so a direction cannot be chosen on one half of it.
    "probes_detect": {
        "file": "scored_probes.jsonl",
        "where": {"probe": ["tool_return"], "arm": ["corrupted", "contradicted"]},
    },
    "probes_clean": {
        "file": "scored_probes.jsonl",
        "where": {"probe": ["tool_return"], "arm": ["clean"]},
    },
    "probes_clean_corpus": {
        "file": "scored_probes.jsonl",
        "where": {"probe": ["tool_return"], "arm": ["clean_corpus"]},
    },
    "probes_stack_idiom": {
        "file": "scored_probes.jsonl",
        "where": {"probe": ["stack_idiom"]},
    },
    # The two cells above read `correct`, which on a probe row means "did not fabricate a
    # value" on the malformed arms and "did not flag AND used the value" on the clean ones.
    # Useful, and not what the s5.4 reliability gate is written on: that gate is a detection
    # rate and a false-alarm rate, both of them `detail.flagged`. The gate is a threshold on
    # rates the graders already computed, but whether one arm genuinely detects more than
    # the reference is a paired question like every other, so it gets a cell rather than a
    # table row. Adding them enlarges the Holm family and therefore makes every claim in
    # this run harder to make, which is the direction an amendment made after the rates were
    # known should push.
    "probes_flag_detect": {
        "file": "scored_probes.jsonl",
        "where": {"probe": ["tool_return"], "arm": ["corrupted", "contradicted"]},
        "verdict": "detail.flagged",
    },
    # Inverted: on a clean return a raised flag is a false alarm, so not-flagged is the
    # success. Without the inversion the sign of every clean-arm delta would be backwards.
    "probes_flag_clean_corpus": {
        "file": "scored_probes.jsonl",
        "where": {"probe": ["tool_return"], "arm": ["clean_corpus"]},
        "verdict": "detail.flagged",
        "invert": True,
    },
}


def log(msg):
    print(str(msg), flush=True)
    try:
        lab.log(str(msg))
    except Exception:                                              # noqa: BLE001
        pass


def check(name, cond, detail=""):
    """Record an assertion. A failure is reported and does not stop the pass.

    The comparison is worth having when one arm's file is missing or one rate does not
    reconcile; what is not worth having is a comparison that hides either. Every failure is
    named in the summary and in the job's completion message.
    """
    if not cond:
        FAILURES.append("%s: %s" % (name, detail))
        log("ASSERTION FAILED %s: %s" % (name, detail))
    return bool(cond)


def save(obj, name):
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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


def dotted(obj, path):
    """`a.b.c` out of nested dicts, or None if any step is missing."""
    cur = obj
    for part in str(path).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# --------------------------------------------------------------------------- inputs

def read_items(path, where=None, group_field=None, verdict="correct", invert=False):
    """id -> (verdict, group) for the rows a component selects.

    `correct` is the per-item verdict every grader writes: a matched call for BFCL, a valid
    parse for structured output, a satisfied instruction for IFEval, the detector's verdict
    for probes. A row with no verdict field is counted and skipped rather than coerced,
    since a missing verdict is not a wrong answer.

    A repeated id is kept, as `<id>#2`, `<id>#3`, and reported. BFCL v3 ships one: two
    different `live_relevance` questions both carry the id `live_relevance_3-3-0`, and the
    model answers them differently, so keying on the bare id would drop a real item from
    every arm and leave the paired denominator one short of the rate each arm reports. The
    kth occurrence in one arm is paired with the kth in another, which is exact as long as
    both arms walked the corpus in the same order; the sha of the id sequence is returned
    so the caller can assert that rather than assume it.

    `verdict` is a dotted path, so a component can pair on a field the grader wrote inside
    `detail` instead of on `correct`. That is not a convenience: on the probe file `correct`
    answers "did the model avoid inventing a value", and the guardrail bar this sweep is
    judged against is written on `detail.flagged`, "did the model say the return was
    broken". They are different questions and they disagree by twenty points or more, so a
    comparison that only read `correct` would leave the bar itself untested.

    `invert` makes a true verdict the failure, for the clean populations where raising a
    flag is the error. It is applied after the lookup so a missing field is still a missing
    verdict rather than silently becoming a pass.
    """
    out, no_verdict, repeats, filtered = {}, 0, 0, 0
    seen, order = {}, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:                                      # noqa: BLE001
                no_verdict += 1
                continue
            if not isinstance(row, dict) or "id" not in row:
                no_verdict += 1
                continue
            v = row.get("correct") if verdict == "correct" else dotted(row, verdict)
            if v is None:
                no_verdict += 1
                continue
            if where and not all(row.get(k) in vals for k, vals in where.items()):
                filtered += 1
                continue
            rid = str(row["id"])
            seen[rid] = seen.get(rid, 0) + 1
            key = rid if seen[rid] == 1 else "%s#%d" % (rid, seen[rid])
            if seen[rid] > 1:
                repeats += 1
            order.append(rid)
            out[key] = (bool(v) != bool(invert),
                        row.get(group_field) if group_field else None)
    return out, {"no_verdict": no_verdict, "repeats": repeats, "filtered_out": filtered,
                 "id_sha": hashlib.sha1("\n".join(order).encode("utf-8")).hexdigest()}


def load_arm(arm, prefix, components):
    """Every component's per-item verdicts for one arm, plus the arm's own score.json."""
    data, meta = {}, {}
    for comp, spec in components.items():
        obj = "%s/%s/%s" % (prefix.rstrip("/"), arm, spec["file"])
        try:
            path = lab.storage_download(obj)
        except Exception as exc:                                   # noqa: BLE001
            NOTES.append("%s has no %s in storage (%s)" % (arm, comp, exc))
            log("%s: %s missing (%s)" % (arm, comp, obj))
            continue
        items, counts = read_items(path, spec.get("where"), spec.get("group_field"),
                                   spec.get("verdict", "correct"),
                                   bool(spec.get("invert", False)))
        if counts["no_verdict"]:
            NOTES.append("%s/%s: %d row(s) carried no verdict"
                         % (arm, comp, counts["no_verdict"]))
        ID_SHA[(arm, comp)] = counts["id_sha"]
        if counts["repeats"]:
            NOTES.append("%s/%s: %d repeated id(s), kept as <id>#k and paired by occurrence"
                         % (arm, comp, counts["repeats"]))
        if not items:
            NOTES.append("%s/%s selected no rows out of %s"
                         % (arm, comp, spec["file"]))
            continue
        data[comp] = items
        log("%s: %-20s %5d items, %.4f correct"
            % (arm, comp, len(items),
               sum(1 for v, _g in items.values() if v) / len(items)))
    # Both summaries, because they carry different things. `score.json` is the flat card the
    # arm queue writes; `eval_summary.json` is the harness's own nested output and is the only
    # place a per-style item-weighted BFCL rate exists. A component's `check` path is resolved
    # against this merged view, so it has to name which one it means.
    for name in ("score.json", "eval_summary.json"):
        try:
            with open(lab.storage_download("%s/%s/%s" % (prefix.rstrip("/"), arm, name)),
                      encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception:                                          # noqa: BLE001
            NOTES.append("%s has no %s in storage, so any rate it carries is not "
                         "cross-checked" % (arm, name))
            continue
        if name == "score.json":
            meta.update(doc if isinstance(doc, dict) else {})
        else:
            meta["eval_summary"] = doc
    return data, meta


def macro_delta(ref, cand):
    """Mean over groups of the per-group paired difference. Descriptive, no test.

    Every group present on both sides gets equal weight, which is what BFCL's own composite
    does across its eleven categories. A group that appears on one side only is dropped and
    reported, because an unmatched group is not a difference.
    """
    groups, dropped = {}, 0
    for item, (v, g) in ref.items():
        if item not in cand:
            continue
        if g is None:
            dropped += 1
            continue
        b = groups.setdefault(g, [0, 0, 0])          # n, ref correct, cand correct
        b[0] += 1
        b[1] += 1 if v else 0
        b[2] += 1 if cand[item][0] else 0
    if not groups:
        return None, 0, dropped
    per = [(b[2] - b[1]) / b[0] for b in groups.values()]
    return sum(per) / len(per), len(groups), dropped


# --------------------------------------------------------------------------- the pass

def parse_components(raw):
    """Config value to the component table. A bare filename is shorthand for {"file": ...}."""
    if not raw:
        return dict(DEFAULT_COMPONENTS)
    spec = json.loads(raw) if isinstance(raw, str) else dict(raw)
    out = {}
    for name, val in spec.items():
        out[name] = {"file": val} if isinstance(val, str) else dict(val)
        if "file" not in out[name]:
            raise SystemExit("component %r has no file" % name)
    return out


def main():
    reset()
    arms = [a.strip() for a in str(C("arms", "")).split(",") if a.strip()]
    ref_arm = str(C("reference", ""))
    prefix = str(C("completions_prefix", "tidepool/s5.3/arms"))
    resamples = int(C("resamples", 10000))
    seed = int(C("seed", 20260826))
    components = parse_components(C("components", ""))
    if len(arms) < 2:
        raise SystemExit("a paired comparison needs at least two arms, got %r" % arms)
    if ref_arm not in arms:
        raise SystemExit("reference %r is not one of the arms %r: the reference has to be "
                         "an arm being compared" % (ref_arm, arms))

    log("comparing %d arm(s) against %s over %d component(s), %d resamples, seed %d"
        % (len(arms) - 1, ref_arm, len(components), resamples, seed))

    loaded = {arm: load_arm(arm, prefix, components) for arm in arms}
    ref_data, _ref_meta = loaded[ref_arm]

    # Pairing the kth occurrence of a repeated id against the kth in another arm is exact only
    # if both arms walked the corpus in the same order. Every arm here was scored by the same
    # harness over the same files, so this should hold everywhere and is asserted rather than
    # assumed: it is also a free check that no arm was scored over a different corpus.
    #
    # An arm that legitimately has no rows for a component is skipped. B1 is scored without a
    # `clean_control_object`, so its corpus components select nothing and hash to the sha of the
    # empty sequence; the cell is already dropped for that arm with a note, and asserting order
    # against an empty selection only manufactures a failure out of an absence.
    for (arm, comp), sha in sorted(ID_SHA.items()):
        if arm == ref_arm or (ref_arm, comp) not in ID_SHA:
            continue
        if not loaded[arm][0].get(comp) or not ref_data.get(comp):
            continue
        check("%s and %s read %s in the same item order" % (arm, ref_arm, comp),
              sha == ID_SHA[(ref_arm, comp)],
              "%s reads %s, %s reads %s"
              % (arm, sha[:12], ref_arm, ID_SHA[(ref_arm, comp)][:12]))

    # ---- paired comparison, one cell per arm and component
    cells, pvals = [], []
    for arm in arms:
        if arm == ref_arm:
            continue
        data, _meta = loaded[arm]
        for comp in components:
            if comp not in ref_data or comp not in data:
                NOTES.append("%s against %s on %s skipped: one side is missing"
                             % (arm, ref_arm, comp))
                continue
            r = {k: v for k, (v, _g) in ref_data[comp].items()}
            c = {k: v for k, (v, _g) in data[comp].items()}
            minus, zero, plus, n = stats.paired_counts(r, c)
            check("%s and %s cover the same %s items" % (arm, ref_arm, comp),
                  n == len(r) == len(c),
                  "reference %d, arm %d, matched %d" % (len(r), len(c), n))
            if n == 0:
                NOTES.append("%s against %s on %s: no shared item ids"
                             % (arm, ref_arm, comp))
                continue
            point, lo, hi = stats.bootstrap_ci(minus, zero, plus, n,
                                               resamples=resamples,
                                               seed=seed + len(cells))
            p = stats.mcnemar_exact(minus, plus)
            shared = [k for k in r if k in c]
            cell = {
                "arm": arm, "component": comp, "n_matched": n,
                "reference_rate": round(sum(1 for k in shared if r[k]) / n, 6),
                "arm_rate": round(sum(1 for k in shared if c[k]) / n, 6),
                "delta": round(point, 6),
                "ci95_low": round(lo, 6), "ci95_high": round(hi, 6),
                "only_reference_correct": minus, "only_arm_correct": plus,
                "agree": zero,
                "p_mcnemar_exact": p,
                "separates_by_interval": bool(lo > 0.0 or hi < 0.0),
            }
            if components[comp].get("group_field"):
                md, n_groups, dropped = macro_delta(ref_data[comp], data[comp])
                cell["macro_delta_descriptive"] = None if md is None else round(md, 6)
                cell["macro_groups"] = n_groups
                if dropped:
                    NOTES.append("%s/%s: %d item(s) had no %s and are outside the macro "
                                 "figure" % (arm, comp, dropped,
                                             components[comp]["group_field"]))
            cells.append(cell)
            pvals.append(p)

    for cell, adj in zip(cells, stats.holm(pvals)):
        cell["p_holm"] = adj
        # Two readings of the same cell, kept apart on purpose. An interval that excludes
        # zero and a family-corrected p below 0.05 are different claims, and a cell that
        # passes one and fails the other is exactly the cell to be careful about.
        cell["significant_family_corrected"] = bool(adj < 0.05)

    # ---- cross-check the join against each arm's own reported rate
    for arm in arms:
        data, meta = loaded[arm]
        for comp, items in data.items():
            path = components[comp].get("check")
            if not path or not isinstance(meta, dict):
                continue
            reported = dotted(meta, path)
            if not isinstance(reported, (int, float)):
                NOTES.append("%s/%s: no %s in either summary, so the rate is not "
                             "cross-checked" % (arm, comp, path))
                continue
            mine = sum(1 for v, _g in items.values() if v) / len(items)
            check("%s/%s recomputed rate matches the arm's own" % (arm, comp),
                  abs(mine - float(reported)) < 1e-3,
                  "recomputed %.6f, %s reports %.6f" % (mine, path, float(reported)))

    summary = {
        "reference": ref_arm,
        "arms": arms,
        "components": components,
        "completions_prefix": prefix,
        "resamples": resamples,
        "seed": seed,
        "cells": cells,
        "n_comparisons": len(cells),
        "n_separating_by_interval": sum(1 for c in cells if c["separates_by_interval"]),
        "n_significant_family_corrected": sum(
            1 for c in cells if c.get("significant_family_corrected")),
        "notes": NOTES,
        "assertion_failures": FAILURES,
        "method": ("Paired per-item comparison against the reference arm. The interval is a "
                   "percentile bootstrap over the three paired-difference counts, which is "
                   "exact for a statistic that is a function of those counts alone. The test "
                   "is McNemar's exact two-sided test on the discordant pairs. The family "
                   "correction is Holm-Bonferroni across every arm-by-component cell in this "
                   "run. Deltas are item-weighted; the macro column is descriptive and "
                   "carries no test."),
    }
    save(summary, "comparison.json")
    save(render(summary), "comparison.md")

    msg = ("%d comparison(s), %d arm(s) against %s: %d separate by interval, %d survive "
           "family correction, %d assertion failure(s)"
           % (len(cells), len(arms) - 1, ref_arm, summary["n_separating_by_interval"],
              summary["n_significant_family_corrected"], len(FAILURES)))
    log(msg)
    try:
        lab.finish(message=msg, score=summary["n_significant_family_corrected"])
    except Exception:                                              # noqa: BLE001
        pass
    return summary


def render(s):
    """The same numbers the JSON carries, as a table a human reads."""
    out = ["# s5.3 arms against %s, paired by item" % s["reference"], "",
           s["method"], "",
           "Bootstrap resamples %d, seed %d, %d comparison(s) in the corrected family."
           % (s["resamples"], s["seed"], s["n_comparisons"]), "",
           "| arm | component | n | ref | arm | delta | 95% CI | macro | McNemar p | Holm p |"
           " reads as |",
           "|---|---|--:|--:|--:|--:|---|--:|--:|--:|---|"]
    for c in s["cells"]:
        if c.get("significant_family_corrected"):
            reads = "separates"
        elif c["separates_by_interval"]:
            reads = "separates before correction only"
        else:
            reads = "indistinguishable"
        macro = c.get("macro_delta_descriptive")
        out.append("| %s | %s | %d | %.4f | %.4f | %+.4f | [%+.4f, %+.4f] | %s | %.4g | %.4g "
                   "| %s |"
                   % (c["arm"], c["component"], c["n_matched"], c["reference_rate"],
                      c["arm_rate"], c["delta"], c["ci95_low"], c["ci95_high"],
                      ("%+.4f" % macro) if macro is not None else "",
                      c["p_mcnemar_exact"], c.get("p_holm", float("nan")), reads))
    out += ["",
            "`delta` is the arm's rate minus the reference's over the items both scored. "
            "`macro` averages the per-group deltas with equal weight per group and has no "
            "interval, so it is a description of the same items and not a second test."]
    if s["notes"]:
        out += ["", "## Notes", ""] + ["- %s" % n for n in s["notes"]]
    if s["assertion_failures"]:
        out += ["", "## Assertion failures", ""] + ["- %s" % f for f in s["assertion_failures"]]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
