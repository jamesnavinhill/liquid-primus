"""s6.4 — reading the failures behind three headline numbers, item by item.

`s6.1` and `s6.2` established what moved and where. Three of those movements are still
descriptions rather than explanations, and each one changes what the paper is allowed to
claim depending on which explanation is true.

**The grader edit.** `s6.1` found `B1` and `B3` both shifting by an identical -0.0051 of
native tool-calling composite when re-scored by the current graders, and recorded the
reading that fit -- the same small set of items reclassified in both -- as a reading, not
a finding. Two rows moving by the same amount is either that, or a bug that scales with
nothing. Naming the items settles it, and the arithmetic is checkable: the composite is a
macro average over eleven categories, so a single item flipping in a category of size n
moves it by exactly 1/(11n) and by nothing else.

**The structured-output regression.** `s6.2` localized it entirely to JSON: -0.0950 for
R3-F16 against -0.0040 on YAML. A model that has become worse at JSON *schemas* and a model
that has become worse at closing a fenced code block produce the same cell and license
completely different sentences in a paper. The grader already wrote the errors it raised
per item; joining them to the completion and separating envelope failures from schema
failures answers it without generating anything.

**The native tool-calling loss.** `s6.2` split the flat cell into -0.086 where a call is
warranted and +0.159 where abstaining is. The losing half is what a deployment feels, and
the grader's own `reason` string already says whether the arm called the wrong function,
called with the wrong arguments, or did not call at all -- three failures with three
different fixes.

Nothing here generates and nothing here loads a model. Every input is a file some earlier
job already wrote: the verdicts, and the completions they were computed from. It is a job
rather than a local script for the usual reason -- a number in a report cites a job id.
"""

import hashlib
import json
import os
import re

from lab import lab

lab.init()
CFG = lab.get_config() or {}

OUT = "out"
NOTES = []
FAILURES = []

# The eight BFCL categories where a call is the right answer, and the three where abstaining
# is. Fixed here rather than passed in, because which half a category belongs to is a
# property of the benchmark and not of a run: `s6.2` drew the same line and any number that
# moves when this list moves is a number about this list.
CALL_WARRANTED = ("simple", "multiple", "parallel", "parallel_multiple",
                  "live_simple", "live_multiple", "live_parallel",
                  "live_parallel_multiple")
ABSTAIN_WARRANTED = ("irrelevance", "live_irrelevance", "live_relevance")

# The two IFStruct grader messages that are about the wrapper around the answer rather than
# the answer. Everything else the validator raises is about the parsed document: a missing
# required field, a wrong type, an extra key. The split is the whole point of the component,
# so the membership is spelled out rather than pattern-matched loosely.
ENVELOPE_ERRORS = ("Response must use a code block but none was found",
                   "Unclosed code block")

FENCE_OPEN = re.compile(r"^[ \t]*```([A-Za-z0-9_+-]*)[ \t]*$", re.M)


def log(msg):
    print(msg, flush=True)
    try:
        lab.log(msg)
    except Exception:                                              # noqa: BLE001
        pass


def C(k, default):
    v = CFG.get(k)
    return default if v is None or v == "" else v


def jparam(k):
    """A config value that is a JSON object, accepted as either a dict or its text."""
    raw = C(k, "")
    if not raw:
        return {}
    val = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if not isinstance(val, dict):
        raise SystemExit("%s must be a JSON object, got %r" % (k, val))
    return val


def check(name, cond, detail=""):
    """Record an assertion. A failure is reported and does not stop the pass."""
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


def fetch(obj):
    """A storage object as a local path. A directory means the object landed inside one."""
    local = lab.storage_download(obj)
    if os.path.isdir(local):
        local = os.path.join(local, os.path.basename(obj))
    return local


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def by_id(rows):
    """key -> row, where a repeated id becomes `<id>#2`, `<id>#3`, and so on.

    BFCL v3 ships exactly one repeat: two different `live_relevance` questions both carry
    the id `live_relevance_3-3-0`, the model answers them differently, and the graders score
    them differently. The comparison job at `s5.3` already pairs by occurrence for that
    reason, and the first pass of THIS job did not: it kept the first of each repeat, which
    dropped the second occurrence from every join. That is not a cosmetic loss. The one item
    the grader edit reclassified is the second occurrence, so an id-keyed join reported zero
    flips against a composite that had visibly moved, and the arithmetic check caught it.

    The kth occurrence in one file is paired with the kth in another, which is exact as long
    as both files walked the corpus in the same order. `order_sha` is returned so the caller
    can assert that rather than assume it.
    """
    out, repeats, order = {}, 0, []
    seen = {}
    for r in rows:
        rid = str(r.get("id"))
        seen[rid] = seen.get(rid, 0) + 1
        key = rid if seen[rid] == 1 else "%s#%d" % (rid, seen[rid])
        if seen[rid] > 1:
            repeats += 1
        order.append(rid)
        out[key] = r
    out_sha = hashlib.sha1("\n".join(order).encode("utf-8")).hexdigest()
    return out, repeats, out_sha


# ---------------------------------------------------------------- grader delta


def macro(rows, group_field="category", verdict="correct"):
    """The BFCL composite as `score.json` computes it: equal weight per category."""
    groups = {}
    for r in rows:
        groups.setdefault(r.get(group_field), []).append(bool(r.get(verdict)))
    per = {g: sum(v) / len(v) for g, v in groups.items() if v}
    return (sum(per.values()) / len(per) if per else 0.0), per, {g: len(v)
                                                                for g, v in groups.items()}


def grader_delta(pre_edit, post_prefix_for, file_name, recorded):
    """Which items the grader edit reclassified, and whether they explain the shift.

    Two things are computed and they are not the same claim. Where an arm has BOTH its
    pre-edit and post-edit verdicts, the flip set is observed directly. Where it has only
    the post-edit file and a composite recorded before the edit -- which is `B1`, whose
    per-item files were lost to the typed-save bug -- the flip set is a PREDICTION: take
    the items that flipped in an arm that has both, assume they flipped the same way here,
    and check the composite that implies against the one actually recorded. A prediction
    that lands to four decimal places over a quantity nobody tuned is evidence; it is
    reported as a separate field from the observed one either way.
    """
    out = {"observed": {}, "predicted": {}, "flips": []}
    seen_flips = []
    for arm, obj in sorted(pre_edit.items()):
        try:
            old = read_jsonl(fetch(obj))
        except Exception as exc:                                   # noqa: BLE001
            NOTES.append("%s has no pre-edit file at %s (%s)" % (arm, obj, exc))
            continue
        new = read_jsonl(fetch("%s/%s/%s" % (post_prefix_for(arm).rstrip("/"),
                                             arm, file_name)))
        o, o_rep, _sha_o = by_id(old)
        n, n_rep, _sha_n = by_id(new)
        shared = sorted(set(o) & set(n))
        check("grader_delta_%s_joins" % arm, len(shared) >= 0.99 * min(len(o), len(n)),
              "only %d of %d/%d ids are shared, so the two passes are not the same items"
              % (len(shared), len(o), len(n)))
        # Occurrence pairing is only exact if both passes walked the corpus in the same
        # order. The sha of the id sequence says so rather than leaving it assumed, and a
        # mismatch would make every `<id>#k` key on a repeated id meaningless.
        check("grader_delta_%s_order" % arm, _sha_o == _sha_n,
              "the pre-edit and post-edit files list their ids in different orders, so "
              "pairing the kth occurrence with the kth is not exact")
        flips = []
        for rid in shared:
            if bool(o[rid].get("correct")) != bool(n[rid].get("correct")):
                # `calls` and `n_calls` travel with the flip because the interesting
                # question is not that a verdict moved but WHY. A flip where both passes
                # parsed the same call is a scoring-rule change; a flip where one pass
                # parsed a call and the other parsed none is a change in what counts as a
                # call at all, which is a much larger edit wearing a one-item delta.
                flips.append({"arm": arm, "id": rid,
                              "category": n[rid].get("category"),
                              "was": bool(o[rid].get("correct")),
                              "now": bool(n[rid].get("correct")),
                              "old_reason": o[rid].get("reason") or "",
                              "new_reason": n[rid].get("reason") or "",
                              "old_n_calls": o[rid].get("n_calls"),
                              "new_n_calls": n[rid].get("n_calls"),
                              "old_calls": json.dumps(o[rid].get("calls"))[:600],
                              "new_calls": json.dumps(n[rid].get("calls"))[:600],
                              "parse_changed": (o[rid].get("n_calls")
                                                != n[rid].get("n_calls"))})
        old_macro, _op, _oc = macro(old)
        new_macro, _np, sizes = macro(new)
        # One flip in a category of size n moves an eleven-category macro by 1/(11n). The
        # per-flip contribution is signed and summed, so several flips in one category and
        # flips in different categories are both handled by the same arithmetic.
        k = len(sizes)
        predicted = sum((1.0 if f["now"] else -1.0) / (k * sizes[f["category"]])
                        for f in flips)
        out["observed"][arm] = {
            "n_shared": len(shared), "n_flips": len(flips),
            "pre_edit_macro": round(old_macro, 6), "post_edit_macro": round(new_macro, 6),
            "observed_delta": round(new_macro - old_macro, 6),
            "delta_implied_by_flips": round(predicted, 6),
            "categories": k, "repeated_ids_pre": o_rep, "repeated_ids_post": n_rep,
            "same_id_order": _sha_o == _sha_n,
            "arithmetic_reconciles": abs((new_macro - old_macro) - predicted) < 1e-9,
        }
        check("grader_delta_%s_arithmetic" % arm,
              out["observed"][arm]["arithmetic_reconciles"],
              "the flips imply %+.6f and the composite moved %+.6f, so something other "
              "than a reclassification moved this row"
              % (predicted, new_macro - old_macro))
        out["flips"] += flips
        seen_flips += flips
        log("%s: %d flip(s) of %d shared items, composite %+.6f, flips imply %+.6f"
            % (arm, len(flips), len(shared), new_macro - old_macro, predicted))

    # The prediction, for arms whose pre-edit per-item file no longer exists.
    ids = sorted({f["id"] for f in seen_flips})
    for arm, pre_composite in sorted(recorded.items()):
        try:
            new = read_jsonl(fetch("%s/%s/%s" % (post_prefix_for(arm).rstrip("/"),
                                                 arm, file_name)))
        except Exception as exc:                                   # noqa: BLE001
            NOTES.append("%s has no post-edit file (%s)" % (arm, exc))
            continue
        n, _rep, _sha_n = by_id(new)
        now_macro, _p, sizes = macro(new)
        k = len(sizes)
        share = []
        for rid in ids:
            row = n.get(rid)
            if row is None:
                NOTES.append("%s does not contain %s, so it cannot have flipped on it"
                             % (arm, rid))
                continue
            # The flip transfers only if this arm's CURRENT verdict matches the flipped
            # arm's current verdict. An arm that is right where the other went wrong was
            # never affected by the edit on that item.
            same = {f["now"] for f in seen_flips if f["id"] == rid}
            if bool(row.get("correct")) in same:
                share.append({"id": rid, "category": row.get("category"),
                              "now": bool(row.get("correct")),
                              "reason": row.get("reason") or "",
                              "macro_contribution":
                                  round((1.0 if row.get("correct") else -1.0)
                                        / (k * sizes[row.get("category")]), 6)})
        predicted_delta = sum(s["macro_contribution"] for s in share)
        out["predicted"][arm] = {
            "recorded_pre_edit_macro": round(float(pre_composite), 6),
            "post_edit_macro": round(now_macro, 6),
            "observed_delta": round(now_macro - float(pre_composite), 6),
            "delta_implied_by_shared_flips": round(predicted_delta, 6),
            "residual": round((now_macro - float(pre_composite)) - predicted_delta, 6),
            "items": share, "categories": k,
        }
        log("%s: recorded %.4f -> %.4f, observed %+.6f, shared flips imply %+.6f"
            % (arm, float(pre_composite), now_macro,
               now_macro - float(pre_composite), predicted_delta))
    return out


# ------------------------------------------------------------ structured output


def envelope(text):
    """What the wrapper around a structured answer looks like, ignoring its contents."""
    fences = FENCE_OPEN.findall(text or "")
    n = len(fences)
    return {
        "fences": n,
        "opened": n >= 1,
        "closed": n >= 2 and n % 2 == 0,
        "lang": (fences[0] or "").lower() if fences else "",
        "chars": len(text or ""),
        "empty": not (text or "").strip(),
    }


def ifstruct_errors(arms, prefix_for, completions_for, file_name, comp_name, top_n):
    """Where the JSON regression actually lives: the wrapper, or the document inside it.

    An error list per item is already in the scored file. The only thing added here is the
    completion it was computed from, which is what separates "raised no valid document"
    from "raised no document at all". The two failure modes have the same grader verdict
    and different fixes, and the cell `s6.2` reported cannot tell them apart.
    """
    out = {}
    for arm in arms:
        try:
            scored = read_jsonl(fetch("%s/%s/%s" % (prefix_for(arm).rstrip("/"),
                                                    arm, file_name)))
        except Exception as exc:                                   # noqa: BLE001
            NOTES.append("%s has no %s (%s)" % (arm, file_name, exc))
            continue
        comps = {}
        try:
            comps, _r, _s = by_id(read_jsonl(fetch("%s/%s/%s"
                                               % (completions_for.rstrip("/"),
                                                  arm, comp_name))))
        except Exception as exc:                                   # noqa: BLE001
            NOTES.append("%s has no %s, so its envelope columns are empty (%s)"
                         % (arm, comp_name, exc))
        joined = sum(1 for r in scored if r.get("id") in comps)
        if comps:
            check("ifstruct_%s_joins" % arm, joined >= 0.99 * len(scored),
                  "only %d of %d scored ids have a completion, so the two files are not "
                  "the same run" % (joined, len(scored)))
        fmts = {}
        for r in scored:
            fmt = r.get("output_format") or "unknown"
            d = r.get("detail") if isinstance(r.get("detail"), dict) else {}
            errs = [e for e in (d.get("errors") or [])]
            env = envelope((comps.get(r.get("id")) or {}).get("completion", "")) \
                if comps else None
            b = fmts.setdefault(fmt, {"n": 0, "valid": 0, "score": 0.0, "errors": {},
                                      "envelope_only": 0, "schema_any": 0, "no_errors": 0,
                                      "fence_missing": 0, "fence_unclosed": 0,
                                      "lang": {}, "empty": 0, "chars": []})
            b["n"] += 1
            b["valid"] += 1 if r.get("correct") else 0
            b["score"] += float(d.get("score") or 0.0)
            for e in errs:
                b["errors"][e] = b["errors"].get(e, 0) + 1
            if not errs:
                b["no_errors"] += 1
            elif all(e in ENVELOPE_ERRORS for e in errs):
                b["envelope_only"] += 1
            else:
                b["schema_any"] += 1
            if env:
                b["chars"].append(env["chars"])
                b["empty"] += 1 if env["empty"] else 0
                b["lang"][env["lang"] or "(none)"] = b["lang"].get(env["lang"] or "(none)",
                                                                  0) + 1
                if not env["opened"]:
                    b["fence_missing"] += 1
                elif not env["closed"]:
                    b["fence_unclosed"] += 1
        for fmt, b in fmts.items():
            ch = sorted(b.pop("chars"))
            b["validity"] = round(b["valid"] / b["n"], 6)
            b["mean_partial_score"] = round(b["score"] / b["n"], 6)
            b.pop("score")
            b["median_chars"] = ch[len(ch) // 2] if ch else None
            b["p90_chars"] = ch[int(0.9 * (len(ch) - 1))] if ch else None
            b["errors"] = dict(sorted(b["errors"].items(), key=lambda kv: -kv[1])[:top_n])
            b["lang"] = dict(sorted(b["lang"].items(), key=lambda kv: -kv[1])[:top_n])
            b["envelope_share_of_failures"] = round(
                b["envelope_only"] / max(1, b["n"] - b["valid"]), 6)
            # The load-bearing column. Raw validity mixes two failures: the model did not
            # understand the requested schema, and the model understood it and did not close
            # a code fence. Dropping the items whose ENTIRE error list is about the fence
            # leaves the question "when the wrapper was fine, was the content right", which
            # is the one a schema regression would show up in. An arm whose raw validity
            # falls while this column holds lost a wrapper, not a schema, and the two want
            # completely different fixes.
            clean = b["n"] - b["envelope_only"]
            b["clean_envelope_items"] = clean
            b["validity_given_clean_envelope"] = round(b["valid"] / max(1, clean), 6)
        out[arm] = {"n": len(scored), "joined_completions": joined, "by_format": fmts}
        log("%s: ifstruct %s" % (arm, ", ".join(
            "%s validity %.4f (%.4f given a clean envelope), envelope-only %d of %d "
            "failures" % (f, b["validity"], b["validity_given_clean_envelope"],
                          b["envelope_only"], b["n"] - b["valid"])
            for f, b in sorted(fmts.items()))))
    return out


# --------------------------------------------------------------- native losses


def reason_family(reason):
    """The grader's own reason string, collapsed to the failure it describes."""
    r = (reason or "").lower()
    if not r:
        return "correct"
    if "parsed 0" in r or "no call" in r:
        return "did not call"
    if "made no call when one applies" in r:
        return "did not call"
    if "called when none applies" in r or "should not have called" in r:
        return "called when abstaining was right"
    # The grader writes a per-call verdict as `gt<i>/call<j>: <what went wrong>`, and one
    # reason string can carry several of them. Reading only the first clause is enough to
    # name the family, and keeping the function name in the key would split "wrong function"
    # into one bucket per function, which is how the first pass reported it: forty singleton
    # rows that are all the same failure.
    head = r.split(";")[0]
    head = re.sub(r"^gt\d+/call\d+:\s*", "", head).strip()
    if head.startswith("wrong function") or "function name" in head:
        return "wrong function"
    if head.startswith("wrong number") or ("expected" in head and "call" in head):
        return "wrong number of calls"
    if head.startswith("wrong argument") or "arg" in head or "param" in head \
            or "value" in head or "type" in head:
        return "wrong arguments"
    if "name" in head:
        return "wrong function"
    return "other: " + head[:60]


def native_losses(arms, ref_arm, prefix_for, completions_for, file_name, comp_name,
                  sample_n):
    """What an arm emits on the call-warranted items it lost against the reference.

    Restricted to the half of BFCL where a call is the right answer, because that is the
    half `s6.2` found the loss in and the half a deployment feels. Gains are tabulated too
    and on the same items: an arm that trades six losses for five gains inside one category
    is a different object from one that only loses.
    """
    ref = {}
    try:
        ref, _r, _s = by_id(read_jsonl(fetch("%s/%s/%s" % (prefix_for(ref_arm).rstrip("/"),
                                                       ref_arm, file_name))))
    except Exception as exc:                                       # noqa: BLE001
        NOTES.append("reference %s has no %s (%s)" % (ref_arm, file_name, exc))
        return {}
    out = {"reference": ref_arm, "arms": {}}
    for arm in arms:
        if arm == ref_arm:
            continue
        try:
            mine, _r, _s = by_id(read_jsonl(fetch("%s/%s/%s" % (prefix_for(arm).rstrip("/"),
                                                            arm, file_name))))
        except Exception as exc:                                   # noqa: BLE001
            NOTES.append("%s has no %s (%s)" % (arm, file_name, exc))
            continue
        comps = {}
        try:
            comps, _r, _s = by_id(read_jsonl(fetch("%s/%s/%s" % (completions_for.rstrip("/"),
                                                             arm, comp_name))))
        except Exception as exc:                                   # noqa: BLE001
            NOTES.append("%s has no %s, so its emitted text is not tabulated (%s)"
                         % (arm, comp_name, exc))
        lost, gained = [], []
        for rid, row in mine.items():
            r0 = ref.get(rid)
            if r0 is None or row.get("category") not in CALL_WARRANTED:
                continue
            if bool(r0.get("correct")) and not row.get("correct"):
                lost.append((rid, row))
            elif not r0.get("correct") and row.get("correct"):
                gained.append((rid, row))
        fam, cats, emitted = {}, {}, {"emitted a call": 0, "emitted prose only": 0,
                                      "emitted nothing": 0}
        for rid, row in lost:
            f = reason_family(row.get("reason"))
            fam[f] = fam.get(f, 0) + 1
            cats[row.get("category")] = cats.get(row.get("category"), 0) + 1
            text = (comps.get(rid) or {}).get("completion")
            if text is None:
                continue
            if row.get("n_calls"):
                emitted["emitted a call"] += 1
            elif (text or "").strip():
                emitted["emitted prose only"] += 1
            else:
                emitted["emitted nothing"] += 1
        out["arms"][arm] = {
            "n_call_warranted": sum(1 for _i, r in mine.items()
                                    if r.get("category") in CALL_WARRANTED),
            "lost": len(lost), "gained": len(gained), "net": len(gained) - len(lost),
            "by_reason": dict(sorted(fam.items(), key=lambda kv: -kv[1])),
            "by_category": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
            "what_it_emitted": emitted,
            "examples": [{"id": rid, "category": row.get("category"),
                          "reason": row.get("reason"),
                          "completion": ((comps.get(rid) or {}).get("completion")
                                         or "")[:400]}
                         for rid, row in lost[:sample_n]],
        }
        log("%s: lost %d and gained %d of %d call-warranted items against %s"
            % (arm, len(lost), len(gained), out["arms"][arm]["n_call_warranted"], ref_arm))
    return out


# ---------------------------------------------------------------------- render


def render(s):
    o = ["# s6.4 — error analysis", "",
         "Reference `%s`. Every input is a file an earlier job wrote; nothing here "
         "generated a token." % s.get("reference", "(none)"), ""]
    g = s.get("grader_delta") or {}
    if g:
        o += ["## What the grader edit reclassified", "",
              "| arm | shared items | flips | pre-edit | post-edit | observed | "
              "implied by flips | reconciles |", "|---|--:|--:|--:|--:|--:|--:|:-:|"]
        for arm, v in sorted((g.get("observed") or {}).items()):
            o.append("| %s | %d | %d | %.4f | %.4f | %+.6f | %+.6f | %s |"
                     % (arm, v["n_shared"], v["n_flips"], v["pre_edit_macro"],
                        v["post_edit_macro"], v["observed_delta"],
                        v["delta_implied_by_flips"],
                        "yes" if v["arithmetic_reconciles"] else "NO"))
        if g.get("flips"):
            o += ["", "| arm | item | category | was | now | calls before | calls after | "
                  "new reason |", "|---|---|---|:-:|:-:|--:|--:|---|"]
            for f in g["flips"]:
                o.append("| %s | `%s` | %s | %s | %s | %s | %s | %s |"
                         % (f["arm"], f["id"], f["category"],
                            "right" if f["was"] else "wrong",
                            "right" if f["now"] else "wrong",
                            f.get("old_n_calls"), f.get("new_n_calls"),
                            f["new_reason"] or "—"))
            if any(f.get("parse_changed") for f in g["flips"]):
                o += ["", "At least one flip changed the NUMBER OF CALLS PARSED, not just "
                      "the verdict on a fixed parse. The completions are the same bytes in "
                      "both passes, checked by prompt hash at replay, so an item that "
                      "yielded a call under the old grader and none under the new one says "
                      "the edit changed what counts as a call. Read the delta as an "
                      "extraction change rather than a scoring-rule change."]
                for f in g["flips"]:
                    if not f.get("parse_changed"):
                        continue
                    o += ["", "`%s` (%s), %s calls before and %s after:"
                          % (f["id"], f["category"], f.get("old_n_calls"),
                             f.get("new_n_calls")),
                          "", "```", "before: " + (f.get("old_calls") or ""),
                          "after:  " + (f.get("new_calls") or ""), "```"]
        for arm, v in sorted((g.get("predicted") or {}).items()):
            o += ["", "**%s has no pre-edit per-item file, so this row is a prediction.** "
                  "Its recorded pre-edit composite is %.4f and it now scores %.4f, a move "
                  "of %+.6f. Assuming it flipped on the same items gives %+.6f, leaving a "
                  "residual of %+.6f."
                  % (arm, v["recorded_pre_edit_macro"], v["post_edit_macro"],
                     v["observed_delta"], v["delta_implied_by_shared_flips"],
                     v["residual"])]
    i = s.get("ifstruct_errors") or {}
    if i:
        o += ["", "## Where the structured-output regression lives", "",
              "| arm | format | n | validity | validity given a clean envelope | "
              "partial | failures | envelope only | envelope share | no fence | "
              "unclosed fence | median chars |",
              "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for arm, v in sorted(i.items()):
            for fmt, b in sorted(v["by_format"].items()):
                o.append("| %s | %s | %d | %.4f | %.4f | %.4f | %d | %d | %.4f | %d | "
                         "%d | %s |"
                         % (arm, fmt, b["n"], b["validity"],
                            b["validity_given_clean_envelope"], b["mean_partial_score"],
                            b["n"] - b["valid"], b["envelope_only"],
                            b["envelope_share_of_failures"], b["fence_missing"],
                            b["fence_unclosed"],
                            b["median_chars"] if b["median_chars"] is not None else "—"))
        o += ["", "`envelope only` counts failures whose entire error list is about the "
              "code fence rather than the document inside it. A high share says the model "
              "lost the wrapper; a low one says it lost the schema."]
        for arm, v in sorted(i.items()):
            for fmt, b in sorted(v["by_format"].items()):
                if not b["errors"]:
                    continue
                o += ["", "**%s / %s** — %s" % (arm, fmt, "; ".join(
                    "%s ×%d" % (k, n) for k, n in b["errors"].items()))]
    n = s.get("native_losses") or {}
    if n.get("arms"):
        o += ["", "## What the call-warranted losses look like", "",
              "| arm | call-warranted | lost | gained | net |", "|---|--:|--:|--:|--:|"]
        for arm, v in sorted(n["arms"].items()):
            o.append("| %s | %d | %d | %d | %+d |"
                     % (arm, v["n_call_warranted"], v["lost"], v["gained"], v["net"]))
        for arm, v in sorted(n["arms"].items()):
            o += ["", "**%s** — by reason: %s. By category: %s. Emitted: %s."
                  % (arm,
                     "; ".join("%s ×%d" % (k, c) for k, c in v["by_reason"].items()),
                     "; ".join("%s ×%d" % (k, c) for k, c in v["by_category"].items()),
                     "; ".join("%s ×%d" % (k, c)
                               for k, c in v["what_it_emitted"].items() if c))]
    if s.get("notes"):
        o += ["", "## Notes", ""] + ["- %s" % x for x in s["notes"]]
    if s.get("assertion_failures"):
        o += ["", "## Assertion failures", ""] + ["- %s" % x
                                                  for x in s["assertion_failures"]]
    return "\n".join(o) + "\n"


def main():
    arms = [a.strip() for a in str(C("arms", "")).split(",") if a.strip()]
    ref_arm = str(C("reference", ""))
    prefix = str(C("scored_prefix", "tidepool/s5.3/arms"))
    arm_prefixes = jparam("arm_prefixes")
    completions_prefix = str(C("completions_prefix", "tidepool/s6.4/completions"))
    pre_edit = jparam("pre_edit_objects")
    recorded = jparam("recorded_pre_edit_composite")
    want = [c.strip() for c in str(C("components", "")).split(",") if c.strip()]
    top_n = int(C("top_errors", 20))
    sample_n = int(C("examples", 12))

    def prefix_for(a):
        return str(arm_prefixes.get(a) or prefix)

    if not arms:
        raise SystemExit("no arms: this pass reads per-item files and needs to be told whose")
    unknown = sorted(set(arm_prefixes) - set(arms))
    if unknown:
        raise SystemExit("arm_prefixes names arm(s) not in this pass: %r" % unknown)
    log("s6.4 error analysis over %d arm(s), reference %s, components %s"
        % (len(arms), ref_arm or "(none)", ",".join(want) or "(all)"))
    for a in arms:
        log("%-14s <- %s" % (a, prefix_for(a)))

    s = {"arms": arms, "reference": ref_arm, "scored_prefix": prefix,
         "arm_prefixes": arm_prefixes, "completions_prefix": completions_prefix,
         "call_warranted": list(CALL_WARRANTED), "abstain_warranted":
         list(ABSTAIN_WARRANTED)}
    run = (lambda c: not want or c in want)
    if run("grader_delta") and (pre_edit or recorded):
        s["grader_delta"] = grader_delta(pre_edit, prefix_for,
                                         "scored_bfcl_native_tools.jsonl", recorded)
    if run("ifstruct_errors"):
        s["ifstruct_errors"] = ifstruct_errors(arms, prefix_for, completions_prefix,
                                               "scored_ifstruct.jsonl",
                                               "completions_ifstruct.jsonl", top_n)
    if run("native_losses") and ref_arm:
        s["native_losses"] = native_losses(arms, ref_arm, prefix_for, completions_prefix,
                                           "scored_bfcl_native_tools.jsonl",
                                           "completions_bfcl_native_tools.jsonl", sample_n)
    s["notes"] = list(NOTES)
    s["assertion_failures"] = list(FAILURES)
    save(s, "errors.json")
    save(render(s), "errors.md")
    msg = ("s6.4 error analysis over %d arm(s). %d assertion failure(s), %d note(s)."
           % (len(arms), len(FAILURES), len(NOTES)))
    log(msg)
    # `finish`, not `job_complete`: the latter is absent from the SDK on the worker, and a
    # job that raises after its artifacts are written is a correct result under a status that
    # says not to quote it. The score is the assertion-failure count, so the dashboard row the
    # operator opens says at a glance whether the pass checked out.
    try:
        lab.finish(message=msg, score=len(FAILURES))
    except Exception:                                              # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
