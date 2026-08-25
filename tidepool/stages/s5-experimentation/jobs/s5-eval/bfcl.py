"""BFCLv3, restricted to the categories that can be checked without a live API.

Roughly a fifth of BFCLv3 (`exec_*`, `rest`) scores a model by *executing* the call it
produced against third-party HTTP APIs with real credentials. None of that is
reproducible here, and a composite that quietly dropped it while keeping the benchmark's
name would not be comparable to anything. So the number this module produces is defined,
named and reported as what it is:

  BFCLv3-AST composite = the unweighted mean over the included categories' accuracies.

Included: the abstract-syntax categories (a predicted call is checked against a set of
acceptable ground-truth calls) plus the two that score restraint — `irrelevance`, where
the right answer is to make no call, and `live_relevance`, where it is to make one.
Excluded, and recorded as excluded in every summary: `exec_*` and `rest` (need live
endpoints and keys), `java`/`javascript`/`sql` (language-specific value parsing that
would need a tree-sitter grammar per language for no gain on an agent stack that calls
Python-shaped tools), and `multi_turn_*` (BFCL scores those against a stateful
environment whose class instances are part of the harness, not the data).

Equal weight per category is a choice, not a convention: the categories differ in size by
two orders of magnitude, and a size-weighted mean would make the composite mostly a
report on `live_simple`. Both are computed; the equal-weighted one is the headline and
the item-weighted one is printed beside it.

The checker's rules, which are the ones that matter for reproducing a number:
  * the call count must equal the ground truth's count;
  * calls are matched to ground-truth calls as a bijection, so parallel calls in a
    different order still score;
  * a function name must match exactly;
  * every ground-truth parameter must be present with an accepted value, unless its
    accepted list contains an empty value, which makes it optional;
  * a parameter the ground truth does not mention is an error;
  * strings compare case-insensitively after stripping, numbers numerically, lists
    element-wise in order, dicts recursively.
"""

import json
import math

AST_CATEGORIES = [
    "simple", "multiple", "parallel", "parallel_multiple",
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
]
RESTRAINT_CATEGORIES = ["irrelevance", "live_irrelevance", "live_relevance"]
DEFAULT_CATEGORIES = AST_CATEGORIES + RESTRAINT_CATEGORIES
EXCLUDED = {
    "exec_simple, exec_multiple, exec_parallel, exec_parallel_multiple, rest":
        "scored by executing the call against live third-party APIs with credentials",
    "java, javascript, sql":
        "language-specific value parsing; needs a grammar per language",
    "multi_turn_base, multi_turn_miss_func, multi_turn_miss_param, multi_turn_long_context":
        "scored against a stateful environment that is part of the BFCL harness, not the data",
}
NO_CALL_EXPECTED = {"irrelevance", "live_irrelevance"}
CALL_EXPECTED = {"live_relevance"}

_EMPTY = ("", None, [], {}, "none", "null")
_TYPE_FIX = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


# ---------------------------------------------------------------- data

def _jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fix_types(node):
    """BFCL writes JSON-schema-ish types with its own names; normalize in place."""
    if isinstance(node, dict):
        t = node.get("type")
        if isinstance(t, str) and t in _TYPE_FIX:
            node["type"] = _TYPE_FIX[t]
        for v in node.values():
            fix_types(v)
    elif isinstance(node, list):
        for v in node:
            fix_types(v)
    return node


def to_openai_tools(functions):
    out = []
    for f in functions or []:
        spec = json.loads(json.dumps(f))          # never mutate the loaded row
        params = fix_types(spec.get("parameters") or {"type": "object", "properties": {}})
        out.append({"type": "function",
                    "function": {"name": spec.get("name", ""),
                                 "description": spec.get("description", ""),
                                 "parameters": params}})
    return out


def load_category(data_dir, cat, limit=0):
    """Returns a list of items: id, messages, tools, ground_truth (None for restraint)."""
    rows = _jsonl("%s/BFCL_v3_%s.json" % (data_dir, cat))
    answers = {}
    if cat not in NO_CALL_EXPECTED and cat not in CALL_EXPECTED:
        for a in _jsonl("%s/possible_answer/BFCL_v3_%s.json" % (data_dir, cat)):
            answers[a["id"]] = a.get("ground_truth")
    items = []
    for r in rows:
        turns = r.get("question") or []
        msgs = [m for turn in turns for m in turn] if turns and isinstance(turns[0], list) else turns
        msgs = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in msgs]
        if not msgs:
            continue
        gt = answers.get(r["id"])
        if cat not in NO_CALL_EXPECTED and cat not in CALL_EXPECTED and gt is None:
            continue                                  # no ground truth, not scoreable
        items.append({"id": r["id"], "category": cat, "messages": msgs,
                      "tools": to_openai_tools(r.get("function")), "ground_truth": gt})
    # Strided, not the first N. The files are ordered, and within a category the harder
    # multi-argument rows cluster; a head slice of `live_multiple` would sample 120 of 1,053
    # rows from one corner of the distribution and call the result a category score. Striding
    # is deterministic, so a screening cap is still an exactly paired comparison across models.
    if limit and len(items) > limit:
        stride = len(items) // limit
        items = items[::stride][:limit]
    return items


# ---------------------------------------------------------------- value matching

def _norm_str(v):
    return " ".join(str(v).strip().lower().split())


def _is_empty(v):
    if isinstance(v, str):
        return _norm_str(v) in ("", "none", "null")
    return v is None or v == [] or v == {}


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip().replace(",", ""))
        except Exception:                                          # noqa: BLE001
            return None
    return None


def value_matches(got, want):
    """One model value against one acceptable ground-truth value."""
    if isinstance(want, bool) or isinstance(got, bool):
        return bool(got) == bool(want) and isinstance(got, bool) == isinstance(want, bool)
    if isinstance(want, (list, tuple)):
        if not isinstance(got, (list, tuple)) or len(got) != len(want):
            return False
        return all(value_matches(g, w) for g, w in zip(got, want))
    if isinstance(want, dict):
        if not isinstance(got, dict) or set(got) != set(want):
            return False
        return all(value_matches(got[k], want[k]) for k in want)
    gn, wn = _num(got), _num(want)
    if gn is not None and wn is not None:
        return math.isclose(gn, wn, rel_tol=1e-6, abs_tol=1e-9)
    return _norm_str(got) == _norm_str(want)


def call_matches(call, gt_call):
    """gt_call: {func_name: {param: [acceptable, ...]}}. Returns (ok, reason)."""
    if not gt_call or len(gt_call) != 1:
        return False, "malformed ground truth"
    name = next(iter(gt_call))
    if call["name"] != name:
        return False, "wrong function: %s" % call["name"]
    spec = gt_call[name] or {}
    for param, accepted in spec.items():
        accepted = accepted if isinstance(accepted, list) else [accepted]
        # An empty accepted list is BFCL's way of saying this parameter must be left
        # empty (7 of 3,351 parameters in the screening categories do it), so it is
        # optional and admits only an empty value.
        optional = (not accepted) or any(_is_empty(a) for a in accepted)
        if param not in call["args"]:
            if optional:
                continue
            return False, "missing required parameter: %s" % param
        got = call["args"][param]
        if _is_empty(got) and optional:
            continue
        if not any(value_matches(got, a) for a in accepted):
            return False, "bad value for %s: %r" % (param, got)
    extra = [k for k in call["args"] if k not in spec]
    if extra:
        return False, "parameters not in the contract: %s" % ",".join(sorted(extra))
    return True, ""


def score_item(item, calls):
    """(correct, reason). `calls` is the parsed output of prompting.parse_calls."""
    cat = item["category"]
    if cat in NO_CALL_EXPECTED:
        return (not calls), ("called %s when no tool applies" % calls[0]["name"] if calls else "")
    if cat in CALL_EXPECTED:
        return bool(calls), ("" if calls else "made no call when one applies")
    gt = item["ground_truth"] or []
    if len(calls) != len(gt):
        return False, "expected %d call(s), parsed %d" % (len(gt), len(calls))
    used, reasons = set(), []
    for gi, gt_call in enumerate(gt):
        hit = None
        for ci, call in enumerate(calls):
            if ci in used:
                continue
            ok, why = call_matches(call, gt_call)
            if ok:
                hit = ci
                break
            reasons.append("gt%d/call%d: %s" % (gi, ci, why))
        if hit is None:
            return False, "; ".join(reasons[-3:]) or "no matching call"
        used.add(hit)
    return True, ""


def summarize(per_item):
    """per_item: list of dicts with category + correct. Returns the composite block."""
    by_cat = {}
    for r in per_item:
        b = by_cat.setdefault(r["category"], {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += 1 if r["correct"] else 0
    cats = {c: {"n": b["n"], "accuracy": (b["correct"] / b["n"]) if b["n"] else 0.0}
            for c, b in sorted(by_cat.items())}
    n = sum(b["n"] for b in by_cat.values())
    correct = sum(b["correct"] for b in by_cat.values())
    mean = (sum(c["accuracy"] for c in cats.values()) / len(cats)) if cats else 0.0
    return {"composite_category_mean": mean,
            "item_weighted": (correct / n) if n else 0.0,
            "n_items": n, "categories": cats, "excluded": EXCLUDED}
