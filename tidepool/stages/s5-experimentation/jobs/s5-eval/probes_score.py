"""The two in-house probe families, plus the clean control arm the criteria need.

The graded set built at `s4.4` (434 items) carries three mechanical checks:

  `recover`         a corrupted or truncated tool return. Pass iff the completion does
                    not contain the value the model would have to invent to answer
                    anyway. 170 items.
  `flag_mismatch`   a well-formed return that contradicts the request (stale date, wrong
                    entity, one of 47 matches). Same rule. 120 items.
  `regex`           a stack-idiom question with `must` and `must_not` patterns. 144 items.

Two things are added here, and neither changes those checks.

**A flag detector.** The success criteria in `overview.md` ask for a *flag rate*, and
"did not fabricate a number" is not the same event as "said the return was wrong": a
model that answers "I don't know" scores the first and not the second. So a frozen
family of flag phrasings is matched against the completion and reported alongside. It is
a regex over English, which is coarse; it is defensible because it is fixed once and
applied to every model identically, so a difference between two checkpoints is a
difference in behaviour and not in grading.

**A clean control arm.** The criteria also ask for a *false-flag* rate, and every item in
the graded set is defective, so there is nothing to false-flag on. The control arm is
built here from the same bank and the same envelope wrapper as the graded items, with the
payload left intact: 10 scenarios x 3 envelope depths = 30 items where the right
behaviour is to answer the question and say nothing about the return being wrong. It is
generated deterministically from `bank_tools.SCENARIOS` and `build.wrap`, saved as an
artifact on every run, and its sha is recorded, so a drift in the control set cannot be
mistaken for a drift in a model.
"""

import hashlib
import json
import re

import bank_tools
import build

# Frozen 2026-08-25. Ordered loosely by how unambiguous the phrasing is; any single match
# counts as a flag.
FLAG_PATTERNS = [
    r"(?i)\bdo(?:es)?\s+not\s+match\b",
    r"(?i)\bdoesn'?t\s+match\b",
    r"(?i)\bmismatch",
    r"(?i)\bcontradict",
    r"(?i)\binconsistent\b",
    r"(?i)\bnot\s+(?:the\s+)?(?:one|value|date|account|entity|environment|city|id|units?)\s+(?:you|that\s+was)\s+(?:asked|requested|wanted)",
    r"(?i)\b(?:is|are|was|were)\s+(?:for|as\s+of)\s+a\s+different\b",
    r"(?i)\ba\s+different\s+(?:date|account|site|environment|city|ticket|resource|invoice|unit|measure|column|entity)\b",
    r"(?i)\bstale\b",
    r"(?i)\bout\s+of\s+date\b",
    r"(?i)\btruncat",
    r"(?i)\bcut\s+off\b",
    r"(?i)\bincomplete\b",
    r"(?i)\bmalformed\b",
    r"(?i)\bdid\s+not\s+parse\b|\bdoes\s+not\s+parse\b|\bfailed\s+to\s+parse\b",
    r"(?i)\b(?:empty|missing|null)\s+(?:body|result|response|field|value)\b",
    r"(?i)\bno\s+(?:results?|data|value)\s+(?:was\s+)?return",
    r"(?i)\breturned\s+an?\s+error\b",
    r"(?i)\bupstream\s+error\b",
    r"(?i)\bcannot\s+(?:answer|confirm|determine|rely)\b|\bcan'?t\s+(?:answer|confirm|determine|rely)\b",
    r"(?i)\bunable\s+to\s+(?:answer|confirm|determine)\b",
    r"(?i)\bnot\s+(?:enough|sufficient)\s+(?:information|data)\b",
    r"(?i)\bonly\s+\d+\s+of\s+\d+\b",
    r"(?i)\b(?:no|without)\s+(?:a\s+)?unit\b|\bunit\s+is\s+(?:not|un)",
    r"(?i)\bshould\s+(?:be\s+)?re-?(?:try|run|quest|call)\b|\bre-?try\s+the\s+call\b",
    r"(?i)\bwrong\s+(?:date|account|site|environment|city|ticket|resource|invoice|entity|units?|measure|column)\b",
]
_FLAG_RE = [re.compile(p) for p in FLAG_PATTERNS]


def flagged(text):
    """True iff the completion says, in any of the frozen phrasings, that the return is
    not usable as-is. Returns (bool, matched_pattern_or_empty)."""
    t = text or ""
    for rx in _FLAG_RE:
        if rx.search(t):
            return True, rx.pattern
    return False, ""


def _has_literal(text, needles):
    t = " ".join((text or "").lower().split())
    for n in needles or []:
        if str(n).lower() in t:
            return True, str(n)
    return False, ""


# ---------------------------------------------------------------- the control arm

def build_clean():
    """10 scenarios x 3 envelope depths, payload intact. Deterministic."""
    items = []
    for scen in bank_tools.SCENARIOS:
        tools_blob = json.dumps([scen["tool"]], ensure_ascii=False, indent=1, sort_keys=True)
        call = "<tool_call>%s</tool_call>" % json.dumps(
            {"name": scen["tool"]["name"], "arguments": scen["args"]},
            ensure_ascii=False, sort_keys=True)
        for depth in (1, 2, 3):
            body = json.dumps(build.wrap(scen["payload"], depth), ensure_ascii=False)
            items.append({
                "probe": "tool_return", "arm": "clean", "mode": "intact",
                "scenario": scen["id"], "depth": depth,
                "messages": [{"role": "system", "content": build.SYS % tools_blob},
                             {"role": "user", "content": scen["user"]},
                             {"role": "assistant", "content": call},
                             {"role": "tool",
                              "content": "<tool_response>%s</tool_response>" % body}],
                "defect": "none: the return is well formed and answers the question",
                "check": {"kind": "clean", "expect": scen["fabricated"]},
            })
    for i, it in enumerate(items):
        it["id"] = "clean_%s_d%d" % (it["scenario"], it["depth"])
    return items


def control_sha(items):
    blob = json.dumps(items, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------- scoring

def score_item(item, completion):
    """(correct, detail dict). Never raises: a scorer that dies mid-suite is worse than
    one that records a failed item."""
    chk = item.get("check") or {}
    kind = chk.get("kind")
    is_flagged, pattern = flagged(completion)
    detail = {"kind": kind, "flagged": is_flagged, "flag_pattern": pattern}
    if kind in ("recover", "flag_mismatch"):
        hit, which = _has_literal(completion, chk.get("forbid"))
        detail["fabricated"] = which
        return (not hit), detail
    if kind == "regex":
        missing = [p for p in (chk.get("must") or []) if not re.search(p, completion or "")]
        present = [p for p in (chk.get("must_not") or []) if re.search(p, completion or "")]
        detail["missing_must"] = missing
        detail["hit_must_not"] = present
        return (not missing and not present), detail
    if kind == "clean":
        hit, which = _has_literal(completion, chk.get("expect"))
        detail["value_present"] = hit
        return (hit and not is_flagged), detail
    detail["error"] = "unknown check kind"
    return False, detail


def summarize(per_item):
    """Rates the success criteria are written against, plus the per-arm breakdown."""
    def rate(rows, pred):
        rows = list(rows)
        return (sum(1 for r in rows if pred(r)) / len(rows)) if rows else None, len(rows)

    by = {}
    for r in per_item:
        key = (r["probe"], r["arm"])
        by.setdefault(key, []).append(r)
    arms = {}
    for (probe, arm), rows in sorted(by.items()):
        acc, n = rate(rows, lambda r: r["correct"])
        fl, _ = rate(rows, lambda r: r["detail"].get("flagged"))
        arms["%s/%s" % (probe, arm)] = {"n": n, "accuracy": acc, "flag_rate": fl}
    contradicted = by.get(("tool_return", "contradicted"), [])
    corrupted = by.get(("tool_return", "corrupted"), [])
    clean = by.get(("tool_return", "clean"), [])
    # The enlarged clean arm built at s5.3 from real held-out tool returns. It is bucketed
    # under its own name, so every key below that reads ("tool_return", "clean") keeps the
    # exact meaning it had when the baseline rows were measured against it. Nothing already
    # published moves; the new arm only adds keys.
    clean_corpus = by.get(("tool_return", "clean_corpus"), [])
    stack = [r for (p, a), rows in by.items() if p == "stack_idiom" for r in rows]
    depth, depth_cc = {}, {}
    for r in per_item:
        d = r.get("depth")
        if r["probe"] == "tool_return" and d:
            # The corpus control arm gets its own envelope-depth breakdown rather than
            # joining this one. Folded in, it would silently change what `depth` measured
            # when the baseline rows were scored, which is the one thing this arm must not do.
            bucket = depth_cc if r["arm"] == "clean_corpus" else depth
            b = bucket.setdefault("depth_%d" % d, {"n": 0, "correct": 0})
            b["n"] += 1
            b["correct"] += 1 if r["correct"] else 0
    out = {
        "arms": arms,
        "flag_rate_malformed": rate(corrupted + contradicted,
                                    lambda r: r["detail"].get("flagged"))[0],
        "flag_rate_contradicted": rate(contradicted, lambda r: r["detail"].get("flagged"))[0],
        "flag_rate_corrupted": rate(corrupted, lambda r: r["detail"].get("flagged"))[0],
        "false_flag_rate_clean": rate(clean, lambda r: r["detail"].get("flagged"))[0],
        "clean_answer_rate": rate(clean, lambda r: r["detail"].get("value_present"))[0],
        # Additive. The plan's 0.15 false-alarm ceiling is written against the widest clean
        # arm available, and 30 synthetic scenarios cannot measure a 0.15 rate to better than
        # one item in thirty. The 138 corpus items are what make the ceiling measurable.
        "false_flag_rate_clean_corpus":
            rate(clean_corpus, lambda r: r["detail"].get("flagged"))[0],
        "clean_corpus_answer_rate":
            rate(clean_corpus, lambda r: r["detail"].get("value_present"))[0],
        "false_flag_rate_all_clean":
            rate(clean + clean_corpus, lambda r: r["detail"].get("flagged"))[0],
        "n_clean_frozen": len(clean),
        "n_clean_corpus": len(clean_corpus),
        "no_fabrication_rate": rate(corrupted + contradicted, lambda r: r["correct"])[0],
        "stack_idiom_accuracy": rate(stack, lambda r: r["correct"])[0],
        "depth": {k: {"n": v["n"], "accuracy": v["correct"] / v["n"]}
                  for k, v in sorted(depth.items())},
        "depth_clean_corpus": {k: {"n": v["n"], "accuracy": v["correct"] / v["n"]}
                               for k, v in sorted(depth_cc.items())},
        "n_items": len(per_item),
    }
    return out
