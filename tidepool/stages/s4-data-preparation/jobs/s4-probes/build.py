"""Expand the two banks into probe items, deterministically.

Every item is generated from a bank entry by an explicit transform, so the set is
reproducible from the bank plus this file and nothing else. No sampling, no randomness,
no model in the loop: a probe whose contents depend on a generation run is a probe whose
score cannot be compared across checkpoints.
"""

import copy
import json


# ------------------------------------------------------- probe A: malformed returns

def wrap(payload, depth):
    """Bury the payload under `depth` extra levels of plausible envelope.

    The hypothesis the depth strata test is that recovery degrades with nesting: a broken
    scalar at the top level is visible, and the same break three levels down reads as
    plausible. The envelopes are the shapes real gateways add, so depth is not a synthetic
    tax on parsing.
    """
    out = payload
    for k in (["data"], ["result", "body"], ["envelope", "response", "content"])[depth - 1]:
        out = {k: out}
    return out


def _paths(obj, prefix=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _paths(v, prefix + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _paths(v, prefix + (str(i),))
    else:
        yield prefix, obj


def _get(obj, path):
    for p in path:
        obj = obj[int(p)] if isinstance(obj, list) else obj[p]
    return obj


def _set(obj, path, val):
    for p in path[:-1]:
        obj = obj[int(p)] if isinstance(obj, list) else obj[p]
    last = path[-1]
    if isinstance(obj, list):
        obj[int(last)] = val
    else:
        obj[last] = val


def _deepest_number(payload):
    """The path to the numeric leaf a model would quote, which is what corruption targets."""
    best = None
    for path, v in _paths(payload):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if best is None or len(path) > len(best):
                best = path
    return best or next(iter(_paths(payload)))[0]


CORRUPTION_MODES = ["truncated_json", "type_swapped", "error_envelope",
                    "empty_body", "null_leaf"]


def corrupt(payload, mode, depth=1):
    """Damage an already-wrapped payload. `depth` is the envelope depth it was wrapped at.

    Every mode has to remain a function of depth, or the depth strata stop being strata.
    The first version of this returned a bare error object and a bare empty string for two
    of the five modes, which made those two identical at every depth: 40 of 290 items were
    byte-for-byte copies of another item, and the build's own duplicate check caught it.
    Both now sit inside the same envelope the success payload would have arrived in, which
    is also what a real gateway does with an upstream failure.
    """
    blob = json.dumps(payload, ensure_ascii=False)
    if mode == "truncated_json":
        return blob[: int(len(blob) * 0.62)], \
            "the response is cut off mid-object and does not parse"
    if mode == "type_swapped":
        p = _deepest_number(payload)
        bad = copy.deepcopy(payload)
        _set(bad, p, "N/A")
        return json.dumps(bad, ensure_ascii=False), \
            "the field at %s is a string where a number is required" % ".".join(p)
    if mode == "null_leaf":
        p = _deepest_number(payload)
        bad = copy.deepcopy(payload)
        _set(bad, p, None)
        return json.dumps(bad, ensure_ascii=False), \
            "the field at %s is null" % ".".join(p)
    if mode == "error_envelope":
        err = {"error": {"code": "upstream_timeout",
                         "message": "backend did not respond in 30s", "retryable": True}}
        return json.dumps(wrap(err, depth), ensure_ascii=False), \
            "the envelope carries an upstream error where the result should be"
    if mode == "empty_body":
        return json.dumps(wrap({}, depth), ensure_ascii=False), \
            "the envelope is well formed and its body is empty"
    raise KeyError(mode)


CONTRADICTIONS = ["wrong_entity", "silently_truncated", "stale_as_of", "unit_swapped"]


def contradict(payload, mode, scen):
    """Well-formed responses that answer a different question than the one asked.

    Takes the *unwrapped* payload, because the bank's contradiction paths are written
    against the scenario's own shape; the depth envelope goes on afterwards.
    """
    bad = copy.deepcopy(payload)
    if mode == "wrong_entity":
        c = scen["contradiction"]
        _set(bad, tuple(c["path"].split(".")), c["wrong"])
        return bad, c["why"]
    if mode == "silently_truncated":
        return ({"partial": True, "returned": 1, "total_matching": 47, "result": bad},
                "the response covers 1 of 47 matches and says so in a field a careless read skips")
    if mode == "stale_as_of":
        return ({"cached": True, "generated_at": "2019-11-02T04:15:00Z", "result": bad},
                "the response is a cache entry from 2019 and the question is about now")
    p = _deepest_number(payload)
    return ({"units": "unspecified", "note": "values are in the provider's base units",
             "result": bad},
            "the magnitude at %s is given with no unit the caller can resolve" % ".".join(p))


SYS = ("You are a function-calling assistant. You are given a set of tools inside "
       "<tools></tools>. When a tool applies, reply with one <tool_call></tool_call> block "
       "per call, containing a JSON object with \"name\" and \"arguments\". When a tool's "
       "response does not answer the question, say so; do not report a value the response "
       "does not contain.\n<tools>\n%s\n</tools>")


def build_tools(scenarios):
    items = []
    for scen in scenarios:
        tools_blob = json.dumps([scen["tool"]], ensure_ascii=False, indent=1, sort_keys=True)
        call = "<tool_call>%s</tool_call>" % json.dumps(
            {"name": scen["tool"]["name"], "arguments": scen["args"]},
            ensure_ascii=False, sort_keys=True)
        for depth in (1, 2, 3):
            payload = wrap(scen["payload"], depth)
            for mode in CORRUPTION_MODES:
                body, why = corrupt(payload, mode, depth)
                items.append({
                    "probe": "tool_return", "arm": "corrupted", "mode": mode,
                    "scenario": scen["id"], "depth": depth,
                    "messages": [{"role": "system", "content": SYS % tools_blob},
                                 {"role": "user", "content": scen["user"]},
                                 {"role": "assistant", "content": call},
                                 {"role": "tool", "content": "<tool_response>%s</tool_response>" % body}],
                    "defect": why,
                    "check": {"kind": "recover", "forbid": scen["fabricated"]},
                })
            for mode in CONTRADICTIONS:
                bad, why = contradict(scen["payload"], mode, scen)
                body = json.dumps(wrap(bad, depth), ensure_ascii=False)
                items.append({
                    "probe": "tool_return", "arm": "contradicted", "mode": mode,
                    "scenario": scen["id"], "depth": depth,
                    "messages": [{"role": "system", "content": SYS % tools_blob},
                                 {"role": "user", "content": scen["user"]},
                                 {"role": "assistant", "content": call},
                                 {"role": "tool", "content": "<tool_response>%s</tool_response>" % body}],
                    "defect": why,
                    "check": {"kind": "flag_mismatch",
                              "forbid": scen["fabricated"] if mode == "wrong_entity" else []},
                })
        # The control arm. Same defect, no tool anywhere, so a model that handles this and
        # fails the corrupted arm has a tool-output-reading deficit rather than a reasoning one.
        for mode, note in (("garbled_prose", "the source text breaks off mid-sentence"),
                           ("wrong_entity_prose", scen["contradiction"]["why"])):
            text = (scen["prose"][: int(len(scen["prose"]) * 0.6)] if mode == "garbled_prose"
                    else scen["prose"].replace(
                        str(scen["args"].get("date") or scen["args"].get("as_of") or ""), "") +
                    " (This note is about %s.)" % scen["contradiction"]["wrong"])
            items.append({
                "probe": "tool_return", "arm": "text_only", "mode": mode,
                "scenario": scen["id"], "depth": 0,
                "messages": [{"role": "user",
                              "content": "%s\n\nHere is what I found written down:\n%s"
                                         % (scen["user"], text)}],
                "defect": note,
                "check": {"kind": "recover", "forbid": scen["fabricated"]},
            })
    return items


# ------------------------------------------------------------ probe B: stack idioms

# Four surface forms per bank item. The graders do not change, so a model that answers the
# direct question and fails the code-review framing has a robustness gap rather than a
# knowledge gap, and the set separates the two instead of averaging over them.
FORMS = [
    ("direct", "%s"),
    ("review", "A colleague wrote an answer to this and I am not sure it is right. "
               "The question was: %s\nAnswer it correctly yourself first, then say in one "
               "line what the most likely mistake would be."),
    ("terse", "Answer in as few words as the question allows. %s"),
    ("agentic", "You are wiring this into a service that must not break in production. %s\n"
                "Give the answer and one sentence on what would break if it were wrong."),
]


def build_stack(families):
    items = []
    for fam in families:
        for i, it in enumerate(fam["items"]):
            for form, tmpl in FORMS:
                items.append({
                    "probe": "stack_idiom", "arm": fam["family"], "mode": form,
                    "scenario": "%s_%02d" % (fam["family"], i), "depth": 0,
                    "messages": [{"role": "user", "content": tmpl % it["q"]}],
                    "defect": it["trap"],
                    "check": {"kind": "regex", "must": it["must"],
                              "must_not": it.get("must_not") or []},
                })
    return items
