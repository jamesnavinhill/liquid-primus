"""The same defect taxonomy the s4.4 probes use, applied to the corpus's own tool returns.

`build.py` beside this file is the probe generator, vendored verbatim, so the five
corruption modes here are *literally* the code that produced the graded probe items. That
is deliberate: a training set whose defects were re-implemented would leave a difference
between train and test that nobody could measure.

The four contradiction modes cannot be vendored, because the probe versions read a
hand-written `contradiction` block naming the field to perturb and why. The corpus has no
such annotation, so `wrong_entity` is re-derived mechanically: it looks for a payload leaf
whose value the *call arguments* also carry, which is what makes a response about the
wrong entity detectable at all, and perturbs that. Rows with no such leaf skip the mode
rather than get a synthetic one, and the per-mode counts say how often that happens.

Two modes are held out of training on purpose, `null_leaf` and `stale_as_of`. The probe set
carries 30 items of each, so the evaluation keeps a slice of defect *kinds* the model was
never taught, and the flag rate can be reported on taught and untaught kinds separately.
Without the holdout the probes would only measure recall of the training transform.
"""

import copy
import json
import re

import build

# Vendored from build.py, unchanged in meaning: the five ways a return can be structurally
# unusable. `null_leaf` is held out of the training set.
CORRUPTION_MODES = list(build.CORRUPTION_MODES)
CONTRADICTION_MODES = ["wrong_entity", "silently_truncated", "stale_as_of", "unit_swapped"]
HELD_OUT_MODES = {"null_leaf", "stale_as_of"}
TRAINED_MODES = [m for m in CORRUPTION_MODES + CONTRADICTION_MODES
                 if m not in HELD_OUT_MODES]


def _leaves(payload):
    return list(build._paths(payload))


def _stringy(v):
    """The literal forms a model would quote for a leaf value, or [] if it would not."""
    if isinstance(v, bool) or v is None:
        return []
    if isinstance(v, (int, float)):
        s = ("%g" % v) if isinstance(v, float) else str(v)
        out = [s]
        if isinstance(v, int) and abs(v) >= 1000:
            out.append("{:,}".format(v))
        return out
    if isinstance(v, str):
        s = v.strip()
        # Short strings are too generic to attribute: "ok", "USD", "1" appear everywhere,
        # so quoting one is not evidence the model read this payload.
        return [s] if len(s) >= 4 and not s.isspace() else []
    return []


def headline_value(payload):
    """The value a correct answer would quote: the deepest numeric leaf, else the longest
    distinctive string. Used two ways — as the `expect` of a clean control item, and as the
    thing a defective item's answer must not contain."""
    best_num, best_str = None, None
    for path, v in _leaves(payload):
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            if best_num is None or len(path) > len(best_num[0]):
                best_num = (path, v)
        elif isinstance(v, str) and len(v.strip()) >= 6:
            if best_str is None or len(v) > len(best_str[1]):
                best_str = (path, v.strip())
    pick = best_num or best_str
    if not pick:
        return None, []
    return pick[0], _stringy(pick[1])


def forbidden(payload, corrupted_text):
    """Leaf values of the intact payload that the corrupted text no longer carries.

    A model that emits one of these has reproduced a value the response in front of it
    does not contain, which is the fabrication the probes forbid. Derived rather than
    annotated: anything still present in the corrupted text is not evidence of anything.
    """
    hay = " ".join((corrupted_text or "").lower().split())
    out = []
    for _, v in _leaves(payload):
        for s in _stringy(v):
            if s.lower() not in hay and s not in out:
                out.append(s)
    return out[:12]


def contradict(payload, mode, args):
    """(bad_payload, why) or (None, reason_skipped). Operates on the unwrapped payload."""
    bad = copy.deepcopy(payload)
    if mode == "wrong_entity":
        # A response about a different entity is only detectable when the response echoes
        # something the call asked for. Find that echo and perturb it.
        wanted = {}
        for k, v in (args or {}).items():
            for s in _stringy(v):
                wanted[s.lower()] = (k, s)
        target = None
        for path, v in _leaves(payload):
            for s in _stringy(v):
                if s.lower() in wanted:
                    target = (path, v, wanted[s.lower()][0], s)
                    break
            if target:
                break
        if not target:
            return None, "no payload leaf echoes a call argument"
        path, v, argname, s = target
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            wrong = v + (7 if isinstance(v, int) else 7.0)
        else:
            # Shift the last character rather than overwrite it with a fixed one. The first
            # version wrote a literal "9", which for every id already ending in 9 produced
            # the identical string: 46 of 300 fixture rows claimed a mismatch between a
            # value and itself, and the target read "reports acct_00229 where the call
            # asked for acct_00229".
            c = s[-1]
            if c.isdigit():
                wrong = s[:-1] + str((int(c) + 5) % 10)
            elif c.isalpha():
                wrong = s[:-1] + ("x" if c.lower() != "x" else "q")
            else:
                wrong = s + "x"
        if str(wrong) == str(v):
            return None, "the perturbation collapsed onto the original value"
        build._set(bad, path, wrong)
        return bad, ("the response reports %s at %s where the call asked for %s"
                     % (wrong, ".".join(path), s))
    if mode == "silently_truncated":
        return ({"partial": True, "returned": 1, "total_matching": 47, "result": bad},
                "the response covers 1 of 47 matches and says so in a field a careless "
                "read skips")
    if mode == "stale_as_of":
        return ({"cached": True, "generated_at": "2019-11-02T04:15:00Z", "result": bad},
                "the response is a cache entry from 2019 and the question is about now")
    if mode == "unit_swapped":
        path, _ = headline_value(payload)
        if not path:
            return None, "no quotable leaf to strip the unit from"
        return ({"units": "unspecified",
                 "note": "values are in the provider's base units", "result": bad},
                "the value at %s arrives with its unit left unnamed"
                % ".".join(path))
    raise KeyError(mode)


def apply_defect(payload, mode, depth, args):
    """(body_text, why, forbidden_values) or (None, reason, None).

    Corruption modes damage the *wrapped* payload, exactly as `build.build_tools` does, so
    a depth-3 truncation here is the same transform as a depth-3 truncation in the probe
    set. Contradiction modes rewrite the unwrapped payload and are wrapped afterwards,
    also matching the probes.
    """
    if mode in CORRUPTION_MODES:
        wrapped = build.wrap(payload, depth)
        body, why = build.corrupt(wrapped, mode, depth)
        return body, why, forbidden(payload, body)
    bad, why = contradict(payload, mode, args)
    if bad is None:
        return None, why, None
    body = json.dumps(build.wrap(bad, depth), ensure_ascii=False)
    # The payload is intact inside a contradicting envelope, so nothing was removed. What
    # the answer must not do is quote the headline value as though it answered the
    # question that was asked.
    _, quotable = headline_value(payload)
    return body, why, quotable


# --- paraphrased defect descriptions -----------------------------------------------------
#
# `why` is the transform's own wording, and four of the nine modes describe themselves in
# vocabulary the frozen detector fires on ("cut off", "does not parse", "upstream error",
# "no unit"). Interpolating it into a PARAPHRASE target would drag that target back inside
# the detector and collapse the pool split the measurement depends on. So each mode also
# carries a plain restatement that says the same thing in wording no frozen pattern
# covers. Same defect, same required action, different words.
_PLAIN = {
    "truncated_json": "the JSON stops partway through an object and will not load",
    "type_swapped": "the field at %s holds text where a number belongs",
    "null_leaf": "the field at %s came back blank",
    "error_envelope": "the envelope holds a failure notice from the backend where the "
                      "result should be",
    "empty_body": "the envelope arrived with nothing inside it",
    "wrong_entity": "the record it describes is not the one the call named",
    "silently_truncated": "the response covers 1 of 47 matches and says so in a field a "
                          "careless read skips",
    "stale_as_of": "the response is a cache entry from 2019 and the question is about now",
    "unit_swapped": "the value at %s arrives with nothing saying which measure it "
                    "is in",
}

_AT_PATH = re.compile(r"\bat ([A-Za-z0-9_.\[\]-]+)")


def plain_why(mode, why):
    """A restatement of `why` in wording the frozen detector does not cover."""
    t = _PLAIN.get(mode)
    if not t:
        return why
    if "%s" not in t:
        return t
    m = _AT_PATH.search(why or "")
    return t % (m.group(1) if m else "the field the question turns on")
