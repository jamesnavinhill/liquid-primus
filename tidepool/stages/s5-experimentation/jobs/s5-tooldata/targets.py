"""The assistant turn a defective tool return should get, derived from the transform.

No teacher model and no sampling. The defect was applied by a known transform, so the
correct reply is a function of which transform ran, and the phrasing is drawn from a fixed
pool by a hash of the row id. Two consequences worth stating: the set is reproducible from
the code, and every label is auditable against the transform that produced it.

The pool is split on purpose. `DETECTED` phrasings are ones the frozen 26-pattern flag
detector fires on; `PARAPHRASE` phrasings say the same thing in wording the detector was
never written to catch. Training on both keeps the measured flag rate from becoming a
recall test on the training vocabulary: if every target matched the detector, a trained
model's flag rate would report how well it memorised our phrasing rather than whether it
noticed the defect. The job measures the split it actually produced and fails if the pool
has collapsed to one side.
"""

import hashlib

# Defect mode -> the family whose phrasings apply. `why` carries the specifics, so the
# families only have to differ where the *action* differs.
FAMILY = {
    "truncated_json": "unusable",
    "empty_body": "unusable",
    "error_envelope": "errored",
    "null_leaf": "missing",
    "type_swapped": "missing",
    "wrong_entity": "mismatch",
    "silently_truncated": "partial",
    "stale_as_of": "stale",
    "unit_swapped": "unitless",
}

# Each entry: (pool, template). `{tool}` is the called function, `{why}` the transform's
# own description of the defect. Templates never quote a value from the payload, which is
# what makes the forbidden-value check pass by construction rather than by luck.
TEMPLATES = {
    "unusable": [
        ("DETECTED", "The `{tool}` response came back malformed: {why}. I can't answer "
                     "from it. Re-run the call and I'll read the fresh result."),
        ("DETECTED", "I can't rely on that result: {why}, so `{tool}` did not return "
                     "anything I can use. Worth retrying the call."),
        ("DETECTED", "That response is incomplete: {why}. Rather than guess at what was "
                     "in the missing part, I'd re-issue the `{tool}` call."),
        ("PARAPHRASE", "Hold on: {plain}. So there is nothing in the `{tool}` output for me "
                       "to work from. Send the call again and I'll pick it up."),
    ],
    "errored": [
        ("DETECTED", "`{tool}` returned an error rather than a result: {why}. Nothing to "
                     "report yet. It looks retryable, so I'd call it again."),
        ("DETECTED", "The call failed upstream: {why}. I cannot answer until it succeeds."),
        ("PARAPHRASE", "`{tool}` came back with a failure instead of data: {plain}. I have "
                       "no figure to give you. Try the call once more."),
    ],
    "missing": [
        ("DETECTED", "The field I need is missing from the `{tool}` response: {why}, so I "
                     "cannot determine the answer from it."),
        ("DETECTED", "That value did not come back usable: {why}. I cannot answer without "
                     "inventing a number, so I'd re-run `{tool}`."),
        ("PARAPHRASE", "The `{tool}` payload arrived without the value your question turns "
                       "on: {why}. I would be making it up if I answered. Re-issue the call."),
    ],
    "mismatch": [
        ("DETECTED", "This response does not match what was asked: {why}. It answers about "
                     "a different record, so I'd re-run `{tool}` with the right argument."),
        ("DETECTED", "The `{tool}` result is for the wrong entity: {why}. I won't report it "
                     "as though it answered your question."),
        ("PARAPHRASE", "What came back describes something else: {plain}. Answering from it "
                       "would give you the figure for the wrong record, so let me re-issue "
                       "the `{tool}` call."),
    ],
    "partial": [
        ("DETECTED", "The `{tool}` response is incomplete: {why}. Any answer from it "
                     "would describe a fraction of the matches, so I'd page through the "
                     "rest first."),
        ("DETECTED", "That result is incomplete by its own admission: {why}. I cannot treat "
                     "it as the full answer."),
        ("PARAPHRASE", "`{tool}` handed back a slice rather than the whole set: {plain}. "
                       "Summarising it would overstate what was found. Let me fetch the "
                       "remaining matches."),
    ],
    "stale": [
        ("DETECTED", "The `{tool}` response is stale: {why}. It cannot answer a question "
                     "about the present, so I'd re-run the call uncached."),
        ("DETECTED", "That result is out of date: {why}. I won't present it as current."),
        ("PARAPHRASE", "What came back is an old cached copy: {plain}. Using it would give "
                       "you last decade's figure. Re-issue the `{tool}` call and skip the "
                       "cache."),
    ],
    "unitless": [
        ("DETECTED", "The `{tool}` response gives a magnitude with no unit: {why}, so I "
                     "cannot determine what the number means."),
        ("DETECTED", "The provider did not say what this number is measured in: {why}. I "
                     "cannot determine whether it is a currency or a count, so reporting "
                     "either would be a guess."),
        ("PARAPHRASE", "`{tool}` returned a bare magnitude: {plain}. Until the provider says "
                       "which base unit it used, the number does not mean anything I can "
                       "pass on."),
    ],
}


def pick(row_id, mode, tool, why, plain=None):
    """(text, pool). Deterministic in `row_id` and `mode`, so the set is reproducible."""
    fam = FAMILY[mode]
    pool = TEMPLATES[fam]
    h = hashlib.blake2b(("%s|%s" % (row_id, mode)).encode("utf-8"), digest_size=8).digest()
    which, tmpl = pool[int.from_bytes(h, "big") % len(pool)]
    # `why` is a sentence fragment from the transform; lower-case its first letter so it
    # reads inside the template rather than as a second sentence starting mid-clause.
    w = (why or "").strip().rstrip(".")
    w = (w[0].lower() + w[1:]) if w else "the response is not usable"
    pw = (plain or w).strip().rstrip(".")
    pw = (pw[0].lower() + pw[1:]) if pw else w
    return tmpl.format(tool=tool or "the tool", why=w, plain=pw), which
