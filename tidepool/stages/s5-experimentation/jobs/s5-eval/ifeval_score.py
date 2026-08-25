"""IFEval, scored by Google's own verifiable-instruction registry.

There is no installable package for this: PyPI's `instruction-following-eval` does not
exist and the `ifeval` name belongs to something unrelated, so the three registry modules
are vendored from `google-research/google-research/instruction_following_eval` (Apache
2.0) with their package imports rewritten to flat ones. Scoring follows the official
`evaluation_main.py`: strict is the raw completion, loose retries a fixed set of harmless
transformations (drop the first line, drop the last line, remove emphasis asterisks) and
counts the instruction as followed if any of them passes. Both prompt-level (every
instruction in the prompt followed) and instruction-level accuracy are reported, which is
the pair the published tables quote.

The prompt is the sole user turn, verbatim, with no system prompt, for the same reason as
IFStruct: a benchmark number is only worth having if it is comparable to the published
one.
"""

import json
import random
import zlib

import ife_instructions_registry as registry


def load(path, limit=0):
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            items.append({"id": str(r.get("key")),
                          "messages": [{"role": "user", "content": r["prompt"]}],
                          "tools": None,
                          "spec": {"prompt": r["prompt"],
                                   "instruction_id_list": r.get("instruction_id_list") or [],
                                   "kwargs": r.get("kwargs") or []}})
    # Strided, not the first N: the file is ordered (prompts arrive grouped by instruction family), so a head slice would
    # screen one slice of the benchmark and report it as the benchmark. Deterministic, so a
    # capped pass is still exactly paired across models.
    if limit and len(items) > limit:
        stride = len(items) // limit
        items = items[::stride][:limit]
    return items


def _variants(response):
    r = (response or "").strip()
    lines = [l for l in r.split("\n")]
    out = [r,
           r.replace("*", ""),
           "\n".join(lines[1:]).strip(),
           "\n".join(lines[:-1]).strip(),
           "\n".join(lines[1:-1]).strip()]
    out += [v.replace("*", "") for v in out[2:]]
    seen, uniq = set(), []
    for v in out:
        if v and v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def _follows(spec, response, variants, item_id):
    """Per-instruction booleans for one item.

    A dozen of the 25 instruction classes fall back to `random.choice`/`random.randint`
    when the input row leaves a keyword argument out, so an unseeded run grades the strict
    and loose passes against two different instructions and can score loose below strict.
    The module-level generator is therefore seeded per (item, instruction) from a stable
    checksum, which makes both passes see one description and makes reruns reproducible.
    """
    flags = []
    for i, iid in enumerate(spec["instruction_id_list"]):
        cls = registry.INSTRUCTION_DICT.get(iid)
        if cls is None:
            flags.append(False)
            continue
        try:
            random.seed(zlib.crc32(("%s|%d|%s" % (item_id, i, iid)).encode("utf-8")))
            inst = cls(iid)
            kw = {}
            if i < len(spec["kwargs"]) and isinstance(spec["kwargs"][i], dict):
                # `if v` and not `if v is not None`: this is the official filter, and a
                # falsy argument there means the row did not specify it.
                kw = {k: v for k, v in spec["kwargs"][i].items() if v}
            inst.build_description(**kw)
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=spec["prompt"])
            ok = any(bool(v) and inst.check_following(v) for v in variants)
        except Exception:                                          # noqa: BLE001
            ok = False
        flags.append(ok)
    return flags


def score_item(item, completion):
    spec = item["spec"]
    strict = _follows(spec, completion, [(completion or "").strip()], item["id"])
    loose = _follows(spec, completion, _variants(completion), item["id"])
    detail = {"strict": strict, "loose": loose,
              "instruction_ids": spec["instruction_id_list"]}
    return (bool(strict) and all(strict)), detail


def summarize(per_item):
    n = len(per_item)
    if not n:
        return {"n_items": 0}
    ps = sum(1 for r in per_item if all(r["detail"]["strict"]) and r["detail"]["strict"])
    pl = sum(1 for r in per_item if all(r["detail"]["loose"]) and r["detail"]["loose"])
    istrict = [f for r in per_item for f in r["detail"]["strict"]]
    iloose = [f for r in per_item for f in r["detail"]["loose"]]
    return {"prompt_level_strict": ps / n,
            "prompt_level_loose": pl / n,
            "instruction_level_strict": (sum(istrict) / len(istrict)) if istrict else 0.0,
            "instruction_level_loose": (sum(iloose) / len(iloose)) if iloose else 0.0,
            "n_items": n, "n_instructions": len(istrict)}
