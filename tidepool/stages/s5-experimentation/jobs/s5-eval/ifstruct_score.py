"""IFStruct v1.0, scored by Liquid's own validator.

The benchmark is 2,000 prompts that each specify a structured output in prose — format,
top-level shape, wrapper key, code fence, whether commentary is allowed — plus the JSON
schema the body has to satisfy. `ifstruct_validator.py` is vendored verbatim from
`Liquid4All/ifstruct` (its only dependency is PyYAML) so the number is the benchmark's
own and not our reading of it. The repo's `eval.py` is deliberately not used: it assumes
an OpenAI-compatible endpoint, and generation here goes through the same frozen local
path as every other component.

The data is read from the repo's `data/test.jsonl` rather than the Hugging Face parquet
mirror, because the jsonl keeps `json_schema` as a real object (the parquet stores it as a
JSON-encoded string) and needs no parquet reader in the job image.

The prompt is used verbatim as the sole user turn. No system prompt is added, including
our own structured-output convention from `s4`: the point of a published benchmark is
comparability with the published number, and a system prompt that tells the model to
answer in JSON would flatter every model on it.
"""

import json

import ifstruct_validator as V

FIRST_ATTEMPT = "first_attempt_validity"


def load(path, limit=0):
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            schema = r.get("json_schema")
            if isinstance(schema, str):
                try:
                    schema = json.loads(schema)
                except Exception:                                  # noqa: BLE001
                    schema = None
            items.append({
                "id": "%s#%s" % (r.get("entity_type", "?"), r.get("seed", "?")),
                "messages": [{"role": "user", "content": r["prompt"]}],
                "tools": None,
                "spec": {"json_schema": schema,
                         "top_level_count": r.get("top_level_count"),
                         "require_no_commentary": bool(r.get("require_no_commentary")),
                         "output_format": r.get("output_format") or "json",
                         "top_level_key": r.get("top_level_key"),
                         "require_wrapper_key": bool(r.get("require_wrapper_key")),
                         "require_code_block": bool(r.get("require_code_block"))},
            })
    # Strided, not the first N: the file is ordered (1,000 JSON rows then 1,000 YAML rows), so a head slice would
    # screen one slice of the benchmark and report it as the benchmark. Deterministic, so a
    # capped pass is still exactly paired across models.
    if limit and len(items) > limit:
        stride = len(items) // limit
        items = items[::stride][:limit]
    return items


def score_item(item, completion):
    s = item["spec"]
    try:
        res = V.validate_response(response=completion or "", **s)
        return bool(res.passed), {"score": float(getattr(res, "score", 0.0)),
                                  "errors": list(getattr(res, "errors", []))[:6]}
    except Exception as exc:                                       # noqa: BLE001
        return False, {"score": 0.0, "errors": ["validator raised: %s" % exc]}


def summarize(per_item):
    n = len(per_item)
    passed = sum(1 for r in per_item if r["correct"])
    mean = (sum(r["detail"].get("score", 0.0) for r in per_item) / n) if n else 0.0
    by_fmt = {}
    for r in per_item:
        f = r.get("output_format", "?")
        b = by_fmt.setdefault(f, {"n": 0, "passed": 0})
        b["n"] += 1
        b["passed"] += 1 if r["correct"] else 0
    return {FIRST_ATTEMPT: (passed / n) if n else 0.0,
            "mean_partial_score": mean, "n_items": n,
            "by_format": {k: {"n": v["n"], "validity": v["passed"] / v["n"]}
                          for k, v in sorted(by_fmt.items())}}
