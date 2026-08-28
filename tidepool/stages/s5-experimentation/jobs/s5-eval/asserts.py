"""The harness's assertion list, split out of `main.py` so it can be tested without pulling
in `gen.py`'s hard `import torch` (main.py -> gen -> torch, unavailable in a CPU-only shell).

`check` records item-count assertions: how many prompts went in. `check_completions` records
what came back. The distinction is the whole reason this file exists: attempt 2 of s5.6's
GGUF quality pack (job 46bd54cb, 2026-08-28) packed three llama-servers onto one L4, the
concurrent compute contention stalled requests to two of the three arms past the client's
patience, and `gen_gguf.py` folded every timeout into a silent empty-string completion with
no retry and no abort. Both arms finished with correct item counts and
`completion_status: success` -- every assertion that existed passed, on a run where
ifstruct, probes and bfcl_native_tools were 100% empty. `check_completions` closes that gap.
"""

ASSERTS = []

EMPTY_COMPLETION_MAX_RATE = 0.02

_log = print


def set_logger(fn):
    """main.py wires its own `log()` in here so a failed assertion still lands on the job's
    console/log stream instead of just stdout."""
    global _log
    _log = fn


def check(name, ok, detail=""):
    ASSERTS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:300]})
    if not ok:
        _log("ASSERTION FAILED %s: %s" % (name, detail))
    return bool(ok)


def check_completions(tag, outs):
    n = len(outs)
    empty = sum(1 for o in outs if not (o or "").strip())
    rate = (empty / n) if n else 0.0
    return check("%s_completions_nonempty" % tag, rate <= EMPTY_COMPLETION_MAX_RATE,
                 "%d/%d completions empty (%.1f%%), max allowed %.0f%%"
                 % (empty, n, 100.0 * rate, 100.0 * EMPTY_COMPLETION_MAX_RATE))
