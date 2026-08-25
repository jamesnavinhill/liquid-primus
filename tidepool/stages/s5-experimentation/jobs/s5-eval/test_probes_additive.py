"""Proves the enlarged clean arm cannot move a number that has already been published.

B1 through B6 are scored against the frozen 30-item clean arm. The 138-item corpus arm built
at s5.3 is what makes the plan's 0.15 false-alarm ceiling measurable, and adding it must not
change what any pre-existing summary key means, or the baseline table stops being comparable
to everything measured after it.

Run: python test_probes_additive.py
"""

import sys

import probes_score as ps

PRE = ["flag_rate_malformed", "flag_rate_contradicted", "flag_rate_corrupted",
       "false_flag_rate_clean", "clean_answer_rate", "no_fabrication_rate",
       "stack_idiom_accuracy", "depth"]


def row(arm, flagged, correct=True, present=True, probe="tool_return", depth=1):
    return {"id": "x", "probe": probe, "arm": arm, "mode": "intact", "depth": depth,
            "correct": correct,
            "detail": {"kind": "clean", "flagged": flagged, "value_present": present}}


base = ([row("clean", False, depth=1 + i % 3) for i in range(29)] + [row("clean", True)] +
        [row("corrupted", True, depth=1 + i % 3) for i in range(20)] +
        [row("contradicted", False, correct=False, depth=1 + i % 3) for i in range(10)])
extra = ([row("clean_corpus", False, depth=1 + i % 3) for i in range(130)] +
         [row("clean_corpus", True, depth=2) for _ in range(8)])

without = ps.summarize(base)
with_arm = ps.summarize(base + extra)
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
    print(("  ok   " if cond else "  FAIL ") + msg)


drift = [k for k in PRE if without[k] != with_arm[k]]
check(not drift, "no pre-existing summary key moves when the arm is added (%s)"
      % (", ".join(drift) if drift else "none moved"))
check(with_arm["n_clean_frozen"] == 30, "the frozen arm still reports 30 items")
check(with_arm["n_clean_corpus"] == 138, "the corpus arm reports its 138 items separately")
check(abs(with_arm["false_flag_rate_clean_corpus"] - 8 / 138) < 1e-9,
      "the corpus false-flag rate is computed over the corpus arm alone")
check(abs(with_arm["false_flag_rate_all_clean"] - 9 / 168) < 1e-9,
      "the combined rate pools both arms")
check("tool_return/clean_corpus" in with_arm["arms"],
      "the corpus arm appears in the per-arm breakdown under its own name")
check(without["false_flag_rate_clean_corpus"] is None and without["n_clean_corpus"] == 0
      and without["depth_clean_corpus"] == {},
      "the new keys read empty when the arm is not configured, so an unconfigured run is "
      "byte-comparable to the rows already measured")
check(sum(v["n"] for v in with_arm["depth"].values())
      == sum(v["n"] for v in without["depth"].values()),
      "the envelope-depth breakdown stays scoped to the original arms")
check(sum(v["n"] for v in with_arm["depth_clean_corpus"].values()) == 138,
      "the corpus arm gets its own depth breakdown")

print()
if fails:
    print("%d FAILURES" % len(fails))
    sys.exit(1)
print("all checks passed")
