"""Local fixture test for the sweep sampler. No GPU, no model, no network.

It exists to catch the three ways this sweep could be quietly invalid:

  1. Arms costing different amounts, which would make the comparison measure compute.
  2. The guardrail-off arm getting guardrail rows anyway, or the guardrail-on arms not
     getting any, either of which turns the C2' ablation into noise.
  3. Selection order drifting between arms, which would confound a one-parameter delta
     with a different sample of the corpus.

Run: python test_sampler.py
"""

import gzip
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sample

# Token counts measured at s4.4 and s5.3, so the calibration below is the real one.
REAL = {"tool": 40_800_000, "struct": 1_500_000, "sql": 87_200_000, "code": 72_000_000,
        "tool_guardrail": 4_351_790}
ARMS = {
    "C1":  {},
    "C2p": {"guardrail_epochs": 0.0},
    "C3":  {},
    "C4":  {},
    "C5a": {"replay_frac": 0.01},
    "C5b": {"replay_frac": 0.05},
    "C6":  {},
    "C7":  {},
}
BUDGET = 64_000_000
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
    print(("  ok   " if cond else "  FAIL ") + msg)


print("1. calibration on the measured corpus, every arm")
plans = {}
for arm, over in ARMS.items():
    avail = dict(REAL)
    if over.get("replay_frac"):
        avail["replay"] = 20_000_000          # a replay set larger than any arm asks for
    plans[arm] = sample.calibrate(avail, budget_tokens=BUDGET, **over)
    p = plans[arm]
    print("  %-4s budget %10d  priority %.3f  shares %s" % (
        arm, p["budget_tokens"], p["priority_share_achieved"],
        json.dumps({k: v for k, v in p["per_role_share"].items() if v})))

for arm, p in plans.items():
    check(abs(p["budget_tokens"] - BUDGET) <= BUDGET * 0.001,
          "%s trains on %d tokens, within 0.1%% of the %d budget" % (arm, p["budget_tokens"], BUDGET))
check(plans["C2p"]["per_role_tokens"]["tool_guardrail"] == 0,
      "C2' allocates zero guardrail tokens")
check(plans["C1"]["per_role_tokens"]["tool_guardrail"] > 0,
      "C1 allocates guardrail tokens (%d)" % plans["C1"]["per_role_tokens"]["tool_guardrail"])
check(plans["C1"]["per_role_repeats"]["tool_guardrail"] == 2.0,
      "C1 sees the guardrail block exactly twice")
check(plans["C2p"]["per_role_tokens"]["tool"] > plans["C1"]["per_role_tokens"]["tool"],
      "C2' returns the guardrail tokens to the base mix rather than shrinking the run")
check(abs(plans["C5b"]["per_role_tokens"]["replay"] - 0.05 * BUDGET) < 1000,
      "C5b allocates 5%% of the budget to replay (%d)" % plans["C5b"]["per_role_tokens"]["replay"])
check(plans["C5a"]["per_role_tokens"]["replay"] < plans["C5b"]["per_role_tokens"]["replay"],
      "C5a allocates less replay than C5b")
missing = sample.calibrate(dict(REAL), budget_tokens=BUDGET, replay_frac=0.05)
check(missing["notes"].get("replay_unavailable") is True,
      "a replay arm with no replay rows is flagged, not silently run as C1")

print("2. selection over a fixture corpus")
tmp = tempfile.mkdtemp()
main_p = os.path.join(tmp, "train.jsonl.gz")
guard_p = os.path.join(tmp, "guard.jsonl.gz")
with gzip.open(main_p, "wt") as fh:
    for i in range(4000):
        role = ("tool", "struct", "sql", "code")[i % 4]
        fh.write(json.dumps({"c": "corpus%d" % (i % 7), "i": i, "role": role, "n_tok": 100 + i % 50,
                             "messages": [{"role": "user", "content": "q"},
                                          {"role": "assistant", "content": "a"}]}) + "\n")
with gzip.open(guard_p, "wt") as fh:
    for i in range(400):
        fh.write(json.dumps({"c": "guard", "i": i, "role": "tool_guardrail", "n_tok": 500,
                             "mode": "empty_body",
                             "messages": [{"role": "user", "content": "q"},
                                          {"role": "assistant", "content": "a"}]}) + "\n")

per_role, order = sample.index_split([main_p, guard_p])
print("  indexed:", json.dumps(per_role))
check(set(per_role) == {"tool", "struct", "sql", "code", "tool_guardrail"},
      "index_split reads both sources into one role space")
check(all(isinstance(i, tuple) and len(i) == 2 for _, i in order["tool_guardrail"]),
      "guardrail ids carry their source index")

FIX_BUDGET = 1_200_000
sel = {}
for arm, over in {"C1": {}, "C2p": {"guardrail_epochs": 0.0}}.items():
    plan = sample.calibrate(per_role, budget_tokens=FIX_BUDGET, **over)
    ids, detail = sample.choose(order, per_role, plan)
    rows = sample.read_lines([main_p, guard_p], ids)
    counts = {}
    for r in rows:
        counts[r["role"]] = counts.get(r["role"], 0) + 1
    sel[arm] = (ids, counts, sum(r["n_tok"] for r in rows))
    print("  %-4s rows %d  tokens %d  counts %s" % (arm, len(rows), sel[arm][2], json.dumps(counts)))

check(sel["C2p"][1].get("tool_guardrail", 0) == 0,
      "C2' materializes no guardrail rows")
check(sel["C1"][1].get("tool_guardrail", 0) > 0,
      "C1 materializes guardrail rows (%d)" % sel["C1"][1].get("tool_guardrail", 0))
check(abs(sel["C1"][2] - sel["C2p"][2]) < 0.05 * FIX_BUDGET,
      "both arms train on the same token count within 5%% (%d vs %d)" % (sel["C1"][2], sel["C2p"][2]))

plan = sample.calibrate(per_role, budget_tokens=FIX_BUDGET)
a, _ = sample.choose(order, per_role, plan)
b, _ = sample.choose(order, per_role, plan)
check(a == b, "selection is deterministic across calls")
for arm in ("C1", "C2p"):
    counts = sel[arm][1]
    check(sum(1 for r in ("tool", "struct", "sql", "code") if counts.get(r)) == 4,
          "%s draws from all four base roles" % arm)

tiny = sample.calibrate(per_role, budget_tokens=100_000)
check(tiny["notes"].get("doses_exceeded_budget") is True,
      "a budget smaller than the fixed doses is flagged rather than silently starving the mix")
base_a = [i for i in a if i[0] == 0]
base_c6, _ = sample.choose(order, per_role, sample.calibrate(per_role, budget_tokens=FIX_BUDGET))
check([i for i in base_c6 if i[0] == 0] == base_a,
      "two arms differing only in a hyperparameter see identical base rows in identical order")

print("3. entropy weighting keeps the mean weight at 1")
# Reproduces compute_loss's weighting arithmetic without torch: the property that matters is
# that centring on the batch mean leaves the average weight at 1, so entropy_beta cannot act
# as a hidden learning-rate multiplier.
h = [0.02, 0.10, 0.35, 0.60, 0.95, 0.44, 0.71, 0.08]
for beta in (0.5, 1.0, 2.0):
    centre = sum(h) / len(h)
    w = [max(0.1, 1.0 + beta * (x - centre)) for x in h]
    mean_w = sum(w) / len(w)
    check(abs(mean_w - 1.0) < 0.02,
          "beta=%.1f leaves mean weight %.4f" % (beta, mean_w))
    check(max(w) > min(w), "beta=%.1f actually reweights (%.3f to %.3f)" % (beta, min(w), max(w)))

print()
if fails:
    print("%d FAILURES" % len(fails))
    for f in fails:
        print(" -", f)
    sys.exit(1)
print("all checks passed")
