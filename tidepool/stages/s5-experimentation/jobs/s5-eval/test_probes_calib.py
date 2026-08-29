"""Contract tests for the guardrail calibration scalar. No model, no GPU, fixtures only."""

import probes_calib as pc

N, FAILED = 0, []


def ok(name, cond, detail=""):
    global N
    N += 1
    print(("ok   " if cond else "FAIL ") + name + ("" if cond else "  <- %s" % (detail,)))
    if not cond:
        FAILED.append(name)


# ---- AUROC. Rank-based, so ties must land at exactly 0.5 and not at a trapezoid's guess.
ok("a perfectly separating score scores 1.0", pc.auroc([0.9, 0.8], [0.2, 0.1]) == 1.0)
ok("a perfectly inverted score scores 0.0", pc.auroc([0.1, 0.2], [0.8, 0.9]) == 0.0)
ok("an all-ties score scores exactly 0.5", pc.auroc([0.5] * 4, [0.5] * 6) == 0.5,
   pc.auroc([0.5] * 4, [0.5] * 6))
# 4 pairs: (.9,.5) (.9,.1) (.5,.1) all won, (.5,.5) tied -> 3.5/4.
ok("a single tied pair among four counts as half a win",
   pc.auroc([0.5, 0.9], [0.5, 0.1]) == 0.875, pc.auroc([0.5, 0.9], [0.5, 0.1]))
ok("an empty side has no AUROC rather than a default",
   pc.auroc([], [0.1]) is None and pc.auroc([0.1], []) is None)

# ---- Operating points. The whole s6.3 question is whether a threshold exists, so a budget
# that cannot be met must report None and not the best-effort point that busts it.
pos = [0.9, 0.8, 0.7, 0.6]
neg = [0.5, 0.4, 0.3, 0.2]
rows = {r["false_alarm_budget"]: r for r in pc.operating_points(pos, neg)}
ok("a separable score detects everything inside every budget",
   all(rows[b]["detection_rate"] == 1.0 for b in rows), rows)
ok("the reported false-alarm rate never exceeds its budget",
   all(r["false_alarm_rate"] <= r["false_alarm_budget"] for r in rows.values()), rows)
tight = pc.operating_points([0.4], [0.9, 0.9, 0.9], budgets=(0.0,))[0]
ok("when no useful threshold fits the budget, flag-nothing is reported, not None",
   tight["detection_rate"] == 0.0 and tight["false_alarm_rate"] == 0.0
   and tight["threshold"] is not None, tight)
overlap = pc.operating_points([0.6, 0.4], [0.5, 0.3], budgets=(0.0, 0.5))
ok("a wider budget can only help",
   overlap[1]["detection_rate"] >= overlap[0]["detection_rate"], overlap)

# ---- ECE. A ranking can be perfect while the number is meaningless; the two must not be
# collapsed into one figure, which is exactly the failure this asserts against.
sc = [0.99, 0.98, 0.97, 0.96]
e, bins = pc.ece(sc, [True, True, False, False])
ok("a confidently wrong-half score has large ECE despite perfect confidence",
   e is not None and e > 0.45, e)
ok("its bins carry the counts they were computed from",
   sum(b["n"] for b in bins) == 4, bins)
e2, _ = pc.ece([0.0, 0.0, 1.0, 1.0], [False, False, True, True])
ok("a perfectly calibrated score has zero ECE", e2 == 0.0, e2)
ok("a score of exactly 1.0 is not dropped off the top bin",
   pc.ece([1.0], [True])[1][0]["n"] == 1, pc.ece([1.0], [True]))

# ---- The scalar itself.
ok("suspicion is the no-share of the two words",
   abs(pc.suspicion({"yes": 0.25, "no": 0.75}) - 0.75) < 1e-9)
ok("suspicion normalises away the total mass on the two words",
   abs(pc.suspicion({"yes": 0.025, "no": 0.075})
       - pc.suspicion({"yes": 0.25, "no": 0.75})) < 1e-9)
ok("an item with no mass on either word has no verdict, not 0.5",
   pc.suspicion({"yes": 0.0, "no": 0.0}) is None)

# ---- Token resolution. A vocabulary where the two words share a first token would make
# every score meaningless, so the overlap has to come back rather than be tolerated.
class Tok:
    def __init__(self, table):
        self.table = table

    def encode(self, w, add_special_tokens=False):
        return self.table.get(w, [])


good = Tok({w: [10 + i] for i, w in enumerate(pc.YES_WORDS)}
           | {w: [50 + i] for i, w in enumerate(pc.NO_WORDS)})
ch, res = pc.choice_ids(good)
ok("every spelling contributes its first token", len(ch["yes"]) == len(pc.YES_WORDS)
   and len(ch["no"]) == len(pc.NO_WORDS), (ch, res))
ok("a clean vocabulary reports no overlap", res["overlapping_token_ids"] == [], res)
bad = Tok({w: [7] for w in pc.YES_WORDS + pc.NO_WORDS})
ok("a shared first token is reported, not swallowed",
   pc.choice_ids(bad)[1]["overlapping_token_ids"] == [7], pc.choice_ids(bad)[1])

# ---- summarize(). The end-to-end shape a run reports.
rows = ([{"id": "c%d" % i, "arm": "corrupted", "mode": "truncated_json", "score": 0.9}
         for i in range(10)]
        + [{"id": "s%d" % i, "arm": "contradicted", "mode": "stale_as_of", "score": 0.1}
           for i in range(10)]
        + [{"id": "n%d" % i, "arm": "clean_corpus", "mode": "intact", "score": 0.1}
           for i in range(10)]
        + [{"id": "x", "arm": "corrupted", "mode": "truncated_json", "score": None}])
out = pc.summarize(rows)
ok("an unreadable item is counted and left out of every statistic",
   out["n_no_verdict"] == 1 and out["n_positive"] == 20, out["n_no_verdict"])
ok("a mode indistinguishable from the clean arm scores at chance, not at the pooled figure",
   out["by_mode"]["truncated_json"]["auroc"] == 1.0
   and out["by_mode"]["stale_as_of"]["auroc"] == 0.5,
   {k: v["auroc"] for k, v in out["by_mode"].items()})
ok("the pooled AUROC sits between the two modes it pools",
   out["by_mode"]["stale_as_of"]["auroc"] < out["auroc"] < 1.0, out["auroc"])
ok("the frozen auditor turn travels with the result",
   out["audit_sha"] == pc.audit_sha() and len(out["audit_sha"]) == 12, out["audit_sha"])
ff = pc.summarize(rows, {"c%d" % i: True for i in range(10)}
                  | {"s%d" % i: False for i in range(10)}
                  | {"n%d" % i: False for i in range(10)})
ok("the scalar is also ranked against the model's own free-form flags",
   ff["auroc_vs_own_free_form_flag"] == 1.0 and ff["n_free_form_flagged"] == 10,
   ff.get("auroc_vs_own_free_form_flag"))
ok("clean items are negatives for AUROC and the frozen 30 are excluded from it",
   "clean" not in pc.NEGATIVE_ARMS and pc.NEGATIVE_ARMS == ("clean_corpus",),
   pc.NEGATIVE_ARMS)

# ---------------------------------------------------------------- by_arm
# The synthetic clean arm and the text-only arm are scored so their distributions can be
# compared against the corpus negatives, and they must not leak into the rank statistic.
mixed = rows[:-1] + [{"id": "y%d" % i, "arm": "clean", "mode": "intact", "score": 0.95}
                     for i in range(10)]
mx = pc.summarize(mixed)
plain = pc.summarize(rows[:-1])
ok("a scored-only arm is reported with its own role and quartiles",
   mx["by_arm"]["clean"]["role"] == "scored_only" and mx["by_arm"]["clean"]["n"] == 10
   and mx["by_arm"]["clean"]["median"] == 0.95, mx["by_arm"].get("clean"))
ok("adding a scored-only arm changes no statistic the negatives define",
   mx["auroc"] == plain["auroc"] and mx["n_negative"] == plain["n_negative"]
   and mx["operating_points"] == plain["operating_points"],
   (mx["auroc"], plain["auroc"]))
ok("every arm carries the role its scores were used under",
   {a: v["role"] for a, v in mx["by_arm"].items()}
   == {"corrupted": "positive", "contradicted": "positive",
       "clean_corpus": "negative", "clean": "scored_only"},
   {a: v["role"] for a, v in mx["by_arm"].items()})

print("\n%d checks, %d failed" % (N, len(FAILED)))
raise SystemExit(1 if FAILED else 0)
