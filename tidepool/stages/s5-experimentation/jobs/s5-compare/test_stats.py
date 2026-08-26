"""The statistics that will decide s5.4, checked against values worked out by hand.

A wrong interval here does not fail loudly: it picks a direction for the rest of the
project and looks like evidence while doing it. So the exact test is checked against
binomial tail probabilities computed independently, the interval against its own analytic
standard error, and the multiple-comparison correction against a worked example.

Run: python3 test_stats.py
"""

import math

import stats

FAILS = []


def check(name, cond, detail=""):
    print("%-4s %s%s" % ("ok" if cond else "FAIL", name, "" if cond else "  <- " + str(detail)))
    if not cond:
        FAILS.append(name)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ------------------------------------------------------- McNemar, against hand values
check("mcnemar: no discordant pairs is p=1", close(stats.mcnemar_exact(0, 0), 1.0))
check("mcnemar: one discordant pair cannot separate anything",
      close(stats.mcnemar_exact(1, 0), 1.0), stats.mcnemar_exact(1, 0))
# 5 discordant pairs all one way: 2 * (1/2)^5 = 0.0625
check("mcnemar: 0 vs 5 is 2*(1/2)^5 = 0.0625",
      close(stats.mcnemar_exact(0, 5), 0.0625), stats.mcnemar_exact(0, 5))
# 2 vs 8: 2 * (C(10,0)+C(10,1)+C(10,2)) / 2^10 = 2 * 56/1024
check("mcnemar: 2 vs 8 is 2*56/1024 = 0.109375",
      close(stats.mcnemar_exact(2, 8), 2 * 56 / 1024.0), stats.mcnemar_exact(2, 8))
check("mcnemar: symmetric in its arguments",
      close(stats.mcnemar_exact(3, 11), stats.mcnemar_exact(11, 3)))
check("mcnemar: never exceeds 1", stats.mcnemar_exact(50, 50) <= 1.0,
      stats.mcnemar_exact(50, 50))
# A large, lopsided split must be tiny but still a real number.
p = stats.mcnemar_exact(5, 60)
check("mcnemar: 5 vs 60 is small and finite", 0.0 < p < 1e-10, p)

# ------------------------------------------------------------- paired counts
ref = {"a": True, "b": True, "c": False, "d": False, "e": True}
cand = {"a": True, "b": False, "c": True, "d": False, "e": True, "z": True}
minus, zero, plus, n = stats.paired_counts(ref, cand)
check("counts: only-reference items", minus == 1, minus)          # b
check("counts: only-candidate items", plus == 1, plus)            # c
check("counts: agreements", zero == 3, zero)                      # a, d, e
check("counts: unmatched ids are dropped", n == 5, n)             # z is not in ref

# ------------------------------------------------- interval, against its own std error
# d_i in {-1,0,+1} with 40 minus and 60 plus out of 1000: point 0.02, and
# var(d) = 100/1000 - 0.02^2, so se = sqrt(var/1000).
point, lo, hi = stats.bootstrap_ci(40, 900, 60, 1000, resamples=20000, seed=7)
se = math.sqrt((100 / 1000.0 - 0.02 ** 2) / 1000.0)
check("interval: point estimate is (plus-minus)/n", close(point, 0.02, 1e-12), point)
check("interval: half-width tracks the analytic 1.96 se (%.4f)" % (1.96 * se),
      abs((hi - lo) / 2.0 - 1.96 * se) < 0.15 * 1.96 * se, (lo, hi))
check("interval: brackets the point estimate", lo < point < hi, (lo, point, hi))
# 0.02 +/- 1.96*0.00998 is [0.0004, 0.0396], so it clears zero by a hair and nothing
# more. Pinning the lower bound just above zero is the check that matters: an interval
# this marginal must not be reported as a comfortable win, and a bug that widened or
# narrowed the interval materially would move this bound off the floor.
check("interval: a 2-point margin at n=1000 clears zero only barely",
      0.0 < lo < 0.005, (lo, hi))

# A wide margin must clear zero.
point2, lo2, hi2 = stats.bootstrap_ci(10, 700, 290, 1000, resamples=20000, seed=7)
check("interval: a 28-point margin clears zero", lo2 > 0.0, (point2, lo2, hi2))

# No discordance at all is a degenerate but legal input.
check("interval: perfect agreement gives a zero-width interval at zero",
      stats.bootstrap_ci(0, 500, 0, 500, resamples=200, seed=1) == (0.0, 0.0, 0.0))
check("interval: an empty intersection does not divide by zero",
      stats.bootstrap_ci(0, 0, 0, 0) == (0.0, 0.0, 0.0))

# Sign symmetry: swapping the two arms must mirror the interval.
pa, la, ha = stats.bootstrap_ci(40, 900, 60, 1000, resamples=8000, seed=3)
pb, lb, hb = stats.bootstrap_ci(60, 900, 40, 1000, resamples=8000, seed=3)
check("interval: swapping arms negates the point estimate", close(pa, -pb, 1e-12), (pa, pb))
check("interval: swapping arms mirrors the bounds to within resampling noise",
      abs(la + hb) < 0.01 and abs(ha + lb) < 0.01, (la, ha, lb, hb))

# Reproducibility: the same seed must give the same interval, or a rerun of the analysis
# reports different bounds for the same data.
check("interval: same seed, same bounds",
      stats.bootstrap_ci(40, 900, 60, 1000, resamples=4000, seed=11)
      == stats.bootstrap_ci(40, 900, 60, 1000, resamples=4000, seed=11))
check("interval: different seed, different bounds",
      stats.bootstrap_ci(40, 900, 60, 1000, resamples=4000, seed=11)
      != stats.bootstrap_ci(40, 900, 60, 1000, resamples=4000, seed=12))

# ------------------------------------------------------------- Holm, worked by hand
# p = [0.01, 0.02, 0.03], m = 3. Sorted: 0.01*3=0.03; 0.02*2=0.04; 0.03*1=0.03 -> 0.04.
adj = stats.holm([0.01, 0.02, 0.03])
check("holm: worked example [0.03, 0.04, 0.04]",
      all(close(a, b, 1e-12) for a, b in zip(adj, [0.03, 0.04, 0.04])), adj)
check("holm: never decreases as raw p increases",
      all(adj[i] <= adj[i + 1] + 1e-12 for i in range(len(adj) - 1)), adj)
check("holm: caps at 1", stats.holm([0.4, 0.5, 0.9]) == [1.0, 1.0, 1.0],
      stats.holm([0.4, 0.5, 0.9]))
check("holm: a single test is unadjusted",
      close(stats.holm([0.023])[0], 0.023, 1e-12))
check("holm: 28 comparisons blunt a nominal 0.03 into nothing",
      stats.holm([0.03] + [0.5] * 27)[0] > 0.5,
      stats.holm([0.03] + [0.5] * 27)[0])

# ------------------------------------------------------------------ binomial helper
rng = __import__("random").Random(0)
check("binom: p=0 draws nothing", stats._binom(rng, 100, 0.0) == 0)
check("binom: p=1 draws everything", stats._binom(rng, 100, 1.0) == 100)
check("binom: stays inside [0, n] on the normal branch",
      all(0 <= stats._binom(rng, 10000, 0.5) <= 10000 for _ in range(200)))
check("binom: small-n branch is exact-support",
      all(0 <= stats._binom(rng, 5, 0.5) <= 5 for _ in range(200)))

print()
if FAILS:
    print("FAILED: %s" % ", ".join(FAILS))
    raise SystemExit(1)
print("the s5.4 statistics hold: the exact test matches binomial tails computed by hand, "
      "the interval matches its own analytic standard error and is reproducible from its "
      "seed, and 28 comparisons against one reference are corrected rather than counted "
      "as 28 independent chances to find something")
