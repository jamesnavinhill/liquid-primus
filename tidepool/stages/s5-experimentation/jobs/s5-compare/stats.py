"""Paired comparison of two arms over the same items, and nothing more.

The sweep's own selection metric failed: three of its four finished arms sit within 0.0006
of each other on validation loss at one seed apiece. So s5.4 has to be decided on task
scores, and the differences it will be reading are small enough that the honest question is
not "which number is bigger" but "is this difference distinguishable from zero at all".

Two things make that answerable here that would not be answerable from summary rates alone.
Every component writes a per-item `{"id", "correct"}` row, so two arms can be joined item by
item; and every arm sees the same items in the same order, because item selection is a
deterministic function of the corpus and the caps. A paired test over matched items removes
the variance of item difficulty, which is the dominant term when two arms agree on most
items and differ on a few dozen.

The paired difference per item is d_i in {-1, 0, +1}, so the whole joint distribution is
three counts: items only the reference got, items both got or both missed, items only the
challenger got. Bootstrapping the mean of d is therefore a multinomial draw over those three
counts rather than a resample of tens of thousands of rows, which is exact and costs nothing.
McNemar's exact test is the matching significance statement: conditional on a pair being
discordant, the challenger wins it with probability 1/2 under the null.
"""

import math
import random


def paired_counts(ref, cand):
    """(only ref correct, both agree, only cand correct, n matched) over shared ids.

    Both arguments map item id -> bool. Ids missing from either side are dropped and
    counted by the caller, because a component one arm scored and the other did not is a
    harness problem to report and not a difference to measure.
    """
    shared = set(ref) & set(cand)
    minus = sum(1 for i in shared if ref[i] and not cand[i])
    plus = sum(1 for i in shared if cand[i] and not ref[i])
    return minus, len(shared) - minus - plus, plus, len(shared)


def mcnemar_exact(minus, plus):
    """Two-sided exact p for a paired binary comparison. 1.0 when nothing is discordant.

    Computed in log space through lgamma so a few thousand discordant pairs does not turn
    into big-integer arithmetic. The one-sided tail is summed on the smaller side and
    doubled, which is the standard two-sided convention for a symmetric null.
    """
    n = minus + plus
    if n == 0:
        return 1.0
    k = min(minus, plus)
    log_half_n = -n * math.log(2.0)
    tail = 0.0
    for i in range(k + 1):
        log_c = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1))
        tail += math.exp(log_c + log_half_n)
    return min(1.0, 2.0 * tail)


def bootstrap_ci(minus, zero, plus, n, resamples=10000, seed=0, level=0.95):
    """Percentile CI for the paired accuracy difference, cand minus ref.

    The statistic is (plus - minus) / n, so a resample only needs the three category
    counts. Drawn with the stdlib rather than numpy: three multinomial cells at ten
    thousand resamples is a few million operations, and one fewer dependency in a job whose
    whole purpose is to be reproducible.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    point = (plus - minus) / n
    rng = random.Random(seed)
    p_minus, p_plus = minus / n, plus / n
    diffs = []
    for _ in range(resamples):
        # Two independent binomials conditioned on n is not the same as a 3-cell
        # multinomial, so the minus cell is drawn first and the plus cell is drawn from
        # what is left, which is exactly the multinomial factorization.
        b_minus = _binom(rng, n, p_minus)
        rest = n - b_minus
        p_rest = 0.0 if p_minus >= 1.0 else min(1.0, p_plus / (1.0 - p_minus))
        b_plus = _binom(rng, rest, p_rest)
        diffs.append((b_plus - b_minus) / n)
    diffs.sort()
    lo = diffs[int((1.0 - level) / 2.0 * resamples)]
    hi = diffs[min(resamples - 1, int((1.0 + level) / 2.0 * resamples))]
    return point, lo, hi


def _binom(rng, n, p):
    """Binomial draw. Exact by summation for small n, normal-approximated above it.

    The cut is at n*p*(1-p) >= 9, where the normal approximation to a binomial is
    conventionally considered safe, and the result is clamped into [0, n] because a
    Gaussian tail can land outside the support.
    """
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    if n * p * (1.0 - p) < 9.0:
        return sum(1 for _ in range(n) if rng.random() < p)
    v = rng.gauss(n * p, math.sqrt(n * p * (1.0 - p)))
    return int(max(0, min(n, round(v))))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in the caller's order.

    Twenty-eight arm-by-component comparisons against one reference will produce a
    nominally significant result by chance about once even if every arm is identical, and
    the whole point of this pass is that the effects being looked for are small. Holm is
    used rather than Bonferroni because it is uniformly more powerful and needs no extra
    assumption, and rather than Benjamini-Hochberg because the decision at s5.4 picks a
    single direction and wants family-wise control, not a controlled false-discovery share.
    """
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (len(pvals) - rank) * pvals[i])
        running = max(running, adj)          # enforce monotonicity
        out[i] = running
    return out
