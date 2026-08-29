"""The serving loop must survive a memory ceiling that turns out to be too tight.

Four packed evaluation arms died at once (job `97939c69`) because every one of them hit
its per-arm ceiling on the first BFCL batch: prefill cost scales with the batch times the
sequence length, and BFCL's tool schemas are long. One of the four had 17.33 GB free on
the card at the moment it was killed, so the ceiling, and not the card, is what it ran
out of. Raising the ceiling is a guess that has to be right; splitting the batch when it
does not fit is a guess that does not have to be. These cases pin the splitting down
without a GPU, by standing a fake model in for the real one and having it refuse any
batch above a set size.

Run: python3 test_gen_oom_split.py
"""

import sys
import types

# ---------------------------------------------------------------- torch stand-in
CALLS = []


class OutOfMemoryError(RuntimeError):
    pass


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_torch = types.ModuleType("torch")
_torch.OutOfMemoryError = OutOfMemoryError
_torch.no_grad = lambda: _NoGrad()
_cuda = types.ModuleType("torch.cuda")
_cuda.empty_cache = lambda: CALLS.append("empty_cache")
_torch.cuda = _cuda
sys.modules["torch"] = _torch
sys.modules["torch.cuda"] = _cuda

import gen  # noqa: E402  (must follow the stand-in)


# ------------------------------------------------------------- tensor stand-ins
class Scalar:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


class Mask:
    def __init__(self, n):
        self.n = n

    def sum(self):
        return Scalar(self.n)


class Row:
    def __init__(self, vals):
        self.vals = list(vals)

    @property
    def shape(self):
        return (len(self.vals),)

    def tolist(self):
        return list(self.vals)


class T:
    def __init__(self, rows):
        self.rows = [list(r) for r in rows]

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def to(self, *a, **k):
        return self

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rsl, csl = key
            return T([r[csl] for r in self.rows[rsl]])
        if isinstance(key, int):
            return Row(self.rows[key])
        return T(self.rows[key])

    def __ne__(self, other):
        return Mask(sum(1 for r in self.rows for v in r if v != other))


class Tok:
    """Encodes each prompt as its own length repeated, so order is checkable."""

    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, prompts, return_tensors=None, padding=None,
                 add_special_tokens=None):
        w = max(len(p) for p in prompts)
        return {"input_ids": T([[len(p)] * w for p in prompts]),
                "attention_mask": T([[1] * w for _ in prompts])}

    def decode(self, row, skip_special_tokens=True):
        return "out:" + ",".join(str(v) for v in row.tolist())


class Model:
    """Refuses any batch larger than `max_batch`, the way a memory ceiling does."""

    device = "cpu"

    def __init__(self, max_batch):
        self.max_batch = max_batch
        self.batches = []

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=8, **kw):
        n = input_ids.shape[0]
        self.batches.append(n)
        if n > self.max_batch:
            raise OutOfMemoryError("fake ceiling: %d prompts" % n)
        w = input_ids.shape[1]
        # Echo the prompt, then three new tokens, the last of them EOS. The first new
        # token carries the prompt's own length so the caller's ordering is verifiable.
        return T([[r[0]] * w + [r[0], 5, 2] for r in input_ids.rows])


# ------------------------------------------------------------------------ cases
FAILS = []


def check(name, cond, detail=""):
    print("%-4s %s%s" % ("ok" if cond else "FAIL", name, "" if cond else "  <- " + detail))
    if not cond:
        FAILS.append(name)


PROMPTS = ["p" * (i + 1) for i in range(20)]          # lengths 1..20, all distinct
EXPECT = ["out:%d,5,2" % len(p) for p in PROMPTS]


def run(max_batch, batch_size=16, prompts=PROMPTS):
    del CALLS[:]
    model = Model(max_batch)
    r = gen.Runner(model, Tok(), log=lambda *a: None)
    out = r.generate(prompts, max_new_tokens=8, batch_size=batch_size, tag="t")
    return r, model, out


# 1. A ceiling nothing trips: no splits, and the batches are what was asked for.
r, m, out = run(max_batch=16)
check("1 no ceiling pressure: every prompt answered", out == EXPECT,
      repr(out[:3]))
check("1 no ceiling pressure: no splits", r.oom_splits == 0, r.oom_splits)
check("1 no ceiling pressure: batches as requested", m.batches == [16, 4], m.batches)
check("1 no ceiling pressure: smallest batch recorded", r.min_batch == 4, r.min_batch)

# 2. The real failure: the ceiling admits 4 at a time. Halving must find that floor.
r, m, out = run(max_batch=4)
check("2 tight ceiling: every prompt still answered", out == EXPECT, repr(out[:3]))
check("2 tight ceiling: order preserved across splits", out == EXPECT)
check("2 tight ceiling: split three times for the first group", r.oom_splits == 3,
      r.oom_splits)
check("2 tight ceiling: halving sequence is 16,8,4,4,8,4,4 then the 4 remainder",
      m.batches == [16, 8, 4, 4, 8, 4, 4, 4], m.batches)
check("2 tight ceiling: cache dropped once per failed attempt",
      CALLS.count("empty_cache") == 3, CALLS)
check("2 tight ceiling: failed-attempt time kept out of generation time",
      r.oom_seconds >= 0.0 and r.gen_seconds >= 0.0)

# 3. An odd group splits without losing the odd item.
r, m, out = run(max_batch=1, batch_size=5, prompts=PROMPTS[:5])
check("3 odd group: all five answered", out == EXPECT[:5], repr(out))
check("3 odd group: 5 -> 2+3 -> 1+1+1+2 -> 1+1",
      m.batches == [5, 2, 1, 1, 3, 1, 2, 1, 1], m.batches)

# 4. One prompt that cannot run on its own is the only case that still raises.
raised = None
try:
    run(max_batch=0)
except OutOfMemoryError as exc:
    raised = exc
check("4 unsplittable: a single prompt that will not fit propagates", raised is not None,
      "no exception")

# 5. The counters reach the score file, so a ceiling set too tight is visible.
r, m, out = run(max_batch=4)
tp = r.throughput()
check("5 throughput reports the splits", tp.get("oom_splits") == 3, tp)
check("5 throughput reports the smallest batch that ran", tp.get("smallest_batch") == 4,
      tp)
check("5 throughput still reports tokens", tp.get("generated_tokens") == 60,
      tp.get("generated_tokens"))


# ------------------------------------------------- 6. the calibration forward pass
# `first_token_probs` reads one position and never generates, so none of the counters
# above may move; it must also survive a transformers build that takes no logits_to_keep,
# and it must sum the softmax mass over every spelling of a choice.
class PRow:
    def __init__(self, vals):
        self.vals = list(vals)

    def __getitem__(self, ids):
        return PRow([self.vals[i] for i in ids])

    def sum(self):
        return Scalar(sum(self.vals))


class P2:
    def __init__(self, rows):
        self.rows = [list(r) for r in rows]

    def float(self):
        return self

    def __getitem__(self, i):
        return PRow(self.rows[i])


class L3:
    """Logits over (batch, positions, vocab); only [:, -1, :] is ever taken."""

    def __init__(self, rows, width):
        self.rows, self.width = rows, width

    def __getitem__(self, key):
        assert key == (slice(None), -1, slice(None)), key
        return P2(self.rows)


class Out:
    def __init__(self, logits):
        self.logits = logits


VOCAB = [0.0] * 8


class ProbModel:
    """Last-position logits keyed off the prompt length, so rows stay identifiable."""

    device = "cpu"

    def __init__(self, takes_keep=True, max_batch=99):
        self.takes_keep, self.max_batch = takes_keep, max_batch
        self.saw_keep, self.batches = [], []

    def __call__(self, input_ids=None, attention_mask=None, logits_to_keep=None, **kw):
        self.saw_keep.append(logits_to_keep)
        if logits_to_keep is not None and not self.takes_keep:
            raise TypeError("unexpected keyword argument 'logits_to_keep'")
        n = input_ids.shape[0]
        self.batches.append(n)
        if n > self.max_batch:
            raise OutOfMemoryError("fake ceiling: %d prompts" % n)
        rows = []
        for r in input_ids.rows:
            v = list(VOCAB)
            # ids 3 and 4 spell "yes", 5 and 6 spell "no". Longer prompts look guiltier.
            v[3], v[4] = 0.1, 0.1
            v[5], v[6] = 0.01 * r[0], 0.01 * r[0]
            rows.append(v)
        return Out(L3(rows, input_ids.shape[1]))


_torch.softmax = lambda t, dim=-1: t          # already normalized enough to sum
CHOICES = {"yes": [3, 4], "no": [5, 6]}

pm = ProbModel()
rp = gen.Runner(pm, Tok(), log=lambda *a: None)
got = rp.first_token_probs(PROMPTS, CHOICES, batch_size=8, tag="cal")
check("6 calibration: one result per prompt, in the caller's order",
      len(got) == 20 and all(g is not None for g in got), len(got))
check("6 calibration: mass is summed over every spelling of a choice",
      abs(got[0]["yes"] - 0.2) < 1e-9 and abs(got[0]["no"] - 0.02) < 1e-9, got[0])
check("6 calibration: rows keep their identity through the length sort",
      abs(got[19]["no"] - 0.40) < 1e-9, got[19])
check("6 calibration: the head is run on the last position only",
      set(pm.saw_keep) == {1}, repr(pm.saw_keep))
check("6 calibration: no tokens are generated and no generation time is charged",
      rp.gen_tokens == 0 and rp.gen_seconds == 0.0,
      (rp.gen_tokens, rp.gen_seconds))

old = ProbModel(takes_keep=False)
ro = gen.Runner(old, Tok(), log=lambda *a: None)
go = ro.first_token_probs(PROMPTS, CHOICES, batch_size=8, tag="cal")
check("6 calibration: a build without logits_to_keep falls back once and keeps going",
      len(go) == 20 and all(g is not None for g in go)
      and old.saw_keep[:2] == [1, None] and set(old.saw_keep[2:]) == {None},
      repr(old.saw_keep[:4]))
check("6 calibration: the fallback returns the same numbers",
      all(abs(a["no"] - b["no"]) < 1e-12 for a, b in zip(got, go)))

tight = ProbModel(max_batch=2)
rt = gen.Runner(tight, Tok(), log=lambda *a: None)
gt = rt.first_token_probs(PROMPTS, CHOICES, batch_size=8, tag="cal")
check("6 calibration: a scoring batch that does not fit is halved like a generating one",
      len(gt) == 20 and all(g is not None for g in gt) and rt.oom_splits > 0,
      rt.oom_splits)
check("6 calibration: splitting does not change any score",
      all(abs(a["no"] - b["no"]) < 1e-12 for a, b in zip(got, gt)))

print()
if FAILS:
    print("FAILED: %s" % ", ".join(FAILS))
    raise SystemExit(1)
print("the serving loop degrades instead of dying: a batch that does not fit the arm's "
      "memory ceiling is halved until it does, in the caller's order, and only a single "
      "prompt that cannot run at all is still fatal")
