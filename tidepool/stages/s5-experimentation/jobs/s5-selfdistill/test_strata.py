"""The stratified prompt sampler, checked against a fixture pool.

Two properties carry the whole s5.5 design and neither is obvious from reading the code:
the default has to reproduce the unstratified sample byte for byte, or the frozen s5.3
buffer stops being reproducible from this file; and a stratum has to be filled from its own
hash order, or the reweighted buffer is not a superset of anything and two runs at different
N stop being comparable.

Runs the sampler block out of `main.py` rather than reimplementing it, so the test cannot
drift away from the code it is testing.
"""
import gzip
import hashlib
import json
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "main.py"), encoding="utf-8").read()
START = SRC.index('SALT = str(C("sample_salt"')
END = SRC.index("from transformers import")
BLOCK = SRC[START:END]

FAILED = []
RAN = []


def ok(name, cond, detail=""):
    RAN.append(name)
    print("%-4s %s" % ("ok" if cond else "FAIL", name), "" if cond else detail)
    if not cond:
        FAILED.append(name)


def pool(path):
    """A fixture pool: 400 'a' rows, 100 'autoif' rows, 20 'ifstruct' rows."""
    with gzip.open(path, "wt") as fh:
        for i in range(400):
            fh.write(json.dumps({"c": "antidoom", "i": "antidoom:plain_train:%d" % i}) + "\n")
        for i in range(100):
            fh.write(json.dumps({"c": "antidoom",
                                 "i": "antidoom:open_perfectblend_autoif:%d" % i}) + "\n")
        for i in range(20):
            fh.write(json.dumps({"c": "antidoom",
                                 "i": "antidoom:ifstruct_train_generated:%d" % i}) + "\n")


def run(path, n, strata):
    """Execute the sampler block with the same names it sees inside main.py."""
    env = {"gzip": gzip, "json": json, "hashlib": hashlib,
           "local": path, "N_PROMPTS": n, "fails": [],
           "log": lambda m: None,
           "C": lambda k, d: {"sample_salt": "tidepool-replay-v1",
                              "strata": json.dumps(strata)}.get(k, d)}
    exec(compile(BLOCK, "sampler", "exec"), env)
    return env["picked"], env["fails"]


def sub(rows):
    return [r["i"].split(":")[1] for r in rows]


with tempfile.TemporaryDirectory() as d:
    P = os.path.join(d, "pool.jsonl.gz")
    pool(P)

    # ---- the default has to be the old behaviour, exactly
    plain, f0 = run(P, 40, {})
    ref = sorted(
        (hashlib.blake2b(("antidoom|%s|tidepool-replay-v1" % r["i"]).encode(),
                         digest_size=8).hexdigest(), r["i"])
        for r in [json.loads(l) for l in gzip.open(P, "rt")])
    ok("an empty strata map reproduces the unstratified hash-rank sample",
       [r["i"] for r in plain] == [i for _, i in ref[:40]],
       [r["i"] for r in plain][:3])
    ok("the default raises nothing", not f0, f0)

    # ---- reweighting moves the composition and nothing else
    strata = {"open_perfectblend_autoif": 0.45, "ifstruct_train_generated": 0.05}
    wt, f1 = run(P, 40, strata)
    got = sub(wt)
    ok("the sample is still exactly N", len(wt) == 40, len(wt))
    ok("the constraint stratum lands on its requested share",
       got.count("open_perfectblend_autoif") == 18, got.count("open_perfectblend_autoif"))
    ok("the small stratum lands on its requested share",
       got.count("ifstruct_train_generated") == 2, got.count("ifstruct_train_generated"))
    ok("the remainder goes to the unmatched rows",
       got.count("plain_train") == 20, got.count("plain_train"))
    ok("reweighting raises nothing", not f1, f1)

    # ---- each stratum is drawn in its own hash order, which is what makes N nested
    small, _ = run(P, 20, strata)
    big, _ = run(P, 40, strata)
    per_small = [r["i"] for r in small if "autoif" in r["i"]]
    per_big = [r["i"] for r in big if "autoif" in r["i"]]
    ok("a stratum at larger N is a superset of the same stratum at smaller N",
       set(per_small) <= set(per_big), (len(per_small), len(per_big)))

    # ---- order must not depend on the strata map, so two buffers are diffable
    ids_wt = [r["i"] for r in wt]
    ok("the emitted order is the global hash order, not stratum-by-stratum",
       ids_wt == sorted(ids_wt,
                        key=lambda i: hashlib.blake2b(
                            ("antidoom|%s|tidepool-replay-v1" % i).encode(),
                            digest_size=8).hexdigest()),
       ids_wt[:4])

    # ---- a stratum the pool cannot fill is an assertion failure, not a short sample
    _, f2 = run(P, 400, {"ifstruct_train_generated": 0.5})
    ok("a stratum the pool cannot fill is reported",
       any("could not be filled" in x for x in f2), f2)

    # ---- shares that sum past 1.0 are refused outright
    try:
        run(P, 40, {"open_perfectblend_autoif": 0.7, "ifstruct_train_generated": 0.5})
        ok("shares summing past 1.0 are refused", False, "no SystemExit")
    except SystemExit as exc:
        ok("shares summing past 1.0 are refused", "more than the whole sample" in str(exc), exc)

    # ---- the shares s5.5 actually queues, against the supply the real pool actually has
    #
    # `antidoom-mix-v1.0` reaches the prompt pool decontaminated, and the decontamination is
    # not uniform across sub-sources: `ifstruct_train_generated` starts at 20,000 rows and
    # 18,721 of them are dropped, because IFStruct is one of this project's evaluation sets
    # and almost every training row collides with it. What is left is 1,279 rows, so the
    # ceiling on that stratum is 4.0% of a 32,000-prompt buffer and not the 5% the design was
    # first written with. A fixture at the real counts is the cheap way to keep that number
    # honest: if a later change to the pool moves the supply, this fails here in seconds
    # rather than forty minutes into a generation arm on a rented card.
    SUPPLY = {"open_perfectblend_autoif": 46734, "ifstruct_train_generated": 1279,
              "other": 399040}
    R = os.path.join(d, "real.jsonl.gz")
    with gzip.open(R, "wt") as fh:
        for name, n in SUPPLY.items():
            tag = "plain_train" if name == "other" else name
            for i in range(n):
                fh.write(json.dumps({"c": "antidoom",
                                     "i": "antidoom:%s:%d" % (tag, i)}) + "\n")

    QUEUED = {"open_perfectblend_autoif": 0.47, "ifstruct_train_generated": 0.03}
    real, f3 = run(R, 32000, QUEUED)
    got = sub(real)
    ok("the queued s5.5 shares fill from the real pool", not f3, f3)
    ok("the queued s5.5 buffer is exactly half constraint-bearing",
       got.count("open_perfectblend_autoif") + got.count("ifstruct_train_generated") == 16000,
       (got.count("open_perfectblend_autoif"), got.count("ifstruct_train_generated")))

    # And the share the design was first written with is the one the pool refuses, which is
    # the whole reason the split between the two strata moved.
    _, f4 = run(R, 32000, {"open_perfectblend_autoif": 0.45,
                           "ifstruct_train_generated": 0.05})
    ok("a 5% ifstruct share is refused by the real supply",
       any("could not be filled" in x and "ifstruct" in x for x in f4), f4)

print("\n%d checks, %d failed" % (len(RAN), len(FAILED)))
raise SystemExit(1 if FAILED else 0)
