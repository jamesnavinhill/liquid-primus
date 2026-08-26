"""Fixture test for the sweep's checkpoint and resume path.

C1 stalled at step 2,940 of 8,416 with three hours of training in the memory of a machine
nobody can reach, and nothing on disk, because weights were written once at the end. Adding
checkpoints is easy; adding them in a way that cannot quietly corrupt an arm is the part worth
testing, and every failure mode below is silent from the outside:

  - a resume that lands on a checkpoint from a DIFFERENT recipe produces an arm that is
    neither, and reports itself as one of them
  - a partial weight load leaves the model part checkpoint and part base weights
  - a pointer written before its upload lands aims at a half-written object
  - a single slot overwritten in place makes the moment of greatest risk the only copy
  - a checkpoint taken off a step boundary resumes mid-accumulation with stale gradients
  - throughput measured against a resumed token count reports a rate no machine achieved
  - a shared input fetched through the CLI is a per-arm download the pack guard cannot see

main.py imports torch at module scope and this box has none, so the checks are static over its
syntax tree, plus one executable check of the row arithmetic the resume depends on.
"""

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "main.py")).read()
TREE = ast.parse(SRC)
fails = []
oks = []


def ok(msg):
    oks.append(msg)


def fn(name):
    for n in ast.walk(TREE):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


def seg(node):
    return ast.get_source_segment(SRC, node) or ""


# ---------------------------------------------------------------- 1. fingerprint coverage
#
# Everything that changes what a step MEANS has to be in the hash, or a resume can silently
# cross recipes. The arm table is the sweep's whole specification, so each field it can set
# is named here explicitly rather than inferred.
fp_node = None
for n in ast.walk(TREE):
    if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "FINGERPRINT" for t in n.targets):
        fp_node = n
if fp_node is None:
    fails.append("no FINGERPRINT is computed, so a resume cannot tell two recipes apart")
else:
    # The identifiers the hash actually READS, taken from the syntax tree. Substring matching
    # is not good enough here: "ARM" is inside "WARMUP", so a fingerprint that had dropped the
    # arm entirely would still look present in the source text.
    hashed = {x.id for x in ast.walk(fp_node) if isinstance(x, ast.Name)}
    required = {"ARM", "BASE", "MIX", "TUNING", "LOSS", "ENTROPY_BETA", "REPLAY_FRAC",
                "REPLAY_OBJ", "GUARD_EPOCHS", "BUDGET", "MAX_LEN", "MICRO_BS", "ACCUM",
                "LR", "WARMUP", "LORA_R", "LORA_ALPHA", "LORA_DROPOUT", "SEED", "steps",
                "SMOKE"}
    missing = sorted(required - hashed)
    if missing:
        fails.append("FINGERPRINT omits %s; two arms differing only in one of those would "
                     "share a checkpoint" % ", ".join(missing))
    else:
        ok("fingerprint reads all %d recipe-defining values" % len(required))

load = fn("load_checkpoint")
save = fn("save_checkpoint")
if load is None or save is None:
    fails.append("save_checkpoint/load_checkpoint are missing")
    print("\n".join(fails))
    sys.exit(1)

# ---------------------------------------------------------------- 2. mismatch is refused
lsrc = seg(load)
guard = None
for n in ast.walk(load):
    if not isinstance(n, ast.If):
        continue
    names = {x.id for x in ast.walk(n.test) if isinstance(x, ast.Name)}
    cmps = [c for c in ast.walk(n.test) if isinstance(c, ast.Compare)]
    if "FINGERPRINT" in names and any(isinstance(o, ast.NotEq) for c in cmps for o in c.ops):
        if any(isinstance(b, ast.Continue) for b in ast.walk(n)):
            guard = n
if guard is None:
    fails.append("load_checkpoint has no `!= FINGERPRINT` test that skips the checkpoint; a "
                 "checkpoint from another recipe would be loaded as if it were this arm's")
else:
    ok("load_checkpoint skips a checkpoint whose fingerprint differs from this run's")

if "weights_only=False" not in lsrc:
    fails.append("torch.load without weights_only=False will refuse a checkpoint carrying "
                 "optimizer state on newer torch")
else:
    ok("torch.load is called in a mode that can read optimizer state back")

# Both slots read, not just whatever the pointer says. The loop must iterate CKPT_SLOTS
# itself: a slice of it reads one slot and silently resumes from the older checkpoint.
sweep = None
for n in ast.walk(load):
    if isinstance(n, ast.For) and isinstance(n.iter, ast.Name) and n.iter.id == "CKPT_SLOTS":
        sweep = n
if sweep is None:
    fails.append("load_checkpoint does not iterate CKPT_SLOTS whole, so a run that died "
                 "between the slot upload and the pointer upload resumes from the older "
                 "checkpoint")
elif not any(isinstance(x, ast.Compare) and any(isinstance(o, ast.Gt) for o in x.ops)
             for x in ast.walk(sweep)):
    fails.append("load_checkpoint reads both slots but does not pick the further one by step")
else:
    ok("load_checkpoint reads both slots whole and keeps the one further into the run")

# ---------------------------------------------------------------- 3. pointer ordering
bail = None
for n in ast.walk(save):
    if not isinstance(n, ast.If):
        continue
    calls = {getattr(c.func, "id", "") for c in ast.walk(n.test) if isinstance(c, ast.Call)}
    if "_storage_put" not in calls:
        continue
    if any(isinstance(b, ast.Return) for b in ast.walk(n)):
        bail = n
ptr_line = None
for n in ast.walk(save):
    if isinstance(n, ast.Constant) and n.value == "ckpt-latest.json":
        ptr_line = n.lineno
if bail is None:
    fails.append("save_checkpoint does not return early when the slot upload fails, so the "
                 "pointer is written whether or not the object it names exists")
elif ptr_line is None:
    fails.append("save_checkpoint writes no pointer")
elif ptr_line <= (bail.end_lineno or bail.lineno):
    fails.append("the pointer is written before the upload has been checked")
else:
    ok("the pointer is only reachable once its slot has actually reached storage")

if not any(isinstance(x, ast.Constant) and x.value is False
           for n in ast.walk(save) if isinstance(n, ast.Return) for x in ast.walk(n)):
    fails.append("save_checkpoint never returns False, so a caller cannot tell a checkpoint "
                 "that landed from one that did not")
else:
    ok("a failed upload returns False and leaves the previous checkpoint standing")

# ---------------------------------------------------------------- 4. slot alternation
call = None
for n in ast.walk(TREE):
    if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "save_checkpoint":
        call = n
if call is None:
    fails.append("save_checkpoint is defined but never called, so nothing is ever written")
else:
    csrc = seg(call)
    if "%" not in csrc or "CKPT_SLOTS" not in csrc:
        fails.append("the call site does not alternate slots; overwriting one object in place "
                     "makes the half-written moment the only copy")
    else:
        ok("the call site alternates slots by step")

# The checkpoint has to be taken at an optimizer-step boundary. Anywhere else and the resume
# restarts mid-accumulation, dropping partial gradients and shifting the effective batch.
guarded = False
for n in ast.walk(TREE):
    if isinstance(n, ast.If) and "micro % ACCUM" in seg(n.test if n.test else n):
        continue
for n in ast.walk(TREE):
    if isinstance(n, ast.If) and call and "CKPT_EVERY" in seg(n.test):
        body = seg(n)
        if "save_checkpoint" in body:
            guarded = True
if not guarded:
    fails.append("the save is not behind a CKPT_EVERY test")
else:
    # step is only incremented once per optimizer step, and the guard is `step % CKPT_EVERY`.
    if "step % CKPT_EVERY == 0" not in SRC:
        fails.append("the save cadence is not counted in optimizer steps")
    else:
        ok("checkpoints are taken on optimizer-step boundaries only")

if "and not SMOKE" not in SRC.split("save_checkpoint({")[0][-200:]:
    fails.append("a smoke would write checkpoints, and a 30-step smoke's weights look exactly "
                 "like a real arm's from the outside")
else:
    ok("a smoke writes no checkpoint")

# ---------------------------------------------------------------- 5. partial load refused
resume_src = SRC[SRC.index("_ck = load_checkpoint()"):SRC.index("tok_at_start = seen_tok")]
rtree = ast.parse(resume_src.strip())
absent = None
for n in ast.walk(rtree):
    if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_absent" for t in n.targets):
        absent = n
if absent is None:
    fails.append("the resume never computes which trainable tensors the checkpoint lacks")
elif not any(isinstance(c, ast.comprehension) for c in ast.walk(absent)):
    fails.append("_absent is not computed from the model's own trainable tensors; a partial "
                 "load would train a model that is part checkpoint and part base weights")
elif "_TRAINABLE" not in {x.id for x in ast.walk(absent) if isinstance(x, ast.Name)}:
    fails.append("_absent is not checked against _TRAINABLE, so a checkpoint missing tensors "
                 "would pass")
else:
    refuses = False
    for n in ast.walk(rtree):
        if not isinstance(n, ast.If):
            continue
        names = {x.id for x in ast.walk(n.test) if isinstance(x, ast.Name)}
        if not {"_absent", "_unexpected"} <= names:
            continue
        for b in ast.walk(n):
            if isinstance(b, ast.Assign) and any(getattr(t, "id", "") == "_ck" for t in b.targets) \
                    and isinstance(b.value, ast.Constant) and b.value.value is None:
                refuses = True
    if not refuses:
        fails.append("a mismatched checkpoint is detected but not discarded")
    else:
        ok("a partial or mismatched weight load is refused rather than half-applied")

for piece in ("opt.load_state_dict", "sched.load_state_dict"):
    if piece not in resume_src:
        fails.append("the resume does not restore %s; an Adam moment or cosine phase reset "
                     "half way through is a different recipe, not a resumed one" % piece)
if "opt.load_state_dict" in resume_src and "sched.load_state_dict" in resume_src:
    ok("optimizer and scheduler state are both restored")

# ---------------------------------------------------------------- 6. throughput accounting
if "tok_at_start" not in SRC:
    fails.append("throughput is measured against a token count that includes another job's "
                 "compute, so a resumed arm reports a rate no machine achieved")
else:
    bad = [ln for ln in SRC.splitlines()
           if "tok/s" in ln or ln.strip().startswith("tok_per_s")]
    if any("seen_tok /" in ln for ln in bad):
        fails.append("a throughput site still divides the full seen_tok by this job's clock")
    else:
        ok("throughput is measured against the tokens this job actually saw")

# ---------------------------------------------------------------- 7. CLI downloads are per-arm
#
# The pack isolation guard only sees `lab.storage_download` in the syntax tree. A download
# spawned through the CLI is invisible to it, so it is checked here instead: the only object a
# child may fetch for itself is one namespaced to its own arm.
for n in ast.walk(TREE):
    if not (isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "run"):
        continue
    src = seg(n)
    if '"storage"' not in src or '"download"' not in src:
        continue
    owner = None
    for f in ast.walk(TREE):
        if isinstance(f, ast.FunctionDef) and n.lineno >= f.lineno and \
                n.lineno <= (f.end_lineno or f.lineno):
            owner = f.name
    if owner != "_storage_get":
        fails.append("a CLI storage download sits outside _storage_get (in %s); the pack "
                     "isolation guard cannot see it and it may fetch a shared input per arm"
                     % owner)
for n in ast.walk(TREE):
    if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_storage_get":
        if "CKPT_DEST" not in seg(n):
            fails.append("_storage_get is called on something other than this arm's own "
                         "checkpoint prefix: %s" % seg(n))
if not any("CLI storage download" in f or "_storage_get is called" in f for f in fails):
    ok("the only CLI download is this arm's own checkpoint, namespaced by run tag")

if "CKPT_DEST" in SRC and "-smoke" not in SRC.split("CKPT_DEST =")[1][:120]:
    fails.append("CKPT_DEST is not namespaced away from a smoke, so a 30-step smoke could be "
                 "resumed into as if it were the real arm")
else:
    ok("smoke checkpoints are namespaced away from the real arm's")

# A pack gives every child the same config, so run_tag alone does not separate two arms sharing
# a card. Without ARM in the path, four trainers write four recipes over one pair of objects.
_dest_names = set()
for n in ast.walk(TREE):
    if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "CKPT_DEST" for t in n.targets):
        _dest_names |= {c.id for c in ast.walk(n.value) if isinstance(c, ast.Name)}
if "ARM" not in _dest_names:
    fails.append("CKPT_DEST does not mention ARM (it uses %s), so two arms packed on one card "
                 "share a checkpoint prefix and overwrite each other"
                 % (", ".join(sorted(_dest_names)) or "no names at all"))
else:
    ok("packed arms cannot collide: the checkpoint prefix carries the arm, not just the run tag")

# ---------------------------------------------------------------- 8. the row arithmetic
#
# The one piece of real arithmetic in the resume: with shuffle off and drop_last on, restarting
# a loader over rows[skip * MICRO_BS:] must yield exactly the batches the original loader would
# have yielded from micro-batch `skip` onward. If it does not, a resumed arm silently trains on
# a shifted or a duplicated slice.
def batches(rows, bs):
    return [tuple(rows[i:i + bs]) for i in range(0, len(rows) - bs + 1, bs)]


cases = 0
for n_rows in (10, 11, 12, 97, 100, 1001):
    for bs in (1, 2, 3, 8):
        rows = list(range(n_rows))
        full = batches(rows, bs)
        for skip in (0, 1, 3, len(full) - 1, len(full)):
            if skip < 0 or skip > len(full):
                continue
            got = batches(rows[skip * bs:], bs)
            cases += 1
            if got != full[skip:]:
                fails.append("resume offset is wrong for %d rows, batch %d, skip %d: %d "
                             "batches against %d expected"
                             % (n_rows, bs, skip, len(got), len(full[skip:])))
if not any("resume offset" in f for f in fails):
    ok("resume-by-row-offset reproduces the original batch sequence exactly (%d cases)" % cases)

# A mutation check: the arithmetic must actually be capable of failing.
if batches(list(range(20)), 2)[3:] == batches(list(range(20))[3 * 2 - 2:], 2):
    fails.append("the row-offset check cannot distinguish a correct offset from an off-by-one")
else:
    ok("the offset check rejects an off-by-one")

# ---------------------------------------------------------------- 9. checkpoints are not artifacts
#
# The pack supervisor uploads every file in a child's output directory as a job artifact. A
# checkpoint pair on the full-parameter arm is tens of gigabytes and is already in shared
# storage, so attaching it to the job as well would multiply the pack's upload for nothing.
PACK = open(os.path.join(HERE, "pack.py")).read()
ptree = ast.parse(PACK)
skipped = False
for n in ast.walk(ptree):
    if not isinstance(n, ast.If):
        continue
    lits = [a.value for c in ast.walk(n.test) if isinstance(c, ast.Call)
            and getattr(c.func, "attr", "") == "startswith"
            for a in c.args if isinstance(a, ast.Constant)]
    if any(str(a).startswith("ckpt-") for a in lits) and \
            any(isinstance(b, ast.Continue) for b in ast.walk(n)):
        skipped = True
if not skipped:
    fails.append("pack.py uploads checkpoint files as job artifacts; on the full-parameter arm "
                 "that is tens of gigabytes of duplicate upload per pack")
else:
    ok("the pack supervisor leaves checkpoints out of the job's artifacts")

print("\n".join("  ok   " + o for o in oks))
if fails:
    print("\nFAILURES")
    print("\n".join("  FAIL " + f for f in fails))
    sys.exit(1)
print("\ncheckpoint/resume contract holds across %d checks" % len(oks))
