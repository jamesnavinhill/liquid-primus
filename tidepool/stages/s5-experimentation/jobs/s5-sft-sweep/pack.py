"""Run several sweep arms concurrently on one GPU, isolated from each other.

A 1.2B model with a rank-16 adapter does not come close to filling an L40S, so running one
arm per card leaves most of the card idle and makes the eight-arm sweep take four sequential
rounds against a two-GPU cap. This supervisor runs N arms on one card as separate child
processes and reports them as N independent results under one job id.

Isolation is the whole design, because packing is only worth doing if one arm's bad day
cannot cost the others:

  process     each arm is its own `python main.py`. An OOM, an assertion or a segfault takes
              that process and leaves its siblings running. Nothing is shared in-process.
  memory      each child gets a hard allocator ceiling of `headroom / n` via
              set_per_process_memory_fraction, so an arm that tries to grow into its
              neighbours' memory hits its own limit first and fails alone.
  filesystem  each child writes to out/<arm>/. No two children can touch the same path.
  reporting   children never call lab.*. One job has one progress stream and one artifact
              namespace, and concurrent writers to it would interleave into nonsense. The
              supervisor is the only reporter: it collects each score.json at the end and
              uploads every artifact under an <arm>__ prefix.
  shutdown    SIGTERM/SIGINT to the supervisor is forwarded to every child, then escalated
              to SIGKILL after a grace period, so a cancelled job does not leave orphaned
              trainers holding the card.

Not every child is a trainer, and not every child can start at the same time. `pack_scripts`
lets an arm run a different program, and `pack_after` holds an arm back until another arm has
finished successfully -- which is how a buffer that one arm generates becomes the input of
another arm on the same card, with no round trip through shared storage and no second job.
`pack_provides` names the file that hands over: the supervisor registers it as the shared
input its dependants were going to download, and uploads it once on their behalf.

Staggered starts matter more than they look. Every child loads the same base model and peaks
its memory during load; starting four at the same instant stacks four load peaks that the
steady-state footprint would never reach. `stagger_seconds` spaces them out.

What packing must NOT change is the science. A child runs the identical training path a solo
arm runs: same sampler, same seed, same row order, same token budget, same assertions. The
only differences are where it writes and who reports it. That comparability is checked rather
than asserted: C1 ran solo before packing existed, so any packed arm can be compared against
it on tokens seen, priority share and rows-in-order.
"""

import itertools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

from lab import lab

lab.init()
CFG = lab.get_config() or {}


def C(k, default):
    v = CFG.get(k)
    return default if v is None or v == "" else v


ARMS = [a.strip() for a in str(C("arms", "")).split(",") if a.strip()]
HEADROOM = float(C("pack_headroom", 0.92))
WEIGHTS_RAW = str(C("pack_weights", ""))
GB_RAW = str(C("pack_gb", ""))
CARD_GB = float(C("card_gb", 47.7))
STAGGER = float(C("stagger_seconds", 45))
GRACE = float(C("shutdown_grace_seconds", 30))
STALL = float(C("stall_minutes", 45)) * 60.0
POLL = float(C("poll_seconds", 5.0))

if not ARMS:
    lab.error(message="pack.py needs an `arms` list, e.g. arms=C2p,C4,C6")
    raise SystemExit(2)

OUT = "out"
os.makedirs(OUT, exist_ok=True)
N = len(ARMS)

# Shares are weighted, not equal. The arms in this sweep do not have equal appetites: a
# full-parameter arm carries gradients and fp32 optimizer moments for every parameter and
# needs several times what a rank-16 adapter does, so splitting the card evenly would starve
# it while leaving the LoRA arms holding ceilings they never approach. `pack_weights` is a
# comma-separated list parallel to `arms`; an empty list means equal shares.
#
# `pack_gb` states each arm's ceiling in gigabytes directly and is preferred, because that is
# the unit the measurements come in and the unit the failures come in. Relative weights have a
# flaw that only shows up once arms can wait on each other: they divide the card between every
# arm listed, including arms that never coexist, so an arm that runs after another still pays
# for it. `pack_weights` remains for even splits.
if GB_RAW.strip():
    G = [float(x) for x in GB_RAW.split(",") if x.strip()]
    if len(G) != N:
        lab.error(message="pack_gb has %d entries for %d arms" % (len(G), N))
        raise SystemExit(2)
    if min(G) <= 0:
        lab.error(message="pack_gb must all be positive, got %r" % (G,))
        raise SystemExit(2)
    W = G
    GB = dict(zip(ARMS, G))
    MEMFRAC = {a: round(min(0.98, GB[a] / CARD_GB), 4) for a in ARMS}
else:
    if WEIGHTS_RAW.strip():
        W = [float(x) for x in WEIGHTS_RAW.split(",") if x.strip()]
        if len(W) != N:
            lab.error(message="pack_weights has %d entries for %d arms" % (len(W), N))
            raise SystemExit(2)
        if min(W) <= 0:
            lab.error(message="pack_weights must all be positive, got %r" % (W,))
            raise SystemExit(2)
    else:
        W = [1.0] * N
    TOTAL_W = sum(W)
    MEMFRAC = {a: round(HEADROOM * w / TOTAL_W, 4) for a, w in zip(ARMS, W)}
    GB = {a: round(CARD_GB * MEMFRAC[a], 2) for a in ARMS}


def _kv(raw, what):
    """Parse an `arm=value,arm=value` parameter, rejecting names that are not arms."""
    out = {}
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            lab.error(message="%s wants `arm=value` items, got %r" % (what, item))
            raise SystemExit(2)
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        if k not in ARMS:
            lab.error(message="%s names %r, which is not one of arms=%s"
                              % (what, k, ",".join(ARMS)))
            raise SystemExit(2)
        out[k] = v
    return out


# An arm is a program, not necessarily a trainer. The default is the sweep's own main.py.
SCRIPTS = {a: "main.py" for a in ARMS}
SCRIPTS.update(_kv(C("pack_scripts", ""), "pack_scripts"))
for a in ARMS:
    if not os.path.exists(SCRIPTS[a]):
        lab.error(message="arm %s runs %s, which is not in the task directory"
                          % (a, SCRIPTS[a]))
        raise SystemExit(2)

# `pack_after` is deliberately one level deep. Deeper chains would need a real scheduler, and
# a real scheduler inside a supervisor whose whole value is that it is simple to reason about
# is a bad trade; two waves cover producer-then-consumer, which is the case that exists.
AFTER = _kv(C("pack_after", ""), "pack_after")
for a, dep in AFTER.items():
    if dep not in ARMS:
        lab.error(message="%s waits on %r, which is not one of arms=%s"
                          % (a, dep, ",".join(ARMS)))
        raise SystemExit(2)
    if dep == a:
        lab.error(message="%s cannot wait on itself" % a)
        raise SystemExit(2)
    if dep in AFTER:
        lab.error(message="pack_after is one level deep: %s waits on %s, which itself waits "
                          "on %s" % (a, dep, AFTER[dep]))
        raise SystemExit(2)

# `arm=config_key:filename`. When the arm exits clean, out/<arm>/<filename> becomes the local
# stand-in for whatever object CFG[config_key] names, and is uploaded there once.
PROVIDES = {}
for a, spec in _kv(C("pack_provides", ""), "pack_provides").items():
    if ":" not in spec:
        lab.error(message="pack_provides wants `arm=config_key:filename`, got %r" % spec)
        raise SystemExit(2)
    key, fname = spec.split(":", 1)
    key, fname = key.strip(), fname.strip()
    if not CFG.get(key):
        lab.error(message="%s says it provides %s, but %s is not set in this pack's config, "
                          "so nothing would know to look for it" % (a, key, key))
        raise SystemExit(2)
    PROVIDES[a] = (key, fname)


# The ceiling has to hold at the pack's worst moment, not on average and not on the flat sum
# of every arm listed. Once arms can wait on each other the flat sum is simply wrong: a
# producer's memory is gone by the time its consumers hold any, so charging the card for both
# at once refuses packs that fit at every instant they actually exist.
#
# The peak is therefore the largest set of arms that can be alive together. With one level of
# dependency the reachable states are enumerable: a state is a choice of which producers have
# already finished, and in that state the card holds the unfinished no-dependency arms plus
# the consumers those finished producers released. There are a handful of producers, so this
# is a handful of sums.
PRODUCERS = sorted(set(AFTER.values()))
NODEP = [a for a in ARMS if a not in AFTER]
PEAK_GB, PEAK_SET = 0.0, tuple(ARMS)
if len(PRODUCERS) > 12:
    # Not a case that exists, but a combinatorial blowup in a sizing check is a worse failure
    # than an over-conservative number, so fall back to charging for everything at once.
    PEAK_GB, PEAK_SET = sum(GB.values()), tuple(ARMS)
else:
    for r in range(len(PRODUCERS) + 1):
        for finished in itertools.combinations(PRODUCERS, r):
            live = [a for a in NODEP if a not in finished]
            live += [a for a in ARMS if AFTER.get(a) in finished]
            tot = sum(GB[a] for a in live)
            if tot > PEAK_GB:
                PEAK_GB, PEAK_SET = tot, tuple(live)
LIMIT_GB = CARD_GB * HEADROOM
if PEAK_GB > LIMIT_GB + 1e-6:
    lab.error(message="this pack peaks at %.1f GB (%s running together) against a %.1f GB "
                      "limit (%.1f GB card at headroom %.2f); lower pack_gb or drop an arm"
                      % (PEAK_GB, "+".join(PEAK_SET), LIMIT_GB, CARD_GB, HEADROOM))
    raise SystemExit(2)

# The child's whole configuration travels on the environment. A child never calls the job
# API, so it cannot ask for its own config; passing the supervisor's verbatim is what keeps a
# packed arm's recipe byte-identical to the same arm run solo.
#
# `pack_overrides` is the exception, and it exists because not every packable program can
# recognise its own arm. The sweep trainer can: `arm` selects a recipe from a table inside
# main.py, so one shared config describes all eight arms. An evaluation pass cannot, because
# what distinguishes two of its arms is which checkpoint and which component they run, and
# those are values rather than a name. Rather than push an arm table into every future child
# script, an arm may be handed a patch on the shared config:
#
#   -p pack_overrides='{"C6": {"lora_r": 64}, "RB": {"batch_size": 32}}'
#
# It is JSON so the values keep their types; a bare string parameter would turn every number
# into text and every false into a true.
CHILD_CFG = dict(CFG)
OVERRIDES = {}
_ov = str(C("pack_overrides", "")).strip()
if _ov:
    try:
        OVERRIDES = json.loads(_ov)
    except Exception as exc:
        lab.error(message="pack_overrides is not valid JSON (%s): %r" % (exc, _ov[:200]))
        raise SystemExit(2)
    if not isinstance(OVERRIDES, dict):
        lab.error(message="pack_overrides must be an object of arm -> {key: value}")
        raise SystemExit(2)
    for a, patch in OVERRIDES.items():
        if a not in ARMS:
            lab.error(message="pack_overrides names %r, which is not one of arms=%s"
                              % (a, ",".join(ARMS)))
            raise SystemExit(2)
        if not isinstance(patch, dict):
            lab.error(message="pack_overrides[%s] must be an object, got %r" % (a, patch))
            raise SystemExit(2)


def log(msg):
    lab.log(msg)
    print(msg, flush=True)


log("packing %d arms onto one GPU: %s" % (N, ", ".join(ARMS)))
log("per-arm memory ceilings on a %.1f GB card (headroom %.2f, stagger %.0fs): %s"
    % (CARD_GB, HEADROOM, STAGGER,
       ", ".join("%s %.1f GB" % (a, GB[a]) for a in ARMS)))
log("peak concurrent demand %.1f GB of a %.1f GB limit, set by %s running together"
    % (PEAK_GB, LIMIT_GB, " + ".join(PEAK_SET)))
if AFTER:
    log("held until their inputs exist: %s"
        % ", ".join("%s after %s" % (a, AFTER[a]) for a in sorted(AFTER)))
for a in sorted(SCRIPTS):
    if SCRIPTS[a] != "main.py":
        log("%s runs %s rather than the sweep trainer" % (a, SCRIPTS[a]))
for a in sorted(OVERRIDES):
    log("%s takes the shared config with %s patched"
        % (a, ", ".join("%s=%r" % kv for kv in sorted(OVERRIDES[a].items()))))

children = {}          # arm -> Popen
started = {}           # arm -> monotonic start
logs = {}              # arm -> open file handle


# ------------------------------------------------------- shared inputs, fetched once
#
# Every arm on this card reads the same corpus. Downloading it per child would multiply the
# transfer and the disk by the pack size and give four processes four chances to fail on the
# network; the supervisor fetches each object once and hands the paths down.
LOCAL = {}
# An object one of these arms is about to generate must not be fetched: it does not exist yet,
# and the whole reason it is in this pack is so that it never has to make the round trip.
PROVIDED_OBJS = {CFG.get(k) for k, _ in PROVIDES.values() if CFG.get(k)}
for key in ("train_object", "val_object", "guardrail_object", "replay_object", "pool_object"):
    obj = CFG.get(key)
    if not obj or obj in LOCAL:
        continue
    if obj in PROVIDED_OBJS:
        log("%s is produced inside this pack, so it is not fetched" % obj)
        continue
    t = time.time()
    LOCAL[obj] = lab.storage_download(obj)
    log("fetched %s -> %s (%.0fs)" % (obj, LOCAL[obj], time.time() - t))
log("%d shared input(s) staged for %d arms" % (len(LOCAL), N))

# The base model is a shared input too, and it was the one I forgot. Four children calling
# from_pretrained on a cold cache is four concurrent downloads of the same repo into the same
# directory, racing on the hub's locks and partial files; the corpus was already fetched once
# per card and the weights should be as well. Pre-warming here means every child finds the
# snapshot complete and does no network work at all.
BASE = str(CFG.get("base_model") or "")
if BASE:
    try:
        from huggingface_hub import snapshot_download
        t = time.time()
        path = snapshot_download(BASE)
        log("base model cached at %s (%.0fs)" % (path, time.time() - t))
    except Exception as exc:
        # Not fatal: the children can still fetch it themselves, just less pleasantly.
        log("could not pre-warm %s (%s); children will fetch it individually" % (BASE, exc))


def spawn(arm):
    adir = os.path.join(OUT, arm)
    os.makedirs(adir, exist_ok=True)
    env = dict(os.environ)
    env["TIDEPOOL_PACK_CHILD"] = "1"
    # Packing makes fragmentation expensive in a way a solo run never notices: a child that
    # holds 2 GB of reserved-but-unallocated blocks is 2 GB its siblings cannot have, and the
    # first packed trial died on exactly that boundary. Expandable segments let the allocator
    # grow and release a single region instead of stranding fixed-size blocks.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["TIDEPOOL_PACK_ARM"] = arm
    env["TIDEPOOL_PACK_OUT"] = adir
    env["TIDEPOOL_PACK_MEMFRAC"] = str(MEMFRAC[arm])
    cfg = dict(CHILD_CFG)
    cfg.update(OVERRIDES.get(arm) or {})
    env["TIDEPOOL_PACK_CFG"] = json.dumps(cfg)
    env["TIDEPOOL_PACK_LOCAL"] = json.dumps(LOCAL)
    # Distinct W&B run per arm; without this every child reports into one run.
    env["WANDB_RUN_GROUP"] = str(C("run_tag", "s5.3-pack"))
    fh = open(os.path.join(adir, "console.log"), "w")
    logs[arm] = fh
    p = subprocess.Popen([sys.executable, "-u", SCRIPTS[arm]], env=env,
                         stdout=fh, stderr=subprocess.STDOUT)
    children[arm] = p
    started[arm] = time.time()
    log("started %s (pid %d, %s, ceiling %.1f GB)"
        % (arm, p.pid, SCRIPTS[arm], GB[arm]))


_shutting_down = {"v": False}


def shutdown(signum, _frame):
    """Forward the signal to every child, then escalate. A cancelled job must not leave
    trainers holding the GPU."""
    if _shutting_down["v"]:
        return
    _shutting_down["v"] = True
    log("received signal %s, terminating %d child arm(s)" % (signum, len(children)))
    for arm, p in children.items():
        if p.poll() is None:
            try:
                p.terminate()
            except Exception as exc:
                log("could not terminate %s: %s" % (arm, exc))
    deadline = time.time() + GRACE
    while time.time() < deadline and any(p.poll() is None for p in children.values()):
        time.sleep(1)
    for arm, p in children.items():
        if p.poll() is None:
            log("%s did not exit within %.0fs, killing" % (arm, GRACE))
            try:
                p.kill()
            except Exception:
                pass
    raise SystemExit(143)


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

t0 = time.time()
FIRST = [a for a in ARMS if a not in AFTER]
for i, arm in enumerate(FIRST):
    spawn(arm)
    if i < len(FIRST) - 1 and STAGGER > 0:
        time.sleep(STAGGER)

log("%d of %d arms started; supervising" % (len(FIRST), N))

# Trainers count steps, the generator counts completions. Both are the arm's own honest
# statement of where it is, and a pack that reads only one of them shows a generator arm at
# zero for its whole run and then jumping to done.
STEP_RE = re.compile(r"(?:step|generated) (\d+)/(\d+)")


def progress_fraction():
    """Mean training progress across the pack, read from the children's own step lines.

    A pack runs for hours and the arms finish minutes apart, so reporting the fraction of
    arms that have exited would sit at zero for most of the run and then jump. Each child
    already prints `step i/n`; the last such line in its console is its true position. A
    child that has exited counts as complete whatever its console last said, so a crash
    cannot hold the bar down."""
    fracs, live = [], []
    for arm in ARMS:
        if arm in done:
            fracs.append(1.0)
            continue
        cur, tot = 0, 0
        try:
            with open(os.path.join(OUT, arm, "console.log")) as fh:
                tail = fh.readlines()[-400:]
            for line in reversed(tail):
                m = STEP_RE.search(line)
                if m:
                    cur, tot = int(m.group(1)), int(m.group(2))
                    break
        except Exception:
            pass
        fracs.append(cur / tot if tot else 0.0)
        if tot:
            live.append("%s %d/%d" % (arm, cur, tot))
    return (sum(fracs) / len(fracs) if fracs else 0.0), live


def console_size(arm):
    try:
        return os.path.getsize(os.path.join(OUT, arm, "console.log"))
    except Exception:
        return -1


done = {}
last_report = 0.0
# A hung arm is the one failure the process boundary does not contain: it holds its slice of
# the card and its share of the SMs, and the pack sits behind it until the job's own wall
# clock kills everything, siblings included. So each child is watched for silence -- no new
# console output at all -- and killed on its own if it goes quiet for `stall_minutes`. The
# threshold is generous because a legitimate quiet stretch does exist (model download, corpus
# indexing, the validation pass), and killing a slow arm is worse than waiting for it.
grew = {a: (time.time(), console_size(a)) for a in ARMS}
stalled = set()
skipped = {}


def release(arm):
    """An arm finished. Publish whatever it was carrying and start whatever was waiting.

    Handing a file over inside the pack is the point: the buffer arm writes it to its own
    directory, the supervisor registers that path as the object its dependants would have
    downloaded, and they read it off local disk. The upload still happens, once, from here --
    so the buffer lands in shared storage for every later stage exactly as it would have if
    it had been its own job, without any arm waiting on the round trip."""
    if arm in PROVIDES:
        key, fname = PROVIDES[arm]
        obj, src = CFG.get(key), os.path.join(OUT, arm, fname)
        if done.get(arm) == 0 and os.path.exists(src) and os.path.getsize(src) > 128:
            LOCAL[obj] = os.path.abspath(src)
            log("%s produced %s (%.0f MB); it stands in for %s for the arms waiting on it"
                % (arm, fname, os.path.getsize(src) / 1e6, obj))
            up = getattr(lab, "storage_upload", None)
            if callable(up):
                try:
                    up(src, obj)
                    log("placed %s in shared storage" % obj)
                except Exception as exc:
                    # Not fatal to the pack. The dependants read the local copy either way;
                    # what is lost is the copy later stages would load, and saying so here
                    # means it is noticed now rather than at the next stage.
                    log("could not place %s in shared storage (%s); the waiting arms still "
                        "have it locally, but later jobs will not find it there" % (obj, exc))
            else:
                log("this SDK has no storage upload, so %s stays on the job record" % obj)
        else:
            log("%s did not produce a usable %s, so anything waiting on it cannot run"
                % (arm, fname))
    waiting = [a for a in ARMS if AFTER.get(a) == arm and a not in children and a not in done]
    for i, w in enumerate(waiting):
        if done.get(arm) != 0:
            why = "%s exited rc=%s" % (arm, done.get(arm))
        elif arm in PROVIDES and CFG.get(PROVIDES[arm][0]) not in LOCAL:
            why = "%s finished but produced no %s" % (arm, PROVIDES[arm][1])
        else:
            why = None
        if why:
            log("not starting %s: %s" % (w, why))
            done[w] = 97
            skipped[w] = why
            continue
        if i and STAGGER > 0:
            time.sleep(STAGGER)
        spawn(w)
        grew[w] = (time.time(), console_size(w))

while len(done) < N:
    time.sleep(POLL)
    now = time.time()
    for arm, p in list(children.items()):
        if arm in done or p.poll() is not None:
            continue
        sz = console_size(arm)
        last_t, last_sz = grew[arm]
        if sz != last_sz:
            grew[arm] = (now, sz)
        elif now - last_t > STALL:
            log("%s has produced no output for %.0f min; killing it so the rest of the pack "
                "is not held behind it" % (arm, (now - last_t) / 60.0))
            stalled.add(arm)
            grew[arm] = (now, sz)
            try:
                p.terminate()
            except Exception:
                pass
    exited = []
    for arm, p in list(children.items()):
        if arm in done:
            continue
        rc = p.poll()
        if rc is not None:
            done[arm] = rc
            exited.append(arm)
            mins = (time.time() - started[arm]) / 60.0
            log("%s exited rc=%d after %.1f min (%d of %d done)"
                % (arm, rc, mins, len(done), N))
            try:
                logs[arm].flush()
            except Exception:
                pass
            # A child's output goes to its own file, which is only uploaded when the pack
            # finishes. Without this, an arm that dies twenty minutes in is a bare exit code
            # for the remaining nine hours, and there is nothing to act on until the whole
            # job lands. The tail goes to the job log the moment it happens instead.
            if rc != 0:
                try:
                    with open(os.path.join(OUT, arm, "console.log")) as fh:
                        tail = fh.readlines()[-40:]
                    log("---- %s died, last %d line(s) of its console ----" % (arm, len(tail)))
                    for line in tail:
                        log("[%s] %s" % (arm, line.rstrip()))
                    log("---- end %s ----" % arm)
                except Exception as exc:
                    log("could not read %s's console: %s" % (arm, exc))
    for arm in exited:
        release(arm)
    if time.time() - last_report > 60:
        last_report = time.time()
        frac, live = progress_fraction()
        lab.update_progress(max(1, min(95, int(95.0 * frac))))
        if live:
            log("progress %.0f%% | " % (100 * frac) + " ".join(live))

for fh in logs.values():
    try:
        fh.close()
    except Exception:
        pass

# ---------------------------------------------------------------- collect
results, failed, missing = {}, [], []
for arm in ARMS:
    sp = os.path.join(OUT, arm, "score.json")
    if os.path.exists(sp):
        try:
            results[arm] = json.load(open(sp))
        except Exception as exc:
            missing.append("%s (score.json unreadable: %s)" % (arm, exc))
    else:
        missing.append("%s (no score.json; rc=%s)" % (arm, done.get(arm)))
    if done.get(arm) != 0:
        if arm in skipped:
            why = " (never started: %s)" % skipped[arm]
        elif arm in stalled:
            why = " (killed after stalling)"
        else:
            why = ""
        failed.append("%s rc=%s%s" % (arm, done.get(arm), why))

# Every child artifact goes up under an <arm>__ prefix so one job's artifact namespace holds
# N arms without collision and each file still says which arm produced it.
uploaded = 0
for arm in ARMS:
    adir = os.path.join(OUT, arm)
    if not os.path.isdir(adir):
        continue
    for name in sorted(os.listdir(adir)):
        src = os.path.join(adir, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(OUT, "%s__%s" % (arm, name))
        try:
            # Hardlink rather than copy. A full-parameter arm's weight archive runs to a
            # couple of gigabytes and there are N of them; copying every one to give it a
            # prefix would double the pack's disk for a filename.
            if os.path.abspath(src) != os.path.abspath(dst) and not os.path.exists(dst):
                try:
                    os.link(src, dst)
                except OSError:
                    with open(src, "rb") as a, open(dst, "wb") as b:
                        shutil.copyfileobj(a, b, 1 << 22)
            lab.save_artifact(dst)
            uploaded += 1
        except Exception as exc:
            log("could not upload %s: %s" % (dst, exc))

summary = {
    "arms": ARMS,
    "pack_size": N,
    "pack_weights": dict(zip(ARMS, W)),
    "per_arm_ceiling_gb": GB,
    "per_arm_memory_fraction": MEMFRAC,
    "per_arm_script": {a: SCRIPTS[a] for a in ARMS},
    "waits_on": AFTER,
    "overrides": OVERRIDES,
    "provides": {a: "%s:%s" % v for a, v in PROVIDES.items()},
    "card_gb": CARD_GB,
    "peak_concurrent_demand_gb": round(PEAK_GB, 2),
    "peak_set_by": list(PEAK_SET),
    "exit_codes": done,
    "wall_clock_hours": round((time.time() - t0) / 3600.0, 3),
    "gpu_hours_billed": round((time.time() - t0) / 3600.0, 3),
    "gpu_hours_per_arm_share": round((time.time() - t0) / 3600.0 / max(1, N), 3),
    "results": results,
    "failed": failed,
    "stalled": sorted(stalled),
    "never_started": skipped,
    "missing_scores": missing,
    "artifacts_uploaded": uploaded,
}
# What the next pack should be sized at, measured rather than guessed. The spare figure is
# against the same headroom the children were held to, so it is directly actionable: an arm
# only "still fits" if it would fit under the ceiling this pack already respected.
# Reserved, not allocated, and with the CUDA context added on top. The first packed trial
# made the difference concrete: an arm reporting 6.07 GB allocated and 6.98 GB reserved held
# 7.52 GB as far as the driver and its siblings were concerned. Sizing a pack on the
# allocated figure overcommits the card by roughly 1.5 GB per arm, which is how three arms
# that "fit" leave a fourth OOM-ing at its ceiling with free memory still on the card.
CONTEXT_GB = 0.55
peaks = {}
for a, r in results.items():
    v = r.get("peak_gpu_reserved_gb") or r.get("peak_gpu_gb")
    if v:
        peaks[a] = round(v + CONTEXT_GB, 2)
card = next((r.get("card_total_gb") for r in results.values() if r.get("card_total_gb")), None)
if peaks and card:
    worst, used = max(peaks.values()), sum(peaks.values())
    spare = card * HEADROOM - used
    summary["memory"] = {
        "per_arm_footprint_gb": peaks, "context_allowance_gb": CONTEXT_GB,
        "card_total_gb": card,
        "pack_peak_total_gb": round(used, 2),
        "spare_gb_under_headroom": round(spare, 2),
        "further_arms_that_would_fit": int(spare // worst) if worst > 0 else None,
    }
    log("memory: %.1f of %.1f GB used by %d arms (worst arm %.1f GB); %.1f GB spare under "
        "the %.2f headroom, room for about %d more"
        % (used, card, len(peaks), worst, spare, HEADROOM,
           int(spare // worst) if worst > 0 else 0))

sp = os.path.join(OUT, "pack_summary.json")
json.dump(summary, open(sp, "w"), indent=1, default=str)
try:
    lab.save_artifact(sp)
except Exception as exc:
    log("could not upload pack_summary.json: %s" % exc)

agg = sum(r.get("tokens_per_second", 0) or 0 for r in results.values())
log("PACK SUMMARY " + json.dumps({
    "completed": sorted(results), "failed": failed,
    "aggregate_tokens_per_second": round(agg, 1),
    "wall_clock_hours": summary["wall_clock_hours"]}))

lab.update_progress(100)
msg = ("%d/%d arms completed (%s); aggregate %.0f tok/s over %.2f h"
       % (len(results), N, ", ".join(sorted(results)) or "none", agg,
          summary["wall_clock_hours"]))
# A partial pack is a real, usable result: the arms that finished are as valid as if they had
# run alone. It is reported as an error only so the failure is visible in job status, and the
# arms that did land are still in results/ and in the artifacts.
if failed:
    lab.error(message=msg + "; FAILED: " + ", ".join(failed))
else:
    lab.finish(message=msg, score=summary)
