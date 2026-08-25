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

Staggered starts matter more than they look. Every child loads the same base model and peaks
its memory during load; starting four at the same instant stacks four load peaks that the
steady-state footprint would never reach. `stagger_seconds` spaces them out.

What packing must NOT change is the science. A child runs the identical training path a solo
arm runs: same sampler, same seed, same row order, same token budget, same assertions. The
only differences are where it writes and who reports it. That comparability is checked rather
than asserted: C1 ran solo before packing existed, so any packed arm can be compared against
it on tokens seen, priority share and rows-in-order.
"""

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
STAGGER = float(C("stagger_seconds", 45))
GRACE = float(C("shutdown_grace_seconds", 30))
STALL = float(C("stall_minutes", 45)) * 60.0
POLL = 5.0

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

# The child's whole configuration travels on the environment. A child never calls the job
# API, so it cannot ask for its own config; passing the supervisor's verbatim is what keeps a
# packed arm's recipe byte-identical to the same arm run solo.
CHILD_CFG = dict(CFG)


def log(msg):
    lab.log(msg)
    print(msg, flush=True)


log("packing %d arms onto one GPU: %s" % (N, ", ".join(ARMS)))
log("per-arm memory ceilings (headroom %.2f, stagger %.0fs): %s"
    % (HEADROOM, STAGGER,
       ", ".join("%s %.1f%%" % (a, 100 * MEMFRAC[a]) for a in ARMS)))

children = {}          # arm -> Popen
started = {}           # arm -> monotonic start
logs = {}              # arm -> open file handle


# ------------------------------------------------------- shared inputs, fetched once
#
# Every arm on this card reads the same corpus. Downloading it per child would multiply the
# transfer and the disk by the pack size and give four processes four chances to fail on the
# network; the supervisor fetches each object once and hands the paths down.
LOCAL = {}
for key in ("train_object", "val_object", "guardrail_object", "replay_object"):
    obj = CFG.get(key)
    if not obj or obj in LOCAL:
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
    env["TIDEPOOL_PACK_CFG"] = json.dumps(CHILD_CFG)
    env["TIDEPOOL_PACK_LOCAL"] = json.dumps(LOCAL)
    # Distinct W&B run per arm; without this every child reports into one run.
    env["WANDB_RUN_GROUP"] = str(C("run_tag", "s5.3-pack"))
    fh = open(os.path.join(adir, "console.log"), "w")
    logs[arm] = fh
    p = subprocess.Popen([sys.executable, "-u", "main.py"], env=env,
                         stdout=fh, stderr=subprocess.STDOUT)
    children[arm] = p
    started[arm] = time.time()
    log("started %s (pid %d)" % (arm, p.pid))


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
for i, arm in enumerate(ARMS):
    spawn(arm)
    if i < N - 1 and STAGGER > 0:
        time.sleep(STAGGER)

log("all %d arms started; supervising" % N)

STEP_RE = re.compile(r"step (\d+)/(\d+)")


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
while len(done) < N:
    time.sleep(POLL)
    now = time.time()
    for arm, p in children.items():
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
    for arm, p in children.items():
        if arm in done:
            continue
        rc = p.poll()
        if rc is not None:
            done[arm] = rc
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
        failed.append("%s rc=%s%s" % (arm, done.get(arm),
                                      " (killed after stalling)" if arm in stalled else ""))

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
    "per_arm_memory_fraction": MEMFRAC,
    "exit_codes": done,
    "wall_clock_hours": round((time.time() - t0) / 3600.0, 3),
    "gpu_hours_billed": round((time.time() - t0) / 3600.0, 3),
    "gpu_hours_per_arm_share": round((time.time() - t0) / 3600.0 / max(1, N), 3),
    "results": results,
    "failed": failed,
    "stalled": sorted(stalled),
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
