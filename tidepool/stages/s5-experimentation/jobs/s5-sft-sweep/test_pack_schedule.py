"""End-to-end checks on the supervisor's scheduling, run here in about ten seconds.

Everything else about packing is checked statically, but the part that decides *when* a child
starts cannot be: it is a loop over live processes, and the failures it can have — an arm that
never starts, an arm that starts before its input exists, a failed producer whose consumers
run anyway on a stale file — all look like a normal exit code from the outside. The first pack
to exercise this code would otherwise be a fourteen-hour job with four arms riding on it.

So the supervisor is run for real, as a subprocess, against arm scripts that sleep for
fractions of a second and a stand-in for the job API that records what it was asked to do.
No GPU, no model, no network: the scheduling is the only thing under test, and it is the only
thing that is not covered elsewhere.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

FAKE_LAB = '''
import json, os, shutil

class _Lab(object):
    def __init__(self):
        self.rec = os.environ["FAKE_LAB_RECORD"]
    def _put(self, kind, **kw):
        kw["kind"] = kind
        with open(self.rec, "a") as fh:
            fh.write(json.dumps(kw, default=str) + "\\n")
    def init(self): self._put("init")
    def get_config(self): return json.loads(os.environ["FAKE_LAB_CFG"])
    def log(self, m): self._put("log", m=m)
    def update_progress(self, p): self._put("progress", p=p)
    def save_artifact(self, p): self._put("artifact", p=os.path.basename(p))
    def storage_download(self, obj):
        d = os.path.join(os.environ["FAKE_LAB_STORE"], obj.replace("/", "__"))
        open(d, "w").write("corpus for " + obj)
        return d
    def storage_upload(self, path, obj=None, dest=None):
        # The real call takes a destination *prefix* and the file keeps its own name. The
        # stand-in mirrors that exactly, so a caller that passes a whole object path as the
        # destination shows up here as the wrong object rather than as a pass.
        pre = dest if dest is not None else obj
        key = (pre.rstrip("/") + "/" + os.path.basename(path)) if pre else os.path.basename(path)
        shutil.copy(path, os.path.join(os.environ["FAKE_LAB_STORE"], key.replace("/", "__")))
        self._put("upload", obj=key)
    def finish(self, message=None, score=None): self._put("finish", message=message, score=score)
    def error(self, message=None): self._put("error", message=message)

lab = _Lab()
'''

# An arm: sleeps, optionally writes the file it promised, optionally fails. It also records
# the shared-input map it was handed, which is how "did T2 see GEN's output" is checked.
ARM_SCRIPT = '''
import json, os, sys, time
arm = os.environ["TIDEPOOL_PACK_ARM"]
out = os.environ["TIDEPOOL_PACK_OUT"]
spec = json.loads(os.environ.get("ARM_SPEC") or "{}").get(arm, {})
print("step 1/2", flush=True)
with open(os.path.join(out, "saw.json"), "w") as fh:
    json.dump({"local": json.loads(os.environ["TIDEPOOL_PACK_LOCAL"]),
               "started_at": time.time(),
               "memfrac": float(os.environ["TIDEPOOL_PACK_MEMFRAC"])}, fh)
time.sleep(float(spec.get("sleep", 0.2)))
if spec.get("produce"):
    with open(os.path.join(out, spec["produce"]), "w") as fh:
        fh.write("x" * 4096)
# A child cannot reach shared storage itself, so it asks the supervisor to put a file there.
for req in spec.get("mirror") or []:
    with open(os.path.join(out, req["file"]), "w") as fh:
        fh.write("weights" * 64)
    with open(os.path.join(out, "mirror_pending.jsonl"), "a") as fh:
        fh.write(json.dumps(req) + "\\n")
        fh.flush()
    time.sleep(float(spec.get("mirror_pause", 0.0)))
print("step 2/2", flush=True)
if spec.get("rc"):
    sys.exit(int(spec["rc"]))
cfg = json.loads(os.environ["TIDEPOOL_PACK_CFG"])
json.dump({"tokens_per_second": 100.0, "peak_gpu_reserved_gb": 1.0, "card_total_gb": 47.7,
           "saw_batch_size": cfg.get("batch_size"), "saw_arm": cfg.get("arm"),
           # The objects this child could actually resolve locally. The supervisor snapshots
           # the map at spawn time, so this is the only place a consumer's view of a
           # handover is visible from outside.
           "saw_local": sorted(json.loads(os.environ["TIDEPOOL_PACK_LOCAL"]))},
          open(os.path.join(out, "score.json"), "w"))
'''


def run(cfg, specs, arms_scripts=("main.py",)):
    d = tempfile.mkdtemp(prefix="packtest-")
    try:
        os.makedirs(os.path.join(d, "lab"))
        open(os.path.join(d, "lab", "__init__.py"), "w").write(FAKE_LAB)
        store = os.path.join(d, "store")
        os.makedirs(store)
        shutil.copy(os.path.join(HERE, "pack.py"), d)
        for name in arms_scripts:
            open(os.path.join(d, name), "w").write(ARM_SCRIPT)
        rec = os.path.join(d, "record.jsonl")
        env = dict(os.environ)
        env.update({"FAKE_LAB_RECORD": rec, "FAKE_LAB_CFG": json.dumps(cfg),
                    "FAKE_LAB_STORE": store, "ARM_SPEC": json.dumps(specs),
                    "PYTHONPATH": d})
        p = subprocess.run([sys.executable, "-u", "pack.py"], cwd=d, env=env,
                           capture_output=True, text=True, timeout=180)
        events = [json.loads(l) for l in open(rec)] if os.path.exists(rec) else []
        summary = os.path.join(d, "out", "pack_summary.json")
        return {"rc": p.returncode, "events": events, "dir": d, "stdout": p.stdout,
                "summary": json.load(open(summary)) if os.path.exists(summary) else None,
                "store": sorted(os.listdir(store))}
    finally:
        shutil.rmtree(d, ignore_errors=True)


BASE = {"base_model": "", "poll_seconds": 0.1, "stagger_seconds": 0,
        "shutdown_grace_seconds": 2, "stall_minutes": 60, "card_gb": 47.7,
        "train_object": "corpus/train.jsonl.gz"}

fails = []


def check(name, cond, detail=""):
    if not cond:
        fails.append("%s: %s" % (name, detail or "did not hold"))


# 1. A producer hands its file to the arm waiting on it, without a round trip through storage.
r = run(dict(BASE, arms="GEN,T1,T2", pack_gb="12,10,10",
             pack_after="T2=GEN", pack_provides="GEN=replay_object:replay.bin",
             replay_object="tidepool/s5.3/replay/replay.jsonl.gz"),
        {"GEN": {"sleep": 1.0, "produce": "replay.bin"}, "T1": {"sleep": 2.0},
         "T2": {"sleep": 0.2}})
s = r["summary"]
check("handover", s and not s["failed"], "pack reported %s" % (s and s["failed"]))
check("handover", s and s["never_started"] == {}, "arms were skipped: %s"
      % (s and s["never_started"]))
saw = [e for e in r["events"] if e["kind"] == "upload"]
check("handover", any(e["obj"] == "tidepool/s5.3/replay/replay.jsonl.gz" for e in saw),
      "the supervisor never uploaded the produced object; got %s" % saw)
check("handover", "tidepool__s5.3__replay__replay.jsonl.gz" in r["store"],
      "the produced object is not in storage: %s" % r["store"])
# The consumer must have been handed the producer's own file, not a download.
log = r["stdout"]
check("handover", "is produced inside this pack, so it is not fetched" in log,
      "the supervisor fetched an object one of its own arms produces")
check("handover", "it stands in for tidepool/s5.3/replay/replay.jsonl.gz" in log,
      "the produced file was never registered for the waiting arm")

# 2. A producer that fails must not let its consumers run on nothing.
r = run(dict(BASE, arms="GEN,T1,T2", pack_gb="12,10,10",
             pack_after="T2=GEN", pack_provides="GEN=replay_object:replay.bin",
             replay_object="tidepool/s5.3/replay/replay.jsonl.gz"),
        {"GEN": {"sleep": 0.3, "rc": 3}, "T1": {"sleep": 1.0}, "T2": {}})
s = r["summary"]
check("failed producer", s is not None, "no pack summary was written")
check("failed producer", s and "T2" in (s["never_started"] or {}),
      "T2 ran even though GEN failed: %s" % (s and s["never_started"]))
check("failed producer", s and "T1" in s["results"],
      "T1 was collateral damage from GEN's failure: %s" % (s and s["failed"]))
check("failed producer", any(e["kind"] == "error" for e in r["events"]),
      "a pack with a failed arm reported success")

# 3. A producer that exits clean but writes nothing is the same case, and must be caught.
r = run(dict(BASE, arms="GEN,T2", pack_gb="12,10",
             pack_after="T2=GEN", pack_provides="GEN=replay_object:replay.bin",
             replay_object="obj/replay.gz"),
        {"GEN": {"sleep": 0.3}, "T2": {}})
s = r["summary"]
check("empty producer", s and "T2" in (s["never_started"] or {}),
      "T2 ran on a file that was never written: %s" % (s and s["never_started"]))

# 4. The ceiling is checked against the pack's worst moment. GEN+T1 and T1+T2 each fit; the
#    flat sum of all three does not, and rejecting on the flat sum would be the bug.
r = run(dict(BASE, arms="GEN,T1,T2", pack_gb="18,18,18", pack_headroom=0.9,
             pack_after="T2=GEN", pack_provides="GEN=replay_object:r.bin",
             replay_object="obj/r.gz"),
        {"GEN": {"sleep": 0.3, "produce": "r.bin"}, "T1": {"sleep": 0.5}, "T2": {}})
check("peak sizing", r["summary"] is not None,
      "the supervisor rejected a pack that fits at every instant (rc=%s)" % r["rc"])
check("peak sizing", r["summary"] and r["summary"]["peak_concurrent_demand_gb"] == 36.0,
      "peak demand came out %s, expected 36.0"
      % (r["summary"] or {}).get("peak_concurrent_demand_gb"))

# 5. A pack that genuinely does not fit is refused before anything is spawned.
r = run(dict(BASE, arms="A,B,C", pack_gb="20,20,20", pack_headroom=0.9), {})
check("overcommit", r["rc"] == 2, "an over-committed pack was allowed to start (rc=%s)" % r["rc"])
check("overcommit", any(e["kind"] == "error" and "peaks at" in (e.get("message") or "")
                        for e in r["events"]),
      "no error explained the refusal: %s" % [e for e in r["events"] if e["kind"] == "error"])

# 6. Per-arm scripts, and a script that is not there is refused rather than discovered late.
r = run(dict(BASE, arms="A,B", pack_gb="10,10", pack_scripts="B=generate.py"),
        {"A": {}, "B": {}}, arms_scripts=("main.py", "generate.py"))
check("scripts", r["summary"] and r["summary"]["per_arm_script"]["B"] == "generate.py",
      "pack_scripts was not honoured: %s" % (r["summary"] or {}).get("per_arm_script"))
r = run(dict(BASE, arms="A,B", pack_gb="10,10", pack_scripts="B=absent.py"), {})
check("scripts", r["rc"] == 2, "an arm pointing at a missing script was allowed to start")

# 7. Malformed wiring is refused, not silently ignored.
for bad, why in (
    ({"pack_after": "T2=NOPE"}, "a dependency on an arm that is not in the pack"),
    ({"pack_after": "A=A"}, "an arm waiting on itself"),
    ({"pack_provides": "A=nokey"}, "a provides clause with no filename"),
    ({"pack_gb": "10"}, "a ceiling list shorter than the arm list"),
):
    cfg = dict(BASE, arms="A,B", pack_gb="10,10")
    cfg.update(bad)
    r = run(cfg, {})
    check("validation", r["rc"] == 2, "%s was accepted" % why)

# 8. Per-arm config patches, for child scripts that cannot recognise their own arm from a
#    name -- an evaluation arm is a checkpoint and a component, not a recipe in a table.
r = run(dict(BASE, arms="A,B", pack_gb="10,10",
             pack_overrides=json.dumps({"B": {"batch_size": 32, "smoke": True}})),
        {"A": {}, "B": {}})
check("overrides", r["summary"] and r["summary"]["overrides"] == {"B": {"batch_size": 32,
                                                                       "smoke": True}},
      "pack_overrides was not recorded: %s" % (r["summary"] or {}).get("overrides"))
check("overrides", r["summary"] and not r["summary"]["failed"],
      "a pack with overrides failed: %s" % (r["summary"] or {}).get("failed"))
# The point is what the child actually received, not what the supervisor recorded.
res = (r["summary"] or {}).get("results") or {}
check("overrides", res.get("B", {}).get("saw_batch_size") == 32,
      "B did not receive its patched batch_size: %s" % res.get("B"))
check("overrides", res.get("A", {}).get("saw_batch_size") is None,
      "A received B's patch; overrides are leaking across arms: %s" % res.get("A"))
for bad, why in (('{"NOPE": {"x": 1}}', "an override for an arm not in the pack"),
                 ('{"A": 5}', "an override that is not an object"),
                 ('not json', "an override that is not JSON")):
    r = run(dict(BASE, arms="A,B", pack_gb="10,10", pack_overrides=bad), {})
    check("overrides", r["rc"] == 2, "%s was accepted" % why)

# 8b. The same patch, arriving already parsed. A `-p key=<json>` argument is typed on the way
#     in, so the supervisor can be handed a dict rather than the text that was typed; str() on
#     a dict is a Python repr, not JSON, and an earlier version rejected its own valid input
#     this way after the launch had already been paid for. Both shapes must work.
for form, ov in (("a dict", {"B": {"batch_size": 32, "smoke": True}}),
                 ("a python repr", repr({"B": {"batch_size": 32, "smoke": True}}))):
    r = run(dict(BASE, arms="A,B", pack_gb="10,10", pack_overrides=ov), {"A": {}, "B": {}})
    check("overrides", r["rc"] == 0 and r["summary"] and not r["summary"]["failed"],
          "pack_overrides as %s was rejected (rc=%s)" % (form, r["rc"]))
    res = (r["summary"] or {}).get("results") or {}
    check("overrides", res.get("B", {}).get("saw_batch_size") == 32,
          "B did not receive its patch when pack_overrides arrived as %s: %s" % (form, res.get("B")))

# 8c. Mirroring. A child has no job API and the `lab` CLI is not installed on these machines,
#     so the only way its weights and checkpoints reach shared storage is the supervisor doing
#     it on request. Before this existed, three arms of a nine-hour pack reported clean and
#     left nothing behind: the request path is what that failure costs to catch.
r = run(dict(BASE, arms="A,B", pack_gb="10,10"),
        {"A": {"mirror": [{"file": "adapter.zip", "dest": "arms/A"},
                          {"file": "ckpt-a.pt", "dest": "ckpt/A"}], "mirror_pause": 0.05},
         "B": {"mirror": [{"file": "adapter.zip", "dest": "arms/B"}]}})
ups = [e["obj"] for e in r["events"] if e["kind"] == "upload"]
check("mirror", r["rc"] == 0 and r["summary"] and not r["summary"]["failed"],
      "a pack that mirrored files failed: %s" % (r["summary"] or {}).get("failed"))
check("mirror", sorted(ups) == ["arms/A/adapter.zip", "arms/B/adapter.zip",
                               "ckpt/A/ckpt-a.pt"],
      "the supervisor did not upload exactly what the children asked for: %s" % ups)
check("mirror", not (r["summary"] or {}).get("mirror_failures"),
      "mirroring reported failures: %s" % (r["summary"] or {}).get("mirror_failures"))
mir = (r["summary"] or {}).get("mirrored") or {}
check("mirror", len(mir.get("A") or []) == 2 and all(m["ok"] for m in mir.get("A") or []),
      "the summary does not record A's two mirrored files: %s" % mir.get("A"))

#     A request naming a file that is not there is reported, not silently dropped, and does
#     not take the pack down with it.
r = run(dict(BASE, arms="A", pack_gb="10"),
        {"A": {"mirror": [{"file": "adapter.zip", "dest": "arms/A"}]}})
check("mirror", r["rc"] == 0, "a lone mirroring arm failed (rc=%s)" % r["rc"])

# 9. Inputs are staged once per card, by naming convention, and per-arm inputs are found in
#    the patches -- an evaluation pack keeps each arm's checkpoint there, not in the shared
#    config. An object no arm reads must not take the pack down.
r = run(dict(BASE, arms="A,B", pack_gb="10,10", val_object="corpus/train.jsonl.gz",
             pack_overrides=json.dumps({"A": {"adapter_object": "ckpt/a.tar"},
                                        "B": {"adapter_object": "ckpt/b.tar"}})),
        {"A": {}, "B": {}})
out = r["stdout"]
check("staging", out.count("fetched ") == 3,
      "expected 3 distinct objects staged (one shared, two per-arm), saw %d in:\n%s"
      % (out.count("fetched "), out))
check("staging", r["summary"] and not r["summary"]["failed"],
      "staging per-arm objects failed the pack: %s" % (r["summary"] or {}).get("failed"))

# 10. Two producers on one card, each standing in for a DIFFERENT object. s5.5 needs this:
#     one replay buffer sampled proportionally and one reweighted towards verifiable
#     constraints, generated side by side and then read by three trainers. The failure it
#     guards against is quiet and expensive -- `pack_provides` resolves the object PATH out of
#     the pack config by key, so if both generators named the key `replay_object` the second
#     registration would overwrite the first and two of the three trainers would train on a
#     buffer that is not the one their arm is defined by. Nothing would crash; the composition
#     contrast would simply be measuring nothing, after twelve hours on a card.
r = run(dict(BASE, arms="GP,GC,T1,T2,T3", pack_gb="18,18,9.5,9.5,9.5", pack_headroom=0.92,
             pack_after="T1=GC,T2=GP,T3=GC",
             pack_provides="GP=replay_object_proportional:replay.jsonl.gz,"
                           "GC=replay_object_constraint:replay.jsonl.gz",
             replay_object_proportional="tidepool/s5.5/replay_proportional/replay.jsonl.gz",
             replay_object_constraint="tidepool/s5.5/replay_constraint/replay.jsonl.gz"),
        {"GP": {"sleep": 1.2, "produce": "replay.jsonl.gz"},
         "GC": {"sleep": 0.6, "produce": "replay.jsonl.gz"},
         "T1": {"sleep": 0.3}, "T2": {"sleep": 0.3}, "T3": {"sleep": 0.3}})
s = r["summary"]
check("two buffers", s and not s["failed"] and s["never_started"] == {},
      "the two-producer pack did not complete: failed=%s never_started=%s"
      % ((s or {}).get("failed"), (s or {}).get("never_started")))
ups = sorted(e["obj"] for e in r["events"] if e["kind"] == "upload")
check("two buffers", ups == ["tidepool/s5.5/replay_constraint/replay.jsonl.gz",
                             "tidepool/s5.5/replay_proportional/replay.jsonl.gz"],
      "the two buffers did not both reach storage under their own names: %s" % ups)
# What each trainer was actually handed, as the child itself saw it. A consumer spawns with a
# snapshot of the resolved map, so T1 and T3 -- released by the faster GC while GP is still
# generating -- must see the constraint buffer and only that one, and T2 must see both by the
# time GP releases it. The wrong-key bug shows up here as T2 resolving the constraint path.
PROP = "tidepool/s5.5/replay_proportional/replay.jsonl.gz"
CONS = "tidepool/s5.5/replay_constraint/replay.jsonl.gz"
res = (s or {}).get("results") or {}
for arm, want in (("T1", [CONS]), ("T3", [CONS]), ("T2", sorted([CONS, PROP]))):
    got = [o for o in (res.get(arm) or {}).get("saw_local") or [] if "replay" in o]
    check("two buffers", got == want,
          "%s resolved %s, expected %s" % (arm, got, want))
check("two buffers", s and s.get("peak_concurrent_demand_gb") == 37.0,
      "peak came out %s, expected 37.0 (GP still running while GC's two consumers start)"
      % (s or {}).get("peak_concurrent_demand_gb"))

if fails:
    print("FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("pack scheduling holds: handover, failed producer, empty producer, peak sizing, "
      "overcommit, per-arm scripts, per-arm config (typed and as text), input staging, "
      "mirroring on request, two producers providing two distinct objects, "
      "7 validation cases")
