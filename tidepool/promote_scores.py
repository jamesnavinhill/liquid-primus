#!/usr/bin/env python3
"""Move a scoring job's per-arm verdicts into shared storage, where the compare job reads.

A packed arm writes its files locally and the supervisor uploads them as *job artifacts*,
flattened under an `<arm>__` prefix so eight arms' files can share one artifact list. The
comparison job reads *storage objects* under `tidepool/s5.3/arms/<arm>/`, because a
comparison spans several jobs and cannot be pinned to any one of them. So the files have to
be moved across, and the prefix stripped on the way.

This is file movement and nothing else: it computes no number, and it refuses to overwrite
an object whose bytes differ unless told to, because a rescore landing on top of the object
a recorded comparison was drawn from is the one way this step could quietly invalidate a
result. Run it with --dry-run first; it prints exactly what it would move.

    python3 promote_scores.py 385e210a --arms C1,C3 --dry-run
    python3 promote_scores.py 385e210a --arms C1,C3
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

EXPERIMENT = "tidepool"
PREFIX = "tidepool/s5.3/arms"
# score.json carries the arm's own reported rates, which the comparison cross-checks its
# recomputed rates against. eval_summary.json carries the run's provenance: item counts,
# limits, throughput, and the assertion list. Both travel with the verdicts.
WANTED_SUFFIXES = ("score.json", "eval_summary.json")
WANTED_GLOB = "scored_"


def run(cmd, **kw):
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def artifacts(job):
    out = run(["lab", "--format", "json", "job", "artifacts", job, "-e", EXPERIMENT]).stdout
    rows = json.loads(out)
    return [r["filename"] for r in rows if not r.get("is_directory")]


def existing():
    out = run(["lab", "--format", "json", "storage", "ls"]).stdout
    try:
        rows = json.loads(out)
    except Exception:                                              # noqa: BLE001
        return {}
    return {r["relpath"]: r for r in rows if isinstance(r, dict) and "relpath" in r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--arms", required=True, help="comma-separated arm names")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an object that is already there")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    names = artifacts(args.job)
    have = existing()
    plan = []
    for arm in arms:
        pre = arm + "__"
        mine = [n for n in names if n.startswith(pre)]
        if not mine:
            print("!! %s has no artifacts under %s in job %s" % (arm, pre, args.job))
            continue
        for n in mine:
            base = n[len(pre):]
            if not (base.startswith(WANTED_GLOB) or base in WANTED_SUFFIXES):
                continue
            dest = "%s/%s/%s" % (PREFIX, arm, base)
            plan.append((arm, n, base, dest, dest in have))

    if not plan:
        sys.exit("nothing to move: no scored files found for %s in %s"
                 % (",".join(arms), args.job))

    for arm, n, base, dest, clash in plan:
        print("%-4s %-34s -> %s%s" % (arm, n, dest, "   [EXISTS]" if clash else ""))
    clashes = [p for p in plan if p[4]]
    if clashes and not args.force:
        sys.exit("\n%d object(s) already exist. Re-run with --force only if this job is "
                 "meant to replace them, and say so in the run log: a comparison already "
                 "recorded against the old bytes is no longer reproducible from these."
                 % len(clashes))
    if args.dry_run:
        print("\ndry run, nothing moved")
        return

    tmp = tempfile.mkdtemp(prefix="promote-")
    for arm, n, base, dest, _clash in plan:
        run(["lab", "job", "download", args.job, "-e", EXPERIMENT, "--file", n,
             "-o", tmp])
        src = os.path.join(tmp, n)
        if not os.path.exists(src):
            # A single --file download lands under its own name; a zip fallback would not.
            cands = [os.path.join(dp, f) for dp, _d, fs in os.walk(tmp) for f in fs
                     if f == n]
            if not cands:
                sys.exit("downloaded %s but cannot find it under %s" % (n, tmp))
            src = cands[0]
        staged = os.path.join(tmp, arm, base)
        os.makedirs(os.path.dirname(staged), exist_ok=True)
        os.replace(src, staged)
        cmd = ["lab", "storage", "upload", staged, "--dest", "%s/%s" % (PREFIX, arm)]
        if args.force:
            cmd.append("--force")
        run(cmd)
        print("   moved %s (%d bytes)" % (dest, os.path.getsize(staged)))
    print("\n%d file(s) moved into %s" % (len(plan), PREFIX))


if __name__ == "__main__":
    main()
