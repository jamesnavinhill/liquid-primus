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

The same move is what a finished training pack needs for its checkpoints, which were done by
hand for Pack A. `--include` takes the file over instead, so Pack B's four adapters land in
the same place under the same overwrite rule:

    python3 promote_scores.py 1df3bf2b --arms C4,C5a,C5b,C6 --include adapter.zip --dry-run

An overwrite keeps the bytes it replaces. `--force` copies each existing object aside to
`<name>.superseded-by-<job>` before writing over it, which costs one extra round trip per
clashing file and buys two things. An overwrite becomes reversible, and where the two passes
share items the old and new verdicts can be compared: greedy argmax with left padding does not
depend on how prompts were grouped, so an item whose verdict moved between passes is a real
finding about determinism and not a rounding difference. Pass --no-keep-superseded when the
old bytes are genuinely worthless, and say in the run log why.
"""

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import tempfile

EXPERIMENT = "tidepool"
# Where the sweep arms live. The baselines live under `tidepool/s5.2/<row>` instead, so
# the destination is a flag rather than a constant; a baseline promoted into the sweep
# arms' directory would be picked up by a comparison that never meant to include it.
DEFAULT_PREFIX = "tidepool/s5.3/arms"
# score.json carries the arm's own reported rates, which the comparison cross-checks its
# recomputed rates against. eval_summary.json carries the run's provenance: item counts,
# limits, throughput, and the assertion list. Both travel with the verdicts.
DEFAULT_INCLUDE = ("scored_*", "score.json", "eval_summary.json")


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
    ap.add_argument("--no-keep-superseded", dest="keep_superseded",
                    action="store_false", default=True,
                    help="with --force, do not copy the replaced bytes aside first")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX,
                    help="storage directory the arms land under, one subdirectory per arm. "
                         "Default %(default)s, the sweep arms.")
    ap.add_argument("--include", action="append", metavar="GLOB",
                    help="filename glob, after the <arm>__ prefix is stripped, repeatable. "
                         "Default is the scored files, score.json and eval_summary.json.")
    args = ap.parse_args()
    include = tuple(args.include) if args.include else DEFAULT_INCLUDE
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    prefix = args.prefix.strip().rstrip("/")
    if not prefix:
        sys.exit("--prefix cannot be empty: the arms need a directory to land in")

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
            if not any(fnmatch.fnmatch(base, g) for g in include):
                continue
            dest = "%s/%s/%s" % (prefix, arm, base)
            plan.append((arm, n, base, dest, dest in have))

    if not plan:
        sys.exit("nothing to move: no artifact of %s matched %s for %s"
                 % (args.job, "/".join(include), ",".join(arms)))

    for arm, n, base, dest, clash in plan:
        print("%-4s %-34s -> %s%s" % (arm, n, dest, "   [EXISTS]" if clash else ""))
    clashes = [p for p in plan if p[4]]
    if clashes and args.force and args.keep_superseded:
        print("\n%d existing object(s) will be copied to <name>.superseded-by-%s first"
              % (len(clashes), args.job))
    elif clashes and args.force:
        print("\n%d existing object(s) will be overwritten and the old bytes discarded"
              % len(clashes))
    if clashes and not args.force:
        sys.exit("\n%d object(s) already exist. Re-run with --force only if this job is "
                 "meant to replace them, and say so in the run log: a comparison already "
                 "recorded against the old bytes is no longer reproducible from these."
                 % len(clashes))
    if args.dry_run:
        print("\ndry run, nothing moved")
        return

    tmp = tempfile.mkdtemp(prefix="promote-")
    for arm, n, base, dest, clash in plan:
        if clash and args.keep_superseded:
            keep_dir = os.path.join(tmp, "_superseded", arm)
            os.makedirs(keep_dir, exist_ok=True)
            run(["lab", "storage", "download", dest, keep_dir])
            got = os.path.join(keep_dir, base)
            if not os.path.exists(got):
                sys.exit("cannot copy aside %s: downloaded it but it is not at %s. Nothing "
                         "has been overwritten. Re-run with --no-keep-superseded only if "
                         "the old bytes are genuinely worthless." % (dest, got))
            aside = os.path.join(keep_dir, "%s.superseded-by-%s" % (base, args.job))
            os.replace(got, aside)
            run(["lab", "storage", "upload", aside, "--dest", "%s/%s" % (prefix, arm),
                 "--force"])
            print("   kept %s.superseded-by-%s (%d bytes)"
                  % (dest, args.job, os.path.getsize(aside)))
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
        cmd = ["lab", "storage", "upload", staged, "--dest", "%s/%s" % (prefix, arm)]
        if args.force:
            cmd.append("--force")
        run(cmd)
        print("   moved %s (%d bytes)" % (dest, os.path.getsize(staged)))
    print("\n%d file(s) moved into %s" % (len(plan), prefix))


if __name__ == "__main__":
    main()
