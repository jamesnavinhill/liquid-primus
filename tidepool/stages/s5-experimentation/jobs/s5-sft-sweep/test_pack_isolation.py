"""Static checks on the packing contract.

Packing is only safe if a child arm never touches the job API and never reaches for the
network. Both are easy to break by accident: adding one `lab.log(...)` to a progress path,
or one `lab.storage_download(...)` to a new input, restores the exact failure packing was
built to avoid — N processes interleaving into one job's stream, or N copies of the corpus.
Neither shows up in a syntax check and both would only surface on a GPU, hours in, as
garbled logs or a disk-full.

So the contract is checked here, statically, over the syntax tree of every script an arm can
run. There is more than one now: `main.py` trains and `replay.py` generates, and a second
packable script is exactly where the contract quietly stops being checked.

  1. every `lab.<method>(...)` call is either guarded by `if not PACK_CHILD` or lives inside
     the `storage()` resolver
  2. `lab.storage_download` appears exactly once, inside `storage()`
  3. the child stub answers any attribute, so a missed guard degrades to a no-op

These run in this box in under a second and cost nothing.
"""

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Every script pack.py can hand to a child. Adding one here is the only step needed to bring
# a new packable program under the same contract.
# Paths are relative to this file, so a script belonging to a different job directory is
# named by its path: the contract belongs to the packing supervisor, not to one job.
CHILD_SCRIPTS = [f for f in ("main.py", "replay.py", "../s5-eval/main.py")
                 if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), f))]

fails = []


def guarded_by_not_pack_child(node):
    """True when `node` is a test of the form `not PACK_CHILD`."""
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
            and isinstance(node.operand, ast.Name) and node.operand.id == "PACK_CHILD")


def is_pack_child(node):
    """True when `node` is the bare test `PACK_CHILD`."""
    return isinstance(node, ast.Name) and node.id == "PACK_CHILD"


def mark(stmts, out):
    for st in stmts:
        for sub in ast.walk(st):
            if hasattr(sub, "lineno"):
                out.add(sub.lineno)


def audit(fname):
    """Run the whole contract over one child script."""
    SRC = open(os.path.join(HERE, fname)).read()
    TREE = ast.parse(SRC)

    # Three shapes make a call site safe, and the check accepts exactly these:
    #
    #   if not PACK_CHILD:  <call>          the explicit guard
    #   if PACK_CHILD: ... else: <call>     the same thing written the other way round
    #   if PACK_CHILD: return|raise         an early exit, after which the whole rest of the
    #   <call>                              block is unreachable for a child
    #
    # The third is the one worth being careful about: it is the idiom `log()` and `dump()` use,
    # and treating it as unguarded would push the file toward deeper nesting for no safety gain.
    guard_lines = set()
    for parent in ast.walk(TREE):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(parent, field, None)
            if not isinstance(block, list):
                continue
            exited = False
            for i, st in enumerate(block):
                if exited:
                    mark([st], guard_lines)
                    continue
                if isinstance(st, ast.If) and guarded_by_not_pack_child(st.test):
                    mark(st.body, guard_lines)
                elif isinstance(st, ast.If) and is_pack_child(st.test):
                    mark(st.orelse, guard_lines)
                    if st.body and isinstance(st.body[-1], (ast.Return, ast.Raise)):
                        exited = True

    # Lines inside the storage() resolver, the one place a download is legitimate.
    storage_lines = set()
    for n in ast.walk(TREE):
        if isinstance(n, ast.FunctionDef) and n.name == "storage":
            for sub in ast.walk(n):
                if hasattr(sub, "lineno"):
                    storage_lines.add(sub.lineno)
    assert storage_lines, "%s has no storage() resolver; packing cannot fetch inputs" % fname


    downloads = []
    for n in ast.walk(TREE):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
            continue
        f = n.func
        if not (isinstance(f.value, ast.Name) and f.value.id == "lab"):
            continue
        where = "lab.%s() at %s:%d" % (f.attr, fname, n.lineno)
        if f.attr == "storage_download":
            downloads.append(where)
            if n.lineno not in storage_lines:
                fails.append("%s is outside storage(); a packed child would download the corpus "
                             "once per arm" % where)
            continue
        if n.lineno in guard_lines or n.lineno in storage_lines:
            continue
        fails.append("%s is not guarded by `if not PACK_CHILD`; packed children would write "
                     "concurrently to one job's reporting stream" % where)

    if len(downloads) != 1:
        fails.append("expected exactly one lab.storage_download call (inside storage()), found "
                     "%d: %s" % (len(downloads), ", ".join(downloads)))

    # The stub must answer anything, so a guard that is ever missed degrades to a no-op printed
    # once rather than an AttributeError mid-training.
    ns = {}
    stub_src = SRC[SRC.index("class _SupervisedLab"):SRC.index("    lab = _SupervisedLab()")]
    exec(compile(ast.parse("\n".join(l[4:] if l.startswith("    ") else l
                                     for l in stub_src.splitlines())), "<stub>", "exec"), ns)
    stub = ns["_SupervisedLab"]()
    for method in ("log", "update_progress", "save_artifact", "error", "finish", "some_new_call"):
        try:
            stub.__getattr__(method)("x", k=1)
        except Exception as exc:
            fails.append("the child stub raised on lab.%s(): %s" % (method, exc))

    # The mirror image: capability probes must NOT be answered. `getattr(lab, "storage_upload",
    # None)` returning a no-op is worse than it failing, because the caller reports the no-op as
    # a successful upload and the weights are silently never written.
    for method in ("storage_upload", "upload_storage", "storage_put",
                   "storage_download", "get_config", "init"):
        try:
            got = getattr(stub, method, None)
        except Exception:
            got = None
        if got is not None:
            fails.append("the child stub answers lab.%s; a probe would take it for a working "
                         "call and report an upload that never happened" % method)

    return SRC


SOURCES = {f: audit(f) for f in CHILD_SCRIPTS}
SRC = SOURCES["main.py"]
EVAL = "../s5-eval/main.py"

# And pack.py must forward config and staged paths, or children silently train the default arm.
PACK = open(os.path.join(HERE, "pack.py")).read()
for needed in ("TIDEPOOL_PACK_CHILD", "TIDEPOOL_PACK_ARM", "TIDEPOOL_PACK_OUT",
               "TIDEPOOL_PACK_MEMFRAC", "TIDEPOOL_PACK_CFG", "TIDEPOOL_PACK_LOCAL",
               "TIDEPOOL_PACK_INDEX"):
    if needed not in PACK:
        fails.append("pack.py never sets %s" % needed)
for fname, src in SOURCES.items():
    for needed in ("TIDEPOOL_PACK_CHILD", "TIDEPOOL_PACK_ARM", "TIDEPOOL_PACK_OUT",
                   "TIDEPOOL_PACK_MEMFRAC", "TIDEPOOL_PACK_CFG", "TIDEPOOL_PACK_LOCAL"):
        if needed not in src:
            fails.append("%s never reads %s" % (fname, needed))

# The two scripts must share one stub and one resolver, or the contract holds in the file it
# was written in and drifts in the other. Comparing the text is crude and exactly right: the
# preamble is copied, so any edit to one that is not made to the other is the drift itself.
def preamble(src):
    a = src.index("PACK_CHILD = os.environ.get")
    b = src.index("def C(", a)
    return src[a:b].strip()


for other in [f for f in CHILD_SCRIPTS if f != "main.py"]:
    if preamble(SOURCES["main.py"]) != preamble(SOURCES[other]):
        fails.append("main.py and %s no longer share a packing preamble; one of them has a "
                     "stub or a storage() resolver the other does not" % other)

# An evaluation's arms differ in the checkpoint each one reads, and a checkpoint arrives as a
# per-arm override rather than in the shared config. If the supervisor only staged objects
# named in the shared config, every eval child would be handed a config naming an adapter and
# no local path for it, and would exit on the first read.
if "OVERRIDES[a] for a in sorted(OVERRIDES)" not in PACK:
    fails.append("pack.py does not stage objects named in per-arm overrides, so a packed "
                 "evaluation could not reach its checkpoints")

# The benchmark caches are the same hazard as the port, one level quieter: two children
# copying BFCL into one directory means one of them can read a half-written file.
if SOURCES.get(EVAL) and ('"data/bfcl"' in SOURCES[EVAL] or '"data/ifstruct_test.jsonl"' in SOURCES[EVAL]):
    fails.append("the evaluation harness still caches benchmarks in a shared `data/`; two "
                 "packed arms would write each other's files while reading them")

# The 4-bit server is a separate process on a fixed port. Two of them on one card is not an
# OOM: the second fails to bind and the first answers both arms, which yields two plausible
# score sets from one model.
GGUF = (open(os.path.join(HERE, "..", "s5-eval", "gen_gguf.py")).read()
        if os.path.exists(os.path.join(HERE, "..", "s5-eval", "gen_gguf.py")) else "")
if GGUF and ("def load_gguf(cfg, log=print, out_dir=" not in GGUF
             or "port if port is not None" not in GGUF):
    fails.append("gen_gguf.load_gguf does not take a per-arm port and output directory; two "
                 "packed 4-bit arms would collide on the server port")
# The reference `storage = lab.storage_download` is the solo-mode default and is fine; a
# *call* is not, because inside a pack it fetches once per arm from a child that is supposed
# to be isolated.
if "lab.storage_download(" in GGUF:
    fails.append("gen_gguf still calls lab.storage_download directly; inside a pack that is "
                 "one fetch per arm from a child that is supposed to be isolated")

if fails:
    print("FAIL")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
def lab_calls(src):
    return sum(1 for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and isinstance(n.func.value, ast.Name) and n.func.value.id == "lab")


print("pack isolation contract holds across %d child script(s): %s; stub total"
      % (len(CHILD_SCRIPTS),
         ", ".join("%s %d lab call site(s)" % (f, lab_calls(src))
                   for f, src in sorted(SOURCES.items()))))
