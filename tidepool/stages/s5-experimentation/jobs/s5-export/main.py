"""Merge, convert, quantize and price the checkpoints s5.5 selected.

Row G1 of the plan, and the first half of s5.6. It turns an s5.3-style LoRA arm into the two
artifacts the project actually promised -- a GGUF at 1.5 GB or less on disk, and a decode
throughput within 5% of the base's at the same format on the same backend -- and it does so
through the same pinned llama.cpp the three published 4-bit baseline rows were measured on.

WHY THREE POINTS PER ARM AND NOT TWO. A Q4-against-FP delta taken across `transformers` and
`llama.cpp` is a delta between two deployed artifacts, weights and runtime together. That is
the thing the headline promise is about, so it is the right number to report -- but it is not
the number H4 is written against, and the project has already paid once at s5.2 for confusing
them. So every arm is exported at F16 as well as at each 4-bit format. F16-in-llama.cpp
against Q4-in-llama.cpp isolates quantization; the s5-eval full-precision row against
F16-in-llama.cpp prices the runtime. Both fall out of the same conversion and cost one extra
file each.

WHAT THIS JOB DOES NOT DO. It does not score anything. Retention is a benchmark question and
belongs to `s5-eval-pack`, which now serves a stored GGUF through `gguf_object`. This job
produces the files that pass names, plus the two facts a benchmark cannot give: bytes on disk
and tokens per second from `llama-bench`.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import time

from lab import lab

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters  # noqa: E402  -- vendored from s5-eval, same resolution rules

lab.init()
cfg = lab.get_config() or {}
log = lab.log

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
NOTES = []
FACTS = {"arms": {}}
FAILURES = []


def C(key, default=None):
    v = cfg.get(key, default)
    return default if v in (None, "") else v


def sh(cmd, cwd=None, timeout=7200, check=True, quiet=False):
    """Run a command and capture its combined output as text.

    `errors="replace"` is load-bearing, not defensive. 39ccd302 got the whole
    conversion path working and then died decoding llama-quantize's own progress
    output: it writes a raw 0xc4 byte partway through the tensor table, which is not
    valid UTF-8, and a strict decode raises AFTER the child has already finished
    successfully. The quantized file was on disk; the run failed reading about it.
    """
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, timeout=timeout, shell=isinstance(cmd, str),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, errors="replace")
    tail = "\n".join((p.stdout or "").strip().splitlines()[-40:])
    if not quiet:
        log("$ %s  (%.0fs, rc=%d)" % (cmd if isinstance(cmd, str) else " ".join(cmd),
                                      time.time() - t0, p.returncode))
    if p.returncode != 0:
        if check:
            raise RuntimeError("command failed: %s\n%s" % (cmd, tail))
        log(tail)
    return p.returncode, p.stdout or ""


def put(path, obj):
    """Upload, trying the call shapes this SDK has been observed to accept.

    d4a7d46b lost every one of its storage mirrors to
    `Lab.storage_upload() got an unexpected keyword argument 'dest'` and the adapters had to be
    recovered from job artifacts by hand. The signature is not stable across images, so the
    shapes are tried in order and the one that worked is recorded rather than assumed. A file
    that does not reach storage is still on the job record; what must not happen is a job that
    believes it mirrored something and did not.
    """
    up = None
    for name in ("storage_upload", "upload_storage", "storage_put"):
        up = getattr(lab, name, None)
        if callable(up):
            break
        up = None
    if up is None:
        NOTES.append("this SDK exposes no storage upload; %s stays on the job record only" % obj)
        return False
    for args, kwargs in (((path, obj), {}),
                         ((path,), {"dest": os.path.dirname(obj)}),
                         ((path,), {})):
        try:
            up(*args, **kwargs)
            FACTS.setdefault("upload_shape", repr(kwargs or "positional"))
            return True
        except Exception as exc:
            log("upload shape %r failed for %s: %s" % (kwargs or "positional", obj, exc))
    NOTES.append("every upload shape failed for %s; it stays on the job record only" % obj)
    return False


# ---------------------------------------------------------------- 1. the serving path

llama_object = str(C("llama_object", ""))
if not llama_object:
    raise RuntimeError("`llama_object` is required: the converter and the quantizer must be the "
                       "same build as the runtime every 4-bit number is taken on")
log("== fetching the pinned serving path: %s ==" % llama_object)
tarball = lab.storage_download(llama_object)
if os.path.isdir(tarball):
    hits = sorted(glob.glob(os.path.join(tarball, "**", "*.tar.gz"), recursive=True))
    if len(hits) != 1:
        raise RuntimeError("`llama_object` resolved to a directory holding %d archives" % len(hits))
    tarball = hits[0]
BACKEND = os.path.join(OUT, "llama")
os.makedirs(BACKEND, exist_ok=True)
with tarfile.open(tarball) as tf:
    tf.extractall(BACKEND)
roots = [d for d in glob.glob(os.path.join(BACKEND, "*")) if os.path.isdir(d)]
if len(roots) != 1:
    raise RuntimeError("the serving-path archive unpacked to %d directories" % len(roots))
ROOT = roots[0]
QUANT = os.path.join(ROOT, "bin", "llama-quantize")
BENCH = os.path.join(ROOT, "bin", "llama-bench")
CONVERT = os.path.join(ROOT, "convert_hf_to_gguf.py")
for p in (QUANT, BENCH, CONVERT):
    if not os.path.exists(p):
        raise RuntimeError("the serving path is missing %s; it was built before the converter "
                           "travelled with the binaries and cannot be used for an export" % p)
for p in (QUANT, BENCH):
    os.chmod(p, 0o755)
os.environ["PYTHONPATH"] = os.path.join(ROOT, "gguf-py") + os.pathsep + os.environ.get("PYTHONPATH", "")

# THE BINARIES ARE NOT SELF-CONTAINED EITHER. `llama-bench` and `llama-quantize` are linked
# against the CUDA runtime stack -- including `libnccl.so.2` -- and the serving-path archive
# carries the executables without those shared objects, on the assumption that the host running
# them has them on the loader path. The build host did; this card's image does not, and attempt 2
# of this job (22a2d914) converted R3 to F16 cleanly and then hit `error while loading shared
# libraries: libnccl.so.2` on the very next call, twice: once on a bench, where the failure is
# recorded and survivable, and once on the quantizer, where it is fatal.
#
# The libraries are almost always already on the box, inside the Python environment: the pip
# CUDA wheels install them under `site-packages/nvidia/*/lib`, and torch pulls those wheels in.
# So the repair is to put those directories on the loader path rather than to install anything.
# Falling back to a wheel is the second move, and raising BEFORE the first merge is the third,
# for the same reason the converter is import-checked above: an arm that is merged and then
# cannot be quantized costs the merge for nothing.
def _nvidia_lib_dirs():
    bases = set()
    for key in ("purelib", "platlib"):
        d = sysconfig.get_paths().get(key)
        if d:
            bases.add(d)
    try:
        import site
        for d in site.getsitepackages():
            bases.add(d)
    except Exception:
        pass
    out = []
    for b in sorted(bases):
        out.extend(sorted(glob.glob(os.path.join(b, "nvidia", "*", "lib"))))
    # A serving path built after this defect may carry its own shared objects; look there first.
    for d in (os.path.join(ROOT, "lib"), os.path.join(ROOT, "lib64"),
              "/usr/local/cuda/lib64", "/usr/local/cuda/targets/x86_64-linux/lib"):
        if os.path.isdir(d):
            out.append(d)
    seen, uniq = set(), []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def _add_to_loader_path(dirs):
    if not dirs:
        return
    cur = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if d]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [d for d in cur if d not in dirs])


def _loads(binary):
    """Can the dynamic loader satisfy this binary? rc 127 and the loader's own message are the
    signal; any other non-zero exit is the tool talking about its arguments, not about linking."""
    rc, out = sh([binary, "-h"], timeout=180, check=False, quiet=True)
    if rc == 127 or "error while loading shared libraries" in out:
        missing = re.findall(r"([\w.+-]+\.so[\w.]*): cannot open shared object", out)
        return False, (missing[0] if missing else "an unresolved shared library")
    return True, None

FACTS["loader_repair"] = "none needed"
ok, missing = _loads(BENCH)
if not ok:
    dirs = _nvidia_lib_dirs()
    log("the serving path's binaries cannot load %s; adding %d CUDA library directories from "
        "this environment to the loader path" % (missing, len(dirs)))
    _add_to_loader_path(dirs)
    ok, missing = _loads(BENCH)
    FACTS["loader_repair"] = "site-packages CUDA wheels"
if not ok:
    log("still short of %s; installing the NCCL wheel" % missing)
    rc, _o = sh([sys.executable, "-m", "pip", "install", "-q", "nvidia-nccl-cu12"],
                timeout=900, check=False, quiet=True)
    _add_to_loader_path(_nvidia_lib_dirs())
    ok, missing = _loads(BENCH)
    FACTS["loader_repair"] = "nvidia-nccl-cu12 wheel (pip rc=%s)" % rc
if not ok:
    raise RuntimeError("the serving path's binaries cannot be loaded on this host: %s is "
                       "missing and neither this environment's CUDA wheels nor a fresh "
                       "nvidia-nccl-cu12 supplied it. Nothing has been merged." % missing)
for name, path in (("llama-quantize", QUANT),):
    good, miss = _loads(path)
    if not good:
        raise RuntimeError("%s cannot be loaded even after the loader repair: %s" % (name, miss))
FACTS["loader_path"] = os.environ.get("LD_LIBRARY_PATH", "")
log("converter, quantizer and bench all load on this host (loader repair: %s)"
    % FACTS["loader_repair"])
if FACTS["loader_repair"] != "none needed":
    NOTES.append("the serving-path archive carries the executables without their CUDA shared "
                 "objects and this host had none on the default loader path; repaired in-job "
                 "via %s, no rebuild" % FACTS["loader_repair"])

# THE CONVERTER IS NOT ONE FILE. At tag b10622 `convert_hf_to_gguf.py` is a thin front end that
# does `from conversion import ...`; the per-architecture writers, including `lfm2.py`, live in a
# sibling `conversion/` package. `s5-llama-build` copied the front end, `gguf-py` and the
# requirements file out of the source tree and not that package, so the serving path in shared
# storage has always been short of it -- attempt 1 of this job (e7dd289e) died on
# `ModuleNotFoundError: No module named 'conversion'` two seconds into the first conversion,
# after paying for the merge. The build script is fixed for the next rebuild, but rebuilding the
# CUDA runtime to recover 1.2 MB of pure Python would change the backend under every 4-bit number
# already recorded, so the package is carried as its own object cut from the SAME PINNED TAG and
# dropped in beside the front end.
#
# It is a repair, not a substitution: if a future serving path already carries `conversion/`,
# the archive is not fetched and nothing is overwritten.
CONVERSION = os.path.join(ROOT, "conversion")
conversion_object = str(C("conversion_object", ""))
if os.path.isdir(CONVERSION):
    FACTS["conversion_source"] = "serving path"
elif not conversion_object:
    raise RuntimeError("the serving path carries no `conversion/` package and no "
                       "`conversion_object` was given; `convert_hf_to_gguf.py` at this tag "
                       "cannot import its per-architecture writers without it")
else:
    log("the serving path carries no `conversion/` package; fetching %s" % conversion_object)
    got = lab.storage_download(conversion_object)
    if os.path.isdir(got):
        hits = sorted(glob.glob(os.path.join(got, "**", "*.tar.gz"), recursive=True))
        if len(hits) != 1:
            raise RuntimeError("`conversion_object` resolved to a directory holding %d archives"
                               % len(hits))
        got = hits[0]
    with tarfile.open(got) as tf:
        tf.extractall(ROOT)
    if not os.path.isdir(CONVERSION):
        raise RuntimeError("%s did not unpack a `conversion/` directory into the serving path"
                           % conversion_object)
    FACTS["conversion_source"] = conversion_object
    NOTES.append("`conversion/` was carried in from %s because the serving path predates the "
                 "converter's split into a package; same tag, no rebuild" % conversion_object)

# Fail here rather than inside the first conversion. The import is the whole point of the
# package, and an arm that is merged and then cannot be written costs the merge for nothing.
rc, _out = sh([sys.executable, "-c", "import conversion; conversion.get_model_class"],
              cwd=ROOT, check=False, quiet=True)
if rc != 0:
    raise RuntimeError("`conversion` does not import from the serving path (rc=%s); the "
                       "converter cannot run" % rc)
writers = len(glob.glob(os.path.join(CONVERSION, "*.py")))
FACTS["conversion_writers"] = writers
log("the converter's `conversion` package imports cleanly (%d writers, from %s)"
    % (writers, FACTS["conversion_source"]))

FACTS["llama_object"] = llama_object
FACTS["backend_root"] = os.path.basename(ROOT)
log("converter, quantizer and bench are all present under %s" % os.path.basename(ROOT))


# ---------------------------------------------------------------- 2. bench, once per file

BENCH_RE = re.compile(r"tg\d+\s*\|\s*([0-9.]+)\s*±")


def bench(gguf, label):
    """Decode throughput, from llama.cpp's own harness rather than a timer around a server.

    `llama-bench` reports prompt processing and token generation separately with a standard
    deviation over repetitions, which is the figure the efficiency floor is written against.
    A number taken by timing a serving loop would carry the scheduler and the HTTP layer with
    it and would not be comparable to the base's published figure.
    """
    rc, out = sh([BENCH, "-m", gguf, "-p", str(int(C("bench_prompt_tokens", 512))),
                  "-n", str(int(C("bench_new_tokens", 128))),
                  "-r", str(int(C("bench_reps", 3))), "-ngl", "99"],
                 timeout=1800, check=False)
    if rc != 0:
        FAILURES.append("llama-bench failed on %s" % label)
        return None
    hits = BENCH_RE.findall(out)
    if not hits:
        FAILURES.append("llama-bench produced no token-generation row for %s" % label)
        log("\n".join(out.strip().splitlines()[-20:]))
        return None
    return round(float(hits[-1]), 2)


# ---------------------------------------------------------------- 3. one arm at a time

base_model = str(C("base_model", "LiquidAI/LFM2.5-1.2B-Instruct"))
prefix = str(C("arms_prefix", "tidepool/s5.3/arms")).rstrip("/")
dest_prefix = str(C("dest_prefix", "tidepool/s5.6")).rstrip("/")
# `C` folds an empty string back to the default, so a run that wants only the variants asks
# for `formats=none` rather than `formats=""`.
_fraw = str(C("formats", "F16,Q4_0,Q4_K_M"))
formats = ([] if _fraw.strip().lower() == "none"
           else [f.strip() for f in _fraw.split(",") if f.strip()])
arms = [a.strip() for a in str(C("arms", "")).split(",") if a.strip()]
if not arms:
    raise RuntimeError("`arms` is empty; there is nothing to export")
ceiling_gb = float(C("size_ceiling_gb", 1.5))

# QUANT VARIANTS (s5.6 G4a). A plain `formats` entry is a bare `llama-quantize <in> <out> FMT`.
# A variant is the same call with the token-embedding and output tensors pinned to a higher
# precision than the body, which is what every high-quality 4-bit build in the ecosystem does
# and what baseline B3 (unsloth `UD-Q4_K_XL`) already is. The body stays 4-bit, so the
# deliverable is still a 4-bit GGUF; what changes is where the bit budget goes. Recorded here
# rather than hand-rolled at the CLI so the exact recipe travels with the artifact.
#
#   quant_variants = {"Q4_K_L": {"ftype": "Q4_K_M", "output_tensor_type": "q6_K",
#                                "token_embedding_type": "q6_K"}}
#
# The label is what the file and the arm are named; `ftype` is what the quantizer is actually
# asked for. Both must be present, and the label may not collide with a plain format.
_vraw = C("quant_variants", "")
if isinstance(_vraw, str):
    _vraw = json.loads(_vraw) if _vraw.strip() else {}
variants = dict(_vraw or {})
for _label, _spec in variants.items():
    if "__" in _label or "-" in _label:
        raise RuntimeError("variant label %r must survive `<arm>-<label>` naming and the "
                           "artifact prefix strip" % _label)
    if _label in formats:
        raise RuntimeError("variant %r collides with a plain format of the same name" % _label)
    if not _spec.get("ftype"):
        raise RuntimeError("variant %r does not say which quantizer type it is built on"
                           % _label)
if variants and "F16" not in [f.upper() for f in formats]:
    # F16 is the quantizer's input and is produced regardless; saying so keeps the log honest.
    log("F16 is not in `formats`, so it is converted as the quantizer's input and not uploaded")
if not formats and not variants:
    raise RuntimeError("neither `formats` nor `quant_variants` asks for anything")

FACTS["formats"] = formats
FACTS["quant_variants"] = variants
FACTS["size_ceiling_gb"] = ceiling_gb
log("== exporting %s at %s ==" % (", ".join(arms),
                                  ", ".join(formats + ["%s(%s)" % (k, v["ftype"])
                                                       for k, v in sorted(variants.items())])))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

for i, arm in enumerate(arms):
    row = {"formats": {}}
    FACTS["arms"][arm] = row
    obj = "%s/%s/adapter.zip" % (prefix, arm)
    log("-- %s: fetching %s" % (arm, obj))
    got = lab.storage_download(obj)
    src = adapters.resolve(got, os.path.join(OUT, arm, "ckpt"))
    kind = adapters.kind(src)
    row["checkpoint_kind"] = kind
    if kind is None:
        FAILURES.append("%s: %s is neither an adapter nor a full checkpoint" % (arm, obj))
        continue

    merged = os.path.join(OUT, arm, "merged")
    os.makedirs(os.path.dirname(merged), exist_ok=True)
    if kind == adapters.ADAPTER:
        log("%s: merging the rank-16 adapter onto %s" % (arm, base_model))
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)
        model = PeftModel.from_pretrained(model, src).merge_and_unload()
    else:
        log("%s: a full-parameter checkpoint, loaded as the model it is" % arm)
        model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)
    model.save_pretrained(merged, safe_serialization=True)
    # The tokenizer travels with the weights: the converter writes the vocabulary into the GGUF
    # and a GGUF carrying a different tokenizer from the one that rendered the prompts is the
    # quietest way to lose a comparison.
    AutoTokenizer.from_pretrained(base_model, trust_remote_code=True).save_pretrained(merged)
    del model
    torch.cuda.empty_cache()

    f16 = os.path.join(OUT, arm, "%s-F16.gguf" % arm)
    sh([sys.executable, CONVERT, merged, "--outfile", f16, "--outtype", "f16"], timeout=3600)
    row["merged_dir_gb"] = round(sum(os.path.getsize(p) for p in
                                     glob.glob(os.path.join(merged, "*"))) / 1e9, 3)

    plan = [(f, None) for f in formats] + [(k, v) for k, v in sorted(variants.items())]
    for fmt, spec in plan:
        if spec is None and fmt.upper() == "F16":
            path = f16
        else:
            path = os.path.join(OUT, arm, "%s-%s.gguf" % (arm, fmt))
            cmd = [QUANT]
            if spec:
                # Options precede the positional arguments in llama-quantize's parser.
                if spec.get("output_tensor_type"):
                    cmd += ["--output-tensor-type", spec["output_tensor_type"]]
                if spec.get("token_embedding_type"):
                    cmd += ["--token-embedding-type", spec["token_embedding_type"]]
            cmd += [f16, path, (spec or {}).get("ftype", fmt)]
            sh(cmd, timeout=3600)
        gb = round(os.path.getsize(path) / 1e9, 3)
        # F16 is an intermediate on the way to the 4-bit builds, not something anyone ships: a
        # 1.2B model at 16 bits is ~2.3 GB by arithmetic and can never sit under a 1.5 GB
        # ceiling. The pre-registered criterion is about the delivered build, so the ceiling is
        # judged on the quantized formats and F16's size is recorded without a verdict.
        deliverable = spec is not None or fmt.upper() != "F16"
        cell = {"gb": gb, "deliverable": deliverable, "recipe": spec,
                "under_ceiling": (gb <= ceiling_gb) if deliverable else None,
                "object": "%s/%s/%s" % (dest_prefix, arm, os.path.basename(path))}
        cell["tok_s"] = bench(path, "%s %s" % (arm, fmt))
        cell["uploaded"] = put(path, cell["object"])
        lab.save_artifact(path)
        row["formats"][fmt] = cell
        log("%s %s: %.3f GB, %s tok/s, ceiling %s"
            % (arm, fmt, gb, cell["tok_s"],
               ("ok" if cell["under_ceiling"] else "BREACHED") if deliverable
               else "n/a (intermediate)"))

    shutil.rmtree(merged, ignore_errors=True)
    lab.update_progress(int(80 * (i + 1) / len(arms)))


# ---------------------------------------------------------------- 4. the base's own 4-bit row

# The efficiency floor is written against the base's 4-bit throughput at identical format and
# identical backend on the same hardware. That number has to be taken in this job, on this card,
# in this process -- a figure carried over from s5.2 was measured through the serving loop and
# on a different launch, and 5% is tighter than the spread between those two conditions.
ref_repo = str(C("ref_gguf_repo", ""))
ref_file = str(C("ref_gguf_file", ""))
if ref_repo and ref_file:
    from huggingface_hub import hf_hub_download
    log("== the reference 4-bit row: %s :: %s ==" % (ref_repo, ref_file))
    ref = hf_hub_download(repo_id=ref_repo, filename=ref_file,
                          local_dir=os.path.join(OUT, "reference"))
    FACTS["reference"] = {"repo": ref_repo, "file": ref_file,
                          "gb": round(os.path.getsize(ref) / 1e9, 3),
                          "tok_s": bench(ref, "reference %s" % ref_file)}
else:
    NOTES.append("no `ref_gguf_repo`/`ref_gguf_file`, so the efficiency floor has no "
                 "same-process reference and must be read against a carried-over figure")

lab.update_progress(95)


# ---------------------------------------------------------------- 5. what the run claims

ref_tok = (FACTS.get("reference") or {}).get("tok_s")
for arm, row in FACTS["arms"].items():
    for fmt, cell in row["formats"].items():
        if ref_tok and cell.get("tok_s") and fmt.upper() != "F16":
            cell["tok_s_vs_reference"] = round(cell["tok_s"] / ref_tok - 1.0, 4)
            cell["within_5pct"] = cell["tok_s_vs_reference"] >= -0.05
        if cell.get("deliverable") and not cell["under_ceiling"]:
            FAILURES.append("%s %s is %.3f GB, over the %.2f GB ceiling"
                            % (arm, fmt, cell["gb"], ceiling_gb))

FACTS["notes"] = NOTES
FACTS["failures"] = FAILURES
FACTS["assertion_failures"] = len(FAILURES)
summary = os.path.join(OUT, "export_summary.json")
json.dump(FACTS, open(summary, "w"), indent=2)
lab.save_artifact(summary)
log(json.dumps(FACTS, indent=2))
lab.update_progress(100)
if FAILURES:
    raise RuntimeError("export finished with %d failure(s): %s"
                       % (len(FAILURES), "; ".join(FAILURES[:5])))
