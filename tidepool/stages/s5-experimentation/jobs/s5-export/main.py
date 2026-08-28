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
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, timeout=timeout, shell=isinstance(cmd, str),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
formats = [f.strip() for f in str(C("formats", "F16,Q4_0,Q4_K_M")).split(",") if f.strip()]
arms = [a.strip() for a in str(C("arms", "")).split(",") if a.strip()]
if not arms:
    raise RuntimeError("`arms` is empty; there is nothing to export")
ceiling_gb = float(C("size_ceiling_gb", 1.5))
FACTS["formats"] = formats
FACTS["size_ceiling_gb"] = ceiling_gb
log("== exporting %s at %s ==" % (", ".join(arms), ", ".join(formats)))

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

    for fmt in formats:
        if fmt.upper() == "F16":
            path = f16
        else:
            path = os.path.join(OUT, arm, "%s-%s.gguf" % (arm, fmt))
            sh([QUANT, f16, path, fmt], timeout=3600)
        gb = round(os.path.getsize(path) / 1e9, 3)
        cell = {"gb": gb, "under_ceiling": gb <= ceiling_gb,
                "object": "%s/%s/%s" % (dest_prefix, arm, os.path.basename(path))}
        cell["tok_s"] = bench(path, "%s %s" % (arm, fmt))
        cell["uploaded"] = put(path, cell["object"])
        lab.save_artifact(path)
        row["formats"][fmt] = cell
        log("%s %s: %.3f GB, %s tok/s, ceiling %s"
            % (arm, fmt, gb, cell["tok_s"], "ok" if cell["under_ceiling"] else "BREACHED"))

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
        if not cell["under_ceiling"]:
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
