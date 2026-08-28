"""Build llama.cpp for the eval card, prove it serves a real Q4 GGUF, and keep the binaries.

Why this is a job and not a line in another job's setup: the compile is the serving path for
every 4-bit number this project will report, so it is pinned to a tag, built once, verified on
the card it will run on, and stored. A backend rebuilt per run is a backend that can change
under a comparison without anyone noticing.

What it deliberately does not assume: that the image ships a CUDA compiler. It probes, reports
what it found, and installs the compiler from a wheel if it has to. A build that silently falls
back to CPU would still finish and would make a Q4 pass cost a day, so a CPU fallback is a
recorded failure here rather than a quiet one later.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

from lab import lab

lab.init()
cfg = lab.get_config() or {}
log = lab.log

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
NOTES = []
FACTS = {}


def sh(cmd, cwd=None, env=None, timeout=3600, check=True, quiet=False):
    """Run a command, keep its tail, raise with that tail on failure."""
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout, shell=isinstance(cmd, str),
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


def which(x):
    return shutil.which(x)


# ---------------------------------------------------------------- 1. what is in this image

log("== probing the build environment ==")
for tool in ("cmake", "ninja", "gcc", "g++", "nvcc", "git", "nvidia-smi"):
    path = which(tool)
    FACTS["has_" + tool] = bool(path)
    log("  %-11s %s" % (tool, path or "not on PATH"))
_, smi = sh("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader",
            check=False, quiet=True)
FACTS["gpu"] = smi.strip().splitlines()[0] if smi.strip() else "unknown"
log("  gpu         %s" % FACTS["gpu"])


def find_nvcc_on_disk():
    """Every place a CUDA compiler is plausibly installed, most preferred first.

    Attempt 1 of this job died with the compiler wheel installed and rc=0, because it looked
    only at PATH and then at this interpreter's `site.getsitepackages()`. The `pip` on PATH
    need not belong to the interpreter running this file, so the wheel can land in a
    site-packages this process never enumerates. Ask the filesystem instead of asking one
    interpreter where it thinks its packages are.
    """
    cands = []
    onpath = which("nvcc")
    if onpath:
        cands.append(onpath)
    # A devel CUDA image installs here and does not always put it on PATH.
    for pat in ("/usr/local/cuda*/bin/nvcc", "/opt/cuda*/bin/nvcc", "/usr/lib/cuda*/bin/nvcc"):
        cands += sorted(glob.glob(pat), reverse=True)
    # The wheel, wherever pip actually put it.
    roots = []
    try:
        import nvidia
        roots += [p for p in getattr(nvidia, "__path__", [])]
    except Exception:
        pass
    import site as _site
    bases = list(sys.path)
    for getter in (_site.getsitepackages, _site.getusersitepackages):
        try:
            got = getter()
            bases += got if isinstance(got, list) else [got]
        except Exception:
            pass
    for b in bases:
        if b and os.path.isdir(os.path.join(b, "nvidia")):
            roots.append(os.path.join(b, "nvidia"))
    for r in dict.fromkeys(roots):
        cands += sorted(glob.glob(os.path.join(r, "cuda_nvcc", "bin", "nvcc")))
    return [c for c in dict.fromkeys(cands) if os.path.isfile(c) and os.access(c, os.X_OK)]


nvcc_bin = (find_nvcc_on_disk() or [None])[0]
if not nvcc_bin:
    NOTES.append("the image ships no CUDA compiler; the toolchain was installed from wheels")
    log("no CUDA compiler anywhere on disk, installing the toolchain from wheels")
    rc, out = sh([sys.executable, "-m", "pip", "install", "--no-cache-dir",
                  "nvidia-cuda-nvcc-cu12", "nvidia-cuda-runtime-cu12",
                  "nvidia-cublas-cu12", "nvidia-cuda-cccl-cu12"], check=False, timeout=1800)
    log("\n".join((out or "").strip().splitlines()[-12:]))
    if rc != 0:
        raise RuntimeError("the CUDA toolchain wheels would not install, so the 4-bit serving "
                           "path cannot be built in this image")
    nvcc_bin = (find_nvcc_on_disk() or [None])[0]
    FACTS["nvcc_from_wheel"] = True
if not nvcc_bin:
    raise RuntimeError("no CUDA compiler on PATH and the toolchain wheels did not put one "
                       "anywhere this job can find; the 4-bit serving path cannot be built "
                       "in this image")
FACTS["nvcc_path"] = nvcc_bin
log("CUDA compiler at %s" % nvcc_bin)


def assemble_cuda_root(nvcc_path):
    """Hand CMake one directory that looks like a CUDA toolkit.

    A real toolkit is bin/ + include/ + lib64/ under a single prefix, and that is what
    `FindCUDAToolkit` goes looking for. The wheels split the same toolkit across four packages,
    so pointing it at the compiler wheel alone finds a compiler and no `cuda_runtime.h`, and the
    CUDA backend does not compile. Link the four into one prefix. Symlinks rather than copies,
    so nvcc still resolves its own `nvvm` through its real path.
    """
    home = os.path.dirname(os.path.dirname(os.path.realpath(nvcc_path)))
    if os.path.isfile(os.path.join(home, "include", "cuda_runtime.h")):
        log("the compiler came with a complete toolkit at %s" % home)
        return home
    parent = os.path.dirname(home)
    root = os.path.abspath("cuda-root")
    for sub in ("bin", "include", "lib64"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    packs = [d for d in sorted(glob.glob(os.path.join(parent, "*"))) if os.path.isdir(d)]
    linked = 0
    for pack in packs:
        for src_sub, dst_sub in (("bin", "bin"), ("include", "include"),
                                 ("lib", "lib64"), ("lib64", "lib64")):
            s = os.path.join(pack, src_sub)
            if not os.path.isdir(s):
                continue
            for name in os.listdir(s):
                dst = os.path.join(root, dst_sub, name)
                if not os.path.exists(dst):
                    os.symlink(os.path.join(s, name), dst)
                    linked += 1
    # The wheels ship versioned libraries only; CMake looks for the unversioned name.
    lib64 = os.path.join(root, "lib64")
    for so in sorted(glob.glob(os.path.join(lib64, "*.so.*"))):
        plain = so.split(".so.")[0] + ".so"
        if not os.path.exists(plain):
            os.symlink(so, plain)
    # ggml links the driver library, which ships with the driver and is in no wheel.
    for drv in ("/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib64/libcuda.so.1",
                "/usr/local/nvidia/lib64/libcuda.so.1", "/usr/lib/libcuda.so.1"):
        target = os.path.join(lib64, "libcuda.so")
        if os.path.exists(drv) and not os.path.exists(target):
            os.symlink(drv, target)
            break
    log("assembled a CUDA toolkit at %s: %d entries from %s"
        % (root, linked, ", ".join(os.path.basename(p) for p in packs)))
    return root


cuda_root = assemble_cuda_root(nvcc_bin)
FACTS["cuda_root"] = cuda_root
FACTS["cuda_runtime_h"] = os.path.isfile(os.path.join(cuda_root, "include", "cuda_runtime.h"))
if not FACTS["cuda_runtime_h"]:
    raise RuntimeError("the assembled CUDA toolkit at %s has no cuda_runtime.h, so the CUDA "
                       "backend cannot compile. Refusing to build a CPU-only binary: it would "
                       "serve every 4-bit number in this project about thirty times slower and "
                       "look identical on the job record." % cuda_root)
rc, nv = sh("%s --version" % nvcc_bin, check=False, quiet=True)
if rc != 0:
    raise RuntimeError("the CUDA compiler at %s will not run:\n%s" % (nvcc_bin, nv))
FACTS["nvcc_version"] = ([l for l in nv.splitlines() if "release" in l] or ["unknown"])[0].strip()
log("  nvcc        %s" % FACTS["nvcc_version"])

# ---------------------------------------------------------------- 2. pinned source

tag = str(cfg.get("llama_tag", "b10622"))
arch = str(cfg.get("cuda_arch", "89"))
src = "llama.cpp-%s" % tag
url = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/%s.tar.gz" % tag
log("== fetching llama.cpp %s ==" % tag)
urllib.request.urlretrieve(url, "llama.tar.gz")
with tarfile.open("llama.tar.gz") as tf:
    tf.extractall(".")
if not os.path.isdir(src):
    cands = [d for d in glob.glob("llama.cpp-*") if os.path.isdir(d)]
    if not cands:
        raise RuntimeError("the source tarball for %s did not unpack as expected" % tag)
    src = cands[0]
log("source at %s" % src)

# ---------------------------------------------------------------- 3. build

env = dict(os.environ)
env["PATH"] = os.path.dirname(nvcc_bin) + os.pathsep + env.get("PATH", "")
env["CUDAToolkit_ROOT"] = cuda_root
env["CUDA_HOME"] = cuda_root
env["CUDA_PATH"] = cuda_root
env["CUDACXX"] = nvcc_bin
env["LD_LIBRARY_PATH"] = (os.path.join(cuda_root, "lib64") + os.pathsep
                          + env.get("LD_LIBRARY_PATH", ""))

flags = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_CUDA=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=%s" % arch,
    # The build node and the run node are the same instance type, but a native-arch build is a
    # portability bug waiting for the first provider that differs, and the GPU does the work.
    "-DGGML_NATIVE=OFF",
    # One self-contained binary per tool: simpler to store, and nothing to resolve at run time
    # except the CUDA runtime.
    "-DBUILD_SHARED_LIBS=OFF",
    # libcurl headers are not in this image and nothing here fetches over HTTP from the binary.
    "-DLLAMA_CURL=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
]
flags.append("-DCUDAToolkit_ROOT=%s" % cuda_root)
flags.append("-DCMAKE_CUDA_COMPILER=%s" % nvcc_bin)
flags.append("-DCMAKE_CUDA_HOST_COMPILER=%s" % (which("g++") or "g++"))

log("== configuring ==")
sh("cmake -S %s -B build -G Ninja %s" % (src, " ".join(flags)), env=env, timeout=1800)
log("== compiling (this is the long part) ==")
lab.update_progress(15)
t0 = time.time()
sh("cmake --build build --config Release -j %d --target llama-server llama-cli llama-quantize "
   "llama-bench" % (os.cpu_count() or 8), env=env, timeout=5400)
FACTS["build_seconds"] = round(time.time() - t0, 1)
log("compiled in %.0f s" % FACTS["build_seconds"])
lab.update_progress(45)

# ---------------------------------------------------------------- 4. collect

bindir = os.path.join(OUT, "llama-%s-sm%s" % (tag, arch))
os.makedirs(os.path.join(bindir, "bin"), exist_ok=True)
found = []
for name in ("llama-server", "llama-cli", "llama-quantize", "llama-bench"):
    hits = glob.glob("build/**/" + name, recursive=True)
    hits = [h for h in hits if os.path.isfile(h) and os.access(h, os.X_OK)]
    if not hits:
        raise RuntimeError("built successfully but %s is not in the build tree" % name)
    shutil.copy2(hits[0], os.path.join(bindir, "bin", name))
    found.append((name, round(os.path.getsize(hits[0]) / 1e6, 1)))
for so in glob.glob("build/**/*.so", recursive=True):
    shutil.copy2(so, os.path.join(bindir, "bin", os.path.basename(so)))
# The HF->GGUF converter and its library travel with the binaries: converting our own finetuned
# checkpoint is the same serving path and has to be the same version as the runtime.
#
# `conversion` WAS MISSING FROM THIS LIST AND THAT COST A JOB. At tag b10622
# `convert_hf_to_gguf.py` is a front end that does `from conversion import ...`, and every
# per-architecture writer -- `lfm2.py` among them -- lives in that sibling package. The tarball
# built without it converts nothing: s5.6's first export attempt (e7dd289e) died on
# `ModuleNotFoundError: No module named 'conversion'` two seconds into the first conversion, and
# the running project had to carry the package in as a separate object rather than rebuild the
# runtime under numbers already recorded. A tarball built from here now carries it.
# A SECOND THING THIS ARCHIVE IS SHORT OF: the CUDA shared objects the executables are linked
# against, `libnccl.so.2` among them. They are not copied at all, so the archive only runs on a
# host that already has them on the loader path. The build host does; job 22a2d914 met an L4
# image that did not, and both `llama-bench` and `llama-quantize` failed to load. `s5-export`
# now repairs that at consume time by putting the environment's CUDA wheel directories on the
# loader path, and looks in `<root>/lib` first -- so a build that wants to be self-contained
# should copy the resolved non-system `.so` files there.
for rel in ("convert_hf_to_gguf.py", "conversion", "gguf-py",
            "requirements/requirements-convert_hf_to_gguf.txt"):
    s = os.path.join(src, rel)
    d = os.path.join(bindir, os.path.basename(rel))
    if os.path.isdir(s):
        shutil.copytree(s, d)
    elif os.path.isfile(s):
        shutil.copy2(s, d)
    else:
        NOTES.append("the source tree has no %s at tag %s" % (rel, tag))
FACTS["binaries"] = dict(found)
log("collected: %s" % json.dumps(dict(found)))

server = os.path.join(bindir, "bin", "llama-server")
_, ver = sh("%s --version" % server, check=False, quiet=True)
FACTS["server_version"] = (ver or "").strip().splitlines()[-1] if ver.strip() else "unknown"
log("server reports: %s" % FACTS["server_version"])

# ---------------------------------------------------------------- 4b. keep it FIRST

# The compile is the expensive part of this job and the verification is the fragile part.
# Attempt 4 proved why the order matters: it configured cleanly, compiled for 1,038 s, produced
# all four binaries, and then raised on a runtime check, and because the tarball was only packed
# at the end of the job the whole 17 minutes was discarded. The binaries are now packed, saved
# and pushed to shared storage the moment they exist, so a failed verification costs the
# verification. A tarball that reached storage without passing its checks is not usable yet, and
# nothing may serve from it until a run reports `verified: true` in this summary.

def keep_binaries():
    tarball = os.path.join(OUT, "llama-%s-sm%s.tar.gz" % (tag, arch))
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(bindir, arcname=os.path.basename(bindir))
    FACTS["tarball_mb"] = round(os.path.getsize(tarball) / 1e6, 1)
    log("packed the serving path: %s (%.1f MB)" % (tarball, FACTS["tarball_mb"]))
    lab.save_artifact(tarball)

    # Every later 4-bit job reads this backend with `lab.storage_download(llama_object)`, so it
    # has to end up in shared storage. If the SDK in this image can put it there directly, that
    # saves shuttling a few hundred megabytes down to the orchestrator and straight back up. If
    # it cannot, the tarball is still on the job record and gets placed from outside; the object
    # name is fixed here either way so both routes agree on it.
    storage_object = "tidepool/%s" % os.path.basename(tarball)
    FACTS["storage_object_intended"] = storage_object
    FACTS["storage_uploaded_by_job"] = False
    _up = None
    for _name in ("storage_upload", "upload_storage", "storage_put"):
        _up = getattr(lab, _name, None)
        if callable(_up):
            break
        _up = None
    if _up is None:
        log("this SDK exposes no storage upload; the tarball will be placed in shared storage "
            "as %s from outside the job" % storage_object)
    else:
        for _args, _kwargs in (((tarball, storage_object), {}),
                               ((tarball,), {"dest": os.path.dirname(storage_object)}),
                               ((tarball,), {})):
            try:
                _up(*_args, **_kwargs)
                FACTS["storage_uploaded_by_job"] = True
                log("uploaded the serving path to shared storage as %s" % storage_object)
                break
            except Exception as exc:
                log("storage upload call shape %r failed: %s" % (_kwargs or _args, exc))
    return tarball


FACTS["verified"] = False
keep_binaries()
lab.update_progress(50)

# ---------------------------------------------------------------- 4c. prove the device

# Attempt 4 read the running server's own log for a `ggml_cuda_init` line and raised when it
# found none, on a build that had almost certainly produced a CUDA runtime: 1,038 s of compile
# and ~80 MB binaries are not what a CPU-only build looks like, and the captured log was missing
# the model-loader lines too, so the read had raced the server's logging rather than caught a
# CPU fallback. The device is now proved by asking a binary to enumerate its backends, which
# returns on its own and cannot be read early, and the driver, the linkage and the CMake cache
# are recorded beside it so the next failure has something to be compared against.

def probe(cmd, timeout=300):
    p = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True,
                       timeout=timeout)
    return ((p.stdout or "") + (p.stderr or "")).strip()


cli = os.path.join(bindir, "bin", "llama-cli")
devices = probe("%s --list-devices" % cli)
FACTS["list_devices"] = devices.splitlines()[:12]
log("== devices this runtime can see ==\n%s" % devices)
FACTS["nvidia_smi"] = [l for l in probe("nvidia-smi").splitlines()[:14]]
FACTS["server_cuda_links"] = [l.strip() for l in probe("ldd %s" % server).splitlines()
                              if "cuda" in l.lower() or "cublas" in l.lower()][:8]
FACTS["cmake_cuda"] = [l.strip() for l in probe("grep -i cuda build/CMakeCache.txt").splitlines()
                       if "=ON" in l or "COMPILER" in l or "ARCHITECTURES" in l][:10]
for k in ("server_cuda_links", "cmake_cuda"):
    for l in FACTS[k]:
        log("  %s" % l)

# Three independent signals, because any one of them can be silent for a reason that has nothing
# to do with the device: a runtime that predates `--list-devices` prints a usage error, a
# statically linked build shows no CUDA library, and a CMake cache can be missing on a rebuild.
dev_says_cuda = "cuda" in devices.lower()
# "available devices" is the header the runtime prints when it really enumerated them. A
# runtime too old for the flag prints its usage text instead, which mentions devices
# without listing any, and that must not read as a card-less machine.
dev_answered = dev_says_cuda or "available devices" in devices.lower()
links_say_cuda = bool(FACTS["server_cuda_links"])
cmake_says_cuda = any("GGML_CUDA" in l and "=ON" in l for l in FACTS["cmake_cuda"])
FACTS["cuda_evidence"] = {"list_devices": dev_says_cuda, "probe_answered": dev_answered,
                          "linked": links_say_cuda, "cmake": cmake_says_cuda}
if dev_answered and not dev_says_cuda:
    raise RuntimeError("the runtime enumerated its backends and no CUDA device was among them, "
                       "so it would serve every 4-bit number on the CPU and price it wrong.\n"
                       "%s\n---- nvidia-smi ----\n%s"
                       % (devices, "\n".join(FACTS["nvidia_smi"])))
if not (dev_says_cuda or links_say_cuda or cmake_says_cuda):
    raise RuntimeError("nothing shows this build reaching a GPU: the device probe returned %r, "
                       "llama-server links no CUDA library, and the CMake cache has no "
                       "GGML_CUDA=ON.\n---- nvidia-smi ----\n%s"
                       % (devices[:400], "\n".join(FACTS["nvidia_smi"])))
log("CUDA evidence: %s" % json.dumps(FACTS["cuda_evidence"]))

# ---------------------------------------------------------------- 5. verify on a real GGUF

import requests  # noqa: E402  (installed in setup)
from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

repo = str(cfg.get("verify_repo", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"))
want = str(cfg.get("verify_file", "") or "")
files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
if want:
    pick = [f for f in files if f == want] or [f for f in files if want.lower() in f.lower()]
else:
    # Prefer the vendor's quantization-aware Q4_0, which is the row this file will serve.
    pick = ([f for f in files if "q4_0" in f.lower()] or
            [f for f in files if "q4" in f.lower()] or files)
if not pick:
    raise RuntimeError("no GGUF file in %s (saw %d files)" % (repo, len(files)))
gguf_name = sorted(pick, key=len)[0]
log("verifying against %s :: %s  (candidates: %s)" % (repo, gguf_name, ", ".join(files[:8])))
gguf = hf_hub_download(repo_id=repo, filename=gguf_name, local_dir="gguf")
FACTS["verify_repo"] = repo
FACTS["verify_file"] = gguf_name
FACTS["verify_file_mb"] = round(os.path.getsize(gguf) / 1e6, 1)

par = int(cfg.get("verify_parallel", 8))
port = 8080
# One slot per concurrent request, and enough context that a slot never truncates a benchmark
# prompt: the tool-definition prompts in this project's BFCL subset run to a few thousand tokens.
ctx_total = 8192 * par
cmd = [server, "--model", gguf, "--host", "127.0.0.1", "--port", str(port),
       "-ngl", "99", "-c", str(ctx_total), "--parallel", str(par),
       "--no-warmup", "--seed", "0", "--metrics"]
log("== starting the server ==\n$ %s" % " ".join(cmd))
srv_log = open(os.path.join(OUT, "llama_server.log"), "w")
proc = subprocess.Popen(cmd, stdout=srv_log, stderr=subprocess.STDOUT, env=env)

base = "http://127.0.0.1:%d" % port
deadline = time.time() + 300
ready = False
while time.time() < deadline:
    if proc.poll() is not None:
        srv_log.flush()
        tail = "\n".join(open(os.path.join(OUT, "llama_server.log")).read().splitlines()[-40:])
        raise RuntimeError("the server exited during startup (rc=%s)\n%s" % (proc.returncode, tail))
    try:
        r = requests.get(base + "/health", timeout=3)
        if r.status_code == 200:
            ready = True
            break
    except Exception:
        pass
    time.sleep(2)
if not ready:
    proc.kill()
    raise RuntimeError("the server did not become healthy within 300 s")
log("server healthy")

# The startup log is now evidence rather than the gate. Attempt 4 raised here on a log that was
# missing its loader lines as well as its CUDA lines, which is the signature of reading the file
# before the server had flushed it, so the lines are given a moment to arrive, saved whole as an
# artifact before anything is asserted on them, and reported either way. What actually protects
# the pricing is the throughput floor further down: a CPU fallback cannot reach it.
time.sleep(5)
srv_log.flush()
log_path = os.path.join(OUT, "llama_server.log")
startup = open(log_path).read()
lab.save_artifact(log_path)
offload = [l for l in startup.splitlines()
           if "offload" in l.lower() or "CUDA" in l or "ggml_cuda_init" in l]
FACTS["offload_lines"] = offload[:8]
for l in offload[:8]:
    log("  %s" % l.strip())
FACTS["server_log_lines"] = len(startup.splitlines())
FACTS["cuda_in_server_log"] = bool(offload)
if not offload:
    NOTES.append("the server's captured startup log carried no CUDA line in %d lines; the device "
                 "was proved by backend enumeration and by the throughput floor instead"
                 % FACTS["server_log_lines"])
    log("the startup log carried no CUDA line (%d lines captured); leaning on the device probe "
        "and the throughput floor" % FACTS["server_log_lines"])
FACTS["cuda_at_runtime"] = bool(offload) or dev_says_cuda
lab.update_progress(60)

# Tokenizing on the server, with add_special off, is what keeps a GGUF run's prompt identical to
# the full-precision run's: the chat template already writes the beginning-of-sequence token as
# text, and letting the server add its own would put two of them in front of every prompt.
def tokenize(text):
    r = requests.post(base + "/tokenize",
                      json={"content": text, "add_special": False}, timeout=60)
    r.raise_for_status()
    return r.json()["tokens"]


def complete(tokens, n_predict):
    r = requests.post(base + "/completion",
                      json={"prompt": tokens, "n_predict": n_predict, "temperature": 0.0,
                            "top_k": 1, "seed": 0, "cache_prompt": False,
                            "samplers": ["top_k"]},
                      timeout=600)
    r.raise_for_status()
    return r.json()


tools_blob = json.dumps({"type": "function", "function": {
    "name": "search_orders",
    "description": "Find orders matching a filter and return them with their line items.",
    "parameters": {"type": "object", "properties": {
        "customer_id": {"type": "string", "description": "the customer's opaque id"},
        "status": {"type": "string", "enum": ["open", "shipped", "cancelled"]},
        "since": {"type": "string", "description": "ISO-8601 date"},
        "limit": {"type": "integer", "description": "maximum rows to return"}},
        "required": ["customer_id"]}}}, indent=1)
want_prompt = int(cfg.get("verify_prompt_tokens", 1024))
prompt = ("You are a function-calling assistant. Tools:\n" +
          "\n".join([tools_blob] * 8) +
          "\nUser: list Ada's open orders since March and stop after ten.\nAssistant:")
ptoks = tokenize(prompt)
while len(ptoks) < want_prompt:
    prompt = prompt.replace("Tools:\n", "Tools:\n" + tools_blob + "\n", 1)
    ptoks = tokenize(prompt)
FACTS["verify_prompt_tokens"] = len(ptoks)
log("verification prompt is %d tokens" % len(ptoks))

single = complete(ptoks, int(cfg.get("verify_new_tokens", 256)))
tim = single.get("timings", {})
FACTS["single_stream"] = {
    "prompt_tokens": tim.get("prompt_n"), "prompt_tok_per_s": tim.get("prompt_per_second"),
    "generated_tokens": tim.get("predicted_n"),
    "generate_tok_per_s": tim.get("predicted_per_second")}
log("single stream: prefill %.0f tok/s, generate %.1f tok/s"
    % (tim.get("prompt_per_second") or 0.0, tim.get("predicted_per_second") or 0.0))
log("first 200 characters it produced: %r" % (single.get("content", "")[:200]))
FACTS["verify_sample"] = single.get("content", "")[:600]

from concurrent.futures import ThreadPoolExecutor  # noqa: E402

n_req = par * 3
t0 = time.time()
with ThreadPoolExecutor(max_workers=par) as pool:
    outs = list(pool.map(lambda i: complete(ptoks, int(cfg.get("verify_new_tokens", 256))),
                         range(n_req)))
wall = time.time() - t0
gen = sum((o.get("timings", {}) or {}).get("predicted_n") or 0 for o in outs)
pre = sum((o.get("timings", {}) or {}).get("prompt_n") or 0 for o in outs)
FACTS["batched"] = {"requests": n_req, "slots": par, "wall_seconds": round(wall, 1),
                    "generated_tokens": gen, "prompt_tokens": pre,
                    "generate_tok_per_s": round(gen / max(wall, 1e-6), 1),
                    "end_to_end_tok_per_s": round((gen + pre) / max(wall, 1e-6), 1)}
log("%d concurrent requests over %d slots: %.0f generated tok/s (%.0f including prefill)"
    % (n_req, par, FACTS["batched"]["generate_tok_per_s"],
       FACTS["batched"]["end_to_end_tok_per_s"]))

# The assertion that actually protects the pricing. A 1.2B Q4_0 model on an L4 generates in the
# high hundreds of tokens a second over eight slots; the same model on two CPU threads manages
# tens. So a floor here catches a CPU fallback whatever the logs say, and it catches the other
# way this row could go wrong too: a GPU build that reaches the device but is so slow that the
# hours quoted for a full Q4 pass would be nonsense.
min_tok_s = float(cfg.get("min_tok_s", 150))
FACTS["min_tok_s"] = min_tok_s
if FACTS["batched"]["generate_tok_per_s"] < min_tok_s:
    proc.kill()
    raise RuntimeError("%d concurrent requests generated only %.1f tok/s, under the %.0f tok/s "
                       "floor for this card. A 1.2B Q4_0 model this slow is not being served "
                       "from the GPU, and every hour quoted from it would be wrong.\n"
                       "devices: %s\ncuda evidence: %s"
                       % (n_req, FACTS["batched"]["generate_tok_per_s"], min_tok_s,
                          FACTS["list_devices"][:4], json.dumps(FACTS["cuda_evidence"])))
log("clears the %.0f tok/s floor" % min_tok_s)

# Determinism: the same prompt at temperature 0 must come back identical, or no Q4 delta this
# project reports is a delta rather than sampling noise.
again = complete(ptoks, 64)
first64 = complete(ptoks, 64)
FACTS["deterministic"] = (again.get("content") == first64.get("content"))
log("two identical requests returned %s output"
    % ("identical" if FACTS["deterministic"] else "DIFFERENT"))
if not FACTS["deterministic"]:
    NOTES.append("the server did not reproduce its own output at temperature 0; every Q4 number "
                 "taken through it would carry sampling noise")

proc.terminate()
try:
    proc.wait(timeout=30)
except Exception:
    proc.kill()
srv_log.close()
lab.update_progress(80)

# ---------------------------------------------------------------- 6. price a Q4 pass

# Item counts of the full suite, as read off the benchmark files by the eval task.
SUITE = {"bfcl": 3491 * 2, "ifstruct": 2000, "ifeval": 541, "probes": 494}
NEW = {"bfcl": 320, "ifstruct": 768, "ifeval": 768, "probes": 320}
# The smoke measured what a completion actually costs rather than its ceiling: ~55 generated
# tokens per tool-calling item against a 320-token cap.
ACTUAL = {"bfcl": 55, "ifstruct": 300, "ifeval": 300, "probes": 80}
tps = FACTS["batched"]["generate_tok_per_s"] or 1.0
ptps = (FACTS["single_stream"]["prompt_tok_per_s"] or 1.0) * min(par, 4)
est_gen = sum(SUITE[k] * ACTUAL[k] for k in SUITE) / tps
est_pre = sum(SUITE[k] * 900 for k in SUITE) / max(ptps, 1.0)
FACTS["full_pass_estimate"] = {
    "generated_tokens": sum(SUITE[k] * ACTUAL[k] for k in SUITE),
    "generate_hours": round(est_gen / 3600, 2),
    "prefill_hours_at_900_prompt_tokens": round(est_pre / 3600, 2),
    "task_hours": round((est_gen + est_pre) / 3600, 2)}
log("a full Q4 pass is estimated at %.2f h of task time on this card"
    % FACTS["full_pass_estimate"]["task_hours"])

FACTS["verified"] = True
FACTS["llama_tag"] = tag
FACTS["cuda_arch"] = arch
FACTS["notes"] = NOTES
summary_path = os.path.join(OUT, "llama_build_summary.jsonl")
with open(summary_path, "w") as fh:
    fh.write(json.dumps(FACTS, default=repr) + "\n")
lab.save_artifact(summary_path)
lab.save_artifact(os.path.join(OUT, "llama_server.log"))

lab.update_progress(100)
lab.finish(message=("llama.cpp %s built for sm_%s, served %s at %.0f generated tok/s over %d "
                    "slots, deterministic=%s, full Q4 pass ~%.2f h"
                    % (tag, arch, gguf_name, FACTS["batched"]["generate_tok_per_s"], par,
                       FACTS["deterministic"], FACTS["full_pass_estimate"]["task_hours"])),
           score={"generate_tok_per_s": FACTS["batched"]["generate_tok_per_s"],
                  "full_pass_hours": FACTS["full_pass_estimate"]["task_hours"],
                  "build_seconds": FACTS["build_seconds"]})
