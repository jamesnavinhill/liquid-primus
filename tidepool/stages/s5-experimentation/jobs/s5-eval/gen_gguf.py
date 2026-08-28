"""The one serving path every 4-bit number in this project comes from.

The project's headline promise is that a 4-bit build holds the full-precision bar, and three of
the six baseline rows are published GGUF files. Neither can be measured through `transformers`:
GGUF runs in llama.cpp, and dequantizing a GGUF back into bf16 would measure the weights while
throwing away the runtime that actually ships. So the Q4 backend is llama.cpp's own server,
built once at a pinned tag by the `s5-llama-build` job and downloaded here rather than rebuilt.

**Two backends is a confound, and it is the honest one.** A Q4-versus-FP delta taken this way is
a delta between two deployed artifacts, weights and runtime together, which is the thing the
success criteria are about. To separate the two, the same F16 GGUF of the full-precision
checkpoint can be run through this backend as well: FP-in-llama.cpp against Q4-in-llama.cpp
isolates quantization, and FP-in-transformers against FP-in-llama.cpp prices the runtime. Both
are one parameter change, and which comparison a number belongs to is recorded in the summary.

**Prompt bytes are identical to the full-precision run.** Prompts are rendered by the same
Hugging Face tokenizer and the same `Prompter` the FP path uses, then tokenized by the server
with `add_special` off. The chat template already writes the beginning-of-sequence token as
text; letting the server prepend its own would put two in front of every prompt and quietly
change what is being compared.

Determinism: temperature 0, `top_k` 1, a fixed seed, and prompt caching off so a slot's history
cannot change a later completion. The build job asserts the server reproduces its own output
before any of this is used.
"""

import glob
import json
import os
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

import requests


class Server:
    """A llama.cpp server process, and the two endpoints this project uses."""

    def __init__(self, proc, url, log_path, gguf, slots, log=print):
        self.proc = proc
        self.url = url
        self.log_path = log_path
        self.gguf = gguf
        self.slots = slots
        self.log = log

    def tail(self, n=40):
        try:
            return "\n".join(open(self.log_path).read().splitlines()[-n:])
        except Exception:
            return "(no server log)"

    def tokenize(self, text):
        r = requests.post(self.url + "/tokenize",
                          json={"content": text, "add_special": False}, timeout=120)
        r.raise_for_status()
        return r.json()["tokens"]

    def complete(self, tokens, n_predict):
        r = requests.post(self.url + "/completion",
                          json={"prompt": tokens, "n_predict": n_predict, "temperature": 0.0,
                                "top_k": 1, "seed": 0, "cache_prompt": False, "stream": False,
                                "samplers": ["top_k"]},
                          timeout=1200)
        r.raise_for_status()
        return r.json()

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()


def _extract_backend(tarball_path, dest="llama"):
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tarball_path) as tf:
        tf.extractall(dest)
    for root, _dirs, files in os.walk(dest):
        if "llama-server" in files:
            path = os.path.join(root, "llama-server")
            os.chmod(path, 0o755)
            return path
    raise RuntimeError("the serving-path archive %s contains no llama-server" % tarball_path)


def resolve_gguf(cfg, storage, out_dir, log=print, hub=None, list_files=None):
    """Which .gguf this row serves, and where it came from. Returns (path, repo, name, source).

    Two sources, and exactly one of them per row. `gguf_repo`/`gguf_file` is a published
    quantization on the Hub, which is what the three 4-bit baseline rows are. `gguf_object` is
    a GGUF this project quantized itself and put in shared storage, which is what every s5.6
    export row is. They are mutually exclusive on purpose: a row that set both would serve one
    file and label itself with the other, and the label is what a retention number is
    attributed to.

    Its own function because it needs a test and `load_gguf` cannot be imported without the
    serving stack. `hub` and `list_files` are injected for the same reason.
    """
    obj = str(cfg.get("gguf_object", "") or "")
    repo = str(cfg.get("gguf_repo", "") or "")
    name = str(cfg.get("gguf_file", "") or "")
    if obj and repo:
        raise RuntimeError("`gguf_object` and `gguf_repo` are both set (%r, %r); a row serves "
                           "one file and may only be labelled with the one it served"
                           % (obj, repo))
    if obj:
        gguf = storage(obj)
        # Storage hands back either the file or a directory holding it, the same two shapes
        # `adapters.py` already has to distinguish for adapter archives.
        if os.path.isdir(gguf):
            hits = sorted(glob.glob(os.path.join(gguf, "**", "*.gguf"), recursive=True))
            if len(hits) != 1:
                raise RuntimeError("`gguf_object` %s resolved to a directory holding %d .gguf "
                                   "files; name the file, not the directory" % (obj, len(hits)))
            gguf = hits[0]
        size_mb = round(os.path.getsize(gguf) / 1e6, 1)
        log("weights: %s (%.1f MB), quantized by this project" % (obj, size_mb))
        return gguf, obj, os.path.basename(gguf), "storage"
    if not repo:
        raise RuntimeError("the 4-bit path needs `gguf_repo` or `gguf_object`")
    if not name:
        files = [f for f in (list_files(repo) if list_files else []) if f.lower().endswith(".gguf")]
        raise RuntimeError("`gguf_file` is empty and a repo publishes many quantizations, which "
                           "is not a choice a run should make for itself. Files: %s"
                           % ", ".join(sorted(files)))
    gguf = hub(repo_id=repo, filename=name, local_dir=os.path.join(out_dir, "gguf"))
    size_mb = round(os.path.getsize(gguf) / 1e6, 1)
    log("weights: %s :: %s (%.1f MB)" % (repo, name, size_mb))
    return gguf, repo, name, "hub"


def load_gguf(cfg, log=print, out_dir="out", storage=None, port=None):
    """Fetch the backend and the weights, serve them, and return (server, tokenizer, facts).

    `tokenizer_repo` is the full-precision repo the GGUF was quantized from. It is what renders
    the prompts, so a Q4 row and the FP row it is compared against see the same bytes.

    Three arguments exist so that several of these can serve on one card at the same time.
    `port` is the caller's, offset by the arm's position in the pack, because two servers
    that both bind 8080 do not fail loudly: the second dies and the first answers both arms.
    `out_dir` puts the extracted backend, the weights and the server log under the arm's own
    directory, so no two children write the same path. `storage` is the caller's resolver,
    which inside a pack returns a path the supervisor already fetched rather than reaching
    for the network from a child that is supposed to be isolated.
    """
    from huggingface_hub import hf_hub_download as hub_download, list_repo_files
    from transformers import AutoTokenizer

    if storage is None:
        from lab import lab
        storage = lab.storage_download

    backend_object = str(cfg.get("llama_object", "") or "")
    if not backend_object:
        raise RuntimeError("the 4-bit path needs `llama_object`, the stored serving path built "
                           "by s5-llama-build; without it a Q4 number would come from an "
                           "unpinned backend")
    log("downloading the serving path: %s" % backend_object)
    tarball = storage(backend_object)
    server_bin = _extract_backend(tarball, dest=os.path.join(out_dir, "llama"))
    log("serving path at %s" % server_bin)

    gguf, repo, name, gguf_source = resolve_gguf(
        cfg, storage, out_dir, log=log, hub=hub_download, list_files=list_repo_files)
    tok_repo = str(cfg.get("tokenizer_repo", "") or cfg.get("base_model", ""))
    tok = AutoTokenizer.from_pretrained(tok_repo, trust_remote_code=True)
    log("prompts rendered by %s, the same tokenizer the full-precision rows use" % tok_repo)

    slots = int(cfg.get("gguf_parallel", 8))
    per_slot = int(cfg.get("gguf_ctx_per_slot", 8192))
    port = int(port if port is not None else cfg.get("gguf_port", 8080))
    cmd = [server_bin, "--model", gguf, "--host", "127.0.0.1", "--port", str(port),
           "-ngl", "99", "-c", str(per_slot * slots), "--parallel", str(slots),
           "--no-warmup", "--seed", "0"]
    log("$ %s" % " ".join(cmd))
    log_path = os.path.join(out_dir, "llama_server.log")
    os.makedirs(out_dir, exist_ok=True)
    handle_log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=handle_log, stderr=subprocess.STDOUT)
    srv = Server(proc, "http://127.0.0.1:%d" % port, log_path, gguf, slots, log=log)

    deadline = time.time() + 420
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("the server exited during startup (rc=%s)\n%s"
                               % (proc.returncode, srv.tail()))
        try:
            if requests.get(srv.url + "/health", timeout=3).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        srv.stop()
        raise RuntimeError("the server did not become healthy within 420 s\n%s" % srv.tail())
    log("server healthy on %d slots, %d context tokens each" % (slots, per_slot))

    facts = {"backend": "llama.cpp", "llama_object": backend_object,
             "gguf_source": gguf_source, "gguf_repo": repo,
             "gguf_file": name, "gguf_mb": size_mb, "tokenizer_repo": tok_repo,
             "slots": slots, "ctx_per_slot": per_slot, "port": port}
    return srv, tok, facts


class GgufRunner:
    """Same surface as the full-precision `Runner`, so nothing downstream changes."""

    def __init__(self, server, tok=None, log=print):
        self.srv = server
        self.tok = tok
        self.log = log
        self.gen_tokens = 0
        self.prompt_tokens = 0
        self.gen_seconds = 0.0
        self.truncated = 0
        self.errors = 0

    def _one(self, text, max_new_tokens):
        try:
            ids = self.srv.tokenize(text)
            r = self.srv.complete(ids, max_new_tokens)
        except Exception as exc:
            self.errors += 1
            return "", 0, 0, False, repr(exc)
        tim = r.get("timings", {}) or {}
        return (r.get("content", "") or "").strip(), \
            int(tim.get("predicted_n") or 0), int(tim.get("prompt_n") or len(ids)), \
            bool(r.get("stopped_limit")), None

    def generate(self, prompts, max_new_tokens=384, batch_size=16, tag="", ids=None):
        """Greedy completions, in the caller's order. `batch_size` is ignored: concurrency is
        the server's slot count, which is fixed for the run so throughput cannot drift between
        components. `ids` is accepted and ignored, for the same reason as in gen.Runner."""
        out = [None] * len(prompts)
        t0 = time.time()
        done = [0]

        def work(i):
            text, gen, pre, hit, err = self._one(prompts[i], max_new_tokens)
            out[i] = text
            self.gen_tokens += gen
            self.prompt_tokens += pre
            self.truncated += 1 if hit else 0
            done[0] += 1
            if err and self.errors <= 5:
                self.log("  %s request %d failed: %s" % (tag, i, err))
            if done[0] % max(self.srv.slots * 10, 20) == 0:
                el = time.time() - t0
                self.log("  %s generated %d/%d (%.0f tok/s so far)"
                         % (tag, done[0], len(prompts), self.gen_tokens / max(el, 1e-6)))

        with ThreadPoolExecutor(max_workers=self.srv.slots) as pool:
            list(pool.map(work, range(len(prompts))))
        self.gen_seconds += time.time() - t0
        return ["" if o is None else o for o in out]

    def throughput(self):
        return {"generated_tokens": self.gen_tokens,
                "prompt_tokens": self.prompt_tokens,
                "generate_seconds": round(self.gen_seconds, 1),
                "tokens_per_second": round(self.gen_tokens / max(self.gen_seconds, 1e-6), 1),
                "hit_max_new_tokens": self.truncated,
                "failed_requests": self.errors,
                "slots": self.srv.slots}
