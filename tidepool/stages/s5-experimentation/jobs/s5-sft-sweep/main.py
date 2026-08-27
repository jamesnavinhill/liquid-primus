"""s5.3 supervised sweep: the whole C row of the experiment matrix, in one code path.

Every arm below runs this file. An arm that runs different code from the arm it is compared
against is not an ablation, and the cheapest way to guarantee a shared code path is to give
them one. What differs between arms is the `arm` parameter and nothing else, so the recipe
table is the entire specification of the sweep:

  C1   reference: LoRA r16, role-balanced mix, cross-entropy, guardrail block at 2 passes
  C2'  C1 with the guardrail block removed, its tokens returned to the base mix
  C3   full-parameter SFT instead of an adapter
  C4   C1 with an entropy-weighted loss
  C5a  C1 with 1% self-distillation replay
  C5b  C1 with 5% self-distillation replay
  C6   C1 at LoRA rank 64
  C7   C1 with the raw corpus mix, uniform sampling

`C2` as originally specified ("public data only") was retired at `s5.2`: `s4.4` measured the
in-house corpus at zero supervised rows, so the arm would have reproduced C1 byte for byte.
The reasoning is in `../s3-research-plan/plan.md`.

Two invariants hold across every arm, and both are asserted rather than assumed.

  **Equal cost.** Each arm trains on exactly `budget_tokens`. C2' does not get a smaller run
  for dropping the guardrail block and C5 does not get a larger one for adding replay; the
  doses come out of the budget, never on top of it.

  **Equal rows in equal order.** Selection is a stable hash order over the corpus, so two
  arms that differ in one hyperparameter see the same rows in the same sequence and the
  comparison is paired rather than confounded by sampling noise.
"""

import gzip
import hashlib
import json
import math
import os
import shutil
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sample
import textify

# ------------------------------------------------------------------ packing
#
# Several arms share one GPU. `pack.py` is the supervisor; it runs this file once per arm as
# an isolated child process, so an arm that OOMs, asserts or is killed takes down its own
# process and nothing else. A child differs from a solo run in five ways and no others:
#
#   - its arm, output directory, config and memory ceiling come from the environment
#   - it never talks to the job API (one job id, one reporter: concurrent writers to a single
#     job's progress and artifact stream would interleave into nonsense, so the supervisor
#     owns that channel and collects the children's results at the end)
#   - it reads shared-storage objects from paths the supervisor already downloaded, so the
#     corpus is fetched once per GPU rather than once per arm
#   - it prefixes every log line with its arm so the merged stream stays readable
#   - it exits non-zero on assertion failure instead of calling lab.error
#
# The training path itself is untouched, which is what keeps a packed arm comparable to a solo
# one. That comparability is measured, not assumed: C1 ran solo before packing existed, so any
# packed arm can be checked against it on tokens seen, priority share and rows-in-order.
PACK_CHILD = os.environ.get("TIDEPOOL_PACK_CHILD") == "1"
PACK_MEMFRAC = float(os.environ.get("TIDEPOOL_PACK_MEMFRAC") or 0.0)
PACK_LOCAL = json.loads(os.environ.get("TIDEPOOL_PACK_LOCAL") or "{}")

if PACK_CHILD:
    class _SupervisedLab(object):
        """Stands in for the job API inside a packed child.

        Every reporting call in this file is already guarded by `if not PACK_CHILD`; this
        catches anything that is ever added without one, turning it into a visible no-op
        rather than a second writer on the job's stream. It says so once per method so a
        missed guard shows up in the console without flooding it.

        Storage calls are the exception and must raise. The file probes for them with
        `getattr(lab, "storage_upload", None)`, and a stub that answered would hand back a
        no-op the caller then reports as a successful upload: the weights would never reach
        shared storage and the log would say they had. Raising makes the probe fail, which is
        exactly what sends the caller down its CLI fallback -- a separate process writing to
        an arm-specific prefix, which is safe to run concurrently."""

        _warned = set()
        _must_raise = ("storage_upload", "upload_storage", "storage_put",
                       "storage_download", "get_config", "init")

        def __getattr__(self, name):
            if name in _SupervisedLab._must_raise:
                raise AttributeError(
                    "lab.%s is deliberately absent in a packed child; storage goes through "
                    "the CLI and config arrives on the environment" % name)

            def _noop(*a, **k):
                if name not in _SupervisedLab._warned:
                    _SupervisedLab._warned.add(name)
                    print("[pack] suppressed lab.%s() in child mode" % name, flush=True)
            return _noop

    lab = _SupervisedLab()
    CFG = json.loads(os.environ.get("TIDEPOOL_PACK_CFG") or "{}")
else:
    from lab import lab
    lab.init()
    CFG = lab.get_config() or {}


def storage(obj):
    """Resolve a shared-storage object to a local path.

    A packed child downloads nothing. Every arm on the card reads the same corpus, so N
    children fetching it N times would multiply the transfer, the disk and the failure
    surface for no benefit; the supervisor fetches each object once and hands the resolved
    paths down. A missing path is fatal rather than silently re-downloaded, because a child
    that quietly reaches for the network has stopped being isolated."""
    if not PACK_CHILD:
        return lab.storage_download(obj)
    p = PACK_LOCAL.get(obj)
    if not p or not os.path.exists(p):
        raise SystemExit("pack child was given no local path for %r (supervisor should have "
                         "downloaded it before spawning children)" % obj)
    return p


def C(k, default):
    v = CFG.get(k)
    return default if v is None or v == "" else v




# ------------------------------------------------------------------ 0. the arm
#
# The recipe table. Anything absent from an arm's dict takes the parameter default, so a
# row here reads as exactly the delta from C1 and nothing is hidden in a config file.
# C3's learning rate is the one place an arm changes two things at once: LoRA and
# full-parameter tuning have different optimal LR scales by construction, and holding
# 1e-4 fixed across them would compare a tuned recipe against a detuned one.
ARMS = {
    "C1":  {},
    "C2p": {"guardrail_epochs": 0.0},
    "C3":  {"tuning": "full", "learning_rate": 1e-5,
            # Half the micro-batch and twice the accumulation, so the effective batch
            # stays at 16 while the optimizer states for every parameter fit alongside
            # the activations on the same card the LoRA arms use.
            "micro_batch_size": 1, "grad_accum": 16},
    "C4":  {"loss": "entropy_weighted"},
    "C5a": {"replay_frac": 0.01},
    "C5b": {"replay_frac": 0.05},
    "C6":  {"lora_r": 64, "lora_alpha": 128},
    "C7":  {"mix": "raw"},

    # ---- s5.5. C7's recipe with replay, separating the two levers the C sweep confounded.
    #
    # C5a and C5b varied the replay fraction with the buffer's composition left proportional,
    # and moved IFEval two to three points before stalling. Reading the pool afterwards showed
    # why: `open_perfectblend_autoif` is 10.5% of `antidoom-mix-v1.0` and is verifiable-constraint
    # instruction following of exactly IFEval's kind, so a proportional buffer delivers it at
    # 0.1% of the training tokens at 1% replay and 0.5% at 5%. Dose and composition were never
    # separable in the C sweep because only one of them ever moved.
    #
    # Both buffers are sized so that the arm reading them at its own fraction makes at most one
    # pass. C5b's 5% dose was 3.2M tokens against a 3.21M-token buffer, which is one epoch by
    # construction; 20% of the same 64M budget is 12.8M tokens, so a 20% arm needs a buffer four
    # times the size or it replays the same 7,945 completions four times and measures
    # memorization. Hence 32,000 prompts rather than 8,000 for both.
    #
    # The three contrasts, with C7 as the origin at zero replay:
    #   R3 - R2   composition, at a fixed 20% dose and a fixed budget
    #   R3 - R1   dose, over one and the same reweighted buffer
    #   R1 - C7   whether a small reweighted dose moves the axis at all
    "R1":  {"mix": "raw", "replay_frac": 0.05,
            "replay_object": "tidepool/s5.5/replay_constraint/replay.jsonl.gz"},
    "R2":  {"mix": "raw", "replay_frac": 0.20,
            "replay_object": "tidepool/s5.5/replay_proportional/replay.jsonl.gz"},
    "R3":  {"mix": "raw", "replay_frac": 0.20,
            "replay_object": "tidepool/s5.5/replay_constraint/replay.jsonl.gz"},
}
ARM = str(os.environ.get("TIDEPOOL_PACK_ARM") or C("arm", "C1"))
if ARM not in ARMS:
    raise SystemExit("unknown arm %r; the sweep defines %s" % (ARM, sorted(ARMS)))
RECIPE = ARMS[ARM]


def A(k, default):
    """Arm override first, then the task parameter, then the code default."""
    if k in RECIPE:
        return RECIPE[k]
    return C(k, default)


BASE = C("base_model", "LiquidAI/LFM2.5-1.2B-Instruct")
TRAIN_OBJ = C("train_object", "tidepool/s4.4/train.jsonl.gz")
VAL_OBJ = C("val_object", "tidepool/s4.4/val.jsonl.gz")
GUARD_OBJ = C("guardrail_object", "tidepool/s5.3/tooldata/tooldata_train.jsonl.gz")
# `A` rather than `C`, so an arm can name its own buffer. s5.5 runs two buffers at once
# and the arm table is where the sweep says which is which.
REPLAY_OBJ = str(A("replay_object", ""))

MIX = str(A("mix", "role_balanced"))            # role_balanced (C1) | raw (C7)
TUNING = str(A("tuning", "lora"))               # lora | full (C3)
LOSS = str(A("loss", "ce"))                     # ce | entropy_weighted (C4)
ENTROPY_BETA = float(A("entropy_beta", 1.0))
REPLAY_FRAC = float(A("replay_frac", 0.0))
GUARD_EPOCHS = float(A("guardrail_epochs", 2.0))
BUDGET = int(A("budget_tokens", 64_000_000))

SMOKE = bool(C("smoke", False))
SMOKE_ROWS = int(C("smoke_rows", 512))
MAX_LEN = int(A("max_seq_len", 2048))
MICRO_BS = int(A("micro_batch_size", 2))
ACCUM = int(A("grad_accum", 8))
LR = float(A("learning_rate", 1e-4))
LORA_R = int(A("lora_r", 16))
LORA_ALPHA = int(A("lora_alpha", 32))
LORA_DROPOUT = float(A("lora_dropout", 0.05))
MAX_STEPS = int(C("max_steps", 0))              # 0 = one full pass over the sampled mix
VAL_ROWS = int(C("val_rows", 256))
WARMUP = float(A("warmup_ratio", 0.03))
SEED = int(C("seed", 17))
TOOL_EPOCHS = float(A("tool_epochs", 1.0))
STRUCT_MAX_EPOCHS = float(A("struct_max_epochs", 3.0))
PRIORITY_SHARE = float(A("min_priority_share", 0.5))
RUN_TAG = C("run_tag", "s5.3-%s" % ARM)

# Checkpointing. Weights used to be written once, after the final step, so an arm that
# stalled part way through had nothing to recover: its whole state lived in the memory of a
# machine nobody can open a shell into, and the only choices were to keep waiting or to
# throw the hours away. C1 stalled at 2,940 of 8,416 steps and posed exactly that question.
CKPT_EVERY = int(C("ckpt_every_steps", 0))   # 0 = auto, about twenty per arm
RESUME = bool(C("resume", True))

OUT = os.environ.get("TIDEPOOL_PACK_OUT") or "out"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED)


def log(msg):
    if PACK_CHILD:
        print("[%s] %s" % (ARM, msg), flush=True)
        return
    lab.log(msg)
    print(msg, flush=True)


def dump(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    if PACK_CHILD:
        return p                      # the supervisor collects and uploads, arm-prefixed
    try:
        lab.save_artifact(p)
    except Exception as exc:
        log("save_artifact failed for %s: %s" % (name, exc))
    return p


def _mirror_request(local, dest):
    """Ask the supervisor to upload one file on this arm's behalf.

    A packed child cannot reach shared storage itself. It has no SDK storage call by design,
    and the CLI fallback that used to stand in for one does not exist on these machines: the
    `lab` binary is not installed, so every attempt returned
    `[Errno 2] No such file or directory: 'lab'`. Four trained adapters and every checkpoint
    of a nine-hour pack were lost that way, in runs that otherwise looked clean, and it was
    noticed only when the next stage went looking for the weights.

    So the child writes a request and the supervisor, which does have the job API, performs
    the upload on its next poll. Returns whether the request was lodged, which is the only
    thing this process can honestly claim."""
    try:
        if not os.path.exists(local):
            log("no mirror request for %s: it does not exist" % local)
            return False
        rel = os.path.relpath(local, OUT)
        if rel.startswith(".."):
            # The supervisor only reads inside this arm's own directory, so a file outside it
            # is copied in rather than quietly not mirrored.
            local = shutil.copy2(local, os.path.join(OUT, os.path.basename(local)))
            rel = os.path.relpath(local, OUT)
        with open(os.path.join(OUT, "mirror_pending.jsonl"), "a") as fh:
            fh.write(json.dumps({"file": rel, "dest": dest}) + "\n")
            fh.flush()
        return True
    except Exception as exc:
        log("could not ask the supervisor to mirror %s (%s)" % (local, exc))
        return False


def _storage_put(local, dest):
    """Get one local file into a shared-storage prefix, whichever way this process can.

    Returns whether the file is on its way there: uploaded, for a solo run that owns the job
    API, or lodged with the supervisor, for a packed child that does not. A caller that
    reports an upload it did not make is the failure mode this project has been bitten by
    twice, so neither path is allowed to answer True on a guess."""
    if PACK_CHILD:
        return _mirror_request(local, dest)
    try:
        up = getattr(lab, "storage_upload", None)
        if callable(up):
            up(local, dest=dest)
            return True
    except Exception as exc:
        log("SDK storage_upload failed for %s (%s)" % (os.path.basename(local), exc))
    try:
        # Last resort for a solo run on a machine that happens to have the CLI. It is not the
        # packed path any more and must not be relied on: the images these jobs run on do not
        # ship `lab`.
        import subprocess
        r = subprocess.run(["lab", "storage", "upload", local, "--dest", dest,
                            "--no-interactive"], capture_output=True, text=True, timeout=1800)
        if r.returncode == 0:
            return True
        log("CLI storage upload returned %d: %s" % (r.returncode, r.stderr[-400:]))
    except Exception as exc:
        log("CLI storage upload failed for %s (%s)" % (os.path.basename(local), exc))
    return False


def _storage_get(obj, dst):
    """Fetch one shared-storage object, or return None when it is not there.

    Deliberately not `storage()`. That resolver is for inputs the supervisor pre-downloads and
    it is fatal on a miss, which is right for a corpus. A checkpoint is optional by nature --
    the first run of any arm has none -- so absence is an ordinary answer here, not an error.
    The CLI is used in both modes so a packed child can read its own checkpoint back."""
    try:
        import subprocess
        r = subprocess.run(["lab", "storage", "download", obj, dst, "--no-interactive"],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return dst
    except Exception as exc:
        log("checkpoint fetch failed for %s (%s)" % (obj, exc))
    return None


fails = []
t_start = time.time()
log("arm %s: %s" % (ARM, json.dumps(RECIPE) if RECIPE else "reference, no overrides"))

# ------------------------------------------------------------------ 1. the mix

log("downloading splits")
sources, source_names = [], []
train_local = storage(TRAIN_OBJ)
sources.append(train_local)
source_names.append(TRAIN_OBJ)
val_local = storage(VAL_OBJ)

if GUARD_EPOCHS > 0:
    sources.append(storage(GUARD_OBJ))
    source_names.append(GUARD_OBJ)
if REPLAY_FRAC > 0:
    if not REPLAY_OBJ:
        fails.append("arm %s asks for %.0f%% replay but no replay_object is configured, so "
                     "the arm cannot be run as specified" % (ARM, 100 * REPLAY_FRAC))
    else:
        sources.append(storage(REPLAY_OBJ))
        source_names.append(REPLAY_OBJ)

t0 = time.time()
per_role, order = sample.index_split(sources, log=log)
plan = sample.calibrate({k: v for k, v in per_role.items() if k in sample.ROLES},
                        tool_epochs=TOOL_EPOCHS, struct_max_epochs=STRUCT_MAX_EPOCHS,
                        min_priority_share=PRIORITY_SHARE,
                        guardrail_epochs=GUARD_EPOCHS, replay_frac=REPLAY_FRAC,
                        budget_tokens=BUDGET)
log("calibration (%.1fs): %s" % (time.time() - t0, json.dumps(plan)))
if plan["notes"].get("replay_unavailable"):
    fails.append("the replay set carried no rows tagged 'replay', so arm %s trained without "
                 "the replay it is defined by" % ARM)
if plan["notes"].get("doses_exceeded_budget"):
    fails.append("the guardrail and replay doses alone exceed the %d-token budget, so this "
                 "arm saw none of the base mix" % BUDGET)

# A replay buffer is sized by the `n_tok` its generator wrote, and the first buffer built at
# size wrote 2 for every row -- len() over a BatchEncoding's keys rather than its ids. The
# generator asserts against that now and refuses to be trusted, but the bad file is still in
# shared storage under the name the good one takes, and a consumer that believed it would ask
# for a 3.2M-token dose from a set indexed at 15,890 tokens, replay the same 8,000 rows two
# hundred times over, and measure memorization while calling it replay. Nothing about that
# looks wrong from inside the run, so it is checked here rather than left to the file being
# the right one. A self-distilled row is a couple of hundred tokens; anything under sixteen
# is a field counting something else.
if REPLAY_FRAC > 0 and not plan["notes"].get("replay_unavailable"):
    _rows = len(order.get(sample.REPLAY) or [])
    _mean = (per_role.get(sample.REPLAY, 0) / _rows) if _rows else 0.0
    if _mean < 16:
        fails.append("the replay set indexes %d row(s) at %.1f tokens each, which is a miscount "
                     "rather than a corpus: %s was written by a generator that did not tokenize "
                     "its own output, and dosing from it would replay a handful of tokens "
                     "hundreds of times over" % (_rows, _mean, REPLAY_OBJ))

if MIX == "raw":
    # C7. Every base-mix row once, in the same hash order, so the only difference from C1 is
    # which rows and how often, not the order they arrive in. The guardrail and replay blocks
    # are dosed identically to C1: the arm isolates the sampler and nothing else.
    want_base = sum(plan["per_role_tokens"][r] for r in sample.BASE_ROLES)
    pool = [(rk, i) for role in sample.BASE_ROLES for rk, i in (order.get(role) or [])]
    pool.sort()
    mean_tok = (sum(per_role.get(r, 0) for r in sample.BASE_ROLES) / max(1, len(pool)))
    n_want = int(round(want_base / mean_tok)) if mean_tok else 0
    ids = [i for _, i in pool[:n_want]]
    detail = {"mode": "raw", "rows": len(ids), "tokens": int(len(ids) * mean_tok),
              "mean_tokens_per_row": round(mean_tok, 1)}
    for role in (sample.GUARDRAIL, sample.REPLAY):
        want = plan["per_role_tokens"].get(role, 0)
        rows = order.get(role) or []
        if want <= 0 or not rows:
            continue
        mt = per_role.get(role, 0) / len(rows)
        n = int(round(want / mt)) if mt else 0
        taken = []
        while len(taken) < n:
            taken.extend(i for _, i in rows[:n - len(taken)])
        ids.extend(taken)
        detail[role] = {"rows_taken": len(taken), "tokens": int(len(taken) * mt)}
    import hashlib
    ids.sort(key=lambda i: hashlib.blake2b(str(i).encode(), digest_size=8).digest())
else:
    ids, detail = sample.choose(order, per_role, plan, log=log)

if SMOKE:
    ids = ids[:SMOKE_ROWS]
    log("SMOKE: training rows capped at %d; the calibration above is still the real one, "
        "computed over the whole split" % len(ids))

rows = sample.read_lines(sources, ids)
val_rows = []
with gzip.open(val_local, "rt") as fh:
    for i, line in enumerate(fh):
        if i >= VAL_ROWS * 8:
            break
        val_rows.append(json.loads(line))
val_rows = val_rows[::max(1, len(val_rows) // VAL_ROWS)][:VAL_ROWS]
log("materialized %d train rows and %d val rows" % (len(rows), len(val_rows)))
if not rows:
    fails.append("the sampler selected no training rows")

role_counts = {}
for r in rows:
    role_counts[r.get("role", "unknown")] = role_counts.get(r.get("role", "unknown"), 0) + 1
log("realized role counts: %s" % json.dumps(role_counts))
if GUARD_EPOCHS > 0 and not role_counts.get(sample.GUARDRAIL):
    fails.append("arm %s is defined with a guardrail block at %.1f passes but no guardrail "
                 "rows reached the training set" % (ARM, GUARD_EPOCHS))
if GUARD_EPOCHS == 0 and role_counts.get(sample.GUARDRAIL):
    fails.append("arm %s is the guardrail-off ablation and guardrail rows reached the "
                 "training set anyway, so it is not the ablation it claims to be" % ARM)

# ------------------------------------------------------------- 2. model and template

from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

log("loading %s" % BASE)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
enc = textify.Encoder(tok, MAX_LEN, log=log)
enc.pick_mode([r["messages"] for r in rows[:64]] or [[{"role": "user", "content": "hi"}]])

probe_ids, probe_labels = enc.encode(rows[0]["messages"]) if rows else ([], [])
supervised = sum(1 for x in probe_labels if x != -100)
log("mask probe: %d tokens, %d supervised (%.1f%%)"
    % (len(probe_ids), supervised, 100.0 * supervised / max(1, len(probe_ids))))
if rows and supervised == 0:
    fails.append("the loss mask covers no tokens on the first training row, so the run would "
                 "optimize nothing")
if rows and supervised == len(probe_ids):
    fails.append("the loss mask covers every token including the user's turns, which trains "
                 "the model to write the questions")

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                             trust_remote_code=True)
model.config.use_cache = False
if TUNING == "full":
    # C3. No adapter: every parameter trains. AdamW's states are the memory cost here, so
    # the arm is priced on the same card with a smaller micro-batch rather than a larger one.
    for p in model.parameters():
        p.requires_grad_(True)
else:
    peft_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                          bias="none", task_type="CAUSAL_LM", target_modules="all-linear")
    model = get_peft_model(model, peft_cfg)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
log("%s: %d trainable of %d parameters (%.3f%%)"
    % ("full-parameter" if TUNING == "full" else "LoRA attached", trainable, total,
       100.0 * trainable / total))
if trainable == 0:
    fails.append("no parameters are trainable, so this run optimizes nothing")
if TUNING == "lora" and trainable > 0.2 * total:
    fails.append("the LoRA arm has %.1f%% of parameters trainable, which is not an adapter"
                 % (100.0 * trainable / total))
dev = "cuda" if torch.cuda.is_available() else "cpu"
if PACK_CHILD and PACK_MEMFRAC > 0 and dev == "cuda":
    # A hard ceiling per arm, so an arm that would grow into its neighbours' memory hits its
    # own allocator limit and fails alone instead of OOM-ing whichever sibling happened to
    # allocate next. Without this, packing turns one arm's bad batch into everyone's crash.
    torch.cuda.set_per_process_memory_fraction(PACK_MEMFRAC, 0)
    _tot = torch.cuda.get_device_properties(0).total_memory / 1e9
    log("memory ceiling %.2f of %.1f GB (pack fraction %.3f)"
        % (PACK_MEMFRAC * _tot, _tot, PACK_MEMFRAC))
if dev == "cpu":
    fails.append("no CUDA device is visible, so this run is not the run that was requested")
model.to(dev)
model.gradient_checkpointing_enable()
if TUNING != "full":
    model.enable_input_require_grads()


class Conv(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids_, labels = enc.encode(self.data[i]["messages"])
        return {"input_ids": ids_, "labels": labels}


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id or 0
    ids_ = torch.full((len(batch), n), pad, dtype=torch.long)
    lab_ = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids_[i, :k] = torch.tensor(b["input_ids"])
        lab_[i, :k] = torch.tensor(b["labels"])
        att[i, :k] = 1
    return {"input_ids": ids_, "labels": lab_, "attention_mask": att}


def token_entropy(logits, chunk=4096):
    """Per-token predictive entropy, normalized to [0, 1], computed without gradients.

    Chunked over the flattened token axis because a float32 softmax over the full vocabulary
    for a whole micro-batch is a gigabyte-scale temporary, and the arm that needs it is
    already the most memory-hungry loss in the sweep.
    """
    flat = logits.reshape(-1, logits.size(-1))
    out = torch.empty(flat.size(0), device=flat.device, dtype=torch.float32)
    for i in range(0, flat.size(0), chunk):
        lp = F.log_softmax(flat[i:i + chunk].float(), dim=-1)
        out[i:i + chunk] = -(lp.exp() * lp).sum(-1)
    return (out / math.log(logits.size(-1))).reshape(logits.shape[:-1])


def compute_loss(batch):
    """Cross-entropy over the masked tokens, optionally reweighted by predictive entropy.

    C4's weights are centred on the batch mean, so the average weight is 1 and the effective
    learning rate is unchanged from C1. Without the centring, `entropy_beta` would be a
    learning-rate multiplier wearing a different name and the ablation would measure the
    wrong thing.
    """
    if LOSS != "entropy_weighted":
        return model(**batch).loss, None
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                          labels.reshape(-1), reduction="none", ignore_index=-100)
    nll = nll.reshape(labels.shape)
    with torch.no_grad():
        h = token_entropy(logits)
        centre = h[mask].mean() if mask.any() else h.mean()
        w = (1.0 + ENTROPY_BETA * (h - centre)).clamp(min=0.1)
    n = mask.sum().clamp(min=1)
    return (nll * w * mask).sum() / n, float(h[mask].mean()) if mask.any() else None


loader = DataLoader(Conv(rows), batch_size=MICRO_BS, shuffle=False, collate_fn=collate,
                    num_workers=2 if len(rows) > 2000 else 0, drop_last=True)
steps = MAX_STEPS or max(1, len(loader) // ACCUM)
log("%d micro-batches, accumulation %d, %d optimizer steps" % (len(loader), ACCUM, steps))

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR,
                        weight_decay=0.0, betas=(0.9, 0.95))
sched = get_cosine_schedule_with_warmup(opt, int(steps * WARMUP), steps)

# ------------------------------------------------------------------ 2b. checkpoint / resume
#
# The checkpoint goes to shared storage rather than to local disk, which is the whole point:
# the machine is torn down with the job, so a local checkpoint dies with the thing it exists
# to survive. It carries the optimizer and scheduler state as well as the weights, because an
# Adam moment reset half way through a cosine schedule is a different recipe, not a resumed
# one, and this sweep's entire job is to compare recipes.
# Namespaced by ARM as well as by run tag. A pack hands every child the SAME config, so
# every arm on a card shares one run_tag; without the arm in the path, four trainers would
# write four different recipes over one pair of objects and each would find a stranger's
# checkpoint on resume.
CKPT_DEST = "tidepool/s5.3/ckpt/%s/%s%s" % (RUN_TAG, ARM, "-smoke" if SMOKE else "")
CKPT_SLOTS = ("ckpt-a.pt", "ckpt-b.pt")

# Every value that changes what a step MEANS is hashed. Resuming across a recipe change would
# produce an arm that is neither of the two recipes while reporting itself as one of them,
# which is worse than losing the run: the sweep would rank a checkpoint nobody specified.
FINGERPRINT = hashlib.sha256(json.dumps({
    "arm": ARM, "base": BASE, "mix": MIX, "tuning": TUNING, "loss": LOSS,
    "entropy_beta": ENTROPY_BETA, "replay_frac": REPLAY_FRAC, "replay_object": REPLAY_OBJ,
    "guardrail_epochs": GUARD_EPOCHS, "budget": BUDGET, "max_seq_len": MAX_LEN,
    "micro_bs": MICRO_BS, "accum": ACCUM, "lr": LR, "warmup": WARMUP,
    "lora": [LORA_R, LORA_ALPHA, LORA_DROPOUT], "seed": SEED,
    "steps": steps, "rows": len(rows), "smoke": SMOKE,
}, sort_keys=True).encode()).hexdigest()[:16]

_TRAINABLE = [n for n, prm in model.named_parameters() if prm.requires_grad]

# How often to checkpoint is a trade between what a stall costs and what the uploads cost, and
# the two sides are not the same size on every arm. A rank-16 adapter is a few hundred
# megabytes with its Adam moments; a full-parameter arm is a fp32 copy of the whole model plus
# two moments, some fourteen gigabytes, and twenty of those is a quarter of a terabyte pushed
# through the network for a job that is meant to be four hours of arithmetic. So the cadence is
# derived from the payload: as many checkpoints as fit under ckpt_max_gb of total upload,
# capped at twenty and floored at four, and never more than one per fifty steps.
_CKPT_BYTES = sum(prm.numel() for prm in model.parameters() if prm.requires_grad) * 12
_CKPT_GB = _CKPT_BYTES / 1e9
if CKPT_EVERY <= 0:
    _budget_gb = float(C("ckpt_max_gb", 60.0))
    _n = int(_budget_gb / _CKPT_GB) if _CKPT_GB > 0 else 20
    _n = max(4, min(20, _n))
    CKPT_EVERY = max(50, steps // _n)
log("checkpoints: about %.2f GB each (%d trainable tensors), every %d of %d steps, to %s"
    % (_CKPT_GB, len(_TRAINABLE), CKPT_EVERY, steps, CKPT_DEST))


def save_checkpoint(state, slot):
    """Write one checkpoint and, only if it lands, move the pointer to it.

    Two slots are alternated and the pointer is written afterwards, so a run killed during an
    upload leaves the previous checkpoint intact and still pointed at. Writing one slot in
    place would mean the moment of greatest risk -- a half-written object -- was also the only
    copy."""
    path = os.path.join(OUT, slot)
    payload = dict(state)
    payload.update({
        "fingerprint": FINGERPRINT,
        "arm": ARM, "run_tag": RUN_TAG, "total_steps": steps,
        "weights": {n: prm.detach().cpu().clone()
                    for n, prm in model.named_parameters() if prm.requires_grad},
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
    })
    torch.save(payload, path)
    if not _storage_put(path, CKPT_DEST):
        log("checkpoint at step %d is not on its way to storage; the previous one still stands"
            % state["step"])
        return False
    ptr = os.path.join(OUT, "ckpt-latest.json")
    with open(ptr, "w") as fh:
        json.dump({"slot": slot, "step": state["step"], "fingerprint": FINGERPRINT}, fh)
    _storage_put(ptr, CKPT_DEST)
    # "on its way" rather than "saved": a solo run has uploaded it by here, a packed child has
    # handed it to the supervisor, and only the supervisor's own log says it landed.
    log("checkpoint at step %d/%d written for %s/%s (%s)"
        % (state["step"], steps, CKPT_DEST, slot,
           "handed to the supervisor" if PACK_CHILD else "uploaded"))
    return True


def load_checkpoint():
    """Return the furthest usable checkpoint for this arm, or None.

    Both slots are read rather than trusting the pointer. The pointer is written after a slot
    uploads, so it can never be ahead of the data, but it can be behind it if a run died
    between the two uploads -- and the newer slot is the one worth having."""
    best = None
    for name in CKPT_SLOTS:
        got = _storage_get("%s/%s" % (CKPT_DEST, name), os.path.join(OUT, "resume-" + name))
        if not got:
            continue
        try:
            ck = torch.load(got, map_location="cpu", weights_only=False)
        except Exception as exc:
            log("checkpoint %s did not load (%s); ignoring it" % (name, exc))
            continue
        if ck.get("fingerprint") != FINGERPRINT:
            log("checkpoint %s was written by a different recipe (%s against this run's %s); "
                "ignoring it and starting from the base checkpoint"
                % (name, ck.get("fingerprint"), FINGERPRINT))
            continue
        if best is None or int(ck.get("step", 0)) > int(best.get("step", 0)):
            best = ck
    return best


WANDB = None
if os.environ.get("WANDB_API_KEY"):
    try:
        import wandb
        WANDB = wandb.init(project="tidepool", name=RUN_TAG, config={
            "arm": ARM, "recipe": RECIPE, "base_model": BASE, "mix": MIX, "tuning": TUNING,
            "loss": LOSS, "lora_r": LORA_R, "lr": LR, "smoke": SMOKE,
            "budget_tokens": BUDGET, "guardrail_epochs": GUARD_EPOCHS,
            "replay_frac": REPLAY_FRAC, "max_seq_len": MAX_LEN,
            "effective_batch": MICRO_BS * ACCUM, "steps": steps,
            "template_mode": enc.mode})
        log("W&B run: %s" % WANDB.url)
    except Exception as exc:
        log("W&B unavailable (%s); training continues without it" % exc)


def eval_loss():
    """Plain cross-entropy on the held-out split, for every arm.

    Deliberately not the training loss: C4 reweights its own objective, and an arm scored on
    its own reweighted loss cannot be ranked against the others.
    """
    model.eval()
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for b in DataLoader(Conv(val_rows), batch_size=MICRO_BS, collate_fn=collate):
            b = {k: v.to(dev) for k, v in b.items()}
            out = model(**b)
            k = int((b["labels"] != -100).sum())
            tot += float(out.loss) * k
            ntok += k
    model.train()
    return tot / max(1, ntok)


# ------------------------------------------------------------------ 3. train

model.train()
t_train = time.time()

# Resume, if there is anything to resume from. A first run of an arm finds nothing and this
# whole block is a no-op costing one storage miss.
_ck = load_checkpoint() if RESUME else None
RESUMED_FROM = 0
if _ck is not None:
    _res = model.load_state_dict(_ck["weights"], strict=False)
    _unexpected = list(getattr(_res, "unexpected_keys", []) or [])
    _absent = [n for n in _TRAINABLE if n not in _ck["weights"]]
    if _unexpected or _absent:
        # Shapes and names line up or they do not; a partial load would train a model that is
        # part checkpoint and part base weights and report itself as the checkpoint.
        fails.append("checkpoint weights do not match this model (%d unexpected, %d trainable "
                     "tensors missing), so the resume was refused"
                     % (len(_unexpected), len(_absent)))
        _ck = None
if _ck is not None:
    opt.load_state_dict(_ck["opt"])
    sched.load_state_dict(_ck["sched"])
    hist = list(_ck.get("hist") or [])
    step = int(_ck.get("step") or 0)
    seen_tok = int(_ck.get("seen_tok") or 0)
    ent_running = list(_ck.get("ent_running") or [])
    val_before = float(_ck["val_before"])
    start_epoch = int(_ck.get("epoch") or 0)
    skip_micro = int(_ck.get("micro_in_epoch") or 0)
    RESUMED_FROM = step
    _ck["weights"] = None          # several GB on the full-parameter arm; it is loaded now
    log("resumed %s from step %d/%d, %d tokens already seen, skipping %d micro-batches of "
        "epoch %d; val loss before any step was %.4f on the original run"
        % (ARM, step, steps, seen_tok, skip_micro, start_epoch, val_before))
else:
    hist, step, seen_tok, ent_running = [], 0, 0, []
    start_epoch, skip_micro = 0, 0
    val_before = eval_loss()
    log("val loss before any step: %.4f" % val_before)

tok_at_start = seen_tok        # everything before this was another job's compute
running, micro = 0.0, 0
done = step >= steps
micro_in_epoch = skip_micro
for epoch in range(start_epoch, start_epoch + 100):
    if done:
        break
    if skip_micro:
        # The loader is built with shuffle off and drop_last on, so micro-batch i is exactly
        # rows[i * MICRO_BS : (i + 1) * MICRO_BS]. The position is therefore recovered by
        # starting a loader further into the same row list, rather than by spinning the old
        # one forward and discarding batches, which on a half-finished arm would mean
        # collating tens of thousands of batches to throw every one of them away.
        ep_loader = DataLoader(Conv(rows[skip_micro * MICRO_BS:]), batch_size=MICRO_BS,
                               shuffle=False, collate_fn=collate,
                               num_workers=2 if len(rows) > 2000 else 0, drop_last=True)
        log("epoch %d resumes %d micro-batches in, %d rows remaining"
            % (epoch, skip_micro, len(rows) - skip_micro * MICRO_BS))
    else:
        ep_loader = loader
        micro_in_epoch = 0
    skip_micro = 0
    for batch in ep_loader:
        micro_in_epoch += 1
        batch = {k: v.to(dev) for k, v in batch.items()}
        loss, ent = compute_loss(batch)
        (loss / ACCUM).backward()
        running += float(loss)
        if ent is not None:
            ent_running.append(ent)
        seen_tok += int(batch["attention_mask"].sum())
        micro += 1
        if micro % ACCUM:
            continue
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tr = running / ACCUM
        running = 0.0
        hist.append({"step": step, "train_loss": round(tr, 4),
                     "lr": sched.get_last_lr()[0], "tokens": seen_tok})
        if WANDB:
            WANDB.log({"train/loss": tr, "train/lr": sched.get_last_lr()[0],
                       "train/tokens": seen_tok}, step=step)
        if step % max(1, steps // 40) == 0 or step == 1:
            log("step %d/%d  loss %.4f  %d tokens  %.0f tok/s"
                % (step, steps, tr, seen_tok,
                   (seen_tok - tok_at_start) / max(1e-6, time.time() - t_train)))
            if not PACK_CHILD:
                lab.update_progress(min(95, int(90.0 * step / steps)))
        if CKPT_EVERY and step % CKPT_EVERY == 0 and not SMOKE:
            # Slot alternates so the object being overwritten is never the only good copy.
            save_checkpoint({"step": step, "seen_tok": int(seen_tok), "epoch": epoch,
                             "micro_in_epoch": micro_in_epoch, "hist": hist,
                             "ent_running": ent_running, "val_before": val_before},
                            CKPT_SLOTS[(step // CKPT_EVERY) % len(CKPT_SLOTS)])
        if not math.isfinite(tr):
            fails.append("training loss became %s at step %d" % (tr, step))
            done = True
            break
        if step >= steps:
            done = True
            break

val_after = eval_loss()
tok_per_s = (seen_tok - tok_at_start) / max(1e-6, time.time() - t_train)
log("val loss after %d steps: %.4f (was %.4f)" % (step, val_after, val_before))
if step and not (val_after < val_before):
    fails.append("validation loss did not fall over %d steps (%.4f -> %.4f), so the "
                 "optimizer is not learning from this data" % (step, val_before, val_after))
if not SMOKE and seen_tok < 0.9 * BUDGET:
    fails.append("the arm saw %d tokens against a budget of %d, so it is not cost-matched "
                 "to the rest of the sweep" % (seen_tok, BUDGET))

# ------------------------------------------------------------------ 4. artifacts

adir = os.path.join(OUT, "adapter")
model.save_pretrained(adir)
tok.save_pretrained(adir)
zpath = shutil.make_archive(os.path.join(OUT, "adapter"), "zip", adir)
try:
    if not PACK_CHILD:
        lab.save_artifact(zpath)
except Exception as exc:
    fails.append("the weights could not be saved as an artifact (%s), so this run produced "
                 "nothing a later job can load" % exc)
if not os.path.exists(zpath) or os.path.getsize(zpath) < 1024:
    fails.append("the weight archive is missing or empty at %s" % zpath)
# Shared storage as well as the job artifact, because s5.4 merging and s5.6 export both need
# to load several arms at once and reaching across job boundaries for weights is fragile. The
# SDK does not document an upload call, so try it, fall back to the CLI, and treat neither as
# fatal: the artifact attached above is the guaranteed copy.
# Namespaced by mode as well as arm. A smoke trains 480 rows and produces weights that look
# exactly like a real arm's from the outside; writing them to the arm's own prefix would leave
# s5.4 loading a 30-step adapter and never knowing.
DEST = "tidepool/s5.3/arms/%s%s" % (ARM, "-smoke" if SMOKE else "")
mirrored = _storage_put(zpath, DEST)
log("weights bound for %s: %s" % (DEST, "handed to the supervisor" if PACK_CHILD and mirrored
                                  else mirrored))
if not mirrored:
    # Loud, and recorded in metrics below, but not an assertion failure. The job artifact is
    # the guaranteed copy, and marking nine hours of clean training as a failed arm would
    # also stop whatever was waiting on it. What must not happen is silence, which is exactly
    # what happened for four arms before this.
    log("WARNING: the weights are NOT on their way to %s. The job artifact is the only copy, "
        "so a later stage that loads arms by storage path will not find this one." % DEST)

dump("mix_calibration.json", {"arm": ARM, "recipe": RECIPE, "plan": plan, "selection": detail,
                              "mix": MIX, "sources": source_names,
                              "realized_role_counts": role_counts,
                              "rows_trained": len(rows), "smoke": SMOKE})
dump("train_history.json", hist)
# Peak allocation is what sets the pack size, so every run records it whether it was packed
# or not: a solo arm's peak is the budget a packed arm has to fit inside, and a packed arm's
# peak against its own ceiling says how much slack is left for one more.
if dev == "cuda":
    peak_gb = round(torch.cuda.max_memory_allocated(0) / 1e9, 2)
    reserved_gb = round(torch.cuda.max_memory_reserved(0) / 1e9, 2)
    card_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
else:
    peak_gb = reserved_gb = card_gb = None
log("peak GPU memory %s GB allocated, %s GB reserved, of %s GB"
    % (peak_gb, reserved_gb, card_gb))

metrics = {
    "arm": ARM, "recipe": RECIPE, "run_tag": RUN_TAG, "base_model": BASE,
    "packed": PACK_CHILD, "pack_memory_fraction": PACK_MEMFRAC or None,
    "peak_gpu_gb": peak_gb, "peak_gpu_reserved_gb": reserved_gb, "card_total_gb": card_gb,
    "mix": MIX, "tuning": TUNING, "loss": LOSS, "smoke": SMOKE,
    "template_mode": enc.mode, "template_notes": enc.notes,
    "steps": step, "tokens_seen": int(seen_tok), "budget_tokens": BUDGET,
    # Nonzero means this arm was finished by more than one job, so its wall-clock and its
    # tok/s are spliced from two machines and only the token count is comparable.
    "resumed_from_step": RESUMED_FROM, "checkpoint_every_steps": CKPT_EVERY,
    # Whether the weights are on their way to shared storage, and where. False means the job
    # artifact is the only copy; in a pack, True means the request was lodged and the pack
    # summary's `mirrored` block is the record of whether it landed.
    "weights_storage_dest": DEST, "weights_mirror_requested": bool(mirrored),
    "tokens_per_second": round(tok_per_s, 1),
    "val_loss_before": round(val_before, 4), "val_loss_after": round(val_after, 4),
    "val_loss_delta": round(val_after - val_before, 4),
    "mean_token_entropy": round(sum(ent_running) / len(ent_running), 4) if ent_running else None,
    "trainable_params": trainable, "total_params": total,
    "effective_batch": MICRO_BS * ACCUM, "learning_rate": LR,
    "lora": None if TUNING == "full" else {"r": LORA_R, "alpha": LORA_ALPHA,
                                           "dropout": LORA_DROPOUT},
    "guardrail_epochs": GUARD_EPOCHS, "replay_frac": REPLAY_FRAC,
    "priority_share": plan["priority_share_achieved"],
    "gpu_hours": round((time.time() - t_train) / 3600.0, 3),
    "wall_clock_seconds": round(time.time() - t_start, 1),
    "assertion_failures": fails,
}
dump("metrics.json", metrics)
score = {"arm": ARM, "val_loss": round(val_after, 4),
         "val_loss_delta": round(val_after - val_before, 4),
         "tokens_per_second": round(tok_per_s, 1), "tokens_seen": int(seen_tok),
         "steps": step, "template_mode": enc.mode,
         "priority_share": plan["priority_share_achieved"],
         "gpu_hours": metrics["gpu_hours"], "assertion_failures": len(fails),
         "packed": PACK_CHILD, "peak_gpu_gb": peak_gb, "card_total_gb": card_gb,
         # So the pack summary's results table says, per arm, where the weights went. A row
         # with a dest and no landing is the shape of the bug that lost four adapters.
         "weights_storage_dest": DEST, "weights_mirror_requested": bool(mirrored)}
dump("score.json", score)
for f in fails:
    log("ASSERTION FAILURE: %s" % f)
if WANDB:
    WANDB.log({"val/loss_after": val_after, "val/loss_before": val_before})
    WANDB.finish()
if not PACK_CHILD:
    lab.update_progress(100)
log("SCORE " + json.dumps(score))
if PACK_CHILD:
    # The supervisor reads score.json for the numbers and the exit code for pass/fail. It
    # must be able to tell "this arm asserted" from "this arm died", so assertions exit 3
    # and an uncaught exception keeps whatever code Python gives it.
    raise SystemExit(3 if fails else 0)
if fails:
    lab.error(message="%s: %d assertion failures; see metrics.json (val loss %.4f -> %.4f, "
                      "%.0f tok/s, %d tokens)"
                      % (ARM, len(fails), val_before, val_after, tok_per_s, seen_tok))
else:
    lab.finish(message="%s: %d steps, %d tokens, val loss %.4f -> %.4f, %.0f tok/s"
                       % (ARM, step, seen_tok, val_before, val_after, tok_per_s),
               score=score)
