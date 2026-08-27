"""Freeze the base model's own answers over the general-quality prompt pool.

The plan's replay control (row `C5`) protects general quality by mixing a small fraction of
general-instruction data back into the specialization run. The corpus that was supposed to
supply it ships no assistant responses at all: all 447,053 renderable rows end on a user
turn, which was measured at `s4.4`. A prompt set with no responses is what self-distillation
replay is for, so the responses are generated once, here, by the checkpoint the replay is
meant to protect, and then frozen for the whole sweep.

Two properties matter more than the text quality, and both are why this is its own job:

  **On-policy.** The replay target is the base model's own distribution. Mixing in a
  stronger model's answers would not be replay; it would be distillation from a teacher the
  project never declared, and any guardrail that held would be unattributable.

  **Frozen.** Every arm in the sweep replays the same completions, generated once with
  greedy decoding. Regenerating per arm would make the replay buffer a source of variance
  in exactly the comparison it exists to stabilize.

The generation is deliberately unfiltered. A completion the base model gets wrong is still
the right replay target: the goal is to keep the specialized model where the base model
already was, not to improve on it.
"""
#
# This is the sweep's copy of the self-distillation generator, made packable. The buffer it
# writes is an input to two arms of the sweep (C5a and C5b), and generating it as a separate
# job means a card that sits at a fifth of its capacity for an hour and two arms that cannot
# start until a whole second job has been queued, launched and finished. Run as an arm of the
# pack instead, it shares the card with the arms that do not need it, hands its file straight
# to the arms that do, and the supervisor uploads it once so later stages still find it in
# shared storage.
#
# Divergence from ../s5-selfdistill/main.py is the risk here, and it is managed the same way
# main.py's is: sync_pack_task.sh copies both into the task directory from one place, and the
# generation path below -- sampling, templating, greedy decode, n_tok accounting, every
# assertion -- is untouched. What changed is who reports, where it writes, and where its
# inputs come from.

import gzip
import hashlib
import json
import os
import time

import torch

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



def C(k, d):
    v = CFG.get(k)
    return d if v is None or v == "" else v


ARM = str(os.environ.get("TIDEPOOL_PACK_ARM") or C("arm", "RB"))
BASE = C("base_model", "LiquidAI/LFM2.5-1.2B-Instruct")
POOL_OBJ = C("pool_object", "tidepool/s4.4/prompt_pool.jsonl.gz")
N_PROMPTS = int(C("n_prompts", 64))
MAX_NEW = int(C("max_new_tokens", 320))
BATCH = int(C("batch_size", 16))
MAX_PROMPT_TOK = int(C("max_prompt_tokens", 1024))
SMOKE = bool(C("smoke", False))
RUN_TAG = C("run_tag", "s5.1-selfdistill-smoke")
DEST = C("dest_prefix", "")

OUT = os.environ.get("TIDEPOOL_PACK_OUT") or "out"
os.makedirs(OUT, exist_ok=True)
fails = []
t_start = time.time()


def log(m):
    if PACK_CHILD:
        print("[%s] %s" % (ARM, m), flush=True)
        return
    lab.log(m)
    print(m, flush=True)


def dump(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    if PACK_CHILD:
        return p
    try:
        lab.save_artifact(p)
    except Exception as exc:
        log("save_artifact failed for %s: %s" % (name, exc))
    return p


def place(path, dest):
    """Put a file in shared storage if this SDK can, and say so either way."""
    if PACK_CHILD:
        # A packed child does not talk to storage. The supervisor registers this file as the
        # object the waiting arms expect and uploads it once on the pack's behalf, which is
        # both fewer copies and one writer instead of several.
        log("packed: the supervisor places %s and hands it to the arms waiting on it"
            % os.path.basename(path))
        return True
    up = None
    for name in ("storage_upload", "upload_storage", "storage_put"):
        up = getattr(lab, name, None)
        if callable(up):
            break
        up = None
    if up is None:
        log("no storage upload in this SDK; %s stays on the job record for outside placement"
            % os.path.basename(path))
        return False
    obj = "%s/%s" % (dest.rstrip("/"), os.path.basename(path))
    for args, kwargs in (((path, obj), {}), ((path,), {"dest": dest}), ((path,), {})):
        try:
            up(*args, **kwargs)
            log("placed %s in shared storage" % obj)
            return True
        except Exception as exc:
            log("upload shape %r failed: %s" % (kwargs or args, exc))
    return False


# ------------------------------------------------------- 1. a reproducible prompt sample

local = storage(POOL_OBJ)
# Reservoir-free: rank every prompt by a hash of its identity and keep the lowest N. The
# sample is a pure function of the pool and the salt, so a rerun draws the same prompts and
# a larger N is a superset of a smaller one.
SALT = str(C("sample_salt", "tidepool-replay-v1"))

# `strata` reweights the sample by sub-source. Added at s5.5, for a reason the sweep only made
# visible once someone read the pool: `antidoom-mix-v1.0` carries 46,734 rows of
# `open_perfectblend_autoif`, verifiable-constraint instruction following of exactly the kind
# IFEval scores, and a proportional sample puts it at 10.5% of the buffer. At a replay fraction
# of 1% that is about a tenth of one percent of the training tokens, which is why the replay
# axis moved IFEval by two points and stopped. Reweighting buys the same protective dose for a
# fifth of the replay tokens, so it is a lever on composition rather than on budget.
#
# Format: {"<substring of the row id>": <share of N>, ...}. Shares must sum to at most 1.0; the
# remainder is one "other" stratum drawn from every row that matched nothing. An empty map is
# the default and reproduces the unstratified sample exactly, which is what keeps the frozen
# s5.3 buffer reproducible from this code.
STRATA = json.loads(str(C("strata", "{}")) or "{}")
if sum(STRATA.values()) > 1.0 + 1e-9:
    raise SystemExit("strata shares sum to %.4f, which is more than the whole sample"
                     % sum(STRATA.values()))
TARGET = {k: int(round(v * N_PROMPTS)) for k, v in STRATA.items()}
TARGET["__other__"] = N_PROMPTS - sum(TARGET.values())
if TARGET["__other__"] < 0:
    raise SystemExit("rounding the strata shares overshot N; asked for %d of %d"
                     % (sum(TARGET.values()) - TARGET["__other__"], N_PROMPTS))


def stratum_of(row):
    """Which stratum a pool row belongs to. First match wins, in the map's own order."""
    rid = str(row.get("i", ""))
    for k in STRATA:
        if k in rid:
            return k
    return "__other__"


best, n_rows = {k: [] for k in TARGET}, 0
seen = dict.fromkeys(TARGET, 0)
with gzip.open(local, "rt") as fh:
    for line in fh:
        r = json.loads(line)
        n_rows += 1
        st = stratum_of(r)
        seen[st] += 1
        want = TARGET[st]
        if want <= 0:
            continue
        key = hashlib.blake2b(("%s|%s|%s" % (r.get("c"), r.get("i"), SALT)).encode(),
                              digest_size=8).hexdigest()
        bucket = best[st]
        bucket.append((key, r))
        if len(bucket) > want * 4:
            bucket.sort(key=lambda kv: kv[0])
            del bucket[want:]
picked, short = [], []
for st in TARGET:
    bucket = best[st]
    bucket.sort(key=lambda kv: kv[0])
    got = bucket[:TARGET[st]]
    picked.extend(r for _, r in got)
    log("stratum %-28s wanted %5d  pool has %7d  took %5d"
        % (st, TARGET[st], seen[st], len(got)))
    # Two different faults, and the stratum has to name which. A stratum the pool cannot
    # supply is a configuration error in the strata map; a stratum that came up short of what
    # the pool does hold is a bug in the sampler. Either way the buffer's composition is not
    # the one the arm was specified against, so neither is allowed to pass quietly.
    if len(got) < TARGET[st]:
        short.append("%s (wanted %d, pool holds %d, took %d)"
                     % (st, TARGET[st], seen[st], len(got)))
# One deterministic order for the whole sample, so a reweighted run and a proportional run of
# the same N are directly diffable and generation order does not depend on the strata map.
picked.sort(key=lambda r: hashlib.blake2b(
    ("%s|%s|%s" % (r.get("c"), r.get("i"), SALT)).encode(), digest_size=8).hexdigest())
log("pool has %d rows; sampled %d by hash rank across %d stratum/strata"
    % (n_rows, len(picked), len(TARGET)))
if short:
    fails.append("strata could not be filled from the pool: %s" % "; ".join(short))
if len(picked) < min(N_PROMPTS, n_rows):
    fails.append("sampled %d prompts from a pool of %d, wanted %d"
                 % (len(picked), n_rows, N_PROMPTS))

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                            trust_remote_code=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu":
    fails.append("no CUDA device is visible, so this run is not the run that was requested")
if PACK_CHILD and PACK_MEMFRAC > 0 and dev == "cuda":
    # The ceiling is what makes sharing a card safe. An arm that grows past its share fails
    # here, alone, instead of taking a sibling's allocation out from under it.
    torch.cuda.set_per_process_memory_fraction(PACK_MEMFRAC, 0)
    log("memory ceiling %.1f%% of the card" % (100 * PACK_MEMFRAC))
model.to(dev).eval()
log("loaded %s on %s" % (BASE, dev))

texts, keep = [], []
for r in picked:
    msgs = [m for m in r["messages"] if m["role"] in ("system", "user")]
    if not any(m["role"] == "user" for m in msgs):
        continue
    try:
        t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception as exc:
        fails.append("the model's chat template rejected a pool prompt: %s" % exc)
        break
    ids = tok(t, add_special_tokens=False)["input_ids"]
    if len(ids) > MAX_PROMPT_TOK:
        continue
    texts.append(t)
    keep.append(r)
log("%d of %d sampled prompts are usable within %d prompt tokens"
    % (len(texts), len(picked), MAX_PROMPT_TOK))
if not texts:
    fails.append("no pool prompt survived templating and the length cap")

# ------------------------------------------------------------------ 2. greedy generation

# The sweep's sampler reads every source through one code path: it keys rows by `role`, sizes
# each role by the `n_tok` field, and opens a source by filename suffix. A replay file that
# omits `n_tok` indexes as zero tokens and the replay dose silently becomes nothing, so the
# count is written here, with the same tokenizer and the same template the training run uses.
path = os.path.join(OUT, "replay.jsonl.gz")


def _ids(enc):
    # apply_chat_template(tokenize=True) returns a list of ids on some versions and a
    # BatchEncoding/dict on others. len() on the latter counts its KEYS -- two of them,
    # input_ids and attention_mask -- which is how the first run at size wrote n_tok=2 for
    # every row and indexed a 1.85M-token buffer as 15,890 training tokens. Unwrap first.
    if hasattr(enc, "input_ids"):
        enc = enc.input_ids
    elif isinstance(enc, dict):
        enc = enc["input_ids"]
    if len(enc) and isinstance(enc[0], (list, tuple)):
        enc = enc[0]
    return enc


def row_tokens(msgs):
    try:
        return len(_ids(tok.apply_chat_template(msgs, tokenize=True,
                                                add_generation_prompt=False)))
    except Exception:
        return len(tok("\n".join(m.get("content") or "" for m in msgs)).input_ids)


n_out, n_tok, empty, train_tok = 0, 0, 0, 0
with gzip.open(path, "wt") as fh:
    for i in range(0, len(texts), BATCH):
        chunk, meta = texts[i:i + BATCH], keep[i:i + BATCH]
        b = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            out = model.generate(**b, max_new_tokens=MAX_NEW, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, b["input_ids"].shape[1]:]
        for r, g in zip(meta, gen):
            txt = tok.decode(g, skip_special_tokens=True).strip()
            if not txt:
                empty += 1
                continue
            msgs = [m for m in r["messages"] if m["role"] in ("system", "user")]
            full = msgs + [{"role": "assistant", "content": txt}]
            nt = row_tokens(full)
            train_tok += nt
            fh.write(json.dumps({
                "c": r.get("c"), "i": r.get("i"), "g": r.get("g"), "role": "replay",
                "source": "self_distilled_greedy", "base_model": BASE,
                "n_tok": nt, "messages": full,
            }, ensure_ascii=False) + "\n")
            n_out += 1
            n_tok += int((g != tok.pad_token_id).sum())
        if not PACK_CHILD:
            lab.update_progress(min(95, int(90.0 * (i + BATCH) / max(1, len(texts)))))
        if i % (BATCH * 8) == 0:
            log("generated %d/%d (%.0f completion tok/s)"
                % (n_out, len(texts), n_tok / max(1e-6, time.time() - t_start)))

if not PACK_CHILD:
    try:
        lab.save_artifact(path)
    except Exception as exc:
        fails.append("the replay file could not be saved as an artifact (%s), so this run "
                     "produced nothing a later job can load" % exc)
if not os.path.exists(path) or os.path.getsize(path) < 128:
    fails.append("the replay file is missing or effectively empty at %s" % path)
if n_out == 0:
    fails.append("every generation came back empty, so there is nothing to replay")
if empty > 0.2 * max(1, len(texts)):
    fails.append("%d of %d generations came back empty, which is too many to treat as noise"
                 % (empty, len(texts)))

# An exact invariant, not a heuristic: a training row is its prompt plus its completion, so
# the buffer's training-token total can never be below the completion tokens it was built
# from. The first run at size violated this by two orders of magnitude (15,890 against
# 1,850,696) because n_tok was counting dict keys. The C5b ratio check below did catch it,
# but only after inferring the cause from a replay-passes number; this names it directly.
if n_out and train_tok < n_tok:
    fails.append("the buffer reports %d training tokens against %d generated completion "
                 "tokens, which is impossible for rows that carry prompt AND completion "
                 "(%.1f per row): n_tok is being miscounted, not the generation"
                 % (train_tok, n_tok, train_tok / max(1, n_out)))

# The dose the sweep will ask for. `C5b` replays 5% of a 64.0M-token budget, so a buffer
# under 3.2M training tokens is not an error, it just means the sampler repeats rows; the
# ratio is reported so the repeat count is a number in the record rather than a surprise.
C5B_DOSE = int(C("c5b_dose_tokens", 3_200_000))
passes = C5B_DOSE / max(1.0, float(train_tok))
if not SMOKE and passes > 2.0:
    fails.append("the buffer holds %.2fM training tokens, so C5b's %.2fM-token dose would "
                 "replay it %.1f times over and the arm would measure memorization of a "
                 "small set as much as replay" % (train_tok / 1e6, C5B_DOSE / 1e6, passes))

placed = place(path, DEST) if DEST else False
if not SMOKE and DEST and not placed:
    fails.append("the buffer was generated but could not be placed at %s, and the sweep "
                 "loads replay from shared storage rather than from a job artifact" % DEST)

if dev == "cuda":
    peak_gb = round(torch.cuda.max_memory_allocated(0) / 1e9, 2)
    reserved_gb = round(torch.cuda.max_memory_reserved(0) / 1e9, 2)
    card_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
else:
    peak_gb = reserved_gb = card_gb = None

tps = n_tok / max(1e-6, time.time() - t_start)
score = {"completions": n_out, "empty": empty,
         "packed": PACK_CHILD, "peak_gpu_gb": peak_gb,
         "peak_gpu_reserved_gb": reserved_gb, "card_total_gb": card_gb, "completion_tokens": int(n_tok),
         "training_tokens": int(train_tok),
         "c5b_replay_passes": round(passes, 2), "placed_in_storage": placed,
         "completion_tokens_per_second": round(tps, 1),
         "pool_rows": n_rows, "sampled": len(picked), "usable": len(texts),
         "projected_hours_per_100k": round(100000.0 * (n_tok / max(1, n_out))
                                          / max(1.0, tps) / 3600.0, 2),
         "assertion_failures": len(fails)}
dump("score.json", score)
dump("replay_summary.json", {"run_tag": RUN_TAG, "base_model": BASE, "pool_object": POOL_OBJ,
                             "sample_salt": SALT, "n_prompts_requested": N_PROMPTS,
                             "max_new_tokens": MAX_NEW, "greedy": True, "smoke": SMOKE,
                             "wall_clock_seconds": round(time.time() - t_start, 1),
                             "score": score, "assertion_failures": fails})
for f in fails:
    log("ASSERTION FAILURE: %s" % f)
if not PACK_CHILD:
    lab.update_progress(100)
log("SCORE " + json.dumps(score))
if PACK_CHILD:
    # Same contract every packed arm keeps: 3 means it asserted, anything else Python chose.
    raise SystemExit(3 if fails else 0)
# lab.finish() marks SUCCESS and takes (message, score, ...) with no status argument;
# lab.error() marks FAILED and takes message only. A failed run therefore carries its
# numbers in score.json rather than in job_data.score, which is why the message repeats
# the headline figures.
if fails:
    lab.error(message="%d assertion failures; see replay_summary.json "
                      "(%d completions, %d empty, %.0f tok/s)"
                      % (len(fails), n_out, score["empty"], tps))
else:
    lab.finish(message="%d frozen greedy completions over the prompt pool, %.0f tok/s"
                       % (n_out, tps),
               score=score)
