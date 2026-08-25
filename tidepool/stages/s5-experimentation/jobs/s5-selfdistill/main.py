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

import gzip
import hashlib
import json
import os
import time

import torch
from lab import lab

lab.init()
CFG = lab.get_config() or {}


def C(k, d):
    v = CFG.get(k)
    return d if v is None or v == "" else v


BASE = C("base_model", "LiquidAI/LFM2.5-1.2B-Instruct")
POOL_OBJ = C("pool_object", "tidepool/s4.4/prompt_pool.jsonl.gz")
N_PROMPTS = int(C("n_prompts", 64))
MAX_NEW = int(C("max_new_tokens", 320))
BATCH = int(C("batch_size", 16))
MAX_PROMPT_TOK = int(C("max_prompt_tokens", 1024))
SMOKE = bool(C("smoke", False))
RUN_TAG = C("run_tag", "s5.1-selfdistill-smoke")

OUT = "out"
os.makedirs(OUT, exist_ok=True)
fails = []
t_start = time.time()


def log(m):
    lab.log(m)
    print(m, flush=True)


def dump(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    try:
        lab.save_artifact(p)
    except Exception as exc:
        log("save_artifact failed for %s: %s" % (name, exc))
    return p


# ------------------------------------------------------- 1. a reproducible prompt sample

local = lab.storage_download(POOL_OBJ)
# Reservoir-free: rank every prompt by a hash of its identity and keep the lowest N. The
# sample is a pure function of the pool and the salt, so a rerun draws the same prompts and
# a larger N is a superset of a smaller one.
SALT = str(C("sample_salt", "tidepool-replay-v1"))
best, n_rows = [], 0
with gzip.open(local, "rt") as fh:
    for line in fh:
        r = json.loads(line)
        n_rows += 1
        key = hashlib.blake2b(("%s|%s|%s" % (r.get("c"), r.get("i"), SALT)).encode(),
                              digest_size=8).hexdigest()
        best.append((key, r))
        if len(best) > N_PROMPTS * 4:
            best.sort(key=lambda kv: kv[0])
            del best[N_PROMPTS:]
best.sort(key=lambda kv: kv[0])
picked = [r for _, r in best[:N_PROMPTS]]
log("pool has %d rows; sampled %d by hash rank" % (n_rows, len(picked)))
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

path = os.path.join(OUT, "replay.jsonl")
n_out, n_tok, empty = 0, 0, 0
with open(path, "w") as fh:
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
            fh.write(json.dumps({
                "c": r.get("c"), "i": r.get("i"), "g": r.get("g"), "role": "replay",
                "source": "self_distilled_greedy", "base_model": BASE,
                "messages": msgs + [{"role": "assistant", "content": txt}],
            }, ensure_ascii=False) + "\n")
            n_out += 1
            n_tok += int((g != tok.pad_token_id).sum())
        lab.update_progress(min(95, int(90.0 * (i + BATCH) / max(1, len(texts)))))
        if i % (BATCH * 8) == 0:
            log("generated %d/%d (%.0f completion tok/s)"
                % (n_out, len(texts), n_tok / max(1e-6, time.time() - t_start)))

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

tps = n_tok / max(1e-6, time.time() - t_start)
score = {"completions": n_out, "empty": empty, "completion_tokens": int(n_tok),
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
lab.update_progress(100)
log("SCORE " + json.dumps(score))
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
