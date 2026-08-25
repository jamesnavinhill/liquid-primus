"""s5 supervised fine-tuning of LFM2.5-1.2B-Instruct, and the s5.1 smoke run of it.

One task covers the whole C row of the experiment matrix: the smoke run, the reference arm,
and every ablation that differs from it only by configuration. Keeping them in one code path
is deliberate. An arm that runs different code from the arm it is compared against is not an
ablation, and the cheapest way to guarantee they share a code path is to give them one.

What the smoke configuration exists to prove, in the order the failures actually happen:

  1. The base checkpoint loads on the card we asked for, in bf16.
  2. Its own chat template accepts a four-role conversation including `tool` turns. If it
     does not, the run says so and switches to an explicit fallback, because a number
     produced under a different template is not comparable to one produced under this one.
  3. The loss mask covers assistant tokens and nothing else, asserted rather than assumed.
  4. The sampler's calibration runs on the real corpus and produces the mix the plan calls
     for, with the achieved shares recorded.
  5. LoRA attaches, the loss moves, and the optimizer step does not diverge.
  6. **The adapter and the metrics come back as artifacts.** This is the step that fails
     silently and only costs you at full scale, so the run treats a missing artifact as a
     failure even when the training itself succeeded.

Everything above is cheap at 30 steps and expensive to discover at 90 million tokens.
"""

import json
import math
import os
import shutil
import time

import torch
from torch.utils.data import DataLoader, Dataset
from lab import lab

import sample
import textify

lab.init()
CFG = lab.get_config() or {}


def C(k, default):
    v = CFG.get(k)
    return default if v is None or v == "" else v


BASE = C("base_model", "LiquidAI/LFM2.5-1.2B-Instruct")
TRAIN_OBJ = C("train_object", "tidepool/s4.4/train.jsonl.gz")
VAL_OBJ = C("val_object", "tidepool/s4.4/val.jsonl.gz")
MIX = C("mix", "role_balanced")            # role_balanced (C1) | raw (C7)
SMOKE = bool(C("smoke", False))
MAX_LEN = int(C("max_seq_len", 2048))
MICRO_BS = int(C("micro_batch_size", 2))
ACCUM = int(C("grad_accum", 24))
LR = float(C("learning_rate", 1e-4))
LORA_R = int(C("lora_r", 16))
LORA_ALPHA = int(C("lora_alpha", 32))
LORA_DROPOUT = float(C("lora_dropout", 0.05))
MAX_STEPS = int(C("max_steps", 0))          # 0 = one full pass over the sampled mix
VAL_ROWS = int(C("val_rows", 256))
WARMUP = float(C("warmup_ratio", 0.03))
SEED = int(C("seed", 17))
TOOL_EPOCHS = float(C("tool_epochs", 1.0))
STRUCT_MAX_EPOCHS = float(C("struct_max_epochs", 3.0))
PRIORITY_SHARE = float(C("min_priority_share", 0.5))
SMOKE_ROWS = int(C("smoke_rows", 512))
RUN_TAG = C("run_tag", "s5.1-smoke")

OUT = "out"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED)


def log(msg):
    lab.log(msg)
    print(msg, flush=True)


def dump(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    try:
        lab.save_artifact(p)
    except Exception as exc:
        log("save_artifact failed for %s: %s" % (name, exc))
    return p


fails = []
t_start = time.time()

# ------------------------------------------------------------------ 1. the mix

log("downloading splits")
train_local = lab.storage_download(TRAIN_OBJ)
val_local = lab.storage_download(VAL_OBJ)

t0 = time.time()
per_role, order = sample.index_split(train_local, log=log)
plan = sample.calibrate({k: v for k, v in per_role.items() if k in sample.ROLES},
                        tool_epochs=TOOL_EPOCHS, struct_max_epochs=STRUCT_MAX_EPOCHS,
                        min_priority_share=PRIORITY_SHARE)
log("calibration (%.1fs): %s" % (time.time() - t0, json.dumps(plan)))

if MIX == "raw":
    # C7. Every row once, in the same hash order, so the only difference from C1 is which
    # rows and how often — not the order they arrive in.
    lines = [ln for role in sample.ROLES for _, ln in (order.get(role) or [])]
    lines.sort(key=lambda ln: sample._rank({"c": "x", "i": ln}, "mix"))
    detail = {"mode": "raw", "rows": len(lines)}
else:
    lines, detail = sample.choose(order, per_role, plan, log=log)

if SMOKE:
    lines = lines[:SMOKE_ROWS]
    log("SMOKE: training rows capped at %d; the calibration above is still the real one, "
        "computed over the whole split" % len(lines))

rows = sample.read_lines(train_local, lines)
val_rows = []
import gzip
with gzip.open(val_local, "rt") as fh:
    for i, line in enumerate(fh):
        if i >= VAL_ROWS * 8:
            break
        val_rows.append(json.loads(line))
val_rows = val_rows[::max(1, len(val_rows) // VAL_ROWS)][:VAL_ROWS]
log("materialized %d train rows and %d val rows" % (len(rows), len(val_rows)))
if not rows:
    fails.append("the sampler selected no training rows")

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
peft_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      bias="none", task_type="CAUSAL_LM", target_modules="all-linear")
model = get_peft_model(model, peft_cfg)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
log("LoRA attached: %d trainable of %d parameters (%.3f%%)"
    % (trainable, total, 100.0 * trainable / total))
if trainable == 0:
    fails.append("LoRA attached no trainable parameters")
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu":
    fails.append("no CUDA device is visible, so this run is not the run that was requested")
model.to(dev)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()


class Conv(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids, labels = enc.encode(self.data[i]["messages"])
        return {"input_ids": ids, "labels": labels}


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id or 0
    ids = torch.full((len(batch), n), pad, dtype=torch.long)
    lab_ = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        lab_[i, :k] = torch.tensor(b["labels"])
        att[i, :k] = 1
    return {"input_ids": ids, "labels": lab_, "attention_mask": att}


loader = DataLoader(Conv(rows), batch_size=MICRO_BS, shuffle=False, collate_fn=collate,
                    num_workers=2 if len(rows) > 2000 else 0, drop_last=True)
steps = MAX_STEPS or max(1, len(loader) // ACCUM)
log("%d micro-batches, accumulation %d, %d optimizer steps" % (len(loader), ACCUM, steps))

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR,
                        weight_decay=0.0, betas=(0.9, 0.95))
sched = get_cosine_schedule_with_warmup(opt, int(steps * WARMUP), steps)

WANDB = None
if os.environ.get("WANDB_API_KEY"):
    try:
        import wandb
        WANDB = wandb.init(project="tidepool", name=RUN_TAG, config={
            "base_model": BASE, "mix": MIX, "lora_r": LORA_R, "lr": LR, "smoke": SMOKE,
            "max_seq_len": MAX_LEN, "effective_batch": MICRO_BS * ACCUM, "steps": steps,
            "template_mode": enc.mode})
        log("W&B run: %s" % WANDB.url)
    except Exception as exc:
        log("W&B unavailable (%s); training continues without it" % exc)


def eval_loss():
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
hist, step, seen_tok, t_train = [], 0, 0, time.time()
val_before = eval_loss()
log("val loss before any step: %.4f" % val_before)
running, micro = 0.0, 0
done = False
for epoch in range(100):
    if done:
        break
    for batch in loader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        out = model(**batch)
        (out.loss / ACCUM).backward()
        running += float(out.loss)
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
        if step % max(1, steps // 20) == 0 or step == 1:
            log("step %d/%d  loss %.4f  %d tokens  %.0f tok/s"
                % (step, steps, tr, seen_tok, seen_tok / max(1e-6, time.time() - t_train)))
            lab.update_progress(min(95, int(90.0 * step / steps)))
        if not math.isfinite(tr):
            fails.append("training loss became %s at step %d" % (tr, step))
            done = True
            break
        if step >= steps:
            done = True
            break

val_after = eval_loss()
tok_per_s = seen_tok / max(1e-6, time.time() - t_train)
log("val loss after %d steps: %.4f (was %.4f)" % (step, val_after, val_before))
if step and not (val_after < val_before):
    fails.append("validation loss did not fall over %d steps (%.4f -> %.4f), so the "
                 "optimizer is not learning from this data" % (step, val_before, val_after))

# ------------------------------------------------------------------ 4. artifacts

adir = os.path.join(OUT, "adapter")
model.save_pretrained(adir)
tok.save_pretrained(adir)
zpath = shutil.make_archive(os.path.join(OUT, "adapter"), "zip", adir)
try:
    lab.save_artifact(zpath)
except Exception as exc:
    fails.append("the adapter could not be saved as an artifact (%s), so this run produced "
                 "nothing a later job can load" % exc)
if not os.path.exists(zpath) or os.path.getsize(zpath) < 1024:
    fails.append("the adapter archive is missing or empty at %s" % zpath)

dump("mix_calibration.json", {"plan": plan, "selection": detail, "mix": MIX,
                              "rows_trained": len(rows), "smoke": SMOKE})
dump("train_history.json", hist)
metrics = {
    "run_tag": RUN_TAG, "base_model": BASE, "mix": MIX, "smoke": SMOKE,
    "template_mode": enc.mode, "template_notes": enc.notes,
    "steps": step, "tokens_seen": int(seen_tok), "tokens_per_second": round(tok_per_s, 1),
    "val_loss_before": round(val_before, 4), "val_loss_after": round(val_after, 4),
    "val_loss_delta": round(val_after - val_before, 4),
    "trainable_params": trainable, "total_params": total,
    "effective_batch": MICRO_BS * ACCUM, "learning_rate": LR,
    "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
    "epoch_budget_tokens": plan["budget_tokens"],
    "projected_full_run_hours": round(plan["budget_tokens"] / max(1.0, tok_per_s) / 3600.0, 2),
    "wall_clock_seconds": round(time.time() - t_start, 1),
    "assertion_failures": fails,
}
dump("metrics.json", metrics)
score = {"val_loss": round(val_after, 4), "val_loss_delta": round(val_after - val_before, 4),
         "tokens_per_second": round(tok_per_s, 1), "steps": step,
         "template_mode": enc.mode, "priority_share": plan["priority_share_achieved"],
         "projected_full_run_hours": metrics["projected_full_run_hours"],
         "assertion_failures": len(fails)}
dump("score.json", score)
for f in fails:
    log("ASSERTION FAILURE: %s" % f)
if WANDB:
    WANDB.log({"val/loss_after": val_after, "val/loss_before": val_before})
    WANDB.finish()
lab.update_progress(100)
log("SCORE " + json.dumps(score))
# lab.finish() marks SUCCESS and takes (message, score, ...) with no status argument;
# lab.error() marks FAILED and takes message only. A failed run therefore carries its
# numbers in score.json rather than in job_data.score, which is why the message repeats
# the headline figures.
if fails:
    lab.error(message="%d assertion failures; see metrics.json (val loss %.4f -> %.4f, "
                      "%.0f tok/s, template %s)"
                      % (len(fails), val_before, val_after, tok_per_s, enc.mode))
else:
    lab.finish(message="%s: %d steps, val loss %.4f -> %.4f, %.0f tok/s, template %s"
                       % (RUN_TAG, step, val_before, val_after, tok_per_s, enc.mode),
               score=score)
