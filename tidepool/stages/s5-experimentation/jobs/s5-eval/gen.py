"""The one serving path every full-precision number in this project comes from.

`plan.md` fixes one backend for all FP comparisons and one for all Q4, because a
benchmark delta between two checkpoints served differently is not a delta between the
checkpoints. The FP backend is Hugging Face `transformers` with greedy decoding, chosen
because it is the same code path the SFT runs load the model through, needs nothing
installed beyond what training already needs, and has no batching-dependent numerics to
argue about later. It is slower than a dedicated server; a screening pass over ~7,400
prompts still costs well under an hour on one L4, which is inside the budget line for the
whole baseline row. Any future move to a faster server invalidates every FP number taken
here and means rerunning all of them, so it is recorded in the summary as part of the
result.

Determinism: `do_sample=False`, no temperature, no penalties, left padding, and prompts
sorted by length into buckets so a batch is mostly one length. Padding still changes
nothing that greedy argmax would resolve differently, so the completions do not depend
on how the prompts were grouped. That last point is what makes the out-of-memory split
below safe: under a per-arm memory ceiling a group is retried in halves, so batch
composition depends on memory pressure from sibling arms and is not a function of the
item order alone.
"""

import time

import torch

# `torch.OutOfMemoryError` is the modern name and `torch.cuda.OutOfMemoryError` is its
# ancestor, still what older wheels raise. Catch whichever this install actually has,
# rather than pinning the harness to one wheel's spelling.
_OOM = tuple({e for e in (getattr(torch, "OutOfMemoryError", None),
                          getattr(torch.cuda, "OutOfMemoryError", None)) if e is not None})


class Runner:
    def __init__(self, model, tok, log=print):
        self.model = model
        self.tok = tok
        self.log = log
        self.gen_tokens = 0
        self.gen_seconds = 0.0
        self.truncated = 0
        self.oom_splits = 0
        # None = not yet decided; True/False = this transformers build takes logits_to_keep.
        self._logits_to_keep = None
        self.oom_seconds = 0.0
        self.min_batch = None

    def _encode(self, prompts):
        # The chat template already emits BOS, so no extra special tokens.
        return self.tok(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False)

    def generate(self, prompts, max_new_tokens=384, batch_size=16, tag="", ids=None):
        """Greedy completions, returned in the caller's order.

        `ids` is accepted and ignored. It exists because the score-only replay runner
        joins saved completions to items by id, and the component functions call one
        generate() for every backend.
        """
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out = [None] * len(prompts)
        done = 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            self._run_group(idx, prompts, max_new_tokens, out, tag)
            done += len(idx)
            if start % (batch_size * 10) == 0:
                self.log("  %s generated %d/%d (%.0f tok/s so far)"
                         % (tag, done, len(prompts),
                            self.gen_tokens / max(self.gen_seconds, 1e-6)))
        return out

    def _run_group(self, idx, prompts, max_new_tokens, out, tag=""):
        """Generate for these item indices, halving the group on out-of-memory.

        Inside a packed job each arm runs under a memory ceiling chosen before the
        longest prompt in the set is known, and prefill cost scales with the batch times
        the sequence length. A group that does not fit is therefore an ordinary event
        and not a failure: it is split and retried until it fits, or until one prompt on
        its own cannot run, which is the only case that still raises.
        """
        eos = self.tok.eos_token_id
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else eos
        enc = self._encode([prompts[i] for i in idx])
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        t0 = time.time()
        try:
            with torch.no_grad():
                # Named out_ids, not ids: `ids` is a parameter of generate() and
                # shadowing it here would make the next reader of this loop wrong.
                out_ids = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                              do_sample=False, num_beams=1,
                                              eos_token_id=eos, pad_token_id=pad,
                                              use_cache=True)
        except _OOM:
            # Time spent on an attempt that produced nothing is not generation time, so
            # it is banked separately and kept out of the tokens-per-second denominator.
            self.oom_seconds += time.time() - t0
            del enc
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if len(idx) == 1:
                raise
            self.oom_splits += 1
            half = len(idx) // 2
            self.log("  %s out of memory on %d prompts, retrying as %d + %d"
                     % (tag, len(idx), half, len(idx) - half))
            self._run_group(idx[:half], prompts, max_new_tokens, out, tag)
            self._run_group(idx[half:], prompts, max_new_tokens, out, tag)
            return
        self.gen_seconds += time.time() - t0
        new = out_ids[:, enc["input_ids"].shape[1]:]
        self.gen_tokens += int((new != pad).sum().item())
        for j, i in enumerate(idx):
            row = new[j]
            if int(row.shape[0]) >= max_new_tokens and eos not in set(row.tolist()):
                self.truncated += 1
            out[i] = self.tok.decode(row, skip_special_tokens=True).strip()
        if self.min_batch is None or len(idx) < self.min_batch:
            self.min_batch = len(idx)

    def first_token_probs(self, prompts, choices, batch_size=16, tag=""):
        """P(first generated token) collapsed onto named token sets. One forward pass.

        The calibration component needs a scalar per item, not text, and the whole decision
        it wants sits in the first token of a one-word answer. So this prefills and reads
        the last position's logits instead of calling generate(): nothing is sampled,
        nothing is decoded, and the result cannot depend on a stopping rule.

        `choices` maps a name to a list of token ids; the returned probability for a name
        is the summed softmax mass over its ids. Softmax is taken in float32 because the
        mass on two rare tokens under bf16 rounds badly at the tail, and the ratio between
        them is exactly what is being measured.

        Left padding (set in load_model) is what makes position -1 the true last token of
        every row in a mixed-length batch. Groups are halved on out-of-memory for the same
        reason generation is, since a packed arm's ceiling is set before prompt lengths are
        known. Time here is NOT generation time and is kept out of the tokens/second
        denominator: no tokens are generated.
        """
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out = [None] * len(prompts)
        done = 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            self._probs_group(idx, prompts, choices, out, tag)
            done += len(idx)
            if start % (batch_size * 10) == 0:
                self.log("  %s scored %d/%d" % (tag, done, len(prompts)))
        return out

    def _probs_group(self, idx, prompts, choices, out, tag=""):
        enc = self._encode([prompts[i] for i in idx])
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        # Only the last position is read, and the head is the memory here: a batch of 16
        # two-thousand-token tool traces materializes 16 x 2000 x |vocab| logits, several
        # gigabytes, to have all but 16 rows thrown away. `logits_to_keep` tells transformers
        # to run the head on the last position only. It is not universal across versions, so
        # the first call decides whether this build takes it and the rest follow; a build that
        # does not is correct and merely fatter, which the OOM split below already handles.
        kw = dict(enc)
        trying_keep = self._logits_to_keep is not False
        if trying_keep:
            kw["logits_to_keep"] = 1
        try:
            with torch.no_grad():
                try:
                    logits = self.model(**kw).logits[:, -1, :].float()
                except TypeError:
                    if not trying_keep:
                        raise                       # not about the keyword; let it out
                    self._logits_to_keep = False
                    self.log("  %s this transformers build takes no logits_to_keep; "
                             "scoring the full head" % tag)
                    logits = self.model(**enc).logits[:, -1, :].float()
                else:
                    if trying_keep:
                        self._logits_to_keep = True
                probs = torch.softmax(logits, dim=-1)
        except _OOM:
            del enc
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if len(idx) == 1:
                raise
            self.oom_splits += 1
            half = len(idx) // 2
            self.log("  %s out of memory scoring %d prompts, retrying as %d + %d"
                     % (tag, len(idx), half, len(idx) - half))
            self._probs_group(idx[:half], prompts, choices, out, tag)
            self._probs_group(idx[half:], prompts, choices, out, tag)
            return
        for j, i in enumerate(idx):
            row = probs[j]
            out[i] = {name: float(row[ids].sum().item()) for name, ids in choices.items()}

    def sample(self, prompt, n, temperature=0.8, top_p=1.0, max_new_tokens=320,
               chunk=8, seed=0, tag=""):
        """n stochastic completions of ONE prompt. Diagnostic only; never a reported metric.

        Every benchmark number in this project comes from `generate()` above, which is greedy
        and says so at length. This method deliberately breaks that: the peakedness check at
        `s6.4` needs the model's output DISTRIBUTION, which greedy decoding by construction
        cannot show. It is kept as a separate entry point rather than a flag on generate() so
        that no benchmark path can reach sampling by passing the wrong argument.

        The seed is set per chunk from `seed` and the chunk index, so the run reproduces
        regardless of how the chunks were sized. Chunk size only exists because the card holds
        a fixed number of concurrent sequences: 51 samples of a 3k-token BFCL prompt is a
        prefill of batch 51, well past what an L4 carries, and it is halved on out-of-memory
        exactly as a generation group is.
        """
        eos = self.tok.eos_token_id
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else eos
        out = []
        c = 0
        while len(out) < n:
            want = min(chunk, n - len(out))
            out.extend(self._sample_chunk(prompt, want, temperature, top_p, max_new_tokens,
                                          eos, pad, seed + 1000 * c, tag))
            c += 1
        return out[:n]

    def _sample_chunk(self, prompt, k, temperature, top_p, max_new_tokens, eos, pad,
                      seed, tag):
        import time as _time
        enc = self._encode([prompt])
        enc = {kk: v.to(self.model.device) for kk, v in enc.items()}
        t0 = _time.time()
        try:
            torch.manual_seed(seed)
            with torch.no_grad():
                out_ids = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                              do_sample=True, temperature=temperature,
                                              top_p=top_p, top_k=0,
                                              num_return_sequences=k,
                                              eos_token_id=eos, pad_token_id=pad,
                                              use_cache=True)
        except _OOM:
            self.oom_seconds += _time.time() - t0
            del enc
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if k == 1:
                raise
            self.oom_splits += 1
            half = k // 2
            self.log("  %s out of memory on %d samples, retrying as %d + %d"
                     % (tag, k, half, k - half))
            return (self._sample_chunk(prompt, half, temperature, top_p, max_new_tokens,
                                       eos, pad, seed, tag)
                    + self._sample_chunk(prompt, k - half, temperature, top_p, max_new_tokens,
                                         eos, pad, seed + 1, tag))
        self.gen_seconds += _time.time() - t0
        new = out_ids[:, enc["input_ids"].shape[1]:]
        self.gen_tokens += int((new != pad).sum().item())
        return [self.tok.decode(row, skip_special_tokens=True).strip() for row in new]

    def throughput(self):
        return {"generated_tokens": self.gen_tokens,
                "generate_seconds": round(self.gen_seconds, 1),
                "tokens_per_second": round(self.gen_tokens / max(self.gen_seconds, 1e-6), 1),
                "hit_max_new_tokens": self.truncated,
                "oom_splits": self.oom_splits,
                "oom_seconds": round(self.oom_seconds, 1),
                "smallest_batch": self.min_batch}


def load_model(base_model, adapter_dir=None, log=print):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # Decoder-only batched generation needs the padding on the left, or the model
    # continues from pad tokens and every short prompt in the batch is scored on noise.
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
        log("merged LoRA adapter from %s" % adapter_dir)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    log("loaded %s (%d params, bf16, padding_side=%s)" % (base_model, n, tok.padding_side))
    return model, tok, n
