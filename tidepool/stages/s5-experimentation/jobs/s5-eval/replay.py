"""Score-only replay: re-score a finished run's saved completions without regenerating.

Three things make this worth having as a mode of the eval task rather than a task of its
own. A scoring bug found after eight sweep arms have been evaluated costs one CPU pass over
saved text instead of eight GPU passes. Anything the generating run computed but did not
print, or saved somewhere the artifact store then hid, comes back. And because it is the
same file, the same item loaders and the same graders that produced the original numbers,
a re-scored number is comparable to the one it replaces; a second copy of the scorers in a
second task is how one set of completions ends up with two different scores.

What it does not do is guess. A replayed component reads the completion for each item **by
id**, and re-renders the prompt it would have generated to check the recorded `prompt_sha`.
A missing id or a changed prompt hash is an assertion failure, not a silent substitution:
if the published chat template has moved since the run, the completions no longer answer
the prompts this harness builds and the re-score is not comparable. Better to see that on
the assertion line than to publish it.
"""

import hashlib
import json
import os


def load_tokenizer(base_model, log=print):
    """Tokenizer only. A replay loads no weights and needs no GPU."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    log("loaded tokenizer for %s (no weights: this is a replay)" % base_model)
    return tok


def _read(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _filename(tag):
    """`bfcl/native_tools` -> completions_bfcl_native_tools.jsonl, `ifeval` -> completions_ifeval.jsonl."""
    return "completions_%s.jsonl" % tag.replace("/", "_")


class ReplayRunner:
    """Stands in for gen.Runner and gen_gguf.GgufRunner with the same generate() shape."""

    replay = True

    def __init__(self, source_dir, log=print):
        self.dir = source_dir
        self.log = log
        self._cache = {}
        self.replayed = 0
        self.missing = 0
        self.sha_mismatch = 0
        self.files_used = []
        have = sorted(f for f in os.listdir(source_dir) if f.endswith(".jsonl"))
        log("replay source %s: %s" % (source_dir, ", ".join(have) or "(empty)"))

    def _rows(self, tag):
        if tag not in self._cache:
            path = os.path.join(self.dir, _filename(tag))
            if not os.path.exists(path):
                raise RuntimeError("replay source has no %s: this run cannot re-score %r"
                                   % (_filename(tag), tag))
            rows = _read(path)
            by_id = {}
            for r in rows:
                by_id.setdefault(r["id"], r)
            self._cache[tag] = by_id
            self.files_used.append({"file": _filename(tag), "rows": len(rows),
                                    "unique_ids": len(by_id)})
            self.log("  replay %-22s %5d rows (%d unique ids)"
                     % (tag, len(rows), len(by_id)))
        return self._cache[tag]

    def generate(self, prompts, max_new_tokens=384, batch_size=16, tag="", ids=None):
        if ids is None:
            raise RuntimeError("a replay needs item ids to join on; the caller passed none")
        by_id = self._rows(tag)
        out = []
        # Counted per call as well as cumulatively, or the progress line compares this
        # component's total against the last component's and reads as nonsense.
        hit = miss = bad = 0
        for item_id, prompt in zip(ids, prompts):
            row = by_id.get(item_id)
            if row is None:
                miss += 1
                out.append("")
                continue
            want = row.get("prompt_sha")
            got = hashlib.sha256(prompt.encode()).hexdigest()[:12]
            if want and want != got:
                bad += 1
            out.append(row.get("completion", ""))
            hit += 1
        self.replayed += hit
        self.missing += miss
        self.sha_mismatch += bad
        self.log("  %s replayed %d/%d (%d missing, %d prompt-hash mismatches)"
                 % (tag, hit, len(prompts), miss, bad))
        return out

    def verify_control(self, control):
        """The clean probe arm is rebuilt deterministically; check it against the source."""
        path = os.path.join(self.dir, "probes_control.jsonl")
        if not os.path.exists(path):
            return False, "the replay source saved no probes_control.jsonl"
        src = _read(path)
        mine = {r["id"]: json.dumps(r.get("messages"), sort_keys=True) for r in control}
        theirs = {r["id"]: json.dumps(r.get("messages"), sort_keys=True) for r in src}
        if mine == theirs:
            return True, "%d control items identical to the source run" % len(mine)
        only_mine = sorted(set(mine) - set(theirs))[:3]
        only_theirs = sorted(set(theirs) - set(mine))[:3]
        changed = sorted(k for k in set(mine) & set(theirs) if mine[k] != theirs[k])[:3]
        return False, ("control arm differs: %d here vs %d there, new=%s gone=%s changed=%s"
                       % (len(mine), len(theirs), only_mine, only_theirs, changed))

    def throughput(self):
        # No tokens were generated, and no throughput is invented. The replay counts are
        # the honest measure of what this pass did.
        return {"generated_tokens": 0, "generate_seconds": 0.0, "tokens_per_second": 0.0,
                "hit_max_new_tokens": 0, "replayed_completions": self.replayed,
                "replay_missing_ids": self.missing,
                "replay_prompt_hash_mismatches": self.sha_mismatch,
                "replay_files": self.files_used}
