"""Turn a rendered conversation into input ids and a loss mask.

Two things here are easy to get silently wrong, and both would cost a whole sweep before
anyone noticed, so both are checked rather than assumed.

**The template.** The rows carry four roles, `system`, `user`, `assistant` and `tool`, and
not every published chat template accepts `tool`. The job tries the model's own template
first, because training on a format the model was not post-trained in throws away most of
what the checkpoint already knows. If the template rejects a tool turn, an explicit
fallback is used and the run records which of the two it took: a number produced under the
fallback is not comparable to one produced under the model's own template, so the mode is
part of the result and not an implementation detail.

**The mask.** Loss belongs on assistant turns only. A run that trains on the user's turns
learns to write questions, scores worse for a reason that looks like a hyperparameter
problem, and is a well-known way to waste a sweep. The mask is built by rendering each
prefix and taking the token delta, which is exact for any additive template, and the
additivity itself is asserted before training starts.
"""

FALLBACK_ROLE_TAGS = {"system": "system", "user": "user",
                      "assistant": "assistant", "tool": "tool"}


def fallback_render(messages, upto):
    """An explicit, additive ChatML-shaped template. Used only if the model's own rejects
    a tool turn, and recorded in the run when it is."""
    out = []
    for m in messages[:upto]:
        tag = FALLBACK_ROLE_TAGS.get(m["role"], "user")
        out.append("<|im_start|>%s\n%s<|im_end|>\n" % (tag, m["content"]))
    return "".join(out)


class Encoder:
    def __init__(self, tok, max_len, log=print):
        self.tok = tok
        self.max_len = max_len
        self.log = log
        self.mode = None
        self.notes = []

    def _render(self, messages, upto):
        if self.mode == "fallback":
            return fallback_render(messages, upto)
        return self.tok.apply_chat_template(messages[:upto], tokenize=False,
                                            add_generation_prompt=False)

    def pick_mode(self, samples):
        """Decide once, on real rows, and say why."""
        has_tool = [s for s in samples if any(m["role"] == "tool" for m in s)]
        probe = (has_tool or samples)[:8]
        try:
            self.mode = "native"
            for msgs in probe:
                txt = self._render(msgs, len(msgs))
                if not txt or msgs[-1]["content"][:24] not in txt:
                    raise ValueError("the model's template dropped the final turn's content")
            # Additivity: a longer prefix must extend the shorter one, or the delta trick
            # that builds the mask is wrong.
            for msgs in probe:
                if len(msgs) < 2:
                    continue
                short = self.tok(self._render(msgs, len(msgs) - 1),
                                 add_special_tokens=False)["input_ids"]
                long = self.tok(self._render(msgs, len(msgs)),
                                add_special_tokens=False)["input_ids"]
                if long[:len(short)] != short:
                    raise ValueError("the model's template is not additive across turns, so "
                                     "an assistant-only loss mask cannot be derived from it")
            self.notes.append("used the model's own chat template; %d of %d probe rows carried "
                              "a tool turn and it accepted them" % (len(has_tool), len(samples)))
        except Exception as exc:
            self.mode = "fallback"
            self.notes.append("the model's own chat template was rejected (%s), so the explicit "
                              "fallback template is in use and these numbers are not comparable "
                              "to a native-template run" % exc)
        self.log("template mode: %s — %s" % (self.mode, self.notes[-1]))
        return self.mode

    def encode(self, messages):
        """Returns (input_ids, labels). labels are -100 everywhere but assistant content."""
        ids, labels, prev = [], [], 0
        for i, m in enumerate(messages):
            full = self.tok(self._render(messages, i + 1), add_special_tokens=False)["input_ids"]
            delta = full[prev:]
            prev = len(full)
            ids.extend(delta)
            labels.extend(delta if m["role"] == "assistant" else [-100] * len(delta))
        if self.tok.eos_token_id is not None and (not ids or ids[-1] != self.tok.eos_token_id):
            ids.append(self.tok.eos_token_id)
            labels.append(self.tok.eos_token_id if messages[-1]["role"] == "assistant" else -100)
        # Truncate from the left: the assistant turn a row exists to teach is at the end, and
        # a right truncation would drop exactly the supervised tokens.
        if len(ids) > self.max_len:
            ids, labels = ids[-self.max_len:], labels[-self.max_len:]
        return ids, labels
