"""One frozen prompting path, and the two tool-call surface forms it has to read.

The protocol in `plan.md` fixes one serving backend and one template for every
full-precision comparison. That is what this module is: the model's own chat template,
greedy decoding, `add_generation_prompt=True`, and nothing else in the context that the
component's own data did not put there.

Two surface forms exist because the project trains on one and the vendor shipped another:

  `tools_text`    the s4 training convention — the tool contract is a JSON array inside
                  <tools></tools> in the system message, and a call comes back as
                  <tool_call>{"name": ..., "arguments": {...}}</tool_call>.
  `native_tools`  LFM2.5's own — the contract goes through the template's `tools=`
                  argument, and a call comes back as
                  <|tool_call_start|>[f(a=1, b='x')]<|tool_call_end|>, i.e. Python call
                  syntax rather than JSON.

A finetune of ours is fluent in the first and a stock checkpoint in the second, so
scoring a baseline in our convention alone would understate it and inflate every delta
measured against it. Both forms are therefore generated for every tool-calling
component, and the parser accepts either regardless of which form was prompted: a model
that answers in the other convention gets the credit for it.
"""

import ast
import json
import re

# The s4 training convention, copied verbatim from s4-preprocess/render.py. Copied and not
# imported because a drift between the two is the kind of thing that silently changes a
# number, so it is asserted against the rendered training rows instead of shared.
SYS_TOOLS = (
    "You are a function-calling assistant. You are given a set of tools inside "
    "<tools></tools>. When a tool applies, reply with one <tool_call></tool_call> "
    "block per call, containing a JSON object with \"name\" and \"arguments\". When no "
    "tool applies, or a required argument is missing, say so in plain text and do not "
    "guess a call.\n<tools>\n%s\n</tools>"
)

FALLBACK_ROLE_TAGS = {"system": "system", "user": "user",
                      "assistant": "assistant", "tool": "tool"}


def fmt_tools(tools):
    """Exactly the s4 rendering: indent=1, sorted keys, no ASCII escaping."""
    return json.dumps(tools, ensure_ascii=False, indent=1, sort_keys=True)


class Prompter:
    """Renders a conversation to text the way training did, and says which way that was.

    `mode` is picked once, on real rows, and recorded: a number produced under the
    fallback template is not comparable to one produced under the model's own.
    """

    def __init__(self, tok, log=print):
        self.tok = tok
        self.log = log
        self.mode = None
        self.note = ""
        self.supports_tools_arg = None

    # ---------------------------------------------------------------- mode selection
    def _native(self, messages, tools=None):
        kw = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kw["tools"] = tools
        return self.tok.apply_chat_template(messages, **kw)

    def _fallback(self, messages, tools=None):
        msgs = list(messages)
        if tools:
            blob = SYS_TOOLS % fmt_tools(tools)
            if msgs and msgs[0]["role"] == "system":
                msgs[0] = {"role": "system", "content": blob + "\n" + msgs[0]["content"]}
            else:
                msgs = [{"role": "system", "content": blob}] + msgs
        out = ["<|im_start|>%s\n%s<|im_end|>\n" % (FALLBACK_ROLE_TAGS.get(m["role"], "user"),
                                                  m["content"]) for m in msgs]
        return "".join(out) + "<|im_start|>assistant\n"

    def pick_mode(self, samples):
        """samples: a handful of real message lists, at least one carrying a tool turn."""
        try:
            self.mode = "native"
            for msgs in samples:
                txt = self._native(msgs)
                if not txt or msgs[-1]["content"][:24] not in txt:
                    raise ValueError("the model's template dropped the final turn")
            self.note = ("used the model's own chat template; it accepted %d probe rows "
                         "including tool turns" % len(samples))
        except Exception as exc:                                  # noqa: BLE001
            self.mode = "fallback"
            self.note = ("the model's own chat template was rejected (%s), so the explicit "
                         "fallback is in use and these numbers are not comparable to a "
                         "native-template run" % exc)
        # Separately: does the template take a tools= argument at all? Without it the
        # native_tools style cannot be rendered and the run says so rather than silently
        # falling back to the text convention and reporting it as native.
        try:
            probe = [{"role": "user", "content": "hello"}]
            t = [{"type": "function",
                  "function": {"name": "f", "description": "d",
                               "parameters": {"type": "object", "properties": {}}}}]
            txt = self._native(probe, tools=t)
            self.supports_tools_arg = bool(txt) and "f" in txt
        except Exception:                                          # noqa: BLE001
            self.supports_tools_arg = False
        self.log("template mode: %s (%s); tools= argument supported: %s"
                 % (self.mode, self.note, self.supports_tools_arg))
        return self.mode

    # ---------------------------------------------------------------- rendering
    def render(self, messages, tools=None, style="tools_text"):
        """One prompt string. `tools` is a list of OpenAI-shaped function specs."""
        render = self._native if self.mode == "native" else self._fallback
        if not tools:
            return render(messages)
        if style == "native_tools":
            if not self.supports_tools_arg:
                raise RuntimeError("native_tools requested but the template has no tools=")
            return render(messages, tools=tools)
        # tools_text: the contract goes in the system message, as training rendered it.
        blob = SYS_TOOLS % fmt_tools(tools)
        msgs = list(messages)
        if msgs and msgs[0]["role"] == "system":
            msgs = [{"role": "system", "content": blob + "\n" + msgs[0]["content"]}] + msgs[1:]
        else:
            msgs = [{"role": "system", "content": blob}] + msgs
        return render(msgs)


# -------------------------------------------------------------------- call parsing

_JSON_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_NATIVE_CALL = re.compile(r"<\|tool_call_start\|>\s*(.*?)\s*<\|tool_call_end\|>", re.S)
# A model that has been shown the JSON convention sometimes emits it inside a fence.
_FENCED = re.compile(r"```(?:json|tool_call)?\s*(\{.*?\})\s*```", re.S)


def _as_call(obj):
    if not isinstance(obj, dict):
        return None
    if "function" in obj and isinstance(obj["function"], dict):
        obj = obj["function"]
    name = obj.get("name")
    args = obj.get("arguments", obj.get("args", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:                                          # noqa: BLE001
            return None
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return {"name": name, "args": args}


def _parse_python_calls(text):
    """`[f(a=1, b='x'), g()]` -> calls. Values are Python literals per the template."""
    out = []
    src = text.strip()
    if not src:
        return out
    if not src.startswith("["):
        src = "[" + src + "]"
    try:
        node = ast.parse(src, mode="eval").body
    except SyntaxError:
        return out
    items = node.elts if isinstance(node, ast.List) else [node]
    for it in items:
        if not isinstance(it, ast.Call):
            continue
        # Dotted names are legal function names in BFCL (math.factorial).
        parts, f = [], it.func
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if isinstance(f, ast.Name):
            parts.append(f.id)
        else:
            continue
        name = ".".join(reversed(parts))
        args = {}
        ok = True
        for kw in it.keywords:
            if kw.arg is None:
                ok = False
                break
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except Exception:                                      # noqa: BLE001
                ok = False
                break
        if ok:
            out.append({"name": name, "args": args})
    return out


def parse_calls(completion):
    """Every tool call in a completion, in either surface form. [] means none."""
    text = completion or ""
    calls = []
    for blob in _JSON_CALL.findall(text):
        c = _as_call(_loads(blob))
        if c:
            calls.append(c)
    for blob in _NATIVE_CALL.findall(text):
        calls.extend(_parse_python_calls(blob))
    if not calls:
        for blob in _FENCED.findall(text):
            c = _as_call(_loads(blob))
            if c:
                calls.append(c)
    if not calls:
        # A bare JSON object with name+arguments and nothing else around it.
        s = text.strip()
        if s.startswith("{") and s.endswith("}"):
            c = _as_call(_loads(s))
            if c:
                calls.append(c)
        elif s.startswith("[") and "(" in s and s.endswith("]"):
            calls.extend(_parse_python_calls(s))
    return calls


def _loads(blob):
    try:
        return json.loads(blob)
    except Exception:                                              # noqa: BLE001
        try:
            return ast.literal_eval(blob)
        except Exception:                                          # noqa: BLE001
            return None
