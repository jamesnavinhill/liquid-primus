"""Per-corpus adapters: turn a heterogeneous row into one common record.

Every corpus in the tidepool mix stores the same three things in a different
place: the user's prompt, the tool schemas the model may call, and the calls the
label expects. An adapter returns them in one shape so the diagnostics below are
written once instead of seven times.

Record fields
  prompt      str            the user-visible request, used for decontamination
  tools       list[dict]     normalized tool schemas ({name, params, required, types})
  calls       list[dict]     expected calls ({name, args})
  refusal     str | None     the exact refusal string when the label is not a call
  target      str | None     the function the row was generated for, when declared
  group_key   str            what s4.3 splits on
  row_id      str            stable identity for a hit list
  schema_text str            the text that carries the tool schemas (token accounting)
"""

import collections
import hashlib
import json
import re


def _sha(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


def _depth(obj, d=0):
    if isinstance(obj, dict):
        return max([_depth(v, d + 1) for v in obj.values()] or [d])
    if isinstance(obj, list):
        return max([_depth(v, d + 1) for v in obj] or [d])
    return d


def norm_tool(t):
    """One tool schema -> {name, params, required, types, depth}.

    Absorbs the three spellings in the mix: OpenAI's {"function": {...}} wrapper,
    a bare {"name", "parameters"} object, and ToolACE's `"type": "dict"` where the
    others write `"type": "object"`.
    """
    if not isinstance(t, dict):
        return None
    f = t.get("function") if isinstance(t.get("function"), dict) else t
    name = f.get("name")
    if not name:
        return None
    params = f.get("parameters") or f.get("arguments") or {}
    if not isinstance(params, dict):
        params = {}
    props = params.get("properties")
    if not isinstance(props, dict):
        props = {}
    required = params.get("required") or []
    if isinstance(required, str):
        required = [required]
    return {
        "name": name,
        "params": sorted(props.keys()),
        "required": sorted(str(r) for r in required if isinstance(r, (str, int))),
        "types": sorted({str((v or {}).get("type")) for v in props.values() if isinstance(v, dict)}),
        "container_type": str(params.get("type")),
        "depth": _depth(params),
        "n_params": len(props),
    }


def _loads(x, default):
    if x is None:
        return default
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except Exception:
            return default
    return default


def _tools_from(raw):
    parsed = _loads(raw, [])
    if isinstance(parsed, dict):
        parsed = [parsed]
    out = [norm_tool(t) for t in parsed if isinstance(t, dict)]
    return [t for t in out if t]


def _group(tools):
    """Group key = the API surface, so a signature cannot straddle a split."""
    if not tools:
        return "no-tools"
    return _sha(*sorted("%s(%s)" % (t["name"], ",".join(t["params"])) for t in tools))


# ---------------------------------------------------------------- Synth-APIGen

APIGEN_REFUSALS = {
    "no_tools": "The query cannot be answered, no tools were provided.",
    "missing_params": "The given question lacks the parameters required by the function.",
}


def apigen(row):
    tools = _tools_from(row.get("tools"))
    ans = row.get("answers")
    calls, refusal = [], None
    parsed = _loads(ans, None)
    if isinstance(parsed, list):
        for c in parsed:
            if isinstance(c, dict):
                calls.append({"name": c.get("name"), "args": c.get("arguments") or c.get("args") or {}})
    else:
        refusal = (ans or "").strip()
    return {
        "prompt": row.get("query") or "",
        "tools": tools,
        "calls": calls,
        "refusal": refusal,
        "target": row.get("func_name"),
        "group_key": _group(tools),
        "row_id": row.get("hash_id") or _sha(row.get("query")),
        "schema_text": row.get("tools") or "",
        "extra": {"model_name": row.get("model_name"), "func_desc": row.get("func_desc")},
    }


# --------------------------------------------------------------------- ToolACE

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def split_pycalls(text):
    """Split ToolACE's pythonic call list into (name, arg_string) pairs.

    A regex cannot do this. ToolACE tool names contain spaces and hyphens
    ("Market Trends API", "SEC Filings"), so matching the identifier before `(`
    silently truncates the name to its last word, and argument values contain
    commas, brackets and parentheses of their own. Walk the string instead,
    tracking quote state and paren depth: at depth zero a call name runs from the
    opening `[` or a separating comma up to the `(` that opens its arguments.
    """
    s = (text or "").strip()
    if not s.startswith("["):
        return []
    calls, i, n = [], 1, len(s)
    while i < n:
        while i < n and s[i] in " \t\n,":
            i += 1
        if i >= n or s[i] == "]":
            break
        start = i
        depth, quote = 0, None
        name = None
        while i < n:
            ch = s[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch == "(":
                if depth == 0:
                    name = s[start:i].strip()
                    arg_start = i + 1
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    calls.append((name or "", s[arg_start:i]))
                    i += 1
                    break
            elif ch == "]" and depth == 0:
                break
            i += 1
        else:
            break
        if name is None:
            break
    return calls


def count_args(arg_string):
    """Top-level `k=v` pairs in one call's argument string."""
    depth, quote, n = 0, None, 0
    has_any = bool((arg_string or "").strip())
    for i, ch in enumerate(arg_string or ""):
        if quote:
            if ch == "\\":
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            n += 1
    return (n + 1) if has_any else 0


def _toolace_json(system):
    """ToolACE hides its schemas inside the system string.

    The array is followed by prose ("Should you decide to return the function
    call(s)..."), so a regex to the last `]` overshoots and a regex anchored at
    the end never matches. Decode from the opening bracket and let the JSON
    parser report where the array stops.
    """
    if not system:
        return [], ""
    i = system.find("[{")
    if i < 0:
        return [], ""
    try:
        parsed, end = json.JSONDecoder().raw_decode(system[i:])
    except ValueError:
        return [], ""
    blob = system[i:i + end]
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return [], blob
    out = [norm_tool(t) for t in parsed if isinstance(t, dict)]
    return [t for t in out if t], blob


_YAML_NAME = re.compile(r"^\s*tool_name:\s*(.+?)\s*$", re.M)
_XML_NAME = re.compile(r"<tool_name>\s*(.*?)\s*</tool_name>", re.S)
_MD_NAME = re.compile(r"\*\*tool_name\*\*\s*:\s*(.+?)\s*$", re.M)
_TEX_HEADER = re.compile(r"tool_name\s*&")
_TEX_ROW = re.compile(r"^\s*([^&\n\\]{1,80}?)\s*&", re.M)


def toolace_schema(system):
    """ToolACE serializes tool schemas three different ways.

    Roughly one row in six does not use JSON at all: some system prompts declare
    tools in a YAML-ish `tool_name:/definition:/arguments:` block, others in an
    XML-ish `<tool_name>...</tool_name>` block. The dataset varies the call format
    to match, which is deliberate format diversity on its part and a
    normalization requirement on ours. Return (format, tools, names, blob).
    """
    if not system:
        return "none", [], [], ""
    tools, blob = _toolace_json(system)
    if tools:
        return "json", tools, [t["name"] for t in tools], blob
    names = [n.strip() for n in _XML_NAME.findall(system) if n.strip()]
    if names:
        return "xml", [], names, system
    names = [n.strip() for n in _YAML_NAME.findall(system) if n.strip()]
    if names:
        return "yaml_block", [], names, system
    names = [n.strip() for n in _MD_NAME.findall(system) if n.strip()]
    if names:
        return "markdown", [], names, system
    if _TEX_HEADER.search(system):
        names = [n.strip() for n in _TEX_ROW.findall(system)
                 if n.strip() and n.strip() != "tool_name" and not n.strip().startswith("\\")]
        if names:
            return "latex_tabular", [], names, system
    return "unknown", [], [], ""


def _call_format(s):
    if not s.startswith("["):
        return "natural_language"
    if re.match(r"^\[[^\[\]]{1,80}\]\s*[-|~:]", s):
        return "custom_delimited"
    if "(" in s and ")" in s:
        return "pythonic_bracket"
    return "bracket_other"


def toolace(row):
    fmt, tools, names, blob = toolace_schema(row.get("system"))
    conv = _loads(row.get("conversations"), []) or []
    prompt, calls, refusal = "", [], None
    bracketed = nl_turns = lead_space = trail_space = non_ident = 0
    call_formats = collections.Counter()
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        frm = (turn.get("from") or turn.get("role") or "").lower()
        val = turn.get("value") or turn.get("content") or ""
        if frm in ("user", "human") and not prompt:
            prompt = val
        if frm in ("assistant", "gpt"):
            s = val.strip()
            cf = _call_format(s)
            call_formats[cf] += 1
            if cf == "natural_language":
                nl_turns += 1
                if refusal is None:
                    refusal = s[:200]
                continue
            bracketed += 1
            if re.match(r"^\[\s+", s):
                lead_space += 1
            if re.search(r"\s+\]$", s):
                trail_space += 1
            if cf == "pythonic_bracket":
                for name, argstr in split_pycalls(s):
                    if not _IDENT.match(name):
                        non_ident += 1
                    calls.append({"name": name, "args": {}, "n_args": count_args(argstr),
                                  "name_is_identifier": bool(_IDENT.match(name))})
            else:
                # A custom-delimited turn still names its tools; count the
                # declared names it mentions rather than guessing a grammar.
                for nm in names:
                    if nm and nm in s:
                        calls.append({"name": nm, "args": {}, "n_args": 0,
                                      "name_is_identifier": bool(_IDENT.match(nm))})
    return {
        "prompt": prompt,
        "tools": tools,
        "declared_names": names,
        "calls": calls,
        "refusal": refusal if bracketed == 0 else None,
        "target": None,
        "group_key": _group(tools) if tools else (_sha(*sorted(names)) if names else "no-tools"),
        "row_id": _sha(row.get("system"), prompt),
        "schema_text": blob,
        "extra": {"bracketed": bracketed, "nl_turns": nl_turns,
                  "lead_space": lead_space, "trail_space": trail_space,
                  "non_identifier_call_names": non_ident,
                  "n_turns": len(conv)},
        "formats": {"schema_format": fmt, "call_formats": dict(call_formats)},
    }


# ---------------------------------------------------------------------- Hermes


def _hermes_calls(val):
    """Hermes wraps calls in <tool_call> ... </tool_call>."""
    out = []
    for blob in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", val or "", re.S):
        d = _loads(blob, None)
        if isinstance(d, dict):
            out.append({"name": d.get("name"), "args": d.get("arguments") or {}})
    return out


def hermes(row):
    tools = _tools_from(row.get("tools"))
    conv = _loads(row.get("conversations"), []) or []
    prompt, calls = "", []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        frm = (turn.get("from") or turn.get("role") or "").lower()
        val = turn.get("value") or turn.get("content") or ""
        if frm in ("human", "user") and not prompt:
            prompt = val
        if frm in ("gpt", "assistant", "model"):
            calls += _hermes_calls(val)
    schema = row.get("schema")
    return {
        "prompt": prompt,
        "tools": tools,
        "calls": calls,
        "refusal": None,
        "target": None,
        "group_key": _group(tools) if tools else ("json-schema:" + _sha(schema) if schema else "no-tools"),
        "row_id": str(row.get("id") or _sha(prompt)),
        "schema_text": row.get("tools") or schema or "",
        "extra": {"category": row.get("category"), "subcategory": row.get("subcategory"),
                  "has_json_schema": bool(schema), "n_turns": len(conv)},
    }


# ------------------------------------------------------------------ SQL / code


def _norm_sql_schema(context):
    """Group SQL rows by the table surface they query, not by the question."""
    tables = re.findall(r"CREATE TABLE\s+([A-Za-z0-9_\"`.]+)", context or "", re.I)
    cols = re.findall(r"\(([^()]*)\)", context or "")
    return _sha(",".join(sorted(t.strip('"`') for t in tables)), ",".join(sorted(c.strip() for c in cols)))


def sql_create_context(row):
    return {
        "prompt": row.get("question") or "",
        "tools": [], "calls": [], "refusal": None, "target": None,
        "group_key": _norm_sql_schema(row.get("context")),
        "row_id": _sha(row.get("question"), row.get("context")),
        "schema_text": row.get("context") or "",
        "extra": {"answer_len": len(row.get("answer") or "")},
    }


def clinton_sql(row):
    ctx = row.get("input") or ""
    return {
        "prompt": row.get("instruction") or "",
        "tools": [], "calls": [], "refusal": None, "target": None,
        "group_key": _norm_sql_schema(ctx),
        "row_id": _sha(row.get("instruction"), ctx),
        "schema_text": ctx,
        "extra": {"source": row.get("source")},
    }


def codefeedback(row):
    return {
        "prompt": row.get("query") or "",
        "tools": [], "calls": [], "refusal": None, "target": None,
        "group_key": "resource:" + str(row.get("resource")),
        "row_id": _sha(row.get("query")),
        "schema_text": "",
        "extra": {"lang": row.get("lang"), "resource": row.get("resource")},
    }


def prompt_only(row):
    text = row.get("prompt") or row.get("text") or row.get("instruction") or row.get("query") or ""
    return {
        "prompt": text, "tools": [], "calls": [], "refusal": None, "target": None,
        "group_key": "prompt-only", "row_id": _sha(text), "schema_text": "", "extra": {},
    }
