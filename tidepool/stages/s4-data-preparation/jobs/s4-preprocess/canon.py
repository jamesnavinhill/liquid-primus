"""Canonicalization primitives for s4.4.

Three jobs, each small enough to test on a fixture and each with a failure mode
that matters more than its success case:

  parse_pyargs   turn a pythonic argument string into typed arguments, or refuse
  norm_type      map a corpus's type vocabulary onto one JSON Schema vocabulary
  ident          turn a display name like "Market Trends API" into an identifier

Every one of them returns a value *and* a verdict. Nothing here guesses: a call
whose arguments do not parse is reported as unparsed and dropped upstream rather
than being rendered with the arguments the parser happened to recover.
"""

import ast
import re

_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {v: k for k, v in _OPEN.items()}


def split_top_level(s, sep=","):
    """Split on `sep` at bracket depth zero, respecting quotes and escapes."""
    out, buf, depth, quote, esc = [], [], [], None, False
    for ch in s:
        if esc:
            buf.append(ch)
            esc = False
            continue
        if quote:
            buf.append(ch)
            if ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            continue
        if ch in _OPEN:
            depth.append(_OPEN[ch])
            buf.append(ch)
            continue
        if ch in _CLOSE:
            if depth and depth[-1] == ch:
                depth.pop()
            buf.append(ch)
            continue
        if ch == sep and not depth:
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if quote or depth:
        return None                     # unbalanced: refuse rather than guess
    out.append("".join(buf))
    return out


_KEY = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)\s*(.*)$", re.S)


def parse_pyargs(argstr):
    """(args_dict, verdict). verdict is "ok", "empty", "positional" or "unparsed"."""
    if argstr is None:
        return {}, "empty"
    body = argstr.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    if not body.strip():
        return {}, "empty"
    parts = split_top_level(body)
    if parts is None:
        return {}, "unparsed"
    args = {}
    for part in parts:
        if not part.strip():
            continue
        m = _KEY.match(part)
        if not m:
            # A positional argument cannot be named without the schema, and
            # naming it by position would invent a contract the row never stated.
            return {}, "positional"
        key, raw = m.group(1), m.group(2).strip()
        try:
            args[key] = ast.literal_eval(raw)
        except Exception:
            # An unquoted bareword is the common case here and is a string; a
            # value that still will not parse keeps its surface form rather than
            # failing the whole call.
            args[key] = raw.strip("'\"")
    return args, "ok"


# Every type name seen across the eleven corpora, mapped onto the JSON Schema
# vocabulary. Deliberately not a default-to-string map: an unseen type is
# returned unchanged and counted, so it shows up as a survivor.
_TYPES = {
    "dict": "object", "dictionary": "object", "object": "object", "json": "object",
    "map": "object", "struct": "object",
    "list": "array", "array": "array", "tuple": "array", "set": "array",
    "sequence": "array", "vector": "array",
    "str": "string", "string": "string", "text": "string", "char": "string",
    "int": "integer", "integer": "integer", "long": "integer", "int32": "integer",
    "int64": "integer",
    "float": "number", "double": "number", "number": "number", "decimal": "number",
    "bool": "boolean", "boolean": "boolean",
    "none": "null", "null": "null", "nonetype": "null",
    "any": "any",
}


def norm_type(t):
    """(normalized, was_known). Case and whitespace insensitive; keeps unknowns."""
    if not isinstance(t, str):
        return t, False
    key = t.strip().lower()
    # `List[str]`, `dict[str, int]`, `Optional[int]` -> the container name
    outer = re.match(r"^([a-z_][a-z0-9_]*)\s*[\[\(<]", key)
    if outer:
        key = outer.group(1)
    if key in ("optional", "union"):
        return t.strip(), False
    if key in _TYPES:
        return _TYPES[key], True
    return t.strip(), False


_BAD = re.compile(r"[^a-z0-9_]+")


def ident(name, taken=None):
    """(identifier, changed). Stable, lowercase, collision-free within `taken`."""
    if name is None:
        return None, False
    base = _BAD.sub("_", str(name).strip().lower()).strip("_")
    if not base:
        base = "fn"
    if base[0].isdigit():
        base = "fn_" + base
    out, n = base, 1
    while taken is not None and out in taken:
        n += 1
        out = "%s_%d" % (base, n)
    if taken is not None:
        taken.add(out)
    return out, out != str(name)


_IS_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


def ident_map(names):
    """Map a toolset's declared names onto identifiers, collision-free.

    Order matters and is fixed rather than incidental: a name that is already a
    valid lowercase identifier claims itself first, so a display name like
    "Get Cars Information" can never take the slot belonging to a real
    `get_cars_information` declared beside it. Everything else is processed in
    sorted order so the same toolset always produces the same map.
    """
    names = [n for n in names if n is not None]
    already = sorted(n for n in names if _IS_IDENT.match(str(n)))
    rest = sorted(n for n in names if not _IS_IDENT.match(str(n)))
    taken, out = set(), {}
    for n in already + rest:
        if n in out:
            continue
        out[n], _ = ident(n, taken)
    return out


def _walk_schema(node, unknown):
    """Normalize `type` names in place, at every depth, preserving everything else.

    `norm_tool` in the s4.2/s4.3 adapters reduces a schema to a summary: parameter
    names, a set of type strings, a depth. That is the right shape for counting and
    the wrong shape for rendering a contract back out, so the pipeline canonicalizes
    from the raw schema instead and keeps descriptions, enums, defaults and nesting.
    """
    if isinstance(node, list):
        return [_walk_schema(v, unknown) for v in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "type" and isinstance(v, str):
            nt, known = norm_type(v)
            if not known:
                unknown[v.strip()] = unknown.get(v.strip(), 0) + 1
            out[k] = nt
        elif k == "type" and isinstance(v, list):
            vals = []
            for one in v:
                nt, known = norm_type(one)
                if not known and isinstance(one, str):
                    unknown[one.strip()] = unknown.get(one.strip(), 0) + 1
                vals.append(nt)
            out[k] = vals
        else:
            out[k] = _walk_schema(v, unknown)
    return out


def canon_tool(raw, name_map, unknown):
    """One raw tool declaration -> one canonical function object, or None.

    Absorbs the three wrappers the mix uses (OpenAI's {"function": {...}}, a bare
    {"name", "parameters"}, and ToolACE's "arguments" spelling) and emits one:
    {"name", "description", "parameters": {"type": "object", "properties", "required"}}.
    """
    if not isinstance(raw, dict):
        return None
    f = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    name = f.get("name")
    if not name:
        return None
    params = f.get("parameters") or f.get("arguments") or {}
    if not isinstance(params, dict):
        params = {}
    props = params.get("properties")
    if not isinstance(props, dict):
        # Some rows write the properties map directly under `parameters`, with no
        # `properties` key and no `type`. Treat it as the properties map when every
        # value looks like a schema, and as empty otherwise.
        if params and all(isinstance(v, dict) for v in params.values()):
            props = params
            params = {}
        else:
            props = {}
    required = params.get("required") or []
    if isinstance(required, str):
        required = [required]
    required = [str(r) for r in required if isinstance(r, (str, int))]
    return {
        "name": name_map.get(name, name),
        "description": f.get("description") or "",
        "parameters": {
            "type": "object",
            "properties": _walk_schema(props, unknown),
            "required": [r for r in required if r in props],
        },
    }


def render_call(name, args):
    """The one call shape the pipeline trains, whatever shape the contract arrived in."""
    import json
    return "<tool_call>%s</tool_call>" % json.dumps(
        {"name": name, "arguments": args}, ensure_ascii=False, sort_keys=True)
