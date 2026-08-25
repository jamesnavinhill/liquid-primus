"""Raw corpus row -> the one conversation shape s5 trains on.

Why this does not reuse the s4.3 adapters wholesale: `adapters.norm_tool` reduces a
tool schema to a summary (parameter names, a set of type strings, a depth), which is
exactly right for grouping and counting and cannot be rendered back into a contract.
The same is true of ToolACE calls, whose arguments the adapter counts but discards.
So the adapters are still imported, for `row_id` and the prompt, and everything that
becomes model-visible text is re-derived here from the raw row.

Target format, identical across all five roles so the model sees one convention:

  system     the tool contract as a JSON array inside <tools></tools>, when there is one
  user       the request, verbatim
  assistant  <tool_call>{"arguments": {...}, "name": "..."}</tool_call>, one per call,
             or plain text when the turn is a refusal or an answer
  tool       <tool_response>...</tool_response>
"""

import json
import re

import adapters
import canon

SYS_TOOLS = (
    "You are a function-calling assistant. You are given a set of tools inside "
    "<tools></tools>. When a tool applies, reply with one <tool_call></tool_call> "
    "block per call, containing a JSON object with \"name\" and \"arguments\". When no "
    "tool applies, or a required argument is missing, say so in plain text and do not "
    "guess a call.\n<tools>\n%s\n</tools>"
)
SYS_JSON = ("You are a structured-output assistant. Reply with a single JSON object "
            "conforming to the schema inside <schema></schema>, and nothing else."
            "\n<schema>\n%s\n</schema>")
SYS_SQL = ("You are a SQL assistant. Given the schema inside <schema></schema>, reply "
           "with a single SQL statement and nothing else.\n<schema>\n%s\n</schema>")


class Unrenderable(Exception):
    """Raised with a short reason string; the caller counts reasons and drops the row."""


def _tool_block(raw_tools, unknown):
    """Canonical <tools> payload plus the display-name -> identifier map used by calls."""
    out, names = [], []
    for raw in raw_tools:
        t = canon.canon_tool(raw, {}, unknown)
        if t:
            names.append(t["name"])
    name_map = canon.ident_map(names)
    for raw in raw_tools:
        t = canon.canon_tool(raw, name_map, unknown)
        if t:
            out.append(t)
    out.sort(key=lambda t: t["name"])
    return out, name_map


def _raw_tools(val):
    """The tools field arrives as a JSON string, a list, or a single object."""
    v = adapters._loads(val, val)
    if isinstance(v, dict):
        return [v]
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


def _turns(row):
    for turn in (adapters._loads(row.get("conversations"), []) or []):
        if isinstance(turn, dict):
            yield ((turn.get("from") or turn.get("role") or "").lower(),
                   turn.get("value") or turn.get("content") or "")


_ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
         "model": "assistant", "tool": "tool", "function": "tool", "observation": "tool",
         "function_response": "tool", "system": "system"}


def _fmt_tools(tools):
    return json.dumps(tools, ensure_ascii=False, indent=1, sort_keys=True)


# ------------------------------------------------------------------- role: tool


def r_apigen(row, unknown):
    tools, name_map = _tool_block(_raw_tools(row.get("tools")), unknown)
    if not tools:
        raise Unrenderable("no_tools_declared")
    user = (row.get("query") or "").strip()
    if not user:
        raise Unrenderable("empty_prompt")
    parsed = adapters._loads(row.get("answers"), None)
    if isinstance(parsed, list):
        blocks = []
        for c in parsed:
            if not isinstance(c, dict) or not c.get("name"):
                raise Unrenderable("call_without_name")
            args = c.get("arguments") or c.get("args") or {}
            if not isinstance(args, dict):
                raise Unrenderable("call_args_not_object")
            blocks.append(canon.render_call(name_map.get(c["name"], c["name"]), args))
        if not blocks:
            raise Unrenderable("empty_call_list")
        answer = "\n".join(blocks)
    else:
        answer = (row.get("answers") or "").strip()
        if not answer:
            raise Unrenderable("empty_refusal")
    return [{"role": "system", "content": SYS_TOOLS % _fmt_tools(tools)},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer}]


def r_toolace(row, unknown):
    """ToolACE's calls are pythonic and its argument values live only in the raw text.

    A row is kept only when every bracketed turn parses end to end. Half-parsing a
    call would put an argument list into the training target that the schema does not
    justify, and the corpus is large enough that dropping the residue costs little.
    """
    fmt, _, names, blob = adapters.toolace_schema(row.get("system"))
    raw_tools = _raw_tools(blob) if fmt == "json" else []
    tools, name_map = _tool_block(raw_tools, unknown)
    if not tools:
        raise Unrenderable("schema_not_json:" + str(fmt))
    msgs = [{"role": "system", "content": SYS_TOOLS % _fmt_tools(tools)}]
    saw_user = saw_call = False
    for frm, val in _turns(row):
        role = _ROLE.get(frm)
        s = (val or "").strip()
        if not role or not s:
            continue
        if role == "user":
            msgs.append({"role": "user", "content": s})
            saw_user = True
        elif role == "tool":
            msgs.append({"role": "tool", "content": "<tool_response>%s</tool_response>" % s})
        elif role == "assistant":
            if adapters._call_format(s) == "natural_language":
                msgs.append({"role": "assistant", "content": s})
                continue
            pairs = adapters.split_pycalls(s)
            if not pairs:
                raise Unrenderable("unsplittable_call_turn")
            blocks = []
            for name, argstr in pairs:
                args, verdict = canon.parse_pyargs(argstr)
                if verdict in ("positional", "unparsed"):
                    raise Unrenderable("pyargs_" + verdict)
                blocks.append(canon.render_call(name_map.get(name, canon.ident(name)[0]), args))
            msgs.append({"role": "assistant", "content": "\n".join(blocks)})
            saw_call = True
    if not saw_user or not saw_call:
        raise Unrenderable("no_user_turn" if not saw_user else "no_call_turn")
    if msgs[-1]["role"] != "assistant":
        raise Unrenderable("conversation_ends_off_assistant")
    return msgs


def r_hermes(row, unknown):
    raw = _raw_tools(row.get("tools"))
    schema = row.get("schema")
    if raw:
        tools, name_map = _tool_block(raw, unknown)
        if not tools:
            raise Unrenderable("tools_field_unparsed")
        system = SYS_TOOLS % _fmt_tools(tools)
    elif schema:
        name_map = {}
        system = SYS_JSON % json.dumps(adapters._loads(schema, schema), ensure_ascii=False,
                                       indent=1, sort_keys=True, default=str)
    else:
        raise Unrenderable("neither_tools_nor_schema")
    msgs = [{"role": "system", "content": system}]
    saw_user = saw_asst = False
    for frm, val in _turns(row):
        role = _ROLE.get(frm)
        s = (val or "").strip()
        if not role or not s or role == "system":
            continue
        if role == "assistant":
            calls = adapters._hermes_calls(s)
            if calls:
                blocks = []
                for c in calls:
                    if not c.get("name"):
                        raise Unrenderable("call_without_name")
                    args = c.get("args") or {}
                    if not isinstance(args, dict):
                        raise Unrenderable("call_args_not_object")
                    blocks.append(canon.render_call(name_map.get(c["name"], c["name"]), args))
                s = "\n".join(blocks)
            saw_asst = True
        elif role == "tool":
            s = s if s.startswith("<tool_response>") else "<tool_response>%s</tool_response>" % s
        else:
            saw_user = True
        msgs.append({"role": role, "content": s})
    if not saw_user or not saw_asst:
        raise Unrenderable("no_user_turn" if not saw_user else "no_assistant_turn")
    if msgs[-1]["role"] != "assistant":
        raise Unrenderable("conversation_ends_off_assistant")
    return msgs


# --------------------------------------------------------- roles: sql, code, prompts


def _sql(question, context, answer):
    q, a = (question or "").strip(), (answer or "").strip()
    if not q or not a:
        raise Unrenderable("empty_prompt" if not q else "empty_answer")
    ctx = (context or "").strip()
    msgs = [{"role": "system", "content": SYS_SQL % ctx}] if ctx else []
    return msgs + [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


def r_sql_ctx(row, unknown):
    return _sql(row.get("question"), row.get("context"), row.get("answer"))


def r_clinton_sql(row, unknown):
    # Clinton/Text-to-sql-v1 carries the schema in `input` and the answer in `response`.
    return _sql(row.get("instruction"), row.get("input"),
                row.get("response") or row.get("output") or row.get("answer"))


def r_code(row, unknown):
    q, a = (row.get("query") or "").strip(), (row.get("answer") or "").strip()
    if not q or not a:
        raise Unrenderable("empty_prompt" if not q else "empty_answer")
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


class PromptOnly(Exception):
    """Carries a `.messages` list that ends on a user turn: a pool row, not an SFT row."""

    def __init__(self, messages):
        Exception.__init__(self, "prompt_only")
        self.messages = messages


def r_antidoom(row, unknown):
    """The general-quality anchor. Kept verbatim: rewriting it would defeat its purpose.

    The scouting pass described this corpus as a prompt set whose responses are meant
    to be generated rather than shipped, and the s4.3 adapter does read assistant turns
    off it, so which of the two it is has to be settled by looking. A row that ends on a
    user turn is routed to the on-policy prompt pool through `PromptOnly` and counted
    there; a row with a real assistant turn is kept as ordinary SFT data.
    """
    msgs, saw_user, saw_asst = [], False, False
    for frm, val in _turns(row):
        role = _ROLE.get(frm)
        s = (val or "").strip()
        if not role or not s:
            continue
        if role == "user":
            saw_user = True
        elif role == "assistant":
            saw_asst = True
        msgs.append({"role": role, "content": s})
    if not saw_user:
        raise Unrenderable("no_user_turn")
    if not saw_asst:
        raise PromptOnly(msgs)
    while msgs and msgs[-1]["role"] != "assistant":
        msgs.pop()
    return msgs


RENDER = {
    "toolace": r_toolace, "apigen": r_apigen,
    "hermes_fc": r_hermes, "hermes_fc_st": r_hermes, "hermes_glaive": r_hermes,
    "hermes_json_ag": r_hermes, "hermes_json_st": r_hermes,
    "sql_ctx": r_sql_ctx, "sql_clinton": r_clinton_sql,
    "codefeedback": r_code, "antidoom": r_antidoom,
}
