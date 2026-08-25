"""Fixture check for the two bugs that made attempt 1 write nothing. 23 checks, no job needed.

Pure functions over the real probe bank: the identifying-value rule, the token-run matcher, the
forbidden-list filter, the old and new request rules against a corpus-shaped row, the mode
rotation, and the assertion that no target quotes a value its damaged response no longer carries.
A unit test, not a measurement: nothing it prints belongs in a report.

    python3 fixture_test.py     # from this directory
"""
import collections, hashlib, json, re, sys
sys.path.insert(0, ".")
import build, defects
import bank_tools

_PUNCT = re.compile(r"[^\w\s]+")
_WS = re.compile(r"\s+")
N = 13

def grams(text, n=N):
    toks = _WS.sub(" ", _PUNCT.sub(" ", (text or "").lower())).split()
    return [int.from_bytes(hashlib.blake2b(" ".join(toks[i:i+n]).encode(), digest_size=8).digest(), "big")
            for i in range(len(toks) - n + 1)]

def norm(t):
    return " %s " % " ".join(_WS.sub(" ", _PUNCT.sub(" ", (t or "").lower())).split())

def quotes(text, values):
    low = norm(text)
    return [v for v in (values or ()) if norm(v).strip() and norm(v) in low]

def utext(msgs):
    return "\n".join(m.get("content") or "" for m in msgs if m.get("role") == "user")

def rtext(msgs):
    return "\n".join(m.get("content") or "" for m in msgs if m.get("role") in ("system", "user"))

ok = fail = 0
def check(label, cond):
    global ok, fail
    if cond: ok += 1
    else:
        fail += 1
        print("  FAIL:", label)

# 1 --- identifying
for s, want in [("1", False), ("47", False), ("0", False), ("ok", False), ("USD", False),
                ("2019", True), ("acct_00229", True), ("12.5", True), ("1.5", False), ("1234.5", True),
                ("Ada Lovelace", True)]:
    check("identifying(%r)==%s" % (s, want), defects.identifying(s) is want)

# 2 --- quotes is token-bounded
check("bare 0 no longer leaks off a path", quotes("the value at result.0.value is missing",
      [v for v in ["0"] if defects.identifying(v)]) == [])
check("ok does not match broken", quotes("the response is broken", ["ok"]) == [])
check("id still matches", quotes("reports acct_00229 at account.id", ["acct_00229"]) == ["acct_00229"])
check("2019 still matches", quotes("a cache entry from 2019", ["2019"]) == ["2019"])

# 3 --- forbidden() drops non-identifying leaves
pay = {"count": 1, "status": "ok", "account": {"id": "acct_00229", "balance": 4821.55}}
body, why = build.corrupt(build.wrap(pay, 1), "empty_body", 1)
fb = defects.forbidden(pay, body)
check("forbidden excludes small ints and short strings", all(defects.identifying(v) for v in fb))
check("forbidden keeps the id", "acct_00229" in fb)

# 4 --- the real probe set's request surface vs its question surface
items = build.build_tools(bank_tools.SCENARIOS)
req_all = set()
for it in items:
    req_all.update(grams(rtext(it["messages"])))
df = collections.Counter()
for it in items:
    df.update(set(grams(utext(it["messages"]))))
cut = max(1, int(0.50 * len(items)))
qg = {g for g, c in df.items() if c <= cut}
print("  probe items %d | request grams %d | question grams %d (of %d, %d boilerplate)"
      % (len(items), len(req_all), len(qg), len(df), len(df) - len(qg)))
check("question grams are a small identifying set", 0 < len(qg) <= len(req_all))

# A corpus-shaped row: generic hermes system prompt with a JSON schema, unrelated question.
schema = json.dumps([{"type": "function", "function": {"name": "get_weather", "parameters":
        {"type": "object", "properties": {"city": {"type": "string", "description": "the city"},
         "unit": {"type": "string", "description": "the unit"}}, "required": ["city"]}}}], indent=1, sort_keys=True)
# The real collision, measured on the held-out split: the probe system prompt was written
# at s4 to match the corpus's own rendering convention, so its preamble matches verbatim in
# 100% of corpus rows carrying a tool return. The rule read a deliberate match as leakage.
corpus_sys = ("You are a function-calling assistant. You are given a set of tools inside "
              "<tools></tools>. When a tool applies, reply with one <tool_call></tool_call> "
              "block per call, containing a JSON object with \"name\" and \"arguments\". When a "
              "tool's response does not answer the question, say so; do not report a value "
              "the response does not contain.\n<tools>\n%s\n</tools>" % schema)
corpus_msgs = [{"role": "system", "content": corpus_sys},
               {"role": "user", "content": "What is the weather in Lisbon right now?"}]
check("old rule collides on schema boilerplate", bool(req_all.intersection(grams(rtext(corpus_msgs)))))
check("new rule does not", not qg.intersection(grams(utext(corpus_msgs))))

# 5 --- mode rotation covers a payload wrong_entity cannot touch
no_echo = {"temperature_c": 18.4, "conditions": "light rain", "station": "LPPT"}
applied = collections.Counter()
for i in range(210):
    hv = int.from_bytes(hashlib.blake2b(("row%d" % i).encode(), digest_size=8).digest(), "big")
    depth = 1 + (hv >> 17) % 3
    got = None
    for cand in [defects.TRAINED_MODES[(hv + j) % len(defects.TRAINED_MODES)]
                 for j in range(len(defects.TRAINED_MODES))]:
        b, w, f = defects.apply_defect(no_echo, cand, depth, {"city": "Lisbon"})
        if b is not None:
            got = cand
            break
    applied[got] += 1
print("  rotation over 210 sources with no echoing leaf:", dict(applied))
check("every source got a mode", applied[None] == 0)
check("at least 5 families present", len([k for k in applied if k]) >= 5)

# 6 --- targets never quote a forbidden value once identifying+token rules hold
import targets
leaks = 0
for i, scen in enumerate(bank_tools.SCENARIOS):
    for mode in defects.TRAINED_MODES:
        for depth in (1, 2, 3):
            b, w, f = defects.apply_defect(scen["payload"], mode, depth, scen["args"])
            if b is None: continue
            t, pool = targets.pick("fx%d" % i, mode, scen["tool"]["name"], w, defects.plain_why(mode, w))
            if quotes(t, f): leaks += 1
print("  target leaks over %d scenario/mode/depth cells: %d" % (len(bank_tools.SCENARIOS) * 7 * 3, leaks))
check("no target quotes a forbidden value", leaks == 0)

# 7 --- the prompt-scoped forbidden filter, which is what killed empty_body
# Corpus tool responses routinely echo the function name back in the payload. `empty_body`
# removes it, so it lands on the forbidden list, and the target interpolates the same name
# because naming the tool is how the reply says which call failed. Measured on the held-out
# split: 12 of 12 empty_body sources leaked this way, and 4,030 of 4,031 were discarded.
pay2 = {"name": "get_current_weather", "weather": "overcast", "temperature_c": 11.2,
        "station_name": "Lisbon Portela"}
b2, w2, f2 = defects.apply_defect(pay2, "empty_body", 1, {"city": "Lisbon"})
t2, _ = targets.pick("fx-tool", "empty_body", "get_current_weather", w2, defects.plain_why("empty_body", w2))
check("unfiltered list reads the tool name as a leak", quotes(t2, f2) != [])
prompt = ("You are a function-calling assistant.\n"
          "What is the weather at Lisbon Portela right now?\n"
          "<tool_call>{\"name\": \"get_current_weather\", \"arguments\": {\"city\": \"Lisbon\"}}</tool_call>\n"
          "<tool_response>%s</tool_response>" % b2)
f2s = [v for v in f2 if not quotes(prompt, [v])]
check("prompt-scoped list clears the leak", quotes(t2, f2s) == [])
check("prompt-scoped list still forbids a value only the payload had",
      "overcast" in f2s and "get_current_weather" not in f2s)

print("%d checks passed, %d failed" % (ok, fail))
sys.exit(1 if fail else 0)
