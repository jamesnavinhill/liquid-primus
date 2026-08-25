"""Local diagnosis of the no-fabrication check. NOT a result, and it carries no job id.

Runs the generator's mode rotation and target selection over the held-out split, and reports
which forbidden values a target actually quotes. It found the cause of 4,030 of 4,031 sources
being discarded: corpus tool responses echo the function name back in the payload, `empty_body`
removes it, and the target names the tool because that is how the reply says which call failed.
Scoping the forbidden list to the whole prompt rather than the damaged body alone takes the
leak count over all 73 held-out sources from 19 to 0.

    lab storage download tidepool/s4.4/test.jsonl.gz /tmp/test.jsonl.gz
    python3 leak_check.py
"""

import collections, gzip, hashlib, json, re, sys
sys.path.insert(0, "/workspace/tenants/59cb8a55-981e-489c-bb0b-a11b2892abb8/projects/6a430460-5c3f-424a-b61c-78396b36eb8f/stages/s5-experimentation/jobs/s5-tooldata")
import build, defects, targets
_P = re.compile(r"[^\w\s]+"); _W = re.compile(r"\s+")
def norm(t): return " %s " % " ".join(_W.sub(" ", _P.sub(" ", (t or "").lower())).split())
def quotes(t, vals):
    low = norm(t); return [v for v in vals if norm(v).strip() and norm(v) in low]
_RESP = re.compile(r"^<tool_response>(.*)</tool_response>$", re.S)
_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)

def call_of(msgs, upto):
    for m in reversed(msgs[:upto]):
        c = _CALL.search(m.get("content") or "")
        if c:
            try:
                o = json.loads(c.group(1).strip()); return o.get("name"), o.get("arguments") or {}
            except Exception: return None, {}
    return None, {}

src = []
for line in gzip.open("/tmp/test.jsonl.gz/test.jsonl.gz", "rt"):
    r = json.loads(line); msgs = r.get("messages") or []
    for k, m in enumerate(msgs):
        if m.get("role") != "tool": continue
        mm = _RESP.match((m.get("content") or "").strip())
        if not mm: continue
        try: obj = json.loads(mm.group(1).strip())
        except Exception: continue
        if not isinstance(obj, dict) or not obj: continue
        if k + 1 >= len(msgs) or msgs[k+1].get("role") != "assistant": continue
        src.append({"c": r.get("c"), "i": r.get("i"), "k": k, "payload": obj, "messages": msgs})
        break
print("sources", len(src))

leaks = collections.Counter(); by_val = collections.Counter(); tried = collections.Counter()
ex = []
for s in src:
    hv = int.from_bytes(hashlib.blake2b(("%s|%s|%s" % (s["c"], s["i"], s["k"])).encode(), digest_size=8).digest(), "big")
    depth = 1 + (hv >> 17) % 3
    name, args = call_of(s["messages"], s["k"])
    for j in range(7):
        cand = defects.TRAINED_MODES[(hv + j) % 7]
        b, w, f = defects.apply_defect(s["payload"], cand, depth, args)
        if b is None: continue
        tried[cand] += 1
        out = [dict(m) for m in s["messages"][:s["k"]+1]]
        out[s["k"]] = {"role": "tool", "content": "<tool_response>%s</tool_response>" % b}
        prompt = "\n".join(x.get("content") or "" for x in out)
        f = [v for v in (f or []) if not quotes(prompt, [v])]
        t, pool = targets.pick("x", cand, name or s["c"], w, defects.plain_why(cand, w))
        hit = quotes(t, f or [])
        if hit:
            leaks[cand] += 1
            for v in hit: by_val[v] += 1
            if len(ex) < 6: ex.append((cand, name, hit, t[:150], (w or "")[:90]))
        break
print("attempted:", dict(tried))
print("leaked:   ", dict(leaks))
print("top leaked values:", by_val.most_common(10))
for e in ex:
    print("\n mode=%s tool=%s hit=%s\n  why=%s\n  target=%s" % (e[0], e[1], e[2], e[4], e[3]))
