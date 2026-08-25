"""Local diagnosis of the decontamination rule. NOT a result, and it carries no job id.

Run in the orchestration sandbox against downloaded copies of probes.jsonl and the held-out
split, to decide whether the request-surface rule was measuring contamination or boilerplate.
It answered that question: the old rule flagged 156 of 156 candidate rows, the new one flags 0.
The citable counters are the ones the generator job itself logs over the training split.

    lab storage download tidepool/s4.4/probes/probes.jsonl /tmp/probes.jsonl
    lab storage download tidepool/s4.4/test.jsonl.gz /tmp/test.jsonl.gz
    python3 decon_check.py
"""

import collections, gzip, hashlib, json, re, sys

_P = re.compile(r"[^\w\s]+"); _W = re.compile(r"\s+"); N = 13
def toks(t): return _W.sub(" ", _P.sub(" ", (t or "").lower())).split()
def grams(t, n=N):
    k = toks(t)
    return [" ".join(k[i:i+n]) for i in range(len(k)-n+1)]
def rtext(m): return "\n".join(x.get("content") or "" for x in m if x.get("role") in ("system","user"))
def utext(m): return "\n".join(x.get("content") or "" for x in m if x.get("role") == "user")

probes = [json.loads(l) for l in open("/tmp/probes.jsonl") if l.strip()]
print("probe items", len(probes), "| arms", dict(collections.Counter(p.get("arm") for p in probes)))
req = collections.Counter(); usr = collections.Counter()
for p in probes:
    req.update(set(grams(rtext(p["messages"]))))
    usr.update(set(grams(utext(p["messages"]))))
print("request grams", len(req), "| question grams", len(usr))
print("\ntop request grams by document frequency:")
for g, c in req.most_common(6):
    print("  %4d  %s" % (c, g[:110]))

# corpus side: the held-out split, only rows with a tool turn (what the generator harvests)
_RESP = re.compile(r"^<tool_response>(.*)</tool_response>$", re.S)
hits_req = hits_usr = scanned = tool_rows = 0
examples = []
rq = set(req); uq = {g for g, c in usr.items() if c <= max(1, int(0.50 * len(probes)))}
print("\nquestion grams after df<=%d filter: %d (of %d)" % (max(1,int(.5*len(probes))), len(uq), len(usr)))
for line in gzip.open("/tmp/test.jsonl.gz/test.jsonl.gz", "rt"):
    r = json.loads(line); scanned += 1
    msgs = r.get("messages") or []
    if not any(m.get("role") == "tool" and _RESP.match((m.get("content") or "").strip()) for m in msgs):
        continue
    tool_rows += 1
    a = rq.intersection(grams(rtext(msgs)))
    b = uq.intersection(grams(utext(msgs)))
    if a:
        hits_req += 1
        if len(examples) < 3: examples.append(sorted(a)[0])
    if b: hits_usr += 1
print("\nheld-out rows scanned %d, with a tool return %d" % (scanned, tool_rows))
print("  collide on the OLD rule (system+user, any shared gram): %d (%.1f%%)"
      % (hits_req, 100.0*hits_req/max(1,tool_rows)))
print("  collide on the NEW rule (user only, df-filtered):       %d (%.1f%%)"
      % (hits_usr, 100.0*hits_usr/max(1,tool_rows)))
print("\nexample colliding grams:")
for e in examples: print("  ", e[:130])
